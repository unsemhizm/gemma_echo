import sys
import os
import time
from stt.transcriber import Transcriber
from llm.translator import Translator
from tts.synthesizer import Synthesizer


class Orchestrator:
    # ═══════════════════════════════════════════════════════════
    # 4 CALISMA MODU — VRAM BUTCE HARITASI (6GB Limit)
    # ═══════════════════════════════════════════════════════════
    #
    # online       : Whisper GPU(1.5) + Groq(0) + ElevenLabs(0) = ~1.5GB
    # offline      : Whisper CPU(0)   + Gemma(3.1) + XTTS CPU(0) = ~3.1GB
    # turbo        : Groq STT(0)     + Groq(0)    + ElevenLabs(0) = 0GB
    # hybrid_plus  : Groq STT(0)     + Groq(0)    + XTTS GPU(1.8) = ~1.8GB

    VALID_MODES = ("online", "offline", "turbo", "hybrid_plus")

    def __init__(self, transcriber: Transcriber, translator: Translator,
                 synthesizer: Synthesizer, initial_mode: str = "online"):
        """
        Gemma Echo Orkestra Sefi — v8 Quad-State.
        4 calisma modu arasinda guvenli gecis yonetimi saglar.
        """
        print("\n" + "="*50)
        print("[SISTEM] GEMMA ECHO ORKESTRA SEFI v8 BASLATILIYOR")
        print("="*50)

        self.transcriber = transcriber
        self.translator = translator
        self.synthesizer = synthesizer
        self.current_mode = None  # set_mode icinde ayarlanacak

        # Baslangic modunu konfigure et
        self.set_mode(initial_mode)

        print("[SISTEM] Orkestra Sefi hazir!")

    # ═══════════════════════════════════════════════════════════
    # MOD YONETIMI — VRAM GUVENLIK MATRISLI
    # ═══════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        """Sistemi belirtilen moda gecirir.
        VRAM guvenlik adimlarini otomatik yonetir."""
        if mode not in self.VALID_MODES:
            raise ValueError(f"Gecersiz mod: {mode}. Gecerli: {self.VALID_MODES}")

        old_mode = self.current_mode

        if old_mode == mode:
            print(f"[SISTEM] Zaten '{mode}' modunda.")
            return

        print(f"\n[SISTEM] Mod gecisi: {old_mode or 'INIT'} -> {mode}")
        print("-" * 40)

        # Hedef moda gore konfigure et
        if mode == "online":
            self._configure_online()
        elif mode == "offline":
            self._configure_offline()
        elif mode == "turbo":
            self._configure_turbo()
        elif mode == "hybrid_plus":
            self._configure_hybrid_plus()

        self.current_mode = mode
        print(f"[SISTEM] Mod gecisi tamamlandi: {mode}")
        print("-" * 40)

    # ─── MODE 1: ONLINE (Yari-Bulut) ──────────────────────────

    def _configure_online(self):
        """STT: Whisper GPU | LLM: Groq | TTS: ElevenLabs
        VRAM: ~1.5GB (sadece Whisper)"""

        # Oncelik: XTTS GPU'daysa bosalt (VRAM guvenlik)
        self.synthesizer.offload_xtts_from_gpu()

        # STT -> Yerel Whisper GPU
        self.transcriber.set_mode("local_gpu")

        # LLM -> Groq (online)
        self.translator.set_mode("online")
        self.translator.release_ollama_vram()

        # TTS -> ElevenLabs
        self.synthesizer.set_mode("online")

        # Arka planda XTTS'i CPU'ya yukle (fallback hazirlik)
        self.synthesizer.preload_xtts_background(use_gpu=False)

    # ─── MODE 2: OFFLINE (Survival / Tam Yerel) ──────────────

    def _configure_offline(self):
        """STT: Whisper CPU | LLM: Gemma GPU | TTS: XTTS CPU
        VRAM: ~3.1GB (sadece Gemma)"""

        # KRITIK SIRA: Once XTTS GPU'yu bosalt, sonra Gemma yukle
        self.synthesizer.offload_xtts_from_gpu()

        # STT -> CPU (VRAM'i Gemma'ya birak)
        self.transcriber.set_mode("local_cpu")

        # LLM -> Ollama/Gemma (offline — GPU'ya yuklenecek)
        self.translator.set_mode("offline")

        # TTS -> XTTS CPU
        self.synthesizer.set_mode("offline")

    # ─── MODE 3: TURBO (Tam Bulut / 0 VRAM) ─────────────────

    def _configure_turbo(self):
        """STT: Groq STT | LLM: Groq | TTS: ElevenLabs
        VRAM: 0GB (tam bulut)"""

        # XTTS GPU'daysa bosalt
        self.synthesizer.offload_xtts_from_gpu()

        # STT -> Groq Cloud (VRAM bosaltilir)
        self.transcriber.set_mode("cloud_deepgram")

        # LLM -> Groq (online)
        self.translator.set_mode("online")
        self.translator.release_ollama_vram()

        # TTS -> ElevenLabs
        self.synthesizer.set_mode("online")

    # ─── MODE 4: HYBRID PLUS (Yuksek Kalite + Yerel Klon) ────

    def _configure_hybrid_plus(self):
        """STT: Groq STT | LLM: Groq | TTS: XTTS GPU
        VRAM: ~1.8GB (sadece XTTS)"""

        # STT -> Groq Cloud (VRAM bosalt)
        self.transcriber.set_mode("cloud_deepgram")

        # LLM -> Groq (online — Ollama'yi bosalt)
        self.translator.set_mode("online")
        self.translator.release_ollama_vram()

        # TTS -> XTTS GPU (VRAM bos, XTTS'i GPU'ya yukle)
        self.synthesizer.set_mode("gpu")

        # XTTS henuz yuklenmemisse GPU'ya eager load
        if self.synthesizer.xtts_model is None:
            self.synthesizer.preload_xtts_background(use_gpu=True)

    # ═══════════════════════════════════════════════════════════
    # ANA ISLEM HATTI — STT -> LLM -> TTS
    # ═══════════════════════════════════════════════════════════

    def process(self, audio_path: str):
        """Uctan uca ses ceviri hatti. Hata yakalarsa _fallback() tetikler."""
        print(f"\n[ORCHESTRATOR] Ses isleniyor ({self.current_mode}): {audio_path}")
        total_start = time.time()

        try:
            # 1. STT (Speech-to-Text)
            stt_start = time.time()
            stt_result = self.transcriber.transcribe(audio_path)

            # Gurultu kontrolu
            if stt_result.get("no_speech_prob", 0) > 0.6:
                print("[ORCHESTRATOR] Gurultu algilandi, ceviri iptal edildi.")
                return

            text_tr = stt_result.get("text", "")
            if not text_tr:
                print("[ORCHESTRATOR] Bos metin dondu, ceviri iptal edildi.")
                return

            stt_ms = int((time.time() - stt_start) * 1000)
            print(f"[ORCHESTRATOR] STT Suresi: {stt_ms}ms | Metin: '{text_tr}'")

            # 2. LLM (Ceviri)
            llm_result = self.translator.translate(text_tr)
            text_en = llm_result.get("translation", "")
            llm_ms = llm_result.get("latency_ms", 0)

            print(f"[ORCHESTRATOR] Ceviri: '{text_en}' | Motor: {llm_result.get('engine')} | {llm_ms}ms")

            # 3. TTS (Sentez)
            tts_ms = self.synthesizer.speak(text_en) or 0

            total_ms = int((time.time() - total_start) * 1000)
            print(f"[ORCHESTRATOR] Islem tamamlandi. E2E: {total_ms}ms (STT:{stt_ms} + LLM:{llm_ms} + TTS:{tts_ms})")

        except Exception as e:
            print(f"\n[ORCHESTRATOR] HATA YAKALANDI: {e}")
            self._fallback(audio_path)

    # ═══════════════════════════════════════════════════════════
    # FALLBACK — Her Moddan Offline'a Guvenli Gecis
    # ═══════════════════════════════════════════════════════════

    def _fallback(self, audio_path: str):
        """Herhangi bir moddan offline (survival) moduna guvenli gecis.
        Internet kopuklugu veya API hatalarinda tetiklenir."""
        print("\n" + "!"*50)
        print("[SISTEM] BAGLANTI HATASI! OFFLINE (HAYATTA KALMA) MODUNA GECILIYOR...")
        print(f"[SISTEM] Onceki mod: {self.current_mode}")
        print("!"*50)

        # Offline moda gec (VRAM guvenlik adimlarini set_mode yonetir)
        self.set_mode("offline")

        # Islemi offline olarak tekrar dene
        print(f"[ORCHESTRATOR] Offline isleniyor: {audio_path}")

        stt_result = self.transcriber.transcribe(audio_path)
        text_tr = stt_result.get("text", "")

        if text_tr:
            llm_result = self.translator.translate(text_tr)
            text_en = llm_result.get("translation", "")
            print(f"[ORCHESTRATOR] Offline Ceviri: '{text_en}'")

            self.synthesizer.speak(text_en)
            print("[ORCHESTRATOR] Offline Islem tamamlandi.")
