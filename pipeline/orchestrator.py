import sys
import os
import time
from stt.transcriber import Transcriber
from llm.translator import Translator
from tts.synthesizer import Synthesizer


class Orchestrator:
    # ═══════════════════════════════════════════════════════════
    # 6 CALISMA MODU — VRAM BUTCE HARITASI (6GB Limit)
    # ═══════════════════════════════════════════════════════════
    VALID_MODES = (
        "online",
        "online_xtts",
        "interactive",
        "offline",
        "hybrid_cloud_io",
        "hybrid_cloud_stt",
        "online_local_stt"
    )

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
        elif mode == "online_xtts":
            self._configure_online_xtts()
        elif mode == "interactive":
            self._configure_interactive()
        elif mode == "offline":
            self._configure_offline()
        elif mode == "hybrid_cloud_io":
            self._configure_hybrid_cloud_io()
        elif mode == "hybrid_cloud_stt":
            self._configure_hybrid_cloud_stt()
        elif mode == "online_local_stt":
            self._configure_online_local_stt()

        self.current_mode = mode
        print(f"[SISTEM] Mod gecisi tamamlandi: {mode}")
        print("-" * 40)

    # ─── MODE 1: ONLINE (Tam Bulut) ───────────────────────────
    def _configure_online(self):
        """STT: Cloud Auto | LLM: Online | TTS: Online"""
        self.synthesizer.offload_xtts_from_gpu()
        self.transcriber.set_mode("cloud_auto")
        self.translator.set_mode("online")
        self.translator.release_ollama_vram()
        self.synthesizer.set_mode("online")

    # ─── MODE 2: ONLINE_XTTS (Bulut Beyin + Yerel Ses) ────────
    def _configure_online_xtts(self):
        """STT: Cloud Auto | LLM: Online | TTS: GPU"""
        self.transcriber.set_mode("cloud_auto")
        self.translator.set_mode("online")
        self.translator.release_ollama_vram()
        self.synthesizer.set_mode("gpu")
        
        if self.synthesizer.xtts_model is None:
            self.synthesizer.preload_xtts_background(use_gpu=True)

    # ─── MODE 3: INTERACTIVE (Yerel Kulak + Bulut Beyin + Yerel Ses)
    def _configure_interactive(self):
        """STT: Local GPU | LLM: Online | TTS: GPU"""
        self.transcriber.set_mode("local_gpu")
        self.translator.set_mode("online")
        self.translator.release_ollama_vram()
        self.synthesizer.set_mode("gpu")
        
        if self.synthesizer.xtts_model is None:
            self.synthesizer.preload_xtts_background(use_gpu=True)

    # ─── MODE 4: OFFLINE (Tam Yerel / Survival Modu) ──────────
    def _configure_offline(self):
        """STT: Local CPU | LLM: Offline | TTS: Offline"""
        self.synthesizer.offload_xtts_from_gpu()
        self.transcriber.set_mode("local_cpu")
        self.translator.set_mode("offline")
        self.synthesizer.set_mode("offline")

    # ─── MODE 5: HYBRID_CLOUD_IO (Bulut Kulak/Ağız + Yerel Beyin)
    def _configure_hybrid_cloud_io(self):
        """STT: Cloud Auto | LLM: Offline | TTS: Online"""
        self.synthesizer.offload_xtts_from_gpu()
        self.transcriber.set_mode("cloud_auto")
        self.translator.set_mode("offline")
        self.synthesizer.set_mode("online")

    # ─── MODE 6: HYBRID_CLOUD_STT (Bulut Kulak + Yerel Beyin/Ses)
    def _configure_hybrid_cloud_stt(self):
        """STT: Cloud Auto | LLM: Offline | TTS: Offline"""
        self.synthesizer.offload_xtts_from_gpu()
        self.transcriber.set_mode("cloud_auto")
        self.translator.set_mode("offline")
        self.synthesizer.set_mode("offline")

    # ─── MODE 7: ONLINE_LOCAL_STT (Gemma Competition Ideal Modu)
    def _configure_online_local_stt(self):
        """STT: Local GPU | LLM: Online | TTS: Online"""
        self.synthesizer.offload_xtts_from_gpu()
        self.transcriber.set_mode("local_gpu")
        self.translator.set_mode("online")
        self.translator.release_ollama_vram()
        self.synthesizer.set_mode("online")

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
