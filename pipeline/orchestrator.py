import sys
import os
import time
import subprocess
from stt.transcriber import Transcriber
from llm.translator import Translator
from tts.synthesizer import Synthesizer


class Orchestrator:
    # ═══════════════════════════════════════════════════════════
    # 8 CALISMA MODU — VRAM BUTCE HARITASI
    # ═══════════════════════════════════════════════════════════
    VALID_MODES = (
        "online",
        "online_xtts",
        "interactive",
        "interactive_hq",
        "offline",
        "offline_gpu",
        "hybrid_cloud_io",
        "hybrid_cloud_stt",
        "online_local_stt",
        "custom",
    )

    def __init__(self, transcriber: Transcriber, translator: Translator,
                 synthesizer: Synthesizer, initial_mode: str = "online", config=None):
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
        self.config = config
        self.current_mode = None  # set_mode icinde ayarlanacak
        self.history = []  # Kayan Bellek — son 3 Turkce cumle (zamir cozumu icin)
        self.result_queue = None  # GUI koprusu — set edilirse her process() sonucu buraya gider

        # Baslangic modunu konfigure et
        self.set_mode(initial_mode)

        print("[SISTEM] Orkestra Sefi hazir!")

    # ═══════════════════════════════════════════════════════════
    # VRAM MONITORU — ASCII Bar
    # ═══════════════════════════════════════════════════════════

    def _print_vram(self):
        """nvidia-smi ile anlik VRAM kullanimi ASCII bar olarak basar."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode != 0:
                return
            parts = result.stdout.strip().split(",")
            if len(parts) < 2:
                return
            used_mb = int(parts[0].strip())
            total_mb = int(parts[1].strip())
            used_gb = used_mb / 1024
            total_gb = total_mb / 1024
            ratio = used_mb / total_mb if total_mb > 0 else 0
            bar_len = 10
            filled = int(ratio * bar_len)
            bar = "#" * filled + "." * (bar_len - filled)
            print(f"[VRAM] [{bar}] {used_gb:.1f}GB / {total_gb:.1f}GB")
        except Exception:
            pass  # nvidia-smi yoksa sessizce gec

    # ═══════════════════════════════════════════════════════════
    # WARM-UP — Soguk Baslangic Isitici
    # ═══════════════════════════════════════════════════════════

    def warm_up(self):
        """API baglantilarini ve GPU CUDA kernellarini isitir.
        Ilk process() cagrisinin hizli olmasi icin baslangicta cagrilir.

        Isitma sirasi:
          1. STT GPU  — sessiz dummy inference ile CUDA kernel derleme
          2. LLM API  — Gemini/Groq baglanti ve kimlik dogrulama
        """
        print("[SISTEM] Modeller isitiliyor...")

        # 1. STT GPU isitma (sadece lokal modlarda; cloud_auto'da atlanir)
        try:
            self.transcriber.warm_up()
        except Exception as e:
            print(f"[UYARI] STT isitma hatasi (kritik degil): {e}")

        # 2. LLM API isitma (baglanti ve auth on-yukleme)
        try:
            self.translator.translate("Merhaba")
        except Exception as e:
            print(f"[UYARI] LLM isitma hatasi (kritik degil): {e}")

        print("[SISTEM] Isitma tamamlandi.")

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
        elif mode == "interactive_hq":
            self._configure_interactive_hq()
        elif mode == "offline":
            self._configure_offline()
        elif mode == "offline_gpu":
            self._configure_offline_gpu()
        elif mode == "hybrid_cloud_io":
            self._configure_hybrid_cloud_io()
        elif mode == "hybrid_cloud_stt":
            self._configure_hybrid_cloud_stt()
        elif mode == "online_local_stt":
            self._configure_online_local_stt()
        elif mode == "custom":
            self._configure_custom()

        self.current_mode = mode
        print(f"[SISTEM] Mod gecisi tamamlandi: {mode}")
        self._print_vram()
        print("-" * 40)

    # ─── MODE 1: ONLINE (Tam Bulut) ───────────────────────────
    def _configure_online(self):
        """STT: Cloud Auto | LLM: Online | TTS: Online"""
        self.synthesizer.offload_xtts_from_gpu()
        self.transcriber.set_mode("cloud_auto")
        self.translator.set_mode("online")
        self.translator.unload_local_model()
        self.synthesizer.set_mode("online")

    # ─── MODE 2: ONLINE_XTTS (Bulut Beyin + Yerel Ses) ────────
    def _configure_online_xtts(self):
        """STT: Cloud Auto | LLM: Online | TTS: GPU"""
        self.transcriber.set_mode("cloud_auto")
        self.translator.set_mode("online")
        self.translator.unload_local_model()
        self.synthesizer.set_mode("gpu")
        
        if self.synthesizer.xtts_model is None:
            self.synthesizer.preload_xtts_background(use_gpu=True)

    # ─── MODE 3: INTERACTIVE (Yerel Kulak + Bulut Beyin + Yerel Ses)
    def _configure_interactive(self):
        """STT: Local GPU | LLM: Online | TTS: GPU"""
        self.transcriber.set_mode("local_gpu")
        self.translator.set_mode("online")
        self.translator.unload_local_model()
        self.synthesizer.set_mode("gpu")
        
        if self.synthesizer.xtts_model is None:
            self.synthesizer.preload_xtts_background(use_gpu=True)

    # ─── MODE 3b: INTERACTIVE_HQ (Yerel medium Kulak + Bulut Beyin + Yerel Ses)
    def _configure_interactive_hq(self):
        """STT: Local GPU medium | LLM: Online | TTS: GPU
        Interactive modun yuksek kaliteli versiyonu.
        Whisper medium ile Turkce tanima cok daha dogru, biraz daha yavastir."""
        self.transcriber.set_mode("local_gpu_hq")
        self.translator.set_mode("online")
        self.translator.unload_local_model()
        self.synthesizer.set_mode("gpu")

        if self.synthesizer.xtts_model is None:
            self.synthesizer.preload_xtts_background(use_gpu=True)

    # ─── MODE 4: OFFLINE (Tam Yerel / Survival Modu) ──────────
    def _configure_offline(self):
        """STT: Local CPU | LLM: Offline (CPU) | TTS: Offline (CPU)"""
        self.synthesizer.offload_xtts_from_gpu()
        self.transcriber.set_mode("local_cpu")
        self.translator.set_mode("offline")
        self.synthesizer.set_mode("offline")

    # ─── MODE 5: OFFLINE_GPU (Tam Yerel Tam GPU / Yuksek VRAM) ─
    def _configure_offline_gpu(self):
        """STT: Local GPU | LLM: Offline (GPU) | TTS: GPU

        Tum bilesenleri GPU'da calistirir. Internet gerektirmez.
        Gereksinim: ~6GB+ VRAM (RTX 3060 Ti, RTX 4070, vb.)
          - Whisper small  : ~1.5 GB
          - GGUF LLM       : ~2.3 GB  (n_gpu_layers=-1, tam GPU)
          - XTTS-v2        : ~2.5 GB
        """
        self.transcriber.set_mode("local_gpu")
        self.translator.set_mode("offline")
        self.translator.load_local_model()
        self.synthesizer.set_mode("gpu")

        if self.synthesizer.xtts_model is None:
            self.synthesizer.preload_xtts_background(use_gpu=True)

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
        self.translator.unload_local_model()
        self.synthesizer.set_mode("online")

    # ─── MODE 8: CUSTOM (Kullanici Tanimli — config.json > mode > stt/llm/tts)
    def _configure_custom(self):
        """config.json > mode > stt/llm/tts degerlerini okuyarak sistemi konfigure eder.

        VRAM guvenlik sirasi:
          1. Once XTTS GPU'dan bosalt (TTS=gpu degilse) — VRAM'i temizle
          2. Sonra STT yukle (local_gpu ise Whisper VRAM'e girer)
          3. Sonra LLM yukle (offline ise GGUF VRAM'e girer)
          4. Son olarak TTS yukle (gpu ise XTTS VRAM'e girer)
        """
        if self.config is None:
            print("[UYARI] Custom mod: config nesnesi yok, 'online' varsayilani kullaniliyor.")
            self._configure_online()
            return

        stt_backend = self.config.get("mode", "stt", "backend", default="local_gpu")
        llm_backend = self.config.get("mode", "llm", "backend", default="online")
        tts_backend = self.config.get("mode", "tts", "backend", default="online")

        print(f"[CUSTOM] STT={stt_backend} | LLM={llm_backend} | TTS={tts_backend}")

        # 1. VRAM temizligi — TTS GPU'ya gitmeyecekse XTTS'i once bosalt
        if tts_backend != "gpu":
            self.synthesizer.offload_xtts_from_gpu()

        # 2. STT
        self.transcriber.set_mode(stt_backend)

        # 3. LLM
        self.translator.set_mode(llm_backend)
        if llm_backend == "offline":
            self.translator.load_local_model()
        else:
            self.translator.unload_local_model()

        # 4. TTS
        if tts_backend == "gpu":
            self.synthesizer.set_mode("gpu")
            if self.synthesizer.xtts_model is None:
                self.synthesizer.preload_xtts_background(use_gpu=True)
        elif tts_backend == "offline":
            self.synthesizer.set_mode("offline")
        else:
            self.synthesizer.set_mode("online")

    # ═══════════════════════════════════════════════════════════
    # ANA ISLEM HATTI — STT -> LLM -> TTS
    # ═══════════════════════════════════════════════════════════

    def process(self, audio_path: str):
        """Uctan uca ses ceviri hatti. Hata yakalarsa _fallback() tetikler."""
        print(f"\n[ORCHESTRATOR] Ses isleniyor ({self.current_mode}): {audio_path}")
        total_start = time.time()

        try:
            # Dil ayarlarini config'den oku
            src_lang = "tr"
            tgt_lang = "en"
            src_name = "Turkish"
            tgt_name = "English"

            if self.config:
                src_lang = self.config.get("language", "source", default="tr")
                tgt_lang = self.config.get("language", "target", default="en")
                src_name = self.config.get("language", "source_name", default="Turkish")
                tgt_name = self.config.get("language", "target_name", default="English")
                self.translator.set_persona(self.config.get("persona", default="none"))

            # 1. STT (Speech-to-Text)
            stt_start = time.time()
            stt_result = self.transcriber.transcribe(audio_path, source_lang=src_lang)

            # Gurultu kontrolu (esik 0.4: initial_prompt kaldirildiginda Whisper
            # gercek konusma skorlarini daha dogru raporluyor)
            if stt_result.get("no_speech_prob", 0) > 0.4:
                print("[ORCHESTRATOR] Gurultu algilandi, ceviri iptal edildi.")
                return

            text_tr = stt_result.get("text", "")
            if not text_tr:
                print("[ORCHESTRATOR] Bos metin dondu, ceviri iptal edildi.")
                return

            # Akilli Noktalama Filtresi — oksuruk / yutkunma / nefes false-positive engeli
            # Kural: <=2 kelime VE sonda noktalama yok → iptal
            # Istisna: sonda noktalama VARSA gonder (Evet., Tamam! vb.)
            # NOT (Streaming Modu): Streaming chunker aktifken bu esigi len(_words) <= 1
            # olarak esnet veya filtreyi tamamen devre disi birak; aksi halde tek kelimelik
            # tepkiler ("Evet", "Hayir", "Tamam") yanlis positive olarak silinebilir.
            _words = text_tr.strip().split()
            _has_punct = text_tr.strip()[-1] in ".!?" if text_tr.strip() else False
            if len(_words) <= 2 and not _has_punct:
                print(f"[ORCHESTRATOR] Kisa metin + noktalama yok ('{text_tr}') — false-positive, iptal edildi.")
                return

            stt_ms = int((time.time() - stt_start) * 1000)
            print(f"[ORCHESTRATOR] STT Suresi: {stt_ms}ms | Metin: '{text_tr}'")

            # 2. LLM (Ceviri) — Kayan Bellek (context) ile
            llm_result = self.translator.translate(
                text_tr, 
                context=self.history,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_name=src_name,
                tgt_name=tgt_name
            )
            text_en = llm_result.get("translation", "")
            llm_ms = llm_result.get("latency_ms", 0)

            # Tum online motorlar basarisiz oldu → _fallback() tetikle
            if llm_result.get("engine") == "Failed":
                raise RuntimeError("Tum online LLM katmanlari basarisiz (baglanti hatasi?).")

            print(f"[ORCHESTRATOR] Ceviri: '{text_en}' | Motor: {llm_result.get('engine')} | {llm_ms}ms")

            # Kayan Bellegi guncelle — son 3 Turkce cumleyi tut
            self.history.append(text_tr)
            self.history = self.history[-3:]

            # 3. TTS (Sentez)
            tts_ms = self.synthesizer.speak(text_en, language=tgt_lang) or 0

            total_ms = int((time.time() - total_start) * 1000)
            print(f"[ORCHESTRATOR] Islem tamamlandi. E2E: {total_ms}ms (STT:{stt_ms} + LLM:{llm_ms} + TTS:{tts_ms})")

            if self.result_queue is not None:
                self.result_queue.put({
                    "text_tr": text_tr,
                    "text_en": text_en,
                    "engine": llm_result.get("engine"),
                    "latency_ms": total_ms,
                    "stt_ms": stt_ms,
                    "llm_ms": llm_ms,
                    "tts_ms": tts_ms,
                    "error": None,
                })

        except Exception as e:
            print(f"\n[ORCHESTRATOR] HATA YAKALANDI: {e}")
            if self.result_queue is not None:
                self.result_queue.put({"error": str(e)})
            self._fallback(audio_path, src_lang=src_lang, tgt_lang=tgt_lang, 
                           src_name=src_name, tgt_name=tgt_name)

    # ═══════════════════════════════════════════════════════════
    # INBOUND — Karsi Taraftan Gelen Sesi Cevir (TTS yok)
    # ═══════════════════════════════════════════════════════════

    def process_inbound(self, audio_path: str):
        """
        Karsı taraftan gelen sesi kullanicinin diline cevir.
        TTS calmiyor — sadece Overlay'e gonderiyor.
        Dil yonu process() ile tersinedir: target_lang -> source_lang
        """
        print(f"\n[INBOUND] Karsi taraf isleniyor: {audio_path}")
        total_start = time.time()

        try:
            # Dil ayarlarini oku — process() ile ayni config ama ters yon
            src_lang = "en"
            tgt_lang = "tr"
            src_name = "English"
            tgt_name = "Turkish"

            if self.config:
                src_lang = self.config.get("language", "target", default="en")
                tgt_lang = self.config.get("language", "source", default="tr")
                src_name = self.config.get("language", "target_name", default="English")
                tgt_name = self.config.get("language", "source_name", default="Turkish")

            # 1. STT — karsi tarafin sesini metne cevir
            stt_start = time.time()
            stt_result = self.transcriber.transcribe(audio_path, source_lang=src_lang)

            if stt_result.get("no_speech_prob", 0) > 0.4:
                print("[INBOUND] Gurultu algilandi, isleme iptal.")
                return

            text_foreign = stt_result.get("text", "").strip()
            if not text_foreign:
                print("[INBOUND] Bos metin, isleme iptal.")
                return

            _words = text_foreign.split()
            _has_punct = text_foreign[-1] in ".!?" if text_foreign else False
            if len(_words) <= 2 and not _has_punct:
                print(f"[INBOUND] Kisa metin ('{text_foreign}') — false-positive, iptal.")
                return

            stt_ms = int((time.time() - stt_start) * 1000)
            print(f"[INBOUND] STT: {stt_ms}ms | '{text_foreign}'")

            # 2. LLM — karsi tarafin metnini kullanicinin diline cevir
            llm_result = self.translator.translate(
                text_foreign,
                context=[],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_name=src_name,
                tgt_name=tgt_name,
            )
            text_native = llm_result.get("translation", "")
            llm_ms = llm_result.get("latency_ms", 0)

            print(f"[INBOUND] Ceviri: '{text_native}' | {llm_ms}ms")

            total_ms = int((time.time() - total_start) * 1000)

            # 3. TTS YOK — overlay'e "inbound" flag'iyle gonder
            if self.result_queue is not None:
                self.result_queue.put({
                    "direction":  "inbound",
                    "text_tr":    text_native,   # cevrilen (yerli dil) metin
                    "text_en":    text_foreign,  # orijinal (yabanci) metin
                    "engine":     llm_result.get("engine"),
                    "latency_ms": total_ms,
                    "stt_ms":     stt_ms,
                    "llm_ms":     llm_ms,
                    "tts_ms":     0,
                    "error":      None,
                })

        except Exception as e:
            print(f"[INBOUND HATA] {e}")
            if self.result_queue is not None:
                self.result_queue.put({"error": f"[Inbound] {e}"})

    # ═══════════════════════════════════════════════════════════
    # FALLBACK — Her Moddan Offline'a Guvenli Gecis
    # ═══════════════════════════════════════════════════════════

    def _fallback(self, audio_path: str, src_lang="tr", tgt_lang="en", 
                  src_name="Turkish", tgt_name="English"):
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

        try:
            stt_result = self.transcriber.transcribe(audio_path, source_lang=src_lang)
            text_tr = stt_result.get("text", "")

            if text_tr:
                # Yerel model lazily VRAM'e yukle (online moddan geliyorsa yuklu olmayabilir)
                self.translator.load_local_model()
                llm_result = self.translator.translate(
                    text_tr, 
                    src_lang=src_lang, 
                    tgt_lang=tgt_lang,
                    src_name=src_name,
                    tgt_name=tgt_name
                )
                text_en = llm_result.get("translation", "")
                print(f"[ORCHESTRATOR] Offline Ceviri: '{text_en}'")

                tts_ms = self.synthesizer.speak(text_en) or 0
                print("[ORCHESTRATOR] Offline Islem tamamlandi.")

                if self.result_queue is not None:
                    self.result_queue.put({
                        "text_tr": text_tr,
                        "text_en": text_en,
                        "engine": llm_result.get("engine", "offline_fallback"),
                        "latency_ms": None,  # fallback'te E2E suresi olculmez
                        "stt_ms": None,
                        "llm_ms": llm_result.get("latency_ms"),
                        "tts_ms": tts_ms,
                        "error": None,
                    })
        except Exception as e:
            print(f"[KRITIK HATA] Offline ceviri de basarisiz oldu: {e}")
            print("[SISTEM] Yerel model veya XTTS basarisiz. Dongu devam ediyor.")
