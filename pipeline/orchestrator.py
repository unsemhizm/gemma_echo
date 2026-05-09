import sys
import os
import time
import subprocess
from core.logger import get_logger
from stt.transcriber import Transcriber
from llm.translator import Translator
from tts.synthesizer import Synthesizer

log = get_logger(__name__)


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
        log.info("=" * 50)
        log.info("GEMMA ECHO ORKESTRA ŞEFI v8 BAŞLATILIYOR")
        log.info("=" * 50)

        self.transcriber = transcriber
        self.translator = translator
        self.synthesizer = synthesizer
        self.config = config
        self.current_mode = None  # set_mode icinde ayarlanacak
        self.history = []  # Kayan Bellek — son 3 Turkce cumle (zamir cozumu icin)
        self.result_queue = None  # GUI koprusu — set edilirse her process() sonucu buraya gider

        # Baslangic modunu konfigure et
        self.set_mode(initial_mode)

        log.info("Orkestra Şefi hazır!")

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
        """API bağlantılarını ve GPU CUDA kernellarını ısıtır.
        İlk process() çağrısının hızlı olması için başlangıçta çağrılır."""
        log.info("Modeller ısıtılıyor...")

        # 1. STT GPU ısıtma
        try:
            self.transcriber.warm_up()
        except Exception:
            log.warning("STT ısıtma hatası (kritik değil).", exc_info=True)

        # 2. LLM API ısıtma
        try:
            self.translator.translate("Merhaba")
        except Exception:
            log.warning("LLM ısıtma hatası (kritik değil).", exc_info=True)

        log.info("Isıtma tamamlandı.")

    def _handle_llm_vram_failure(self):
        """Yerel GGUF VRAM'e sığmadı; çeviriyi bulut motoruna kaydır."""
        log.error("Yerel LLM yüklenemedi (VRAM). Çeviri motoru buluta kaydırılıyor.")
        self.translator.set_mode("online")
        self.translator.unload_local_model()

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
            log.debug(f"Zaten '{mode}' modunda, geçiş atlanıyor.")
            return

        log.info(f"Mod geçişi: {old_mode or 'INIT'} -> {mode}")

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
        log.info(f"Mod geçişi tamamlandı: {mode}")
        self._print_vram()

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
        if not self.translator.load_local_model():
            self._handle_llm_vram_failure()
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
            if not self.translator.load_local_model():
                self._handle_llm_vram_failure()
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
        log.info(f"Ses işleniyor ({self.current_mode}): {audio_path}")
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
                self.translator.set_persona(self.config.get("persona", default="default"))

            # 1. STT (Speech-to-Text)
            stt_start = time.time()
            stt_result = self.transcriber.transcribe(audio_path, source_lang=src_lang)

            # Gurultu kontrolu (esik 0.4)
            if stt_result.get("no_speech_prob", 0) > 0.4:
                log.debug("Gürültü algılandı, çeviri iptal edildi.")
                return

            text_tr = stt_result.get("text", "")
            if not text_tr:
                log.debug("Boş metin döndü, çeviri iptal edildi.")
                return

            # Akilli Noktalama Filtresi — oksuruk / yutkunma / nefes false-positive engeli
            _words = text_tr.strip().split()
            _has_punct = text_tr.strip()[-1] in ".!?" if text_tr.strip() else False
            if len(_words) <= 2 and not _has_punct:
                log.debug(f"Kısa metin ('{text_tr}') — false-positive, iptal edildi.")
                return

            stt_ms = int((time.time() - stt_start) * 1000)
            log.info(f"STT: {stt_ms}ms | '{text_tr}'")

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
                raise RuntimeError("Tüm online LLM katmanları başarısız (bağlantı hatası?).")

            log.info(f"Çeviri: '{text_en}' | Motor: {llm_result.get('engine')} | {llm_ms}ms")

            # Kayan Bellegi guncelle — son 3 Turkce cumleyi tut
            self.history.append(text_tr)
            self.history = self.history[-3:]

            # 3. TTS (Sentez)
            tts_ms = self.synthesizer.speak(text_en, language=tgt_lang) or 0

            total_ms = int((time.time() - total_start) * 1000)
            log.info(f"E2E: {total_ms}ms (STT:{stt_ms} + LLM:{llm_ms} + TTS:{tts_ms})")

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

        except Exception:
            log.error("ORCHESTRATOR işlem hatası — fallback tetikleniyor.", exc_info=True)
            if self.result_queue is not None:
                import traceback
                self.result_queue.put({"error": traceback.format_exc()})
            self._fallback(audio_path, src_lang=src_lang, tgt_lang=tgt_lang,
                           src_name=src_name, tgt_name=tgt_name)

    # ═══════════════════════════════════════════════════════════
    # INBOUND — Karsi Taraftan Gelen Sesi Cevir (TTS yok)
    # ═══════════════════════════════════════════════════════════

    def process_inbound(self, audio_path: str):
        """
        Karşı taraftan gelen sesi kullanıcının diline çevir.
        TTS çalmıyor — sadece Overlay'e gönderiyor.
        Dil yönü process() ile tersinedir: target_lang -> source_lang
        """
        log.info(f"[Inbound] İşleniyor: {audio_path}")
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
                log.debug("[Inbound] Gürültü algılandı, işleme iptal.")
                return

            text_foreign = stt_result.get("text", "").strip()
            if not text_foreign:
                log.debug("[Inbound] Boş metin, işleme iptal.")
                return

            _words = text_foreign.split()
            _has_punct = text_foreign[-1] in ".!?" if text_foreign else False
            if len(_words) <= 2 and not _has_punct:
                log.debug(f"[Inbound] Kısa metin ('{text_foreign}') — false-positive, iptal.")
                return

            stt_ms = int((time.time() - stt_start) * 1000)
            log.info(f"[Inbound] STT: {stt_ms}ms | '{text_foreign}'")

            # 2. LLM — karşı tarafın metnini kullanıcının diline çevir
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

            log.info(f"[Inbound] Çeviri: '{text_native}' | {llm_ms}ms")

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

        except Exception:
            log.error("[Inbound] İşlem hatası.", exc_info=True)
            if self.result_queue is not None:
                import traceback
                self.result_queue.put({"error": f"[Inbound] {traceback.format_exc()}"})

    # ═══════════════════════════════════════════════════════════
    # FALLBACK — Her Moddan Offline'a Güvenli Geçiş
    # ═══════════════════════════════════════════════════════════

    def _fallback(self, audio_path: str, src_lang="tr", tgt_lang="en",
                  src_name="Turkish", tgt_name="English"):
        """Herhangi bir moddan offline (survival) moduna güvenli geçiş.
        İnternet kopukluğu veya API hatalarında tetiklenir."""
        log.critical(
            f"BAĞLANTI HATASI! OFFLINE (HAYATTA KALMA) MODUNA GEÇİLİYOR "
            f"— önceki mod: {self.current_mode}"
        )

        # Offline moda gec (VRAM guvenlik adimlarini set_mode yonetir)
        self.set_mode("offline")

        log.info(f"Offline işlenecek: {audio_path}")

        try:
            stt_result = self.transcriber.transcribe(audio_path, source_lang=src_lang)
            text_tr = stt_result.get("text", "")

            if text_tr:
                if not self.translator.load_local_model():
                    self._handle_llm_vram_failure()
                llm_result = self.translator.translate(
                    text_tr,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    src_name=src_name,
                    tgt_name=tgt_name
                )
                text_en = llm_result.get("translation", "")
                log.info(f"Offline Çeviri: '{text_en}'")

                tts_ms = self.synthesizer.speak(text_en) or 0
                log.info("Offline işlem tamamlandı.")

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
        except Exception:
            log.critical(
                "Offline çeviri de başarısız oldu — tüm katmanlar çöktü!",
                exc_info=True
            )

