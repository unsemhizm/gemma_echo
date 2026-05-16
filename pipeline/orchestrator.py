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
    # 8 RUNTIME MODES — VRAM BUDGET MAP
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
        Gemma Echo orchestration engine — v8 Quad-State.

        Owns the safe transitions between runtime modes and the associated
        VRAM bookkeeping.
        """
        log.info("=" * 50)
        log.info("GEMMA ECHO ORCHESTRATION ENGINE v8 INITIALIZING")
        log.info("=" * 50)

        self.transcriber = transcriber
        self.translator = translator
        self.synthesizer = synthesizer
        self.config = config
        self.current_mode = None  # Assigned inside set_mode().
        self.history = []  # Rolling memory — last 3 source-language sentences (used for pronoun resolution).
        self.result_queue = None  # GUI bridge — when set, every process() result is pushed here.

        # Apply the requested startup mode.
        self.set_mode(initial_mode)

        log.info("Orchestration engine ready.")

    # ═══════════════════════════════════════════════════════════
    # VRAM MONITOR — ASCII bar
    # ═══════════════════════════════════════════════════════════

    def _print_vram(self):
        """Render the current GPU VRAM usage as an ASCII bar via nvidia-smi."""
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
            pass  # nvidia-smi unavailable — silently skip.

    # ═══════════════════════════════════════════════════════════
    # WARM-UP — Cold-start primer
    # ═══════════════════════════════════════════════════════════

    def warm_up(self):
        """Warm up the API clients and the GPU CUDA kernels.

        Called immediately after bootstrap so the first ``process()`` invocation
        runs at steady-state latency rather than paying the cold-start cost.
        """
        log.info("Warming up models...")

        # 1. STT GPU warm-up.
        try:
            self.transcriber.warm_up()
        except Exception:
            log.warning("STT warm-up failed (non-critical).", exc_info=True)

        # 2. LLM API warm-up.
        try:
            self.translator.translate("Merhaba")
        except Exception:
            log.warning("LLM warm-up failed (non-critical).", exc_info=True)

        log.info("Warm-up complete.")

    def _handle_llm_vram_failure(self):
        """The local GGUF would not fit in VRAM; route translation to the cloud engine instead."""
        log.error("Local LLM failed to load (VRAM exhausted). Routing translation to the cloud engine.")
        self.translator.set_mode("online")
        self.translator.unload_local_model()

    # ═══════════════════════════════════════════════════════════
    # MODE MANAGEMENT — VRAM-safe transition matrix
    # ═══════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        """Transition the system to the requested mode.

        VRAM bookkeeping (eviction order, load order) is fully automated.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Allowed: {self.VALID_MODES}")

        old_mode = self.current_mode

        if old_mode == mode and mode != "custom":
            log.debug(f"Already in '{mode}' mode, skipping transition.")
            return

        log.info(f"Mode transition: {old_mode or 'INIT'} -> {mode}")

        # Dispatch to the appropriate configuration routine.
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
        log.info(f"Mode transition complete: {mode}")
        self._print_vram()

    # ─── MODE 1: ONLINE (fully cloud) ─────────────────────────
    def _configure_online(self):
        """STT: Cloud Auto | LLM: Online | TTS: Online"""
        self.synthesizer.offload_xtts_from_gpu()
        self.transcriber.set_mode("cloud_auto")
        self.translator.set_mode("online")
        self.translator.unload_local_model()
        self.synthesizer.set_mode("online")

    # ─── MODE 2: ONLINE_XTTS (cloud brain + local voice) ──────
    def _configure_online_xtts(self):
        """STT: Cloud Auto | LLM: Online | TTS: GPU"""
        self.transcriber.set_mode("cloud_auto")
        self.translator.set_mode("online")
        self.translator.unload_local_model()
        self.synthesizer.set_mode("gpu")

        if self.synthesizer.xtts_model is None:
            self.synthesizer.preload_xtts_background(use_gpu=True)

    # ─── MODE 3: INTERACTIVE (local ear + cloud brain + local voice) ────
    def _configure_interactive(self):
        """STT: Local GPU | LLM: Online | TTS: GPU"""
        self.transcriber.set_mode("local_gpu")
        self.translator.set_mode("online")
        self.translator.unload_local_model()
        self.synthesizer.set_mode("gpu")

        if self.synthesizer.xtts_model is None:
            self.synthesizer.preload_xtts_background(use_gpu=True)

    # ─── MODE 3b: INTERACTIVE_HQ (Whisper medium ear + cloud brain + local voice) ───
    def _configure_interactive_hq(self):
        """STT: Local GPU medium | LLM: Online | TTS: GPU

        High-quality variant of the interactive mode. Whisper 'medium' yields
        significantly more accurate Turkish recognition at a slightly higher
        latency cost.
        """
        self.transcriber.set_mode("local_gpu_hq")
        self.translator.set_mode("online")
        self.translator.unload_local_model()
        self.synthesizer.set_mode("gpu")

        if self.synthesizer.xtts_model is None:
            self.synthesizer.preload_xtts_background(use_gpu=True)

    # ─── MODE 4: OFFLINE (fully local / survival mode) ────────
    def _configure_offline(self):
        """STT: Local CPU | LLM: Offline (CPU) | TTS: Offline (CPU)"""
        self.synthesizer.offload_xtts_from_gpu()
        self.transcriber.set_mode("local_cpu")
        self.translator.set_mode("offline")
        self.synthesizer.set_mode("offline")

    # ─── MODE 5: OFFLINE_GPU (fully local + fully GPU / high VRAM) ─
    def _configure_offline_gpu(self):
        """STT: Local GPU | LLM: Offline (GPU) | TTS: GPU

        Runs every component on the GPU. No internet required.
        Hardware budget: ~6 GB+ VRAM (RTX 3060 Ti, RTX 4070, etc.)
          - Whisper small  : ~1.5 GB
          - GGUF LLM       : ~2.3 GB  (n_gpu_layers=-1, full GPU)
          - XTTS-v2        : ~2.5 GB
        """
        self.transcriber.set_mode("local_gpu")
        self.translator.set_mode("offline")
        if not self.translator.load_local_model():
            self._handle_llm_vram_failure()
        self.synthesizer.set_mode("gpu")

        if self.synthesizer.xtts_model is None:
            self.synthesizer.preload_xtts_background(use_gpu=True)

    # ─── MODE 5: HYBRID_CLOUD_IO (cloud ear + cloud voice + local brain)
    def _configure_hybrid_cloud_io(self):
        """STT: Cloud Auto | LLM: Offline | TTS: Online"""
        self.synthesizer.offload_xtts_from_gpu()
        self.transcriber.set_mode("cloud_auto")
        self.translator.set_mode("offline")
        self.synthesizer.set_mode("online")

    # ─── MODE 6: HYBRID_CLOUD_STT (cloud ear + local brain + local voice)
    def _configure_hybrid_cloud_stt(self):
        """STT: Cloud Auto | LLM: Offline | TTS: Offline"""
        self.synthesizer.offload_xtts_from_gpu()
        self.transcriber.set_mode("cloud_auto")
        self.translator.set_mode("offline")
        self.synthesizer.set_mode("offline")

    # ─── MODE 7: ONLINE_LOCAL_STT (ideal Gemma-competition mode)
    def _configure_online_local_stt(self):
        """STT: Local GPU | LLM: Online | TTS: Online"""
        self.synthesizer.offload_xtts_from_gpu()
        self.transcriber.set_mode("local_gpu")
        self.translator.set_mode("online")
        self.translator.unload_local_model()
        self.synthesizer.set_mode("online")

    # ─── MODE 8: CUSTOM (user-defined — config.json > mode > stt/llm/tts)
    def _configure_custom(self):
        """Read STT / LLM / TTS backends from ``config.json > mode`` and configure the system accordingly.

        VRAM-safe transition order:
          1. First evict XTTS from the GPU (when TTS != "gpu") to free VRAM.
          2. Load STT next (Whisper enters VRAM when local_gpu is selected).
          3. Then load the LLM (GGUF enters VRAM when offline is selected).
          4. Finally load TTS (XTTS enters VRAM when "gpu" is selected).
        """
        if self.config is None:
            print("[WARN] Custom mode: config object missing, falling back to 'online' default.")
            self._configure_online()
            return

        stt_backend = self.config.get("mode", "stt", "backend", default="local_gpu")
        llm_backend = self.config.get("mode", "llm", "backend", default="online")
        tts_backend = self.config.get("mode", "tts", "backend", default="online")

        print(f"[CUSTOM] STT={stt_backend} | LLM={llm_backend} | TTS={tts_backend}")

        # 1. VRAM hygiene — evict XTTS first when it will not run on the GPU.
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
    # PRIMARY PIPELINE — STT → LLM → TTS
    # ═══════════════════════════════════════════════════════════

    def process(self, audio_path: str):
        """End-to-end audio translation pipeline.

        Any caught exception cascades into ``_fallback()`` so the system
        degrades gracefully rather than failing hard.
        """
        log.info(f"Processing audio ({self.current_mode}): {audio_path}")
        total_start = time.time()

        try:
            # Pull language configuration.
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

            # 1. STT (speech-to-text).
            stt_start = time.time()
            stt_result = self.transcriber.transcribe(audio_path, source_lang=src_lang)

            # Noise gate (threshold 0.4).
            if stt_result.get("no_speech_prob", 0) > 0.4:
                log.debug("Noise detected; translation aborted.")
                return

            text_tr = stt_result.get("text", "")
            if not text_tr:
                log.debug("Empty transcript; translation aborted.")
                return

            # Punctuation-aware filter — suppresses cough / swallow / breath false positives.
            _words = text_tr.strip().split()
            _has_punct = text_tr.strip()[-1] in ".!?" if text_tr.strip() else False
            if len(_words) <= 2 and not _has_punct:
                log.debug(f"Short transcript ('{text_tr}') — false positive, aborting.")
                return

            stt_ms = int((time.time() - stt_start) * 1000)
            log.info(f"STT: {stt_ms} ms | '{text_tr}'")

            # 2. LLM (translation) — with rolling-memory context.
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

            # All online tiers failed → trigger _fallback().
            if llm_result.get("engine") == "Failed":
                raise RuntimeError("All online LLM tiers failed (network outage?).")

            log.info(f"Translation: '{text_en}' | Engine: {llm_result.get('engine')} | {llm_ms} ms")

            # Update the rolling memory — keep the last 3 source sentences.
            self.history.append(text_tr)
            self.history = self.history[-3:]

            # 3. TTS (synthesis).
            tts_ms = self.synthesizer.speak(text_en, language=tgt_lang) or 0

            total_ms = int((time.time() - total_start) * 1000)
            log.info(f"E2E: {total_ms} ms (STT:{stt_ms} + LLM:{llm_ms} + TTS:{tts_ms})")

            if self.result_queue is not None:
                self.result_queue.put({
                    "direction":  "outbound",   # Symmetry: inbound payload uses the same field.
                    "text_tr":    text_tr,
                    "text_en":    text_en,
                    "engine":     llm_result.get("engine"),
                    "latency_ms": total_ms,
                    "stt_ms":     stt_ms,
                    "llm_ms":     llm_ms,
                    "tts_ms":     tts_ms,
                    "error":      None,
                })

        except Exception:
            log.error("ORCHESTRATOR processing error — triggering fallback.", exc_info=True)
            if self.result_queue is not None:
                import traceback
                self.result_queue.put({"error": traceback.format_exc()})
            self._fallback(audio_path, src_lang=src_lang, tgt_lang=tgt_lang,
                           src_name=src_name, tgt_name=tgt_name)

    # ═══════════════════════════════════════════════════════════
    # INBOUND — Translate incoming counterpart audio (no TTS)
    # ═══════════════════════════════════════════════════════════

    def process_inbound(self, audio_path: str):
        """
        Translate audio arriving from the other party into the user's native language.

        TTS is not played — the result is pushed only to the overlay.
        The direction is reversed relative to ``process()``: target_lang -> source_lang.
        """
        log.info(f"[Inbound] Processing: {audio_path}")
        total_start = time.time()

        try:
            # Reuse the same config object but with the direction reversed.
            src_lang = "en"
            tgt_lang = "tr"
            src_name = "English"
            tgt_name = "Turkish"

            if self.config:
                src_lang = self.config.get("language", "target", default="en")
                tgt_lang = self.config.get("language", "source", default="tr")
                src_name = self.config.get("language", "target_name", default="English")
                tgt_name = self.config.get("language", "source_name", default="Turkish")

            # 1. STT — transcribe the counterpart's audio.
            stt_start = time.time()
            stt_result = self.transcriber.transcribe(audio_path, source_lang=src_lang)

            if stt_result.get("no_speech_prob", 0) > 0.4:
                log.debug("[Inbound] Noise detected; processing aborted.")
                return

            text_foreign = stt_result.get("text", "").strip()
            if not text_foreign:
                log.debug("[Inbound] Empty transcript; processing aborted.")
                return

            _words = text_foreign.split()
            _has_punct = text_foreign[-1] in ".!?" if text_foreign else False
            if len(_words) <= 2 and not _has_punct:
                log.debug(f"[Inbound] Short transcript ('{text_foreign}') — false positive, aborting.")
                return

            stt_ms = int((time.time() - stt_start) * 1000)
            log.info(f"[Inbound] STT: {stt_ms} ms | '{text_foreign}'")

            # 2. LLM — translate the foreign transcript into the user's language.
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

            log.info(f"[Inbound] Translation: '{text_native}' | {llm_ms} ms")

            total_ms = int((time.time() - total_start) * 1000)

            # 3. No TTS — push to the overlay with the "inbound" flag.
            if self.result_queue is not None:
                self.result_queue.put({
                    "direction":  "inbound",
                    "text_tr":    text_native,   # Translated (native-language) text.
                    "text_en":    text_foreign,  # Original (foreign) text.
                    "engine":     llm_result.get("engine"),
                    "latency_ms": total_ms,
                    "stt_ms":     stt_ms,
                    "llm_ms":     llm_ms,
                    "tts_ms":     0,
                    "error":      None,
                })

        except Exception:
            log.error("[Inbound] Processing error.", exc_info=True)
            if self.result_queue is not None:
                import traceback
                self.result_queue.put({"error": f"[Inbound] {traceback.format_exc()}"})

    # ═══════════════════════════════════════════════════════════
    # FALLBACK — Safe transition from any mode to offline
    # ═══════════════════════════════════════════════════════════

    def _fallback(self, audio_path: str, src_lang="tr", tgt_lang="en",
                  src_name="Turkish", tgt_name="English"):
        """Safe transition from any mode into the offline (survival) mode.

        Triggered by network outages or cloud API failures.
        """
        log.critical(
            f"CONNECTION FAILURE! Transitioning to OFFLINE (survival) mode "
            f"— previous mode: {self.current_mode}"
        )

        # Switch to offline mode (set_mode handles VRAM bookkeeping).
        self.set_mode("offline")

        log.info(f"Processing offline: {audio_path}")

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
                log.info(f"Offline translation: '{text_en}'")

                tts_ms = self.synthesizer.speak(text_en) or 0
                log.info("Offline processing complete.")

                if self.result_queue is not None:
                    self.result_queue.put({
                        "text_tr": text_tr,
                        "text_en": text_en,
                        "engine": llm_result.get("engine", "offline_fallback"),
                        "latency_ms": None,  # E2E timing is not measured during fallback.
                        "stt_ms": None,
                        "llm_ms": llm_result.get("latency_ms"),
                        "tts_ms": tts_ms,
                        "error": None,
                    })
        except Exception:
            log.critical(
                "Offline translation also failed — every tier of the cascade is down!",
                exc_info=True
            )
