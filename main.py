import sys
import os
import argparse
import threading

# ─── STAGE 0: Initialize the logging subsystem before anything else ───────────
# Must run prior to importing any other module so that failures surfaced during
# their import are captured by the global exception hooks installed below.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.logger import setup_logging, get_logger

setup_logging()                        # Rotating file handler + colored console
log = get_logger("gemma_echo.main")


# ─── GLOBAL EXCEPTION HOOKS (defensive armor layer) ───────────────────────────

def _handle_unhandled_exception(exc_type, exc_value, exc_tb):
    """
    Persist every unhandled exception raised on the main thread to the log file.
    Bound to ``sys.excepthook``; invoked immediately before the Python VM
    terminates the interpreter.

    ``KeyboardInterrupt`` (Ctrl+C) is treated as a user-initiated shutdown
    signal: it is delegated back to the default hook so the process exits
    cleanly without a misleading critical-error entry.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Ctrl+C → clean exit; bypass logging.
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    log.critical(
        "━━━ UNHANDLED EXCEPTION ON MAIN THREAD ━━━",
        exc_info=(exc_type, exc_value, exc_tb)
    )


def _handle_thread_exception(args: threading.ExceptHookArgs):
    """
    Persist every unhandled exception raised on a background worker thread.
    Bound to ``threading.excepthook``.

    Gemma Echo's STT / LLM / XTTS workers run on dedicated threads; without
    this hook a failure would die silently. With it installed every crash
    leaves a full stack trace tied to the originating thread name.
    """
    if args.exc_type is SystemExit:
        return   # sys.exit() → suppress logging.

    thread_name = args.thread.name if args.thread else "<unknown thread>"
    log.critical(
        f"━━━ UNHANDLED THREAD EXCEPTION  thread='{thread_name}' ━━━",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )


# Wire the hooks into the runtime.
sys.excepthook       = _handle_unhandled_exception
threading.excepthook = _handle_thread_exception

log.info("Global exception hooks (sys.excepthook + threading.excepthook) installed.")

# Module paths were appended to sys.path above.

from stt.transcriber import Transcriber
from llm.translator import Translator
from tts.synthesizer import Synthesizer
from pipeline.orchestrator import Orchestrator
from gui.config import ConfigManager


def main():
    parser = argparse.ArgumentParser(description="Gemma Echo v8 — Quad-State Hybrid Translation Engine")
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch the graphical user interface (default behavior)."
    )
    parser.add_argument(
        "--mode", type=str, default="interactive",
        choices=["online", "online_xtts", "interactive", "offline", "offline_gpu", "hybrid_cloud_io", "hybrid_cloud_stt", "online_local_stt"],
        help="Runtime mode: online, online_xtts, interactive, offline, etc."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Audio file to process (for CLI test runs)."
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Live terminal mode driven by microphone VAD."
    )
    args = parser.parse_args()

    # If no CLI input is supplied and live mode is not requested, fall back to the GUI.
    is_cli = args.live or (args.input is not None)

    if not is_cli:
        log.info("No CLI arguments supplied. Launching the graphical user interface...")
        from gui.app import GemmaEchoApp
        app = GemmaEchoApp()
        app.run()
        return

    # --- Otherwise: TERMINAL ENGINE ---
    log.info("═" * 58)
    log.info("      GEMMA ECHO v8 — QUAD-STATE ORCHESTRATION ENGINE")
    log.info(f"      Mode: {args.mode.upper()}" + (" | LIVE MICROPHONE" if args.live else ""))
    log.info("═" * 58)

    # 1. Initialize core components.
    log.info("Bootstrap: [1/3] STT module...")
    transcriber = Transcriber()
    log.info("Bootstrap: [2/3] LLM module...")
    translator = Translator()
    log.info("Bootstrap: [3/3] TTS module...")
    synthesizer = Synthesizer()

    # 2. Wire up the orchestrator.
    cfg = ConfigManager()
    orchestrator = Orchestrator(transcriber, translator, synthesizer, initial_mode=args.mode, config=cfg)

    # 3. Warm up the active pipeline.
    orchestrator.warm_up()

    # 4. Dispatch the workload.
    if args.live:
        from stt.recorder import Recorder
        recorder = Recorder(orchestrator, aggressiveness=2, transcriber=transcriber, config=cfg)
        recorder.run()
    else:
        log.info(f"Target file: {args.input}")
        orchestrator.process(args.input)
        log.info("═" * 58)
        log.info("      JOB COMPLETED")
        log.info("═" * 58)


if __name__ == "__main__":
    main()
