"""
gemma_echo/core/logger.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Centralized thread-safe logging architecture — Gemma Echo v8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage (in any module):
    from core.logger import get_logger
    log = get_logger(__name__)

    log.info("Model loaded.")
    log.warning("VRAM is low!")
    log.error("API failure", exc_info=True)
    log.critical("System crash!", exc_info=True)

Outputs:
  • Terminal  → Colorized, human-readable format
  • File      → gemma_echo.log (10 MB × 5 rotation; timestamp + thread + caller)
"""

import logging
import logging.handlers
import os
import threading

# ─── Constants ───────────────────────────────────────────────────────────────

# Log file lives at the project root (next to main.py).
_LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(_LOG_DIR, "gemma_echo.log")

# Rotation policy: 10 MB per file, up to 5 backups retained.
MAX_BYTES = 10 * 1024 * 1024   # 10 MB
BACKUP_COUNT = 5

# ─── ANSI color codes (terminal handler) ─────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

_COLORS = {
    "DEBUG"    : "\033[36m",    # Cyan
    "INFO"     : "\033[32m",    # Green
    "WARNING"  : "\033[33m",    # Yellow
    "ERROR"    : "\033[31m",    # Red
    "CRITICAL" : "\033[35m",    # Magenta (bold)
}

# ─── Custom formatter: colorized console ─────────────────────────────────────

class _ColoredConsoleFormatter(logging.Formatter):
    """
    Colorizes log records for terminal output.
    Format: [HH:MM:SS] [LEVEL] [ThreadName] message  (file:line)
    """
    _FMT = (
        "{dim}[%(asctime)s]{reset} "
        "{color}{bold}[%(levelname)-8s]{reset} "
        "{dim}[%(threadName)s]{reset} "
        "%(message)s"
        "  {dim}(%(filename)s:%(lineno)d){reset}"
    )

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        bold  = _BOLD if record.levelname in ("CRITICAL", "ERROR") else ""
        fmt = self._FMT.format(
            dim=_DIM, reset=_RESET, color=color, bold=bold
        )
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


# ─── Custom formatter: file (plain text, full detail) ────────────────────────

class _FileFormatter(logging.Formatter):
    """
    Writes plain-text log entries to disk — optimized for grep / log analysis.
    Format: 2026-05-09 23:15:01,234 | ERROR    | STT-Thread | translator.py:203 | message
    """
    _FMT = (
        "%(asctime)s | %(levelname)-8s | %(threadName)-20s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )

    def __init__(self):
        super().__init__(fmt=self._FMT, datefmt="%Y-%m-%d %H:%M:%S")


# ─── Centralized setup (invoked exactly once) ────────────────────────────────

_setup_lock = threading.Lock()
_is_configured = False


def setup_logging(level: int = logging.DEBUG) -> None:
    """
    Initialize the logging subsystem. Called exactly once from main.py during
    application bootstrap; subsequent calls are silently no-op (idempotent).

    Args:
        level: Root logger level. Defaults to DEBUG (capture everything).
               Production deployments may prefer logging.INFO.
    """
    global _is_configured
    with _setup_lock:
        if _is_configured:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # ── Handler 1: colorized console ─────────────────────────────────
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)      # Console shows INFO and above.
        console_handler.setFormatter(_ColoredConsoleFormatter())

        # ── Handler 2: rotating file ─────────────────────────────────────
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=LOG_FILE,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
                delay=False,
            )
            file_handler.setLevel(logging.DEBUG)    # File captures DEBUG and above.
            file_handler.setFormatter(_FileFormatter())
        except OSError as e:
            # Log directory is not writable — continue with console only and emit a warning.
            console_handler.emit(
                logging.LogRecord(
                    name="core.logger", level=logging.ERROR,
                    pathname=__file__, lineno=0,
                    msg=f"[LOGGER] Failed to create log file: {e}",
                    args=(), exc_info=None
                )
            )
            file_handler = None

        # ── Reset any pre-existing handlers and register ours ────────────
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        if file_handler:
            root_logger.addHandler(file_handler)

        # ── Silence noisy third-party loggers ─────────────────────────────
        # llama-cpp, TTS and httpx emit excessive DEBUG output.
        for noisy_lib in ("llama_cpp", "TTS", "httpx", "urllib3", "httpcore",
                          "pdfminer", "pdfplumber"):
            logging.getLogger(noisy_lib).setLevel(logging.WARNING)

        _is_configured = True

        logger = logging.getLogger("core.logger")
        logger.info(
            "━━━ Gemma Echo Logging Initialized ━━━  "
            f"File → {LOG_FILE}"
        )


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-scoped named logger.

    Usage:
        log = get_logger(__name__)

    If ``setup_logging()`` has not been invoked yet, this function transparently
    bootstraps it — a defensive convenience for development and test contexts.
    """
    if not _is_configured:
        setup_logging()
    return logging.getLogger(name)
