"""
gemma_echo/core/logger.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Merkezi Thread-Safe Loglama Mimarisi — Gemma Echo v8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Kullanım (her modülde):
    from core.logger import get_logger
    log = get_logger(__name__)

    log.info("Model yüklendi.")
    log.warning("VRAM düşük!")
    log.error("API hatası", exc_info=True)
    log.critical("Sistem çöktü!", exc_info=True)

Çıktılar:
  • Terminal  → Renkli, okunabilir format
  • Dosya     → gemma_echo.log (10MB x 5 rotasyon, zaman + thread + konum)
"""

import logging
import logging.handlers
import os
import threading

# ─── Sabitler ────────────────────────────────────────────────────────────────

# Log dosyasının konumu: projenin kökü (main.py'nin yanı)
_LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(_LOG_DIR, "gemma_echo.log")

# Rotasyon: her dosya max 10 MB, en fazla 5 yedek
MAX_BYTES = 10 * 1024 * 1024   # 10 MB
BACKUP_COUNT = 5

# ─── ANSI Renk Kodları (terminal için) ───────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

_COLORS = {
    "DEBUG"    : "\033[36m",    # Cyan
    "INFO"     : "\033[32m",    # Yeşil
    "WARNING"  : "\033[33m",    # Sarı
    "ERROR"    : "\033[31m",    # Kırmızı
    "CRITICAL" : "\033[35m",    # Magenta (kalın)
}

# ─── Özel Formatter: Renkli Terminal ─────────────────────────────────────────

class _ColoredConsoleFormatter(logging.Formatter):
    """
    Terminal çıktısını renklendirir.
    Format: [HH:MM:SS] [LEVEL] [ThreadName] mesaj  (file:satır)
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


# ─── Özel Formatter: Dosya (renksiz, tam bilgi) ───────────────────────────────

class _FileFormatter(logging.Formatter):
    """
    Log dosyasına düz metin yazar — grep/analiz için temiz.
    Format: 2026-05-09 23:15:01,234 | ERROR    | STT-Thread | translator.py:203 | mesaj
    """
    _FMT = (
        "%(asctime)s | %(levelname)-8s | %(threadName)-20s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )

    def __init__(self):
        super().__init__(fmt=self._FMT, datefmt="%Y-%m-%d %H:%M:%S")


# ─── Merkezi Setup (bir kez çağrılır) ────────────────────────────────────────

_setup_lock = threading.Lock()
_is_configured = False


def setup_logging(level: int = logging.DEBUG) -> None:
    """
    Loglama sistemini başlatır. main.py içinde uygulama açılırken BİR KEZ çağrılır.
    İkinci çağrı sessizce yok sayılır (idempotent).

    Args:
        level: Kök logger seviyesi. Varsayılan DEBUG (her şeyi yakala).
               Production'da logging.INFO kullanılabilir.
    """
    global _is_configured
    with _setup_lock:
        if _is_configured:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # ── Handler 1: Renkli Terminal ────────────────────────────────────
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)      # Terminalde INFO+ göster
        console_handler.setFormatter(_ColoredConsoleFormatter())

        # ── Handler 2: Rotasyonlu Dosya ───────────────────────────────────
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=LOG_FILE,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
                delay=False,
            )
            file_handler.setLevel(logging.DEBUG)    # Dosyada DEBUG+ kaydet
            file_handler.setFormatter(_FileFormatter())
        except OSError as e:
            # Log dizini yazılabilir değilse devam et ama uyar
            console_handler.emit(
                logging.LogRecord(
                    name="core.logger", level=logging.ERROR,
                    pathname=__file__, lineno=0,
                    msg=f"[LOGGER] Log dosyası oluşturulamadı: {e}",
                    args=(), exc_info=None
                )
            )
            file_handler = None

        # ── Mevcut handler'ları temizle, yenileri ekle ───────────────────
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        if file_handler:
            root_logger.addHandler(file_handler)

        # ── Gürültülü kütüphane loglarını sustur ─────────────────────────
        # llama-cpp, TTS ve httpx çok fazla DEBUG mesajı üretir
        for noisy_lib in ("llama_cpp", "TTS", "httpx", "urllib3", "httpcore"):
            logging.getLogger(noisy_lib).setLevel(logging.WARNING)

        _is_configured = True

        logger = logging.getLogger("core.logger")
        logger.info(
            "━━━ Gemma Echo Loglama Başlatıldı ━━━  "
            f"Dosya → {LOG_FILE}"
        )


def get_logger(name: str) -> logging.Logger:
    """
    Modül için adlandırılmış logger döner.

    Kullanım:
        log = get_logger(__name__)

    setup_logging() henüz çağrılmadıysa otomatik başlatır
    (geliştirme ve test ortamları için güvenli).
    """
    if not _is_configured:
        setup_logging()
    return logging.getLogger(name)
