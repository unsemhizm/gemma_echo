import sys
import os
import argparse
import threading

# ─── ADIM 0: Loglama sistemini her şeyden önce başlat ─────────────────────────
# Diğer modüller import edilmeden önce çalışmalı — aksi halde onların
# ürettiği hatalar yakalanmadan geçer.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.logger import setup_logging, get_logger

setup_logging()                        # Rotasyonlu dosya + renkli terminal
log = get_logger("gemma_echo.main")


# ─── GLOBAL HATA KANCALARI (Zırh Katmanı) ─────────────────────────────────────

def _handle_unhandled_exception(exc_type, exc_value, exc_tb):
    """
    Ana thread'deki yakalanmayan tüm exception'ları log dosyasına yazar.
    sys.excepthook olarak atanır — Python VM çökmeden hemen önce çağrılır.

    KeyboardInterrupt (Ctrl+C) kasıtlı çıkış sinyalidir; loglanmaz,
    normal Python davranışına bırakılır.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Ctrl+C → normal çıkış, loglamadan geç
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    log.critical(
        "━━━ YAKALANMAYAN ANA THREAD HATASI ━━━",
        exc_info=(exc_type, exc_value, exc_tb)
    )


def _handle_thread_exception(args: threading.ExceptHookArgs):
    """
    Arka plan thread'lerindeki yakalanmayan exception'ları log dosyasına yazar.
    threading.excepthook olarak atanır.

    Gemma Echo'da Whisper/LLM/XTTS thread'leri bu kanca sayesinde
    sessizce ölmek yerine tam stack trace bırakır.
    """
    if args.exc_type is SystemExit:
        return   # sys.exit() → loglamadan geç

    thread_name = args.thread.name if args.thread else "<bilinmeyen thread>"
    log.critical(
        f"━━━ YAKALANMAYAN THREAD HATASI  thread='{thread_name}' ━━━",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )


# Kancaları sisteme bağla
sys.excepthook       = _handle_unhandled_exception
threading.excepthook = _handle_thread_exception

log.info("Global hata kancaları (sys.excepthook + threading.excepthook) kuruldu.")

# Modül yolları sys.path'e yukarıda eklendi

from stt.transcriber import Transcriber
from llm.translator import Translator
from tts.synthesizer import Synthesizer
from pipeline.orchestrator import Orchestrator
from gui.config import ConfigManager


def main():
    parser = argparse.ArgumentParser(description="Gemma Echo v8 — 4 Modlu Hibrit Ceviri Sistemi")
    parser.add_argument(
        "--mode", type=str, default="interactive",
        choices=["online", "online_xtts", "interactive", "offline", "offline_gpu", "hybrid_cloud_io", "hybrid_cloud_stt", "online_local_stt"],
        help="Calisma modu: online, online_xtts, interactive (varsayilan), offline, offline_gpu, hybrid_cloud_io, hybrid_cloud_stt, online_local_stt"
    )
    parser.add_argument(
        "--input", type=str, default="audio/Kayıt (3).wav",
        help="Islenecek ses dosyasi (--live kullanilmiyorsa)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Canli mikrofon modunu baslat (VAD tabanli, Ctrl+C ile dur)"
    )
    args = parser.parse_args()

    log.info("═" * 58)
    log.info("      GEMMA ECHO v8 — QUAD-STATE ORKESTRA SEFİ")
    log.info(f"      Mod: {args.mode.upper()}" + (" | CANLI MİKROFON" if args.live else ""))
    log.info("═" * 58)

    # 1. Bileşenleri Başlat
    log.info("Başlatma: [1/3] STT modülü...")
    transcriber = Transcriber()
    log.info("Başlatma: [2/3] LLM modülü...")
    translator = Translator()
    log.info("Başlatma: [3/3] TTS modülü...")
    synthesizer = Synthesizer()

    # 2. Orkestrasyonu Kur (başlangıç moduyla)
    cfg = ConfigManager()
    orchestrator = Orchestrator(transcriber, translator, synthesizer, initial_mode=args.mode, config=cfg)

    # 3. Isıt (Cold-Start Warm-up)
    orchestrator.warm_up()

    # 4. İşlemi Başlat
    if args.live:
        from stt.recorder import Recorder
        recorder = Recorder(orchestrator, aggressiveness=2, transcriber=transcriber, config=cfg)
        recorder.run()
    else:
        log.info(f"Hedef dosya: {args.input}")
        orchestrator.process(args.input)
        log.info("═" * 58)
        log.info("      İŞLEM TAMAMLANDI")
        log.info("═" * 58)


if __name__ == "__main__":
    main()