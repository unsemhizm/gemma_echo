"""
Gemma Echo GUI — Ana Koordinatör

Baslatma sirasi:
  1. ConfigManager yukle
  2. first_run == True  → SetupWizard goster, tamamlaninca devam et
  3. Orchestrator + backend bilesenleri yukle (arka planda)
  4. Overlay penceresini goster
  5. Mod / ayar degisikliklerini ConfigManager uzerinden yonet

Kullanim:
    python -m gui.app          (proje kokunden)
    python gui/app.py          (dogrudan)
"""

import os
import sys
import queue
import threading

import customtkinter as ctk

# Proje kokunu path'e ekle
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gui.config        import ConfigManager
from gui.pages.overlay import Overlay

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_C = {
    "bg":     "#0b0b12",
    "green":  "#23d05e",
    "yellow": "#f5a623",
    "red":    "#f04747",
    "blue":   "#5b9ef9",
    "dim":    "#3a3d52",
}


# ══════════════════════════════════════════════════════════════════════════════
# Ana Uygulama
# ══════════════════════════════════════════════════════════════════════════════

class GemmaEchoApp:
    """
    Uygulamanin yasam dongusunu yoneten sinif.
    Dogrudan bir pencere degil; pencereler arasinda koordinasyon saglar.
    """

    def __init__(self):
        self.cfg             = ConfigManager()
        self._rq             = queue.Queue()   # orchestrator → overlay koprusu
        self._overlay: Overlay | None          = None
        self._main                             = None   # MainWindow
        self._backend_ready                    = False
        self._recorder                         = None
        self._orchestrator                     = None
        self._backend_thread                   = None

    def run(self):
        if self.cfg.is_first_run():
            self._run_wizard()
        self._launch_main()

    # ── Kurulum Sihirbazi ─────────────────────────────────────────────────────

    def _run_wizard(self):
        from gui.pages.setup_wizard import SetupWizard
        wizard = SetupWizard(self.cfg, on_complete=self._on_wizard_done)
        wizard.mainloop()
        # mainloop bitti (wizard destroy edildi) → config yenile
        self.cfg = ConfigManager()

    def _on_wizard_done(self, cfg: ConfigManager):
        # Sadece flag — _launch_main wizard.mainloop() sonrasinda cagrilir
        self.cfg = cfg

    # ── Ana Ekran ─────────────────────────────────────────────────────────────

    def _launch_main(self):
        """Tek pencere (MainWindow) + overlay baslatir."""
        from gui.pages.main_window import MainWindow

        # MainWindow ana event loop'u tasir (CTk)
        self._main    = MainWindow(self.cfg, app=self)
        self._overlay = Overlay(self.cfg, result_queue=self._rq)

        # Backend arka planda yukle
        self._backend_thread = threading.Thread(
            target=self._load_backend, daemon=True
        )
        self._backend_thread.start()

        # Ana pencere mainloop'u baslatir
        self._main.mainloop()

    # ── Backend Yukleme ───────────────────────────────────────────────────────

    def _load_backend(self):
        """
        STT / LLM / TTS modellerini arka planda yukler.
        Tamamlaninca orchestrator'u result_queue'ya baglar.
        """
        try:
            self._overlay.set_status("Modeller yukleniyor...", _C["yellow"])

            from stt.transcriber       import Transcriber
            from llm.translator        import Translator
            from tts.synthesizer       import Synthesizer
            from pipeline.orchestrator import Orchestrator

            transcriber = Transcriber()
            translator  = Translator()
            synthesizer = Synthesizer()

            mode = self.cfg.get("mode", "current", default="online")
            self._orchestrator = Orchestrator(
                transcriber, translator, synthesizer,
                initial_mode=mode
            )
            self._orchestrator.result_queue = self._rq
            self._orchestrator.warm_up()

            self._backend_ready = True
            self._overlay.set_status("Hazir · VAD dinliyor", _C["dim"])
            self._overlay.after(0, lambda: self._overlay._dot.configure(
                text_color=_C["yellow"]
            ))

        except Exception as e:
            self._overlay.set_status(f"Yukleme hatasi: {e}", _C["red"])

    # ── Kayit Kontrolu ─────────────────────────────────────────────────────────

    def start_live(self):
        """VAD veya Bas-Konuş modunda kayıt başlatır."""
        if not self._backend_ready:
            self._overlay.set_status("Backend henüz hazır değil...", _C["yellow"])
            return
        if self._recorder is not None:
            return   # zaten calisiyor

        from stt.recorder import Recorder
        aggressiveness = self.cfg.get("recording", "vad_aggressiveness", default=2)
        self._recorder = Recorder(self._orchestrator, aggressiveness=aggressiveness)

        threading.Thread(target=self._recorder.run, daemon=True).start()
        self._overlay.set_status("Canlı dinleme aktif", _C["green"])

    def stop_live(self):
        if self._recorder:
            self._recorder._stop_event.set()
            self._recorder = None
        if self._overlay:
            self._overlay.set_status("Hazir · durdu", _C["dim"])

    def switch_mode(self, mode: str):
        if self._orchestrator:
            try:
                self._orchestrator.set_mode(mode)
                self.cfg.set_mode(mode)
                if self._overlay:
                    self._overlay.set_status(f"Mod: {mode.upper()}", _C["blue"])
            except Exception as e:
                if self._overlay:
                    self._overlay.set_status(f"Mod hatasi: {e}", _C["red"])


# ══════════════════════════════════════════════════════════════════════════════
# Giris Noktasi
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = GemmaEchoApp()
    app.run()


if __name__ == "__main__":
    main()
