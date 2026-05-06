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
from gui.i18n          import t, set_language

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
        # Dil ayarini baslat
        ui_lang = self.cfg.get("language", "ui_language", default="tr")
        set_language(ui_lang)
        
        self._rq             = queue.Queue()   # orchestrator → overlay koprusu
        self._overlay: Overlay | None          = None
        self._main                             = None   # MainWindow
        self._backend_ready                    = False
        self._recorder                         = None
        self._inbound_recorder                 = None   # loopback recorder
        self._ptt_mode                         = None   # "outbound" | "inbound" | None
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

        # Overlay ayarlar butonunu ana pencereye bagla
        self._overlay._open_settings = lambda: self._main.switch_view("settings")

        # Ana pencere mainloop'u baslatir
        self._main.mainloop()

    # ── Backend Yukleme ───────────────────────────────────────────────────────

    def _load_backend(self):
        """
        STT / LLM / TTS modellerini arka planda yukler.
        Tamamlaninca orchestrator'u result_queue'ya baglar.
        """
        try:
            self._overlay.set_status(t("loading_models"), _C["yellow"])

            from stt.transcriber       import Transcriber
            from llm.translator        import Translator
            from tts.synthesizer       import Synthesizer
            from pipeline.orchestrator import Orchestrator

            self._transcriber = Transcriber()
            transcriber = self._transcriber
            translator  = Translator()
            synthesizer = Synthesizer()

            mode = self.cfg.get("mode", "current", default="online")
            self._orchestrator = Orchestrator(
                transcriber, translator, synthesizer,
                initial_mode=mode, config=self.cfg
            )
            self._orchestrator.result_queue = self._rq
            self._orchestrator.warm_up()

            self._backend_ready = True
            self._overlay.set_status(t("ready_vad"), _C["dim"])

            # ─── BROADCASTER AYARLARI ───────────────────────────
            if self.cfg.get("broadcaster", "enabled", default=False):
                dev_idx = self.cfg.get("broadcaster", "output_device_index")
                self._orchestrator.synthesizer.set_output_device(dev_idx)
            self._overlay.after(0, lambda: self._overlay._dot.configure(
                text_color=_C["yellow"]
            ))

            # ─── GLOBAL HOTKEY KAYDEDICI ────────────────────────
            self._register_hotkeys()

        except Exception as e:
            self._overlay.set_status(t("mode_error", str(e)), _C["red"])

    # ── Kayit Kontrolu ─────────────────────────────────────────────────────────

    def start_live(self):
        """VAD veya Bas-Konuş modunda kayıt başlatır."""
        if not self._backend_ready:
            self._overlay.set_status(t("backend_not_ready"), _C["yellow"])
            return
        if self._recorder is not None:
            return   # zaten calisiyor

        from stt.recorder import Recorder
        aggressiveness = self.cfg.get("recording", "vad_aggressiveness", default=2)
        mic_idx = self.cfg.get("recording", "mic_device_index", default=None)
        self._recorder = Recorder(
            self._orchestrator,
            aggressiveness=aggressiveness,
            transcriber=self._transcriber,
            config=self.cfg,
            device=int(mic_idx) if mic_idx is not None else None,
        )

        threading.Thread(target=self._recorder.run, daemon=True).start()
        self._overlay.set_status(t("live_active"), _C["green"])
        
        # ─── OVERLAY GORUNURLUK ───────────────────────────
        self._overlay.deiconify()
        self._overlay.lift()

    def stop_live(self):
        if self._recorder:
            self._recorder._stop_event.set()
            self._recorder = None
        if self._overlay:
            self._overlay.set_status(t("ready_stopped"), _C["dim"])

    # ── Inbound (Loopback) Kontrolu ────────────────────────────────────────────

    def start_inbound(self):
        """WASAPI Loopback'ten karsi tarafin sesini dinlemeye basla."""
        if not self._backend_ready:
            return
        if self._inbound_recorder is not None:
            return  # zaten calisiyor

        from stt.recorder import LoopbackRecorder

        # process_inbound() cagiran proxy — mevcut Orchestrator'u degistirmiyor
        class _InboundProxy:
            def __init__(self, orch):
                self._orch = orch
            def process(self, audio_path):
                self._orch.process_inbound(audio_path)

        proxy      = _InboundProxy(self._orchestrator)
        dev_name   = self.cfg.get("inbound", "loopback_device_name", default=None)

        self._inbound_recorder = LoopbackRecorder(
            proxy,
            config=self.cfg,
            device_name=dev_name,
        )
        threading.Thread(target=self._inbound_recorder.run, daemon=True).start()

    def stop_inbound(self):
        if self._inbound_recorder:
            self._inbound_recorder._stop_event.set()
            self._inbound_recorder = None

    # ── Dual PTT Anahtarlama ───────────────────────────────────────────────────

    def switch_ptt(self, mode: str):
        """
        mode: "outbound" → sen konusursun (mikrofon aktif, loopback durur)
              "inbound"  → karsi tarafi dinlersin (loopback aktif, mikrofon durur)
        Ayni moda tekrar basilirsa toggle gibi davranir (durur).
        """
        if mode == self._ptt_mode:
            # Ayni tusa tekrar basildi — her ikisini durdur
            self.stop_live()
            self.stop_inbound()
            self._ptt_mode = None
            if self._overlay:
                self._overlay.set_status(t("ready_stopped"), _C["dim"])
            return

        # Once her ikisini de durdur, sonra istenen yonu ac
        self.stop_live()
        self.stop_inbound()
        self._ptt_mode = mode

        if mode == "outbound":
            self.start_live()
            if self._overlay:
                self._overlay.set_status("SEN KONUSUYORSUN", _C["blue"])
        elif mode == "inbound":
            self.start_inbound()
            if self._overlay:
                self._overlay.set_status("KARSI TARAF DINLENIYOR", _C["green"])

    # ── Global Hotkey Kaydedici ────────────────────────────────────────────────

    def _register_hotkeys(self):
        """
        Global klavye kisayollarini kaydeder.
        Varsayilan: SPACE = sen konusursun, ALT = karsi tarafi dinle
        config.json'dan override edilebilir:
          "inbound": { "hotkey_outbound": "space", "hotkey_inbound": "alt" }
        """
        try:
            import keyboard

            key_out = self.cfg.get("inbound", "hotkey_outbound", default="space")
            key_in  = self.cfg.get("inbound", "hotkey_inbound",  default="alt")

            keyboard.add_hotkey(key_out, lambda: self.switch_ptt("outbound"))
            keyboard.add_hotkey(key_in,  lambda: self.switch_ptt("inbound"))

            print(f"[PTT] Hotkey kaydedildi: '{key_out}' = sen | '{key_in}' = karsi taraf")
        except ImportError:
            print("[UYARI] 'keyboard' kutuphanesi bulunamadi. 'pip install keyboard' calistir.")
        except Exception as e:
            print(f"[UYARI] Hotkey kaydedilemedi: {e}")

    def _reregister_hotkeys(self, key_out: str, key_in: str):
        """Mevcut hotkey'leri temizleyip yeni tuslarla yeniden kaydet."""
        try:
            import keyboard
            keyboard.unhook_all_hotkeys()
            keyboard.add_hotkey(key_out, lambda: self.switch_ptt("outbound"))
            keyboard.add_hotkey(key_in,  lambda: self.switch_ptt("inbound"))
            print(f"[PTT] Hotkey yeniden kaydedildi: '{key_out}' = sen | '{key_in}' = karsi taraf")
            if self._overlay:
                self._overlay.set_status(f"Tuslar guncellendi: {key_out} / {key_in}", _C["blue"])
        except Exception as e:
            print(f"[UYARI] Hotkey yeniden kaydedilemedi: {e}")

    def switch_mode(self, mode: str):
        if not self._orchestrator:
            return
        if getattr(self, "_mode_switching", False):
            return  # Zaten mod değiştiriliyor, tekrar tetiklenmesin
        self._mode_switching = True

        def _do_switch():
            try:
                self._orchestrator.set_mode(mode)
                self.cfg.set_mode(mode)
                ok_msg = t("mode_active", mode.upper())
                if self._overlay:
                    self._overlay.set_status(ok_msg, _C["blue"])
                if self._main:
                    self._main.after(0, lambda: self._main.set_status(ok_msg, _C["blue"]))
            except Exception as e:
                err_msg = t("mode_error", str(e))
                if self._overlay:
                    self._overlay.set_status(err_msg, _C["red"])
                if self._main:
                    self._main.after(0, lambda: self._main.set_status(err_msg, _C["red"]))
            finally:
                self._mode_switching = False

        threading.Thread(target=_do_switch, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# Giris Noktasi
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = GemmaEchoApp()
    app.run()


if __name__ == "__main__":
    main()
