import os
import sys
import queue
import threading
import logging
import traceback

import customtkinter as ctk

# Make the project root importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Initialize the central logging subsystem before anything else ────────────
from core.logger import setup_logging, get_logger
setup_logging()
_app_log = get_logger("gemma_echo.gui")
# ─────────────────────────────────────────────────────────────────────────────

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
# Application root
# ══════════════════════════════════════════════════════════════════════════════

class GemmaEchoApp:
    """
    Owns the application lifecycle.

    Not a window itself — coordinates between windows (main + overlay) and the
    backend pipeline.
    """

    def __init__(self):
        self.cfg             = ConfigManager()
        # Apply the persisted UI language.
        ui_lang = self.cfg.get("language", "ui_language", default="tr")
        set_language(ui_lang)

        self._rq             = queue.Queue()   # Orchestrator → overlay bridge.
        self._overlay: Overlay | None          = None
        self._main                             = None   # MainWindow.
        self._backend_ready                    = False
        self._recorder                         = None
        self._inbound_recorder                 = None   # WASAPI loopback recorder.
        self._ptt_mode                         = None   # "outbound" | "inbound" | None
        self._orchestrator                     = None
        self._backend_thread                   = None
        self._ptt_hotkeys_registered          = False  # Space/Alt are bound only while live capture is active.
        self._outbound_hook                   = None   # keyboard.add_hotkey handle.
        self._inbound_hook                    = None   # keyboard.add_hotkey handle.

        # Module-scoped logger — _init_logging() is no longer necessary.
        self.logger = get_logger("gemma_echo.app")
        self.logger.info("Gemma Echo GUI starting up")

        self._install_exception_hooks()

    def _install_exception_hooks(self):
        """Install the main-thread and background-thread exception hooks."""

        def excepthook(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            self._handle_uncaught_exception(exc_type, exc_value, exc_traceback)

        sys.excepthook = excepthook
        if hasattr(threading, "excepthook"):
            threading.excepthook = self._threading_excepthook

    def _handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        self.logger.error("Uncaught exception:\n%s", message)
        if self._main:
            try:
                self._main.after(0, lambda: self._show_error_dialog(
                    t("unexpected_error_title"),
                    message
                ))
            except Exception:
                pass

    def _threading_excepthook(self, args):
        message = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        self.logger.error("Thread exception:\n%s", message)
        if self._main:
            try:
                self._main.after(0, lambda: self._show_error_dialog(
                    t("background_error_title"),
                    message
                ))
            except Exception:
                pass

    def _show_error_dialog(self, title: str, message: str):
        from tkinter import messagebox

        def _show():
            try:
                parent = self._main if self._main is not None else None
                messagebox.showerror(
                    title,
                    f"{message}\n\nDetails were written to '{os.path.basename(os.path.abspath(_ROOT))}.log'.",
                    parent=parent,
                )
            except Exception:
                pass

        if self._main is not None:
            try:
                self._main.after(0, _show)
            except Exception:
                pass

    def _on_vram_issue(self):
        """Local-model VRAM failure — surface a messagebox on the main thread."""
        main = self._main
        if main is None:
            return

        def _show():
            from tkinter import messagebox

            messagebox.showwarning(
                t("vram_insufficient_title"),
                t("vram_insufficient_detail"),
                parent=main,
            )

        try:
            main.after(0, _show)
        except Exception:
            pass

    def run(self):
        if self.cfg.is_first_run():
            self._run_wizard()
        self._launch_main()

    # ── Setup wizard ──────────────────────────────────────────────────────────

    def _run_wizard(self):
        from gui.pages.setup_wizard import SetupWizard
        wizard = SetupWizard(self.cfg, on_complete=self._on_wizard_done)
        wizard.mainloop()
        # mainloop returned (wizard was destroyed) → reload the config.
        self.cfg = ConfigManager()

    def _on_wizard_done(self, cfg: ConfigManager):
        # Pure handoff — _launch_main runs immediately after wizard.mainloop() exits.
        self.cfg = cfg

    # ── Main window ───────────────────────────────────────────────────────────

    def _launch_main(self):
        """Spin up the single main window + overlay."""
        from gui.pages.main_window import MainWindow

        # MainWindow owns the CTk event loop.
        self._main    = MainWindow(self.cfg, app=self)
        self._overlay = Overlay(self.cfg, result_queue=self._rq)
        self._overlay.on_telemetry_update = self._on_telemetry_update

        # Initialize the backend on a background thread.
        self._backend_thread = threading.Thread(
            target=self._load_backend, daemon=True
        )
        self._backend_thread.start()

        # Wire the overlay's "Settings" button to the main-window view switcher.
        self._overlay._open_settings = lambda: self._main.switch_view("settings")

        # Hand control to the main window's mainloop.
        self._main.mainloop()

    def _on_telemetry_update(self, data: dict):
        if self._main:
            live = getattr(self._main, "_views", {}).get("live")
            if live and hasattr(live, "update_telemetry"):
                try:
                    self._main.after(0, lambda: live.update_telemetry(data))
                except Exception:
                    pass

    # ── Backend loading ───────────────────────────────────────────────────────

    def _load_backend(self):
        """
        Load the STT / LLM / TTS modules on a background thread.

        Once everything is ready, wire the orchestrator to the result queue.
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
            translator.vram_issue_callback = self._on_vram_issue
            synthesizer.vram_issue_callback = self._on_vram_issue

            mode = self.cfg.get("mode", "current", default="online")
            self._orchestrator = Orchestrator(
                transcriber, translator, synthesizer,
                initial_mode=mode, config=self.cfg
            )
            self._orchestrator.result_queue = self._rq
            self._orchestrator.warm_up()

            self._backend_ready = True
            self._overlay.set_status(t("ready_vad"), _C["dim"])

            # ─── BROADCASTER ROUTING ───────────────────────────
            if self.cfg.get("broadcaster", "enabled", default=False):
                dev_idx = self.cfg.get("broadcaster", "output_device_index")
                self._orchestrator.synthesizer.set_output_device(dev_idx)
            self._overlay.after(0, lambda: self._overlay._dot.configure(
                text_color=_C["yellow"]
            ))

        except Exception as e:
            error_text = str(e)
            traceback_text = traceback.format_exc()
            self.logger.error("Backend load failure:\n%s", traceback_text)
            self._overlay.set_status(t("mode_error", error_text), _C["red"])
            if self._main is not None:
                try:
                    self._main.after(0, lambda: self._show_error_dialog(
                        "Backend Load Failure",
                        traceback_text
                    ))
                except Exception:
                    pass

    # ── Recording control ─────────────────────────────────────────────────────

    def start_live(self):
        """Start live capture in either VAD or push-to-talk mode."""
        if not self._backend_ready:
            self._overlay.set_status(t("backend_not_ready"), _C["yellow"])
            return
        if self._recorder is not None:
            return   # Already running.

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
        self._ensure_ptt_hotkeys()

        # ─── OVERLAY VISIBILITY ───────────────────────────
        self._overlay.deiconify()
        self._overlay.lift()

    def stop_live(self):
        if self._recorder:
            if hasattr(self._recorder, 'stop'):
                self._recorder.stop()
            else:
                self._recorder._stop_event.set()
            self._recorder = None
        if self._overlay:
            self._overlay.set_status(t("ready_stopped"), _C["dim"])

    # ── Inbound (WASAPI loopback) control ────────────────────────────────────

    def start_inbound(self):
        """Start listening to the counterpart's audio via the WASAPI loopback."""
        if not self._backend_ready:
            return
        if self._inbound_recorder is not None:
            return  # Already running.

        from stt.recorder import LoopbackRecorder

        # Proxy that routes captured audio through process_inbound() —
        # the existing Orchestrator instance stays untouched.
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
            if hasattr(self._inbound_recorder, 'stop'):
                self._inbound_recorder.stop()
            else:
                self._inbound_recorder._stop_event.set()
            self._inbound_recorder = None

    # ── Dual PTT toggling ─────────────────────────────────────────────────────

    def switch_ptt(self, mode: str):
        """
        mode: "outbound" → user is speaking (mic active, loopback paused).
              "inbound"  → user is listening (loopback active, mic paused).

        Pressing the same key again toggles the current mode off.
        """
        if mode == self._ptt_mode:
            # Same key was pressed again — stop both directions.
            self.stop_live()
            self.stop_inbound()
            self._ptt_mode = None
            if self._overlay:
                self._overlay.set_status(t("ready_stopped"), _C["dim"])
            self._notify_live_view_ptt(None)
            self._sync_ptt_hotkeys_state()
            return

        # Stop both first, then start the requested direction.
        self.stop_live()
        self.stop_inbound()
        self._ptt_mode = mode

        if mode == "outbound":
            self.start_live()
            self._ensure_ptt_hotkeys()
            if self._overlay:
                self._overlay.set_status(t("ptt_outbound_active"), _C["blue"])
        elif mode == "inbound":
            self.start_inbound()
            self._ensure_ptt_hotkeys()
            if self._overlay:
                self._overlay.set_status(t("ptt_inbound_active"), _C["green"])

        self._notify_live_view_ptt(mode)

    def _notify_live_view_ptt(self, mode):
        """Refresh the LiveView PTT button visuals in a thread-safe way."""
        if not self._main:
            return
        live = getattr(self._main, "_views", {}).get("live")
        if live and hasattr(live, "_update_ptt_state"):
            self._main.after(0, lambda m=mode: live._update_ptt_state(m))

    # ── Global hotkey registrar (active only during a live capture / PTT session) ───

    def _live_hotkey_session_active(self) -> bool:
        return (
            self._recorder is not None
            or self._inbound_recorder is not None
            or self._ptt_mode is not None
        )

    def _ensure_ptt_hotkeys(self):
        """Bind the Space / Alt hooks (once) when a live session is active."""
        if self._ptt_hotkeys_registered or not self._live_hotkey_session_active():
            return
        self._register_hotkeys()

    def _sync_ptt_hotkeys_state(self):
        """Unbind the hotkeys once every live session has ended."""
        if self._live_hotkey_session_active():
            return
        self._unregister_hotkeys()

    def _clear_hotkey_handles(self):
        """Remove only the hotkey handles we registered ourselves — do not touch other modules.

        Used by both _unregister_hotkeys and _reregister_hotkeys.
        """
        try:
            import keyboard
        except ImportError:
            self._outbound_hook = None
            self._inbound_hook = None
            self._ptt_hotkeys_registered = False
            return

        for attr in ("_outbound_hook", "_inbound_hook"):
            handle = getattr(self, attr, None)
            if handle is None:
                continue
            try:
                keyboard.remove_hotkey(handle)
            except (KeyError, ValueError):
                # Already removed or invalid — ignore.
                pass
            except Exception as e:
                print(f"[WARN] failed to remove {attr}: {e}")
            setattr(self, attr, None)
        self._ptt_hotkeys_registered = False

    def _unregister_hotkeys(self):
        if not self._ptt_hotkeys_registered:
            return
        self._clear_hotkey_handles()
        print("[PTT] Global hotkeys removed (live session ended).")

    def _register_hotkeys(self):
        """
        Register the global keyboard shortcuts.

        Defaults: SPACE = user speaks; ALT = listen to the counterpart.
        Overridable via config.json:
          "inbound": { "hotkey_outbound": "space", "hotkey_inbound": "alt" }
        """
        if self._ptt_hotkeys_registered:
            return
        try:
            import keyboard

            key_out = self.cfg.get("inbound", "hotkey_outbound", default="space")
            key_in  = self.cfg.get("inbound", "hotkey_inbound",  default="alt")

            self._outbound_hook = keyboard.add_hotkey(key_out, lambda: self.switch_ptt("outbound"))
            self._inbound_hook  = keyboard.add_hotkey(key_in,  lambda: self.switch_ptt("inbound"))
            self._ptt_hotkeys_registered = True

            print(f"[PTT] Hotkey registered: '{key_out}' = self | '{key_in}' = counterpart")
        except ImportError:
            print("[WARN] 'keyboard' library not installed. Run 'pip install keyboard'.")
        except Exception as e:
            print(f"[WARN] Hotkey registration failed: {e}")

    def _reregister_hotkeys(self, key_out: str, key_in: str):
        """Settings panel changed the bindings: drop only our own handles and re-register."""
        try:
            import keyboard
        except ImportError:
            print("[WARN] 'keyboard' library not installed.")
            return

        self._clear_hotkey_handles()

        if not self._live_hotkey_session_active():
            # No active session; the new keys are already persisted to config,
            # and _register_hotkeys will pick them up on the next live session.
            print(
                f"[PTT] New bindings persisted (no active session): "
                f"'{key_out}' = self | '{key_in}' = counterpart"
            )
            if self._overlay:
                self._overlay.set_status(
                    f"Bindings saved: {key_out} / {key_in}", _C["blue"]
                )
            return

        try:
            self._outbound_hook = keyboard.add_hotkey(key_out, lambda: self.switch_ptt("outbound"))
            self._inbound_hook  = keyboard.add_hotkey(key_in,  lambda: self.switch_ptt("inbound"))
            self._ptt_hotkeys_registered = True
            print(
                f"[PTT] Hotkey re-registered: '{key_out}' = self | '{key_in}' = counterpart"
            )
            if self._overlay:
                self._overlay.set_status(
                    f"Bindings updated: {key_out} / {key_in}", _C["blue"]
                )
        except Exception as e:
            print(f"[WARN] Hotkey re-registration failed: {e}")

    def switch_mode(self, mode: str):
        if not self._orchestrator:
            return
        if getattr(self, "_mode_switching", False):
            return  # A mode transition is already in flight — debounce.
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
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = GemmaEchoApp()
    app.run()


if __name__ == "__main__":
    main()
