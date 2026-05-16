"""
Gemma Echo — Floating caption overlay window.

Always-on-top, draggable, semi-transparent. Renders the STT + translation
output streamed through ``result_queue`` in near real-time.

Integration:
    import queue
    rq = queue.Queue()
    orchestrator.result_queue = rq
    overlay = Overlay(cfg, result_queue=rq)
    overlay.mainloop()
"""

import sys
import queue
import time
import customtkinter as ctk

from gui.config import ConfigManager
from gui.i18n   import t

# ─── Theme ────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")

_C = {
    "bg":       "#0d1117",   # main background
    "bar":      "#161b22",   # title bar
    "card":     "#1c2128",   # text card
    "border":   "#30363d",   # border
    "blue":     "#58a6ff",   # target-language text color
    "gray":     "#8b949e",   # source-language text color
    "green":         "#3fb950",   # active indicator
    "yellow":        "#d29922",   # idle indicator
    "inbound_text":  "#3fb950",   # counterpart speech (green)
    "outbound_text": "#58a6ff",   # own speech (blue)
    "red":      "#f85149",   # error
    "white":    "#e6edf3",   # primary text
    "dim":      "#484f58",   # button hover
}

WIN_W   = 480
WIN_H   = 162
POLL_MS = 80     # Queue polling interval (ms).
FADE_MS = 4000   # Time before the last message starts fading (ms).


class Overlay(ctk.CTkToplevel):
    """
    Floating caption window (CTkToplevel bound to the main window).

    Parameters:
        cfg          : ConfigManager instance.
        result_queue : orchestrator.result_queue — when None the overlay runs in demo mode.
    """

    def __init__(self, cfg: ConfigManager, result_queue: queue.Queue | None = None):
        super().__init__()
        self.cfg          = cfg
        self._rq          = result_queue
        self._dragging    = False
        self._drag_x      = 0
        self._drag_y      = 0
        self._last_update = 0.0   # Last update timestamp.
        self._fade_job    = None  # Fade timer.

        # Drag state.
        self._last_x = 0
        self._last_y = 0

        self._setup_window()
        self._build_ui()
        self._start_poll()

    # ── Window setup ─────────────────────────────────────────────────────────

    def _setup_window(self):
        self.overrideredirect(True)           # Remove the system title bar.
        self.wm_attributes("-topmost", True)  # Always on top.
        self.configure(fg_color=_C["bg"])

        opacity = self.cfg.get("overlay", "opacity", default=0.92)
        self.wm_attributes("-alpha", opacity)

        # Default position — top-right (does not occlude the taskbar or chat panes).
        self.update_idletasks()
        x = self.cfg.get("overlay", "position_x", default=-1)
        y = self.cfg.get("overlay", "position_y", default=-1)

        if x == -1 or y == -1:
            sw = self.winfo_screenwidth()
            x  = sw - WIN_W - 24
            y  = 40   # Top-right corner.

        self.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

        # Start hidden — surfaced on demand from the main window.
        self.withdraw()

    # ── UI structure ──────────────────────────────────────────────────────────

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_titlebar()
        self._build_content()
        self._build_statusbar()

    def _build_titlebar(self):
        """Draggable title bar — app name + control buttons."""
        bar = ctk.CTkFrame(self, fg_color=_C["bar"], corner_radius=0, height=30)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_propagate(False)

        # Drag bindings.
        bar.bind("<ButtonPress-1>",   self._drag_start)
        bar.bind("<B1-Motion>",       self._drag_move)
        bar.bind("<ButtonRelease-1>", self._drag_end)

        # Left: logo + name.
        left = ctk.CTkFrame(bar, fg_color="transparent")
        left.pack(side="left", padx=10)
        left.bind("<ButtonPress-1>",   self._drag_start)
        left.bind("<B1-Motion>",       self._drag_move)
        left.bind("<ButtonRelease-1>", self._drag_end)

        self._dot = ctk.CTkLabel(
            left, text="●", font=ctk.CTkFont(size=9),
            text_color=_C["yellow"]   # Idle color at startup.
        )
        self._dot.pack(side="left", padx=(0, 5))
        self._dot.bind("<ButtonPress-1>", self._drag_start)
        self._dot.bind("<B1-Motion>",     self._drag_move)

        title = ctk.CTkLabel(
            left, text=t("app_name"),
            font=ctk.CTkFont(family="Helvetica", size=10, weight="bold"),
            text_color=_C["dim"]
        )
        title.pack(side="left")
        title.bind("<ButtonPress-1>", self._drag_start)
        title.bind("<B1-Motion>",     self._drag_move)

        # Direction indicator — "SELF" vs "COUNTERPART".
        self._dir_label = ctk.CTkLabel(
            left, text="",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=_C["dim"]
        )
        self._dir_label.pack(side="left", padx=(8, 0))
        self._dir_label.bind("<ButtonPress-1>", self._drag_start)
        self._dir_label.bind("<B1-Motion>",     self._drag_move)

        # Right: control buttons.
        right = ctk.CTkFrame(bar, fg_color="transparent")
        right.pack(side="right", padx=6)

        ctk.CTkButton(
            right, text="A+", width=22, height=20,
            fg_color="transparent", hover_color=_C["dim"],
            font=ctk.CTkFont(size=10, weight="bold"), text_color=_C["dim"],
            command=self._font_increase
        ).pack(side="left", padx=1)

        ctk.CTkButton(
            right, text="A-", width=22, height=20,
            fg_color="transparent", hover_color=_C["dim"],
            font=ctk.CTkFont(size=10, weight="bold"), text_color=_C["dim"],
            command=self._font_decrease
        ).pack(side="left", padx=1)

        ctk.CTkButton(
            right, text="⚙", width=22, height=20,
            fg_color="transparent", hover_color=_C["dim"],
            font=ctk.CTkFont(size=11), text_color=_C["dim"],
            command=self._open_settings
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            right, text="—", width=22, height=20,
            fg_color="transparent", hover_color=_C["dim"],
            font=ctk.CTkFont(size=11), text_color=_C["dim"],
            command=self._minimize
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            right, text="✕", width=22, height=20,
            fg_color="transparent", hover_color="#3d1014",
            font=ctk.CTkFont(size=11), text_color=_C["dim"],
            command=self._close
        ).pack(side="left", padx=(2, 0))

    def _build_content(self):
        """Source + target text card."""
        card = ctk.CTkFrame(
            self, fg_color=_C["card"], corner_radius=0,
            border_width=1, border_color=_C["border"]
        )
        card.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        card.columnconfigure(0, weight=1)

        # Source-language row.
        tr_row = ctk.CTkFrame(card, fg_color="transparent")
        tr_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 2))

        ctk.CTkLabel(
            tr_row, text=t("tr_label"),
            font=ctk.CTkFont(size=8, weight="bold"),
            text_color=_C["dim"],
            width=20
        ).pack(side="left", anchor="n", padx=(0, 6))

        self._tr_label = ctk.CTkLabel(
            tr_row, text=t("waiting"),
            font=ctk.CTkFont(size=12), text_color=_C["gray"],
            anchor="w", justify="left", wraplength=420
        )
        self._tr_label.pack(side="left", fill="x", expand=True)

        # Divider.
        sep = ctk.CTkFrame(card, fg_color=_C["border"], height=1)
        sep.grid(row=1, column=0, sticky="ew", padx=12, pady=4)

        # Target-language row.
        en_row = ctk.CTkFrame(card, fg_color="transparent")
        en_row.grid(row=2, column=0, sticky="ew", padx=12, pady=(2, 10))

        ctk.CTkLabel(
            en_row, text=t("en_label"),
            font=ctk.CTkFont(size=8, weight="bold"),
            text_color=_C["blue"],
            width=20
        ).pack(side="left", anchor="n", padx=(0, 6))

        # Read the font size from config.
        fsize = self.cfg.get("overlay", "font_size", default=13)
        self._en_label = ctk.CTkLabel(
            en_row, text="—",
            font=ctk.CTkFont(size=fsize, weight="bold"),
            text_color=_C["white"],
            anchor="w", justify="left", wraplength=420
        )
        self._en_label.pack(side="left", fill="x", expand=True)

    def _build_statusbar(self):
        """Bottom status bar — engine + latency information."""
        bar = ctk.CTkFrame(self, fg_color=_C["bar"], corner_radius=0, height=22)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)

        self._status_label = ctk.CTkLabel(
            bar, text=t("ready_vad"),
            font=ctk.CTkFont(size=9), text_color=_C["dim"]
        )
        self._status_label.pack(side="left", padx=10)

        # Engine label (right).
        self._engine_label = ctk.CTkLabel(
            bar, text="",
            font=ctk.CTkFont(size=9), text_color=_C["dim"]
        )
        self._engine_label.pack(side="right", padx=10)

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _start_poll(self):
        self._poll()

    def _poll(self):
        """
        Drain result_queue and update the UI accordingly.

        Reschedules itself every POLL_MS milliseconds.
        """
        if self._rq is not None:
            try:
                while True:                        # Consume every queued message.
                    data = self._rq.get_nowait()
                    self._handle_result(data)
            except queue.Empty:
                pass

        self.after(POLL_MS, self._poll)

    def _handle_result(self, data: dict):
        """Update the UI based on the incoming result dictionary."""
        if data.get("error"):
            self._show_error(str(data["error"]))
            return

        direction = data.get("direction", "outbound")
        text_tr = data.get("text_tr", "")
        text_en = data.get("text_en", "")
        engine  = data.get("engine", "")
        e2e_ms  = data.get("latency_ms", 0)
        stt_ms  = data.get("stt_ms", 0)
        llm_ms  = data.get("llm_ms", 0)

        # Active translation — turn off click-through so the window can still be dragged.
        self._set_clickthrough(False)

        if direction == "inbound":
            # Counterpart is speaking — green theme.
            self._tr_label.configure(text=text_tr, text_color=_C["inbound_text"])
            self._en_label.configure(text=text_en, text_color=_C["white"])
            self._dot.configure(text_color=_C["inbound_text"])
            self._dir_label.configure(text=t("overlay_inbound_label"), text_color=_C["inbound_text"])
            status_color = _C["inbound_text"]
        else:
            # User is speaking — blue theme.
            self._tr_label.configure(text=text_tr, text_color=_C["gray"])
            self._en_label.configure(text=text_en, text_color=_C["white"])
            self._dot.configure(text_color=_C["green"])
            self._dir_label.configure(text=t("overlay_outbound_label"), text_color=_C["outbound_text"])
            status_color = _C["green"]

        # Status bar.
        e2e_s = f"{e2e_ms/1000:.1f}s" if e2e_ms else "—"
        self._status_label.configure(
            text=f"{t('stt')} {stt_ms}ms · {t('llm')} {llm_ms}ms · {t('e2e')} {e2e_s}",
            text_color=status_color
        )
        self._engine_label.configure(text=engine, text_color=_C["blue"])
        self._last_update = time.time()

        # Fade back to the idle indicator after a few seconds.
        if self._fade_job:
            try:
                self.after_cancel(self._fade_job)
            except Exception:
                pass
            self._fade_job = None
        self._fade_job = self.after(FADE_MS, self._fade_to_idle)

    def _show_error(self, msg: str):
        self._tr_label.configure(text=t("error"), text_color=_C["red"])
        self._en_label.configure(text=msg[:80], text_color=_C["red"])
        self._dot.configure(text_color=_C["red"])
        self._status_label.configure(text=t("error_caught"), text_color=_C["red"])

    def _fade_to_idle(self):
        """Revert the indicator to the idle state after a brief delay."""
        self._dot.configure(text_color=_C["yellow"])
        self._dir_label.configure(text="")
        self._status_label.configure(
            text=t("ready_vad"), text_color=_C["dim"]
        )
        self._engine_label.configure(text="")

    def _set_clickthrough(self, enable: bool):
        """
        Windows: toggle WS_EX_TRANSPARENT so the overlay is pass-through to clicks.

        Disabled during an active translation so the user can still drag the window.
        """
        if sys.platform != "win32":
            return
        try:
            import ctypes
            hwnd  = self.winfo_id()
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)   # GWL_EXSTYLE
            if enable:
                ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20)
            else:
                ctypes.windll.user32.SetWindowLongW(hwnd, -20, style & ~0x20)
        except Exception:
            pass

    # ── Window dragging ───────────────────────────────────────────────────────

    def _drag_start(self, event):
        self._dragging = True
        self._last_x = event.x_root
        self._last_y = event.y_root

    def _drag_move(self, event):
        if self._dragging:
            dx = event.x_root - self._last_x
            dy = event.y_root - self._last_y

            new_x = self.winfo_x() + dx
            new_y = self.winfo_y() + dy

            self.geometry(f"+{new_x}+{new_y}")
            self._last_x = event.x_root
            self._last_y = event.y_root

    def _drag_end(self, event):
        self._dragging = False
        # Persist the new position to config.
        self.cfg.set("overlay", "position_x", self.winfo_x())
        self.cfg.set("overlay", "position_y", self.winfo_y())
        self.cfg.save()

    # ── Button actions ────────────────────────────────────────────────────────

    def _minimize(self):
        self.wm_attributes("-alpha", 0.0)
        self.after(200, lambda: self.wm_attributes("-alpha", 0.15))

    def _close(self):
        self.cfg.set("overlay", "position_x", self.winfo_x())
        self.cfg.set("overlay", "position_y", self.winfo_y())
        self.cfg.save()
        self.withdraw()   # Hide instead of destroying — keep the poll loop alive.

    def _open_settings(self):
        """Open the settings panel (overridden by app.py)."""
        pass   # app.py monkey-patches this method.

    def _font_increase(self):
        curr = self._en_label.cget("font").cget("size")
        new_size = min(curr + 1, 24)
        self._en_label.configure(font=ctk.CTkFont(size=new_size, weight="bold"))
        self.cfg.set("overlay", "font_size", new_size)
        self.cfg.save()

    def _font_decrease(self):
        curr = self._en_label.cget("font").cget("size")
        new_size = max(curr - 1, 8)
        self._en_label.configure(font=ctk.CTkFont(size=new_size, weight="bold"))
        self.cfg.set("overlay", "font_size", new_size)
        self.cfg.save()

    # ── External API ─────────────────────────────────────────────────────────

    def _safe(self, fn):
        """Silently swallow after() callbacks delivered after the widget tree is torn down."""
        def _wrapped():
            try:
                if self.winfo_exists():
                    fn()
            except Exception:
                pass
        return _wrapped

    def set_status(self, msg: str, color: str = None):
        """Thread-safe entry point for background threads to push status messages."""
        self.after(0, self._safe(lambda: self._status_label.configure(
            text=msg, text_color=color or _C["dim"]
        )))

    def set_listening(self):
        """Called when recording starts."""
        self.after(0, self._safe(lambda: [
            self._dot.configure(text_color=_C["green"]),
            self._status_label.configure(
                text=t("recording"), text_color=_C["green"]
            ),
        ]))

    def set_processing(self):
        """Processing in progress."""
        self.after(0, self._safe(lambda: [
            self._dot.configure(text_color=_C["blue"]),
            self._status_label.configure(
                text=t("processing"), text_color=_C["blue"]
            ),
        ]))


# ══════════════════════════════════════════════════════════════════════════════
# Demo mode (standalone test)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os, threading
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from gui.config import ConfigManager

    cfg = ConfigManager()
    rq  = queue.Queue()

    overlay = Overlay(cfg, result_queue=rq)

    # Send demo data after 2 seconds.
    def _demo():
        time.sleep(2.0)
        rq.put({
            "text_tr":    "Yapay zeka modelleri gün geçtikçe daha hızlı gelişiyor.",
            "text_en":    "Artificial intelligence models are developing faster day by day.",
            "engine":     "gemma-4",
            "latency_ms": 1820,
            "stt_ms":     940,
            "llm_ms":     380,
            "tts_ms":     500,
            "error":      None,
        })
        time.sleep(6.0)
        rq.put({
            "text_tr":    "Bu proje Hatay'daki kamp için hazırlanıyor.",
            "text_en":    "This project is being prepared for the camp in Hatay.",
            "engine":     "gemini-2.5-flash",
            "latency_ms": 2100,
            "stt_ms":     810,
            "llm_ms":     520,
            "tts_ms":     770,
            "error":      None,
        })

    threading.Thread(target=_demo, daemon=True).start()
    overlay.mainloop()
