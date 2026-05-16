"""
Gemma Echo — Main hub window.

The first screen shown after the setup wizard completes. The user picks the
desired workflow from here:
  • Live translation   — microphone + overlay.
  • Text translation   — type-to-translate.
  • Video translation  — load a video file.
  • Audio translation  — load an audio file.
"""

import sys
import threading
import customtkinter as ctk
from tkinter import messagebox

from gui.config import ConfigManager
from gui.i18n import t

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_C = {
    "bg":     "#0d1117",
    "panel":  "#161b22",
    "card":   "#1c2128",
    "card_h": "#21262d",
    "border": "#30363d",
    "blue":   "#58a6ff",
    "green":  "#3fb950",
    "yellow": "#d29922",
    "red":    "#f85149",
    "gray":   "#8b949e",
    "white":  "#e6edf3",
    "dim":    "#484f58",
    "accent": "#0f3460",
}

WIN_W, WIN_H = 680, 520


class HomeWindow(ctk.CTk):
    """
    Main hub window — the application's central control surface.

    HomeWindow.mainloop() drives the entire event loop.
    """

    def __init__(self, cfg: ConfigManager, app):
        super().__init__()
        self.cfg = cfg
        self.app = app

        self.title(t("app_name"))
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(fg_color=_C["bg"])
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._center()
        self._build()
        self._poll_backend()  # Monitor backend readiness.

    def _center(self):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - WIN_W) // 2
        y = (sh - WIN_H) // 2
        self.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        # Header.
        hdr = ctk.CTkFrame(self, fg_color=_C["accent"], corner_radius=0, height=60)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr, text=t("app_name"),
            font=ctk.CTkFont(family="Helvetica", size=22, weight="bold"),
            text_color="white"
        ).pack(side="left", padx=24, pady=14)

        ctk.CTkLabel(
            hdr, text=t("home_tagline"),
            font=ctk.CTkFont(size=11), text_color="#aabbcc"
        ).pack(side="left", padx=4)

        # Top-right buttons.
        btns = ctk.CTkFrame(hdr, fg_color="transparent")
        btns.pack(side="right", padx=12)
        ctk.CTkButton(
            btns, text=f"⚙ {t('settings')}", width=110, height=32,
            fg_color="transparent", hover_color=_C["accent"],
            border_width=1, border_color="#aabbcc",
            font=ctk.CTkFont(size=11),
            command=self._open_settings
        ).pack(side="left", padx=4)

        # Bottom status bar (packed first — Tk packing order requirement).
        self._status_bar = ctk.CTkFrame(self, fg_color=_C["panel"], corner_radius=0, height=36)
        self._status_bar.pack(fill="x", side="bottom")
        self._status_bar.pack_propagate(False)

        hw = self.cfg.get("hardware")
        gpu_name = hw.get("gpu", {}).get("name", "CPU") if hw else "CPU"
        mode = self.cfg.get("mode", "current", default="online").upper()
        ctk.CTkLabel(
            self._status_bar,
            text=f"  {t('hardware_label')}: {gpu_name}   |   {t('mode_label')}: {mode}",
            font=ctk.CTkFont(size=10), text_color=_C["dim"]
        ).pack(side="left")

        self._backend_lbl = ctk.CTkLabel(
            self._status_bar, text=t("models_loading_short"),
            font=ctk.CTkFont(size=10), text_color=_C["yellow"]
        )
        self._backend_lbl.pack(side="right", padx=12)

        # Prompt label.
        body = ctk.CTkFrame(self, fg_color=_C["bg"])
        body.pack(fill="both", expand=True, padx=32, pady=20)

        ctk.CTkLabel(
            body, text=t("home_question"),
            font=ctk.CTkFont(size=16, weight="bold"), text_color=_C["white"]
        ).pack(anchor="w", pady=(0, 18))

        # 2×2 mode card grid.
        grid = ctk.CTkFrame(body, fg_color="transparent")
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        cards = [
            (0, 0, "🎙", t("home_card_live"),  t("home_card_live_desc"),  self._start_live),
            (0, 1, "📝", t("home_card_text"),  t("home_card_text_desc"),  self._open_text),
            (1, 0, "🎬", t("home_card_video"), t("home_card_video_desc"), self._open_video),
            (1, 1, "🎵", t("home_card_audio"), t("home_card_audio_desc"), self._open_audio),
        ]
        self._live_btn = None

        for row, col, icon, title, desc, cmd in cards:
            btn = _ModeCard(grid, icon=icon, title=title, desc=desc, command=cmd)
            btn.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            if cmd is self._start_live:
                self._live_btn = btn

    # ── Mode actions ──────────────────────────────────────────────────────────

    def _start_live(self):
        if not self.app._backend_ready:
            messagebox.showinfo(t("please_wait"), t("models_loading_short"))
            return

        if self.app._recorder is not None:
            # Already running — stop.
            self.app.stop_live()
            self.app.stop_inbound()
            self.app._ptt_mode = None
            self.app._sync_ptt_hotkeys_state()
            self._live_btn.set_active(False)
            if self.app._overlay:
                self.app._overlay.withdraw()
        else:
            # Start.
            self.app.start_live()
            self._live_btn.set_active(True)
            if self.app._overlay:
                self.app._overlay.deiconify()
                self.app._overlay.lift()

    def _open_text(self):
        TextModeWindow(self)

    def _open_video(self):
        if self.app._file_mode_win:
            self.app._file_mode_win.show(filter_type="video")

    def _open_audio(self):
        if self.app._file_mode_win:
            self.app._file_mode_win.show(filter_type="audio")

    def _open_settings(self):
        if hasattr(self.app, '_panel') and self.app._panel:
            self.app._panel.show()

    # ── Backend polling ───────────────────────────────────────────────────────

    def _poll_backend(self):
        if self.app._backend_ready:
            self._backend_lbl.configure(
                text=t("ready_tick"), text_color=_C["green"]
            )
        else:
            self.after(1000, self._poll_backend)

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        self.app.stop_live()
        self.quit()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# Text translation window (small, standalone)
# ══════════════════════════════════════════════════════════════════════════════

class TextModeWindow(ctk.CTkToplevel):
    """Lightweight type-to-translate window."""

    def __init__(self, master):
        super().__init__(master)
        self.title(f"{t('text_title')} — {t('app_name')}")
        self.geometry("560x360")
        self.resizable(True, True)
        self.configure(fg_color=_C["bg"])
        self.app = master.app

        self._center()
        self._build()
        self.lift()
        self.focus()

    def _center(self):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"560x360+{(sw-560)//2}+{(sh-360)//2}")

    def _build(self):
        hdr = ctk.CTkFrame(self, fg_color=_C["accent"], corner_radius=0, height=44)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(
            hdr, text=t("text_title"),
            font=ctk.CTkFont(size=13, weight="bold"), text_color="white"
        ).pack(side="left", padx=16, pady=10)

        # Input.
        ctk.CTkLabel(self, text=t("source_text"), font=ctk.CTkFont(size=11),
                     text_color=_C["gray"]).pack(anchor="w", padx=16, pady=(12, 2))
        self._in = ctk.CTkTextbox(self, height=90, font=ctk.CTkFont(size=12),
                                  fg_color=_C["card"], wrap="word")
        self._in.pack(fill="x", padx=16)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=8)
        ctk.CTkButton(
            btn_row, text=t("translate"), width=100, height=32,
            fg_color=_C["blue"], command=self._translate
        ).pack(side="left")
        ctk.CTkButton(
            btn_row, text=t("clear"), width=80, height=32,
            fg_color=_C["panel"], command=self._clear
        ).pack(side="left", padx=8)

        # Output.
        ctk.CTkLabel(self, text=t("target_text"), font=ctk.CTkFont(size=11),
                     text_color=_C["blue"]).pack(anchor="w", padx=16, pady=(0, 2))
        self._out = ctk.CTkTextbox(self, height=90, font=ctk.CTkFont(size=12, weight="bold"),
                                   fg_color=_C["card"], wrap="word", state="disabled")
        self._out.pack(fill="x", padx=16, pady=(0, 12))

    def _translate(self):
        text = self._in.get("0.0", "end").strip()
        if not text:
            return
        if not self.app._backend_ready:
            messagebox.showinfo(t("please_wait"), t("models_loading_short"))
            return

        self._out.configure(state="normal")
        self._out.delete("0.0", "end")
        self._out.insert("0.0", t("translating"))
        self._out.configure(state="disabled")

        def _run():
            try:
                result = self.app._orchestrator.translator.translate(text)
                en = result.get("translation", "")
                self.after(0, lambda: self._show_result(en))
            except Exception as e:
                self.after(0, lambda: self._show_result(t("error_message", e)))

        threading.Thread(target=_run, daemon=True).start()

    def _show_result(self, text: str):
        self._out.configure(state="normal")
        self._out.delete("0.0", "end")
        self._out.insert("0.0", text)
        self._out.configure(state="disabled")

    def _clear(self):
        self._in.delete("0.0", "end")
        self._out.configure(state="normal")
        self._out.delete("0.0", "end")
        self._out.configure(state="disabled")


# ══════════════════════════════════════════════════════════════════════════════
# Mode card widget
# ══════════════════════════════════════════════════════════════════════════════

class _ModeCard(ctk.CTkFrame):
    def __init__(self, master, icon: str, title: str, desc: str, command):
        super().__init__(
            master, fg_color=_C["card"], corner_radius=12,
            border_width=1, border_color=_C["border"],
            cursor="hand2"
        )
        self._cmd = command
        self._active = False

        ctk.CTkLabel(
            self, text=icon, font=ctk.CTkFont(size=28)
        ).pack(pady=(18, 4))

        ctk.CTkLabel(
            self, text=title,
            font=ctk.CTkFont(size=14, weight="bold"), text_color=_C["white"]
        ).pack()

        ctk.CTkLabel(
            self, text=desc,
            font=ctk.CTkFont(size=10), text_color=_C["gray"],
            wraplength=240
        ).pack(pady=(2, 16))

        # Bind click events.
        for w in (self, *self.winfo_children()):
            w.bind("<Button-1>", self._on_click)
            w.bind("<Enter>",    self._on_enter)
            w.bind("<Leave>",    self._on_leave)

    def _on_click(self, _=None):
        self._cmd()

    def _on_enter(self, _=None):
        if not self._active:
            self.configure(fg_color=_C["card_h"])

    def _on_leave(self, _=None):
        if not self._active:
            self.configure(fg_color=_C["card"])

    def set_active(self, active: bool):
        self._active = active
        if active:
            self.configure(fg_color="#1a3a1a", border_color=_C["green"])
        else:
            self.configure(fg_color=_C["card"], border_color=_C["border"])
