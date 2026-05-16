"""
Gemma Echo — Main window (single-window SPA).

Sidebar-driven, modular single-shell architecture. Each view (View class) is
managed as its own module.
"""

import customtkinter as ctk
from gui.config import ConfigManager
from gui.i18n import t

from gui.pages._helpers import _C, _header, _card
from gui.pages.live_view import LiveView
from gui.pages.media_view import MediaView
from gui.pages.book_view import BookView
from gui.pages.text_view import TextView
from gui.pages.settings_view import SettingsView

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

WIN_W, WIN_H = 960, 660
SIDEBAR_W    = 210

class MainWindow(ctk.CTk):
    """Single-shell main window — the central router for every feature."""

    def __init__(self, cfg: ConfigManager, app):
        super().__init__()
        self.cfg = cfg
        self.app = app

        self.title(t("app_name"))
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.minsize(800, 560)
        self.configure(fg_color=_C["bg"])
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._center()
        self._build()
        self._poll_backend()

    def _center(self):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x  = (sw - WIN_W) // 2
        y  = (sh - WIN_H) // 2
        self.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

    def _build(self):
        # Sidebar (left, fixed width).
        self._sidebar = _Sidebar(self, on_nav=self.switch_view, app=self.app)
        self._sidebar.pack(side="left", fill="y")

        # Vertical separator.
        ctk.CTkFrame(self, width=1, fg_color=_C["border"]).pack(
            side="left", fill="y"
        )

        # Content area (right, flexible).
        shell = ctk.CTkFrame(self, fg_color=_C["bg"], corner_radius=0)
        shell.pack(side="left", fill="both", expand=True)

        self._views = {
            "live":     LiveView(shell, cfg=self.cfg, app=self.app),
            "media":    MediaView(shell, cfg=self.cfg, app=self.app),
            "book":     BookView(shell, cfg=self.cfg, app=self.app),
            "text":     TextView(shell, cfg=self.cfg, app=self.app),
            "settings": SettingsView(shell, cfg=self.cfg, app=self.app),
        }
        self.switch_view("live")

    def switch_view(self, name: str):
        # Fire the on-leave hook for the page we are leaving so it can cancel work.
        current_view_name = getattr(self._sidebar, "_active", None)
        if current_view_name and current_view_name in self._views:
            curr_view = self._views[current_view_name]
            if hasattr(curr_view, "on_leave"):
                curr_view.on_leave()

        for v in self._views.values():
            v.pack_forget()
        view = self._views[name]
        if hasattr(view, "_update_language_labels"):
            view._update_language_labels()
        view.pack(fill="both", expand=True)
        self._sidebar.set_active(name)

    def set_status(self, text: str, color: str):
        self._sidebar.set_status(text, color)

    def _poll_backend(self):
        if self.app._backend_ready:
            self._sidebar.set_status(t("ready_tick"), _C["green"])
        else:
            self.after(1000, self._poll_backend)

    def _on_close(self):
        self.app.stop_live()
        self.app.stop_inbound()
        self.quit()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

class _Sidebar(ctk.CTkFrame):
    _NAV = [
        ("live",     "\U0001f399",  "nav_live"),
        ("media",    "\U0001f3ac",  "nav_media"),
        ("book",     "\U0001f4d6",  "nav_book"),
        ("text",     "\U0001f4dd",  "nav_text"),
        ("settings", "\u2699",      "nav_settings"),
    ]

    def __init__(self, master, on_nav, app):
        super().__init__(
            master, width=SIDEBAR_W,
            fg_color=_C["sidebar"], corner_radius=0
        )
        self.on_nav  = on_nav
        self.app     = app
        self._active = "live"
        self._btns:  dict = {}
        self.pack_propagate(False)
        self._build()

    def _build(self):
        # ── Logo ──────────────────────────────────────────────────────
        logo_area = ctk.CTkFrame(self, fg_color="transparent", height=78)
        logo_area.pack(fill="x")
        logo_area.pack_propagate(False)

        row = ctk.CTkFrame(logo_area, fg_color="transparent")
        row.place(relx=0.08, rely=0.44, anchor="w")

        ctk.CTkLabel(
            row, text="GEMMA ",
            font=ctk.CTkFont(family="Helvetica", size=16, weight="bold"),
            text_color=_C["blue"]
        ).pack(side="left")
        ctk.CTkLabel(
            row, text="ECHO",
            font=ctk.CTkFont(family="Helvetica", size=16, weight="bold"),
            text_color=_C["text"]
        ).pack(side="left")
        ctk.CTkLabel(
            logo_area, text="AI Translation Suite",
            font=ctk.CTkFont(size=9), text_color=_C["dim"]
        ).place(relx=0.08, rely=0.78, anchor="w")

        # Separator.
        ctk.CTkFrame(self, height=1, fg_color=_C["border"]).pack(fill="x")

        # ── Navigation ────────────────────────────────────────────────
        nav = ctk.CTkFrame(self, fg_color="transparent")
        nav.pack(fill="x", pady=(8, 0))

        ctk.CTkLabel(
            nav, text=t("modes"),
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=_C["dim"]
        ).pack(anchor="w", padx=18, pady=(10, 4))

        for key, icon, label_key in self._NAV:
            btn = ctk.CTkButton(
                nav,
                text=f"  {icon}   {t(label_key)}",
                font=ctk.CTkFont(size=12),
                fg_color="transparent",
                hover_color=_C["surface2"],
                text_color=_C["muted"],
                anchor="w",
                height=46,
                corner_radius=10,
                command=lambda k=key: self.on_nav(k),
            )
            btn.pack(fill="x", padx=8, pady=2)
            self._btns[key] = btn

        # ── Bottom: status bar ────────────────────────────────────────
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=12, pady=14)

        ctk.CTkFrame(bottom, height=1, fg_color=_C["border"]).pack(
            fill="x", pady=(0, 10)
        )
        row2 = ctk.CTkFrame(bottom, fg_color="transparent")
        row2.pack(fill="x")

        self._dot = ctk.CTkLabel(
            row2, text="\u25cf",
            font=ctk.CTkFont(size=9),
            text_color=_C["yellow"]
        )
        self._dot.pack(side="left")

        self._lbl = ctk.CTkLabel(
            row2, text=t("loading_models"),
            font=ctk.CTkFont(size=9),
            text_color=_C["muted"], anchor="w"
        )
        self._lbl.pack(side="left", padx=6, fill="x", expand=True)

        ctk.CTkLabel(
            bottom, text="Gemma Echo  v1.0",
            font=ctk.CTkFont(size=8), text_color=_C["dim"]
        ).pack(anchor="w", pady=(8, 0))

    def set_active(self, name: str):
        self._active = name
        for key, btn in self._btns.items():
            if key == name:
                btn.configure(fg_color=_C["blue_bg"], text_color=_C["blue"])
            else:
                btn.configure(fg_color="transparent", text_color=_C["muted"])

    def set_status(self, text: str, color: str):
        self._dot.configure(text_color=color)
        self._lbl.configure(text=text)
