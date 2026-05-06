"""
Gemma Echo — Ortak Arayüz Yardımcıları (UI Helpers)
"""

import customtkinter as ctk
from gui.i18n import t

# ── Premium Renk Paleti ────────────────────────────────────────────────────────
_C = {
    "bg":       "#0b0b12",
    "sidebar":  "#0d0d18",
    "surface":  "#141422",
    "surface2": "#1c1c2e",
    "border":   "#252540",
    "blue":     "#5b9ef9",
    "blue_bg":  "#12264a",
    "green":    "#23d05e",
    "green_bg": "#0a2318",
    "yellow":   "#f5a623",
    "red":      "#f04747",
    "red_bg":   "#2a0d0d",
    "text":     "#e8ecf1",
    "muted":    "#7a8499",
    "dim":      "#3a3d52",
}

def _header(parent, title: str, subtitle: str):
    hdr = ctk.CTkFrame(parent, fg_color="transparent", height=76)
    hdr.pack(fill="x", padx=28, pady=(22, 6))
    hdr.pack_propagate(False)
    ctk.CTkLabel(
        hdr, text=title,
        font=ctk.CTkFont(size=20, weight="bold"),
        text_color=_C["text"], anchor="w"
    ).pack(anchor="w")
    ctk.CTkLabel(
        hdr, text=subtitle,
        font=ctk.CTkFont(size=11),
        text_color=_C["muted"], anchor="w"
    ).pack(anchor="w")

def _card(parent, title: str) -> ctk.CTkFrame:
    """Baslikli, kenarlıklı kart — ic frame dondurur."""
    outer = ctk.CTkFrame(
        parent, fg_color=_C["surface"],
        corner_radius=14, border_width=1, border_color=_C["border"]
    )
    outer.pack(fill="x", pady=(0, 10))
    ctk.CTkLabel(
        outer, text=title.upper(),
        font=ctk.CTkFont(size=9, weight="bold"),
        text_color=_C["dim"]
    ).pack(anchor="w", padx=18, pady=(12, 0))
    ctk.CTkFrame(outer, height=1, fg_color=_C["border"]).pack(
        fill="x", padx=18, pady=(5, 8)
    )
    inner = ctk.CTkFrame(outer, fg_color="transparent")
    inner.pack(fill="x", padx=18, pady=(0, 14))
    return inner

def _section_lbl(parent, text: str):
    ctk.CTkLabel(
        parent, text=text,
        font=ctk.CTkFont(size=11, weight="bold"),
        text_color=_C["blue"]
    ).pack(anchor="w", padx=20, pady=(14, 2))
    ctk.CTkFrame(parent, height=1, fg_color=_C["border"]).pack(
        fill="x", padx=20, pady=(0, 6)
    )

class _InfoIcon(ctk.CTkButton):
    """Hover'da aciklama balonu gosteren [?] ikonu."""

    _active = None  # Tum ornekler arasinda tek aktif tooltip

    def __init__(self, parent, tooltip_text: str, **kwargs):
        super().__init__(
            parent, text="?", width=18, height=18,
            font=ctk.CTkFont(size=9, weight="bold"),
            corner_radius=9,
            fg_color=_C["surface2"], hover_color=_C["dim"],
            text_color=_C["muted"], border_width=0,
            **kwargs
        )
        self._tip_text = tooltip_text
        self._tip_win = None
        self._show_after_id = None
        self._focus_bind_id = None
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_leave)

    def _on_enter(self, event=None):
        if self._show_after_id:
            self.after_cancel(self._show_after_id)
        if _InfoIcon._active and _InfoIcon._active is not self:
            _InfoIcon._active._do_hide()
        self._show_after_id = self.after(150, self._do_show)

    def _on_leave(self, event=None):
        if self._show_after_id:
            self.after_cancel(self._show_after_id)
            self._show_after_id = None
        self._do_hide()

    def _do_show(self):
        self._show_after_id = None
        if self._tip_win:
            return
        x = self.winfo_rootx() + 22
        y = self.winfo_rooty()
        self._tip_win = ctk.CTkToplevel(self)
        self._tip_win.wm_overrideredirect(True)
        self._tip_win.wm_geometry(f"+{x}+{y}")
        ctk.CTkLabel(
            self._tip_win, text=self._tip_text,
            font=ctk.CTkFont(size=10),
            fg_color=_C["surface2"], text_color=_C["text"],
            corner_radius=8, wraplength=260,
            padx=10, pady=8
        ).pack()
        _InfoIcon._active = self
        try:
            root = self.winfo_toplevel()
            self._focus_bind_id = root.bind("<FocusOut>", self._on_leave, add="+")
        except Exception:
            pass

    def _do_hide(self):
        try:
            if self._focus_bind_id:
                self.winfo_toplevel().unbind("<FocusOut>", self._focus_bind_id)
                self._focus_bind_id = None
        except Exception:
            pass
        if self._tip_win:
            try:
                self._tip_win.destroy()
            except Exception:
                pass
            self._tip_win = None
        if _InfoIcon._active is self:
            _InfoIcon._active = None
