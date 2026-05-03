"""
Gemma Echo — Ana Hub Penceresi

Sihirbaz bitince açılan ilk ekran.
Kullanıcı buradan ne yapmak istediğini seçer:
  • Canlı Çeviri   — mikrofon + overlay
  • Metin Çevirisi — yazarak çeviri
  • Video Çevirisi — video dosyası yükle
  • Ses Çevirisi   — ses dosyası yükle
"""

import sys
import threading
import customtkinter as ctk
from tkinter import messagebox

from gui.config import ConfigManager

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
    Ana hub penceresi — uygulamanın merkezi.
    HomeWindow.mainloop() tüm olay döngüsünü sürer.
    """

    def __init__(self, cfg: ConfigManager, app):
        super().__init__()
        self.cfg = cfg
        self.app = app

        self.title("Gemma Echo")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(fg_color=_C["bg"])
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._center()
        self._build()
        self._poll_backend()  # backend hazır mı izle

    def _center(self):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - WIN_W) // 2
        y = (sh - WIN_H) // 2
        self.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        # Başlık
        hdr = ctk.CTkFrame(self, fg_color=_C["accent"], corner_radius=0, height=60)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr, text="GEMMA ECHO",
            font=ctk.CTkFont(family="Helvetica", size=22, weight="bold"),
            text_color="white"
        ).pack(side="left", padx=24, pady=14)

        ctk.CTkLabel(
            hdr, text="Yapay Zeka Çeviri Asistanı",
            font=ctk.CTkFont(size=11), text_color="#aabbcc"
        ).pack(side="left", padx=4)

        # Sağ üst butonlar
        btns = ctk.CTkFrame(hdr, fg_color="transparent")
        btns.pack(side="right", padx=12)
        ctk.CTkButton(
            btns, text="⚙ Ayarlar", width=90, height=32,
            fg_color="transparent", hover_color=_C["accent"],
            border_width=1, border_color="#aabbcc",
            font=ctk.CTkFont(size=11),
            command=self._open_settings
        ).pack(side="left", padx=4)

        # Alt çubuk (önce pack — tkinter kuralı)
        self._status_bar = ctk.CTkFrame(self, fg_color=_C["panel"], corner_radius=0, height=36)
        self._status_bar.pack(fill="x", side="bottom")
        self._status_bar.pack_propagate(False)

        hw = self.cfg.get("hardware")
        gpu_name = hw.get("gpu", {}).get("name", "CPU") if hw else "CPU"
        mode = self.cfg.get("mode", "current", default="online").upper()
        ctk.CTkLabel(
            self._status_bar,
            text=f"  Donanım: {gpu_name}   |   Mod: {mode}",
            font=ctk.CTkFont(size=10), text_color=_C["dim"]
        ).pack(side="left")

        self._backend_lbl = ctk.CTkLabel(
            self._status_bar, text="Modeller yükleniyor...",
            font=ctk.CTkFont(size=10), text_color=_C["yellow"]
        )
        self._backend_lbl.pack(side="right", padx=12)

        # Soru etiketi
        body = ctk.CTkFrame(self, fg_color=_C["bg"])
        body.pack(fill="both", expand=True, padx=32, pady=20)

        ctk.CTkLabel(
            body, text="Ne yapmak istiyorsunuz?",
            font=ctk.CTkFont(size=16, weight="bold"), text_color=_C["white"]
        ).pack(anchor="w", pady=(0, 18))

        # 2×2 Mod Kartları
        grid = ctk.CTkFrame(body, fg_color="transparent")
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        cards = [
            (0, 0, "🎙", "Canlı Çeviri",   "Toplantı · Zoom · Gerçek zamanlı",  self._start_live),
            (0, 1, "📝", "Metin Çevirisi",  "Yazarak TR→EN anlık çeviri",         self._open_text),
            (1, 0, "🎬", "Video Çevirisi",  "MP4 · MKV · AVI dosyası yükle",     self._open_video),
            (1, 1, "🎵", "Ses Çevirisi",    "WAV · MP3 · M4A dosyası yükle",     self._open_audio),
        ]
        self._live_btn = None

        for row, col, icon, title, desc, cmd in cards:
            btn = _ModeCard(grid, icon=icon, title=title, desc=desc, command=cmd)
            btn.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            if title == "Canlı Çeviri":
                self._live_btn = btn

    # ── Mod Eylemleri ─────────────────────────────────────────────────────────

    def _start_live(self):
        if not self.app._backend_ready:
            messagebox.showinfo(
                "Modeller Hazırlanıyor",
                "Modeller henüz yükleniyor.\nBir dakika bekleyip tekrar deneyin."
            )
            return

        if self.app._recorder is not None:
            # Zaten çalışıyor — durdur
            self.app.stop_live()
            self._live_btn.set_active(False)
            if self.app._overlay:
                self.app._overlay.withdraw()
        else:
            # Başlat
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

    # ── Backend Takibi ────────────────────────────────────────────────────────

    def _poll_backend(self):
        if self.app._backend_ready:
            self._backend_lbl.configure(
                text="Hazır ✓", text_color=_C["green"]
            )
        else:
            self.after(1000, self._poll_backend)

    # ── Kapatma ───────────────────────────────────────────────────────────────

    def _on_close(self):
        self.app.stop_live()
        self.quit()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# Metin Çeviri Penceresi (basit, bağımsız)
# ══════════════════════════════════════════════════════════════════════════════

class TextModeWindow(ctk.CTkToplevel):
    """Kullanıcının yazarak çeviri yaptığı mini pencere."""

    def __init__(self, master):
        super().__init__(master)
        self.title("Metin Çevirisi — Gemma Echo")
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
            hdr, text="Metin Çevirisi  TR → EN",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="white"
        ).pack(side="left", padx=16, pady=10)

        # Giriş
        ctk.CTkLabel(self, text="Türkçe metin:", font=ctk.CTkFont(size=11),
                     text_color=_C["gray"]).pack(anchor="w", padx=16, pady=(12, 2))
        self._in = ctk.CTkTextbox(self, height=90, font=ctk.CTkFont(size=12),
                                  fg_color=_C["card"], wrap="word")
        self._in.pack(fill="x", padx=16)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=8)
        ctk.CTkButton(
            btn_row, text="Çevir  ▶", width=100, height=32,
            fg_color=_C["blue"], command=self._translate
        ).pack(side="left")
        ctk.CTkButton(
            btn_row, text="Temizle", width=80, height=32,
            fg_color=_C["panel"], command=self._clear
        ).pack(side="left", padx=8)

        # Çıktı
        ctk.CTkLabel(self, text="İngilizce çeviri:", font=ctk.CTkFont(size=11),
                     text_color=_C["blue"]).pack(anchor="w", padx=16, pady=(0, 2))
        self._out = ctk.CTkTextbox(self, height=90, font=ctk.CTkFont(size=12, weight="bold"),
                                   fg_color=_C["card"], wrap="word", state="disabled")
        self._out.pack(fill="x", padx=16, pady=(0, 12))

    def _translate(self):
        text = self._in.get("0.0", "end").strip()
        if not text:
            return
        if not self.app._backend_ready:
            messagebox.showinfo("Bekleyin", "Modeller henüz yükleniyor.")
            return

        self._out.configure(state="normal")
        self._out.delete("0.0", "end")
        self._out.insert("0.0", "Çevriliyor...")
        self._out.configure(state="disabled")

        def _run():
            try:
                result = self.app._orchestrator.translator.translate(text)
                en = result.get("translation", "")
                self.after(0, lambda: self._show_result(en))
            except Exception as e:
                self.after(0, lambda: self._show_result(f"Hata: {e}"))

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
# Mod Kartı Widget
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

        # Tıklama bağlamaları
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
