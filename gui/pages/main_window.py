"""
Gemma Echo — Ana Pencere (Single-Window SPA)

Sidebar tabanli, tek govde mimari.
Tum ozellikler (Canli, Medya, Metin, Ayarlar) bu pencere icinde yonetilir.
Overlay penceresi baginmsiz kalir (always-on-top altyazi).
"""

import os
import time
import threading
import webbrowser
import subprocess
import tempfile
import customtkinter as ctk
from tkinter import filedialog, messagebox

from gui.config import ConfigManager

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

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

WIN_W, WIN_H = 960, 660
SIDEBAR_W    = 210
_LLM_CHUNK   = 400
_AUDIO_EXT   = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
_VIDEO_EXT   = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts"}
_ALL_EXT     = _AUDIO_EXT | _VIDEO_EXT


# ── Yardimci widget fabrikaları ────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# Ana Pencere
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(ctk.CTk):
    """Tek govde ana pencere — tum ozellikler bu cerceve icinde."""

    def __init__(self, cfg: ConfigManager, app):
        super().__init__()
        self.cfg = cfg
        self.app = app

        self.title("Gemma Echo")
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
        # Sidebar (sol, sabit)
        self._sidebar = _Sidebar(self, on_nav=self.switch_view, app=self.app)
        self._sidebar.pack(side="left", fill="y")

        # Dikey ayirici
        ctk.CTkFrame(self, width=1, fg_color=_C["border"]).pack(
            side="left", fill="y"
        )

        # Icerik alani (sag, esnek)
        shell = ctk.CTkFrame(self, fg_color=_C["bg"], corner_radius=0)
        shell.pack(side="left", fill="both", expand=True)

        self._views = {
            "live":     LiveView(shell, cfg=self.cfg, app=self.app),
            "media":    MediaView(shell, cfg=self.cfg, app=self.app),
            "text":     TextView(shell, cfg=self.cfg, app=self.app),
            "settings": SettingsView(shell, cfg=self.cfg, app=self.app),
        }
        self.switch_view("live")

    def switch_view(self, name: str):
        for v in self._views.values():
            v.pack_forget()
        self._views[name].pack(fill="both", expand=True)
        self._sidebar.set_active(name)

    def set_status(self, text: str, color: str):
        self._sidebar.set_status(text, color)

    def _poll_backend(self):
        if self.app._backend_ready:
            self._sidebar.set_status("Hazir  \u2713", _C["green"])
        else:
            self.after(1000, self._poll_backend)

    def _on_close(self):
        self.app.stop_live()
        self.quit()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

class _Sidebar(ctk.CTkFrame):
    _NAV = [
        ("live",     "\U0001f399",  "Canli Ceviri"),
        ("media",    "\U0001f3ac",  "Medya"),
        ("text",     "\U0001f4dd",  "Metin"),
        ("settings", "\u2699",      "Ayarlar"),
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

        # Ayirici
        ctk.CTkFrame(self, height=1, fg_color=_C["border"]).pack(fill="x")

        # ── Navigasyon ────────────────────────────────────────────────
        nav = ctk.CTkFrame(self, fg_color="transparent")
        nav.pack(fill="x", pady=(8, 0))

        ctk.CTkLabel(
            nav, text="MODLAR",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=_C["dim"]
        ).pack(anchor="w", padx=18, pady=(10, 4))

        for key, icon, label in self._NAV:
            btn = ctk.CTkButton(
                nav,
                text=f"  {icon}   {label}",
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

        # ── Alt: durum ────────────────────────────────────────────────
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
            row2, text="Yukleniyor...",
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


# ══════════════════════════════════════════════════════════════════════════════
# Canli Ceviri View
# ══════════════════════════════════════════════════════════════════════════════

class LiveView(ctk.CTkFrame):
    def __init__(self, master, cfg: ConfigManager, app):
        super().__init__(master, fg_color=_C["bg"], corner_radius=0)
        self.cfg = cfg
        self.app = app
        self._recording = False
        self._build()

    def _build(self):
        _header(self, "\U0001f399  Canli Ceviri",
                "Mikrofon girdisini gercek zamanli Turkce \u2192 Ingilizce cevirir")

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=24, pady=(0, 12))

        # ── Kayit Kontrolu ────────────────────────────────────────────
        inner = _card(scroll, "Kayit Kontrolu")

        # Durum satirı
        status_row = ctk.CTkFrame(
            inner, fg_color=_C["surface2"],
            corner_radius=10, border_width=1, border_color=_C["border"]
        )
        status_row.pack(fill="x", pady=(0, 14))

        self._sdot = ctk.CTkLabel(
            status_row, text="\u23fa",
            font=ctk.CTkFont(size=13), text_color=_C["dim"]
        )
        self._sdot.pack(side="left", padx=(14, 8), pady=12)

        self._slbl = ctk.CTkLabel(
            status_row,
            text="Hazir  \u2014  Baslamak icin \u25b6 butonuna basin",
            font=ctk.CTkFont(size=11), text_color=_C["muted"], anchor="w"
        )
        self._slbl.pack(side="left", fill="x", expand=True)

        # Butonlar
        btn_row = ctk.CTkFrame(inner, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0, 10))

        self._btn_start = ctk.CTkButton(
            btn_row,
            text="\u25b6  Basla",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=_C["green"], hover_color="#1aad4e",
            height=46, corner_radius=12,
            command=self._start,
        )
        self._btn_start.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self._btn_stop = ctk.CTkButton(
            btn_row,
            text="\u25a0  Durdur",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=_C["surface2"], hover_color=_C["red_bg"],
            text_color=_C["dim"],
            height=46, corner_radius=12,
            state="disabled",
            command=self._stop,
        )
        self._btn_stop.pack(side="left", fill="x", expand=True)

        # Push-to-Talk
        ptt_row = ctk.CTkFrame(inner, fg_color="transparent")
        ptt_row.pack(fill="x", pady=(8, 0))

        lf = ctk.CTkFrame(ptt_row, fg_color="transparent")
        lf.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(
            lf, text="Bas-Konus (Push-to-Talk)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=_C["text"], anchor="w"
        ).pack(anchor="w")
        ctk.CTkLabel(
            lf, text="Kapali iken VAD otomatik konusma algilar",
            font=ctk.CTkFont(size=10), text_color=_C["muted"], anchor="w"
        ).pack(anchor="w")

        self._ptt = ctk.CTkSwitch(
            ptt_row, text="", width=52,
            command=self._on_ptt, onvalue=True, offvalue=False,
            progress_color=_C["blue"],
        )
        if self.cfg.get("recording", "push_to_talk", default=False):
            self._ptt.select()
        self._ptt.pack(side="right")

        # ── Altyazi Kontrolu ──────────────────────────────────────────
        ov = _card(scroll, "Altyazi Penceresi")
        ov_row = ctk.CTkFrame(ov, fg_color="transparent")
        ov_row.pack(fill="x")

        ctk.CTkLabel(
            ov_row,
            text="Gercek zamanli TR/EN altyazi penceresi. "
                 "Kayit basladiginda otomatik gosterilir.",
            font=ctk.CTkFont(size=11), text_color=_C["muted"],
            anchor="w", wraplength=500
        ).pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            ov_row, text="Goster", width=80, height=34,
            fg_color=_C["blue_bg"], hover_color=_C["surface2"],
            text_color=_C["blue"], corner_radius=10,
            font=ctk.CTkFont(size=11),
            command=self._show_overlay,
        ).pack(side="right")

        # ── Ipuclari ──────────────────────────────────────────────────
        tips = _card(scroll, "Ipuclari")
        for tip in [
            "\u2022  Gurultulu ortamda VAD hassasiyetini Ayarlar'dan dusunun.",
            "\u2022  Overlay penceresi diger uygulamalarin ustunde kalir.",
            "\u2022  Kayit sirasinda 'Medya' sekmesinden dosya cevirisi de yapabilirsiniz.",
        ]:
            ctk.CTkLabel(
                tips, text=tip,
                font=ctk.CTkFont(size=10), text_color=_C["muted"],
                anchor="w", wraplength=580
            ).pack(anchor="w", pady=1)

    # ── Olaylar ───────────────────────────────────────────────────────────────

    def _start(self):
        if not self.app._backend_ready:
            messagebox.showinfo(
                "Modeller Hazirlaniyor",
                "Modeller henuz yukleniyor.\nBir dakika bekleyip tekrar deneyin."
            )
            return
        self.app.start_live()
        self._recording = True
        self._btn_start.configure(state="disabled", fg_color=_C["dim"])
        self._btn_stop.configure(
            state="normal", fg_color=_C["red"], text_color=_C["text"]
        )
        self._sdot.configure(text_color=_C["green"])
        self._slbl.configure(
            text="Canli dinleme aktif  \u2014  konusun...",
            text_color=_C["green"]
        )
        if self.app._overlay:
            self.app._overlay.deiconify()
            self.app._overlay.lift()

    def _stop(self):
        self.app.stop_live()
        self._recording = False
        self._btn_start.configure(state="normal", fg_color=_C["green"])
        self._btn_stop.configure(
            state="disabled", fg_color=_C["surface2"], text_color=_C["dim"]
        )
        self._sdot.configure(text_color=_C["dim"])
        self._slbl.configure(
            text="Durduruldu  \u2014  Baslamak icin \u25b6 butonuna basin",
            text_color=_C["muted"]
        )

    def _show_overlay(self):
        if self.app._overlay:
            self.app._overlay.deiconify()
            self.app._overlay.lift()

    def _on_ptt(self):
        self.cfg.set("recording", "push_to_talk", self._ptt.get())
        self.cfg.save()


# ══════════════════════════════════════════════════════════════════════════════
# Medya View
# ══════════════════════════════════════════════════════════════════════════════

class MediaView(ctk.CTkFrame):
    def __init__(self, master, cfg: ConfigManager, app):
        super().__init__(master, fg_color=_C["bg"], corner_radius=0)
        self.cfg = cfg
        self.app = app
        self._processing = False
        self._build()

    def _build(self):
        _header(self, "\U0001f3ac  Medya Cevirisi",
                "Ses (.wav .mp3 .m4a) ve video (.mp4 .mkv .avi) dosyalarini cevir")

        # ── Dosya secim cubugu ─────────────────────────────────────────
        bar = ctk.CTkFrame(
            self, fg_color=_C["surface"],
            corner_radius=12, border_width=1, border_color=_C["border"]
        )
        bar.pack(fill="x", padx=24, pady=(0, 10))

        bar_in = ctk.CTkFrame(bar, fg_color="transparent")
        bar_in.pack(fill="x", padx=14, pady=12)

        self._file_entry = ctk.CTkEntry(
            bar_in,
            placeholder_text="Ses veya video dosyasi secin...",
            font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            height=36, corner_radius=10
        )
        self._file_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(
            bar_in, text="Gozat", width=76, height=36,
            fg_color=_C["surface2"], hover_color=_C["border"],
            corner_radius=10, font=ctk.CTkFont(size=11),
            command=self._browse
        ).pack(side="left", padx=(0, 6))

        self._btn_process = ctk.CTkButton(
            bar_in, text="\u25b6  Cevir", width=90, height=36,
            fg_color=_C["blue"], hover_color="#4080d0",
            corner_radius=10, font=ctk.CTkFont(size=12, weight="bold"),
            command=self._start_processing
        )
        self._btn_process.pack(side="left", padx=(0, 6))

        self._btn_cancel = ctk.CTkButton(
            bar_in, text="\u25a0 Iptal", width=70, height=36,
            fg_color=_C["red_bg"], hover_color=_C["red"],
            text_color=_C["red"], corner_radius=10,
            state="disabled", command=self._cancel
        )
        self._btn_cancel.pack(side="left")

        # ── Ilerleme ──────────────────────────────────────────────────
        prog_frame = ctk.CTkFrame(self, fg_color="transparent", height=32)
        prog_frame.pack(fill="x", padx=24, pady=(0, 8))
        prog_frame.pack_propagate(False)

        self._progress = ctk.CTkProgressBar(
            prog_frame, height=6, mode="determinate",
            progress_color=_C["blue"], fg_color=_C["surface2"]
        )
        self._progress.set(0)
        self._progress.pack(side="left", fill="x", expand=True, pady=12)

        self._prog_lbl = ctk.CTkLabel(
            prog_frame, text="Hazir",
            font=ctk.CTkFont(size=10),
            text_color=_C["dim"], width=180, anchor="e"
        )
        self._prog_lbl.pack(side="left", padx=(10, 0))

        # ── Metin panelleri (TR | EN) ──────────────────────────────────
        mid = ctk.CTkFrame(self, fg_color="transparent")
        mid.pack(fill="both", expand=True, padx=24, pady=(0, 8))
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mid.rowconfigure(1, weight=1)

        ctk.CTkLabel(
            mid, text="Turkce Transkript  (STT)",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_C["muted"]
        ).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=(0, 4))

        ctk.CTkLabel(
            mid, text="Ingilizce Ceviri",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_C["blue"]
        ).grid(row=0, column=1, sticky="w", padx=(6, 0), pady=(0, 4))

        self._tr_box = ctk.CTkTextbox(
            mid, font=ctk.CTkFont(size=12),
            fg_color=_C["surface"], border_color=_C["border"], border_width=1,
            text_color=_C["muted"], wrap="word",
            corner_radius=12, state="disabled"
        )
        self._tr_box.grid(row=1, column=0, sticky="nsew", padx=(0, 6))

        self._en_box = ctk.CTkTextbox(
            mid, font=ctk.CTkFont(size=12),
            fg_color=_C["surface"], border_color=_C["border"], border_width=1,
            text_color=_C["text"], wrap="word",
            corner_radius=12, state="disabled"
        )
        self._en_box.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        # ── Alt kayit cubugu ──────────────────────────────────────────
        bot = ctk.CTkFrame(
            self, fg_color=_C["surface"],
            corner_radius=0, height=44
        )
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)

        for label, cmd in [
            ("TR Kaydet",      lambda: self._save("tr")),
            ("EN Kaydet",      lambda: self._save("en")),
            ("Ikisini Kaydet", lambda: (self._save("tr"), self._save("en"))),
        ]:
            ctk.CTkButton(
                bot, text=label, height=28, width=120,
                fg_color=_C["surface2"], hover_color=_C["border"],
                corner_radius=8, font=ctk.CTkFont(size=10),
                command=cmd
            ).pack(side="left", padx=(12, 4), pady=8)

        self._elapsed = ctk.CTkLabel(
            bot, text="", font=ctk.CTkFont(size=10), text_color=_C["dim"]
        )
        self._elapsed.pack(side="right", padx=16)

    # ── Dosya Secimi ──────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Ses veya Video Dosyasi Sec",
            filetypes=[
                ("Tum medya",    "*.wav *.mp3 *.ogg *.flac *.m4a *.aac "
                                  "*.mp4 *.mkv *.avi *.mov *.webm"),
                ("Ses",          "*.wav *.mp3 *.ogg *.flac *.m4a *.aac"),
                ("Video",        "*.mp4 *.mkv *.avi *.mov *.webm *.ts"),
                ("Tum dosyalar", "*.*"),
            ]
        )
        if path:
            self._file_entry.delete(0, "end")
            self._file_entry.insert(0, path)
            self._clear()

    def _clear(self):
        for box in (self._tr_box, self._en_box):
            box.configure(state="normal")
            box.delete("0.0", "end")
            box.configure(state="disabled")
        self._progress.set(0)
        self._prog_lbl.configure(text="Hazir", text_color=_C["dim"])
        self._elapsed.configure(text="")

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _start_processing(self):
        path = self._file_entry.get().strip()
        if not path:
            messagebox.showwarning("Dosya Secilmedi", "Lutfen once bir dosya secin.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Bulunamadi", f"Dosya mevcut degil:\n{path}")
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in _ALL_EXT:
            messagebox.showwarning(
                "Desteklenmeyen Format",
                f"'{ext}' desteklenmiyor.\nDesteklenenler: "
                f"{', '.join(sorted(_ALL_EXT))}"
            )
            return
        if not getattr(self.app, "_backend_ready", False):
            messagebox.showwarning(
                "Backend Hazir Degil",
                "STT/LLM modelleri henuz yukleniyor."
            )
            return

        self._processing = True
        self._btn_process.configure(state="disabled")
        self._btn_cancel.configure(state="normal")
        self._clear()
        threading.Thread(
            target=self._pipeline, args=(path,), daemon=True
        ).start()

    def _cancel(self):
        self._processing = False
        self._set_progress(0, "Iptal edildi", _C["yellow"])
        self._btn_process.configure(state="normal")
        self._btn_cancel.configure(state="disabled")

    def _pipeline(self, src: str):
        t0 = time.time()
        wav, owns = None, False
        try:
            ext = os.path.splitext(src)[1].lower()

            if ext in _VIDEO_EXT:
                self._set_progress(0.05, "Video'dan ses ayiklaniyor...", _C["yellow"])
                wav, owns = self._to_wav(src)
            elif ext != ".wav":
                self._set_progress(0.05, "Ses donusturuluyor...", _C["yellow"])
                wav, owns = self._to_wav(src)
            else:
                wav = src

            if wav is None or not self._processing:
                return

            self._set_progress(0.20, "Transkript olusturuluyor (Whisper)...", _C["blue"])
            res    = self.app._orchestrator.transcriber.transcribe(wav)
            txt_tr = res.get("text", "").strip()

            if not txt_tr:
                self._set_progress(1.0, "Ses taninamadi.", _C["red"])
                return

            self._set_text(self._tr_box, txt_tr)
            self._set_progress(0.55, "Ceviri yapiliyor...", _C["blue"])

            if not self._processing:
                return

            txt_en = self._translate_chunked(
                self.app._orchestrator.translator, txt_tr
            )
            if not self._processing:
                return

            self._set_text(self._en_box, txt_en, _C["text"])
            elapsed = time.time() - t0
            self._set_progress(1.0, f"Tamamlandi  \u2713", _C["green"])
            self.after(0, lambda: self._elapsed.configure(
                text=f"Sure: {elapsed:.1f}s", text_color=_C["dim"]
            ))

        except Exception as e:
            self._set_progress(0, f"Hata: {e}", _C["red"])
        finally:
            if owns and wav and os.path.exists(wav):
                try:
                    os.remove(wav)
                except OSError:
                    pass
            self._processing = False
            self.after(0, lambda: [
                self._btn_process.configure(state="normal"),
                self._btn_cancel.configure(state="disabled"),
            ])

    def _to_wav(self, src: str):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        out = tmp.name
        try:
            import ffmpeg as ff
            (
                ff.input(src)
                  .output(out, ar=16000, ac=1, acodec="pcm_s16le")
                  .overwrite_output()
                  .run(quiet=True)
            )
            return out, True
        except Exception:
            try:
                r = subprocess.run(
                    ["ffmpeg", "-i", src,
                     "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                     "-y", out],
                    capture_output=True, timeout=600
                )
                if r.returncode == 0:
                    return out, True
                self._set_progress(0, "ffmpeg hatasi", _C["red"])
            except FileNotFoundError:
                self._set_progress(0, "ffmpeg bulunamadi!", _C["red"])
        return None, False

    def _translate_chunked(self, translator, text: str) -> str:
        words = text.split()
        if len(words) <= _LLM_CHUNK:
            return translator.translate(text).get("translation", text)

        chunks = []
        buf = []
        for w in words:
            buf.append(w)
            if len(buf) >= _LLM_CHUNK:
                chunks.append(" ".join(buf))
                buf = []
        if buf:
            chunks.append(" ".join(buf))

        parts = []
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            if not self._processing:
                break
            p = 0.55 + 0.40 * (i / total)
            self._set_progress(p, f"Ceviri: {i+1}/{total} bolum", _C["blue"])
            parts.append(translator.translate(chunk).get("translation", chunk))
        return " ".join(parts)

    def _save(self, lang: str):
        box  = self._tr_box if lang == "tr" else self._en_box
        text = box.get("0.0", "end").strip()
        if not text:
            messagebox.showinfo("Bos", "Kaydedilecek metin yok.")
            return
        src  = self._file_entry.get().strip()
        base = os.path.splitext(os.path.basename(src))[0] if src else "cikti"
        path = filedialog.asksaveasfilename(
            title=f"{'Transkripti' if lang == 'tr' else 'Ceviriyi'} Kaydet",
            initialfile=f"{base}_{lang}.txt",
            initialdir=self.cfg.get("file_mode", "output_dir",
                                    default=os.path.expanduser("~")),
            defaultextension=".txt",
            filetypes=[("Metin", "*.txt"), ("Hepsi", "*.*")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.cfg.set("file_mode", "output_dir", os.path.dirname(path))
            self.cfg.save()

    # ── Thread-safe yardimcilar ───────────────────────────────────────────────

    def _set_progress(self, val: float, msg: str, color: str = None):
        def _u():
            self._progress.set(max(0.0, min(1.0, val)))
            self._prog_lbl.configure(
                text=msg, text_color=color or _C["dim"]
            )
        self.after(0, _u)

    def _set_text(self, box: ctk.CTkTextbox, text: str, color: str = None):
        def _u():
            box.configure(state="normal")
            box.delete("0.0", "end")
            box.insert("0.0", text)
            if color:
                box.configure(text_color=color)
            box.configure(state="disabled")
        self.after(0, _u)


# ══════════════════════════════════════════════════════════════════════════════
# Metin Ceviri View
# ══════════════════════════════════════════════════════════════════════════════

class TextView(ctk.CTkFrame):
    def __init__(self, master, cfg: ConfigManager, app):
        super().__init__(master, fg_color=_C["bg"], corner_radius=0)
        self.cfg = cfg
        self.app = app
        self._build()

    def _build(self):
        _header(self, "\U0001f4dd  Metin Cevirisi",
                "Turkce metin yazin, aninda Ingilizce cevirisini alin")

        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 12))

        # ── Giris ─────────────────────────────────────────────────────
        in_card = _card(body, "Turkce Giris")
        self._in = ctk.CTkTextbox(
            in_card, height=130,
            font=ctk.CTkFont(size=13),
            fg_color=_C["surface2"], border_color=_C["border"], border_width=1,
            text_color=_C["text"], wrap="word", corner_radius=10
        )
        self._in.pack(fill="x")

        # Butonlar
        br = ctk.CTkFrame(in_card, fg_color="transparent")
        br.pack(fill="x", pady=(10, 0))

        self._btn_tr = ctk.CTkButton(
            br, text="\u25b6  Cevir", height=38, width=110,
            fg_color=_C["blue"], hover_color="#4080d0",
            corner_radius=10, font=ctk.CTkFont(size=12, weight="bold"),
            command=self._translate
        )
        self._btn_tr.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            br, text="Temizle", height=38, width=90,
            fg_color=_C["surface2"], hover_color=_C["border"],
            corner_radius=10, font=ctk.CTkFont(size=11),
            command=self._clear
        ).pack(side="left")

        # ── Cikis ─────────────────────────────────────────────────────
        out_card = _card(body, "Ingilizce Ceviri")
        self._out = ctk.CTkTextbox(
            out_card, height=130,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=_C["surface2"], border_color=_C["border"], border_width=1,
            text_color=_C["blue"], wrap="word", corner_radius=10,
            state="disabled"
        )
        self._out.pack(fill="x")

        # Kopyala butonu
        ctk.CTkButton(
            out_card, text="Kopyala", height=32, width=90,
            fg_color=_C["surface2"], hover_color=_C["border"],
            corner_radius=8, font=ctk.CTkFont(size=10),
            command=self._copy
        ).pack(anchor="e", pady=(8, 0))

    def _translate(self):
        text = self._in.get("0.0", "end").strip()
        if not text:
            return
        if not self.app._backend_ready:
            messagebox.showinfo("Bekleyin", "Modeller henuz yukleniyor.")
            return
        self._set_out("Cevriliyor...")
        self._btn_tr.configure(state="disabled")

        def _run():
            try:
                result = self.app._orchestrator.translator.translate(text)
                en = result.get("translation", "")
                self.after(0, lambda: self._set_out(en))
            except Exception as e:
                self.after(0, lambda: self._set_out(f"Hata: {e}"))
            finally:
                self.after(0, lambda: self._btn_tr.configure(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    def _set_out(self, text: str):
        self._out.configure(state="normal")
        self._out.delete("0.0", "end")
        self._out.insert("0.0", text)
        self._out.configure(state="disabled")

    def _clear(self):
        self._in.delete("0.0", "end")
        self._set_out("")

    def _copy(self):
        text = self._out.get("0.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)


# ══════════════════════════════════════════════════════════════════════════════
# Ayarlar View
# ══════════════════════════════════════════════════════════════════════════════

class SettingsView(ctk.CTkFrame):
    _MODES = [
        ("interactive",      "Interactive     \u2014  STT small \u00b7 LLM Cloud \u00b7 TTS GPU  (Hizli)"),
        ("interactive_hq",   "Interactive HQ  \u2014  STT medium \u00b7 LLM Cloud \u00b7 TTS GPU  (Kaliteli)"),
        ("online",           "Online          \u2014  Tumu bulut"),
        ("online_xtts",      "Online XTTS     \u2014  Bulut + Yerel TTS"),
        ("online_local_stt", "Local STT       \u2014  STT small \u00b7 LLM+TTS bulut"),
        ("offline_gpu",      "Offline GPU     \u2014  Tumu yerel GPU"),
        ("offline",          "Offline CPU     \u2014  Tumu yerel CPU"),
        ("hybrid_cloud_io",  "Hybrid I/O      \u2014  Bulut I/O \u00b7 Yerel LLM"),
        ("hybrid_cloud_stt", "Hybrid STT      \u2014  Bulut STT \u00b7 Yerel LLM/TTS"),
    ]

    def __init__(self, master, cfg: ConfigManager, app):
        super().__init__(master, fg_color=_C["bg"], corner_radius=0)
        self.cfg = cfg
        self.app = app
        self._build()

    def _build(self):
        _header(self, "\u2699  Ayarlar",
                "Calisma modu, API anahtarlari ve gorunum tercihleri")

        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 12))

        # ── Calisma Modu ──────────────────────────────────────────────
        mode_card = _card(body, "Calisma Modu")
        mode_vals = [m[1] for m in self._MODES]
        cur       = self.cfg.get("mode", "current", default="online")
        cur_idx   = next(
            (i for i, m in enumerate(self._MODES) if m[0] == cur), 0
        )
        self._mode_combo = ctk.CTkComboBox(
            mode_card, values=mode_vals,
            height=36, corner_radius=10, font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            command=self._on_mode
        )
        self._mode_combo.set(mode_vals[cur_idx])
        self._mode_combo.pack(fill="x")

        # ── Donanim bilgisi ───────────────────────────────────────────
        hw_card = _card(body, "Donanim")
        hw      = self.cfg.get("hardware") or {}
        gpu_n   = hw.get("gpu", {}).get("name", "CPU")
        ram_gb  = hw.get("ram_gb", "?")
        cpu_c   = hw.get("cpu_cores", "?")
        ctk.CTkLabel(
            hw_card,
            text=f"GPU: {gpu_n}   RAM: {ram_gb} GB   CPU: {cpu_c} cekirdek",
            font=ctk.CTkFont(size=11), text_color=_C["muted"], anchor="w"
        ).pack(anchor="w")

        # ── API Anahtarlari ───────────────────────────────────────────
        api_card = _card(body, "API Anahtarlari")
        for svc, lbl, url in [
            ("gemini",     "Gemini",     "https://aistudio.google.com/apikey"),
            ("groq",       "Groq",       "https://console.groq.com/keys"),
            ("elevenlabs", "ElevenLabs", "https://elevenlabs.io/app/settings/api-keys"),
        ]:
            self._api_row(api_card, svc, lbl, url)

        # ── ElevenLabs Ses ────────────────────────────────────────────
        voice_card = _card(body, "ElevenLabs Ses ID")
        vr = ctk.CTkFrame(voice_card, fg_color="transparent")
        vr.pack(fill="x")

        self._voice_entry = ctk.CTkEntry(
            vr, height=34, corner_radius=10, font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            placeholder_text="Voice ID"
        )
        vid = self.cfg.get("elevenlabs_voice_id", default="")
        if vid:
            self._voice_entry.insert(0, vid)
        self._voice_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(
            vr, text="Ses Kutuphanesi \u2192", width=130, height=34,
            fg_color=_C["surface2"], corner_radius=10,
            command=lambda: webbrowser.open("https://elevenlabs.io/app/voice-library")
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            vr, text="Kaydet", width=72, height=34,
            fg_color=_C["blue"], corner_radius=10,
            command=self._save_voice
        ).pack(side="left")

        # ── Overlay Opakligi ──────────────────────────────────────────
        ovl_card = _card(body, "Overlay Gorunumu")
        op_row = ctk.CTkFrame(ovl_card, fg_color="transparent")
        op_row.pack(fill="x")

        ctk.CTkLabel(
            op_row, text="Opaklik:",
            font=ctk.CTkFont(size=11), text_color=_C["muted"]
        ).pack(side="left")

        self._op_slider = ctk.CTkSlider(
            op_row, from_=0.3, to=1.0, width=200,
            button_color=_C["blue"], progress_color=_C["blue"],
            command=self._on_opacity
        )
        self._op_slider.set(self.cfg.get("overlay", "opacity", default=0.92))
        self._op_slider.pack(side="left", padx=12)

        self._op_lbl = ctk.CTkLabel(
            op_row, text=f"{self.cfg.get('overlay','opacity',default=0.92):.0%}",
            font=ctk.CTkFont(size=11), text_color=_C["text"], width=36
        )
        self._op_lbl.pack(side="left")

        # VAD hassasiyet
        vad_row = ctk.CTkFrame(ovl_card, fg_color="transparent")
        vad_row.pack(fill="x", pady=(10, 0))

        ctk.CTkLabel(
            vad_row, text="VAD Hassasiyeti (0-3):",
            font=ctk.CTkFont(size=11), text_color=_C["muted"]
        ).pack(side="left")

        self._vad_seg = ctk.CTkSegmentedButton(
            vad_row, values=["0", "1", "2", "3"],
            command=self._on_vad,
            selected_color=_C["blue"], selected_hover_color="#4080d0",
            unselected_color=_C["surface2"],
            font=ctk.CTkFont(size=11)
        )
        cur_vad = str(self.cfg.get("recording", "vad_aggressiveness", default=2))
        self._vad_seg.set(cur_vad)
        self._vad_seg.pack(side="left", padx=12)

    # ── Yardimci: API satiri ──────────────────────────────────────────────────

    def _api_row(self, parent, svc: str, lbl: str, url: str):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)

        ctk.CTkLabel(
            row, text=f"{lbl}:", width=82,
            font=ctk.CTkFont(size=11), text_color=_C["muted"], anchor="w"
        ).pack(side="left")

        entry = ctk.CTkEntry(
            row, show="\u2022", height=32, corner_radius=8,
            font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"]
        )
        existing = self.cfg.get("api_keys", svc, default="")
        if existing:
            entry.insert(0, existing)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        ctk.CTkButton(
            row, text="Al \u2192", width=60, height=32, corner_radius=8,
            fg_color=_C["surface2"],
            command=lambda u=url: webbrowser.open(u)
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            row, text="Kaydet", width=72, height=32, corner_radius=8,
            fg_color=_C["blue"],
            command=lambda s=svc, e=entry: self.cfg.set_api_key(s, e.get().strip())
        ).pack(side="left")

    # ── Olaylar ───────────────────────────────────────────────────────────────

    def _on_mode(self, display: str):
        key = next((m[0] for m in self._MODES if m[1] == display), None)
        if key:
            self.app.switch_mode(key)

    def _on_opacity(self, val: float):
        v = round(val, 2)
        self.cfg.set("overlay", "opacity", v)
        self.cfg.save()
        self._op_lbl.configure(text=f"{v:.0%}")
        if self.app._overlay:
            self.app._overlay.wm_attributes("-alpha", v)

    def _on_vad(self, val: str):
        self.cfg.set("recording", "vad_aggressiveness", int(val))
        self.cfg.save()

    def _save_voice(self):
        vid = self._voice_entry.get().strip()
        if vid:
            self.cfg.set_voice(vid)
