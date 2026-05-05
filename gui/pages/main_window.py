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
from gui.i18n   import t, get_language

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
_DOC_EXT     = {".txt", ".pdf", ".docx"}

# Desteklenen dil secenekleri (BookView icin)
_LANGS = [
    ("Turkish",  "tr"),
    ("English",  "en"),
    ("German",   "de"),
    ("Spanish",  "es"),
    ("French",   "fr"),
]
_LANG_NAMES = [l[0] for l in _LANGS]


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
            "book":     BookView(shell, cfg=self.cfg, app=self.app),
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
            self._sidebar.set_status(t("ready_tick"), _C["green"])
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

        # Ayirici
        ctk.CTkFrame(self, height=1, fg_color=_C["border"]).pack(fill="x")

        # ── Navigasyon ────────────────────────────────────────────────
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
        _header(self, f"\U0001f399  {t('live_title')}",
                t("live_subtitle"))

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=24, pady=(0, 12))

        # ── Kayit Kontrolu ────────────────────────────────────────────
        inner = _card(scroll, t("rec_control"))

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
            text=t("live_ready_hint"),
            font=ctk.CTkFont(size=11), text_color=_C["muted"], anchor="w"
        )
        self._slbl.pack(side="left", fill="x", expand=True)

        # Butonlar
        btn_row = ctk.CTkFrame(inner, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0, 10))

        self._btn_start = ctk.CTkButton(
            btn_row,
            text=f"\u25b6  {t('start')}",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=_C["green"], hover_color="#1aad4e",
            height=46, corner_radius=12,
            command=self._start,
        )
        self._btn_start.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self._btn_stop = ctk.CTkButton(
            btn_row,
            text=f"\u25a0  {t('stop')}",
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
            lf, text=t("ptt"),
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=_C["text"], anchor="w"
        ).pack(anchor="w")
        ctk.CTkLabel(
            lf, text=t("ptt_hint"),
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
        ov = _card(scroll, t("overlay_title"))
        ov_row = ctk.CTkFrame(ov, fg_color="transparent")
        ov_row.pack(fill="x")

        ctk.CTkLabel(
            ov_row,
            text=t("overlay_hint"),
            font=ctk.CTkFont(size=11), text_color=_C["muted"],
            anchor="w", wraplength=500
        ).pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            ov_row, text=t("show"), width=80, height=34,
            fg_color=_C["blue_bg"], hover_color=_C["surface2"],
            text_color=_C["blue"], corner_radius=10,
            font=ctk.CTkFont(size=11),
            command=self._show_overlay,
        ).pack(side="right")

        # ── Ipuclari ──────────────────────────────────────────────────
        tips = _card(scroll, t("tips"))
        for tip in [
            t("tip1"),
            t("tip2"),
            t("tip3"),
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
                t("modeller_hazirlaniyor"),
                t("modeller_yukleniyor_bekle")
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
            text=t("live_active_hint"),
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
            text=t("stopped_hint"),
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
        self._dubbing    = False
        self._build()

    def _build(self):
        _header(self, f"\U0001f3ac  {t('media_title')}",
                t("media_subtitle"))

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
            placeholder_text=t("select_file"),
            font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            height=36, corner_radius=10
        )
        self._file_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(
            bar_in, text=t("browse"), width=76, height=36,
            fg_color=_C["surface2"], hover_color=_C["border"],
            corner_radius=10, font=ctk.CTkFont(size=11),
            command=self._browse
        ).pack(side="left", padx=(0, 6))

        self._btn_process = ctk.CTkButton(
            bar_in, text=f"\u25b6  {t('translate')}", width=90, height=36,
            fg_color=_C["blue"], hover_color="#4080d0",
            corner_radius=10, font=ctk.CTkFont(size=12, weight="bold"),
            command=self._start_processing
        )
        self._btn_process.pack(side="left", padx=(0, 6))

        self._btn_cancel = ctk.CTkButton(
            bar_in, text=f"\u25a0 {t('cancel')}", width=70, height=36,
            fg_color=_C["red_bg"], hover_color=_C["red"],
            text_color=_C["red"], corner_radius=10,
            state="disabled", command=self._cancel
        )
        self._btn_cancel.pack(side="left", padx=(0, 6))

        self._btn_dub = ctk.CTkButton(
            bar_in, text=f"\U0001f3ac  {t('dubbing')}", width=90, height=36,
            fg_color="#2a1a4a", hover_color="#4a2a7a",
            text_color="#c084fc", corner_radius=10,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._start_dubbing
        )
        self._btn_dub.pack(side="left")

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
            prog_frame, text=t("ready"),
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
            mid, text=t("tr_transcript"),
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_C["muted"]
        ).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=(0, 4))

        ctk.CTkLabel(
            mid, text=t("en_translation"),
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

        for label_key, cmd in [
            ("save_tr",      lambda: self._save("tr")),
            ("save_en",      lambda: self._save("en")),
            ("save_both",    lambda: (self._save("tr"), self._save("en"))),
        ]:
            ctk.CTkButton(
                bot, text=t(label_key), height=28, width=120,
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
            title=t("browse"),
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
        self._prog_lbl.configure(text=t("ready"), text_color=_C["dim"])
        self._elapsed.configure(text="")

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _start_processing(self):
        path = self._file_entry.get().strip()
        if not path:
            messagebox.showwarning(t("file_not_selected"), t("select_file_first"))
            return
        if not os.path.exists(path):
            messagebox.showerror(t("file_not_found"), t("file_exists_error", path))
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in _ALL_EXT:
            messagebox.showwarning(
                t("unsupported_format"),
                t("unsupported_ext", ext, ", ".join(sorted(_ALL_EXT)))
            )
            return
        if not getattr(self.app, "_backend_ready", False):
            messagebox.showwarning(
                t("backend_not_ready_models"),
                t("models_still_loading")
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

    # ── Dublaj ────────────────────────────────────────────────────────────────

    def _start_dubbing(self):
        if self._dubbing or self._processing:
            return
        path = self._file_entry.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Dosya Yok", "Lutfen once bir video dosyasi secin.")
            return
        if os.path.splitext(path)[1].lower() not in _VIDEO_EXT | _AUDIO_EXT:
            messagebox.showwarning("Desteklenmiyor", "Desteklenen format: mp4, mkv, avi, mov ...")
            return
        if not self.app._backend_ready:
            messagebox.showinfo(
                t("modeller_hazirlaniyor"),
                t("modeller_yukleniyor_bekle")
            )
            return

        # Cikti yolu: kaynak_video_dubbed.mp4
        base, _ = os.path.splitext(path)
        output_path = base + "_dubbed.mp4"

        self._dubbing = True
        self._btn_dub.configure(state="disabled", fg_color=_C["dim"])
        self._btn_process.configure(state="disabled")
        self._clear()
        threading.Thread(
            target=self._dubbing_pipeline, args=(path, output_path), daemon=True
        ).start()

    def _dubbing_pipeline(self, src: str, output_path: str):
        from pipeline.dubber import DubbingPipeline

        orch = self.app._orchestrator
        dubber = DubbingPipeline(
            transcriber=orch.transcriber,
            translator=orch.translator,
            synthesizer=orch.synthesizer,
        )

        def on_progress(fraction, msg, color):
            self._set_progress(fraction, msg, color)

        try:
            dubber.process(src, output_path, progress_cb=on_progress)
            self.after(0, lambda: self._elapsed.configure(
                text=f"Cikti: {os.path.basename(output_path)}",
                text_color=_C["green"]
            ))
        except Exception as e:
            self._set_progress(0, t("dubbing_error", str(e)), _C["red"])
        finally:
            self._dubbing = False
            self.after(0, lambda: [
                self._btn_dub.configure(
                    state="normal", fg_color="#2a1a4a"
                ),
                self._btn_process.configure(state="normal"),
            ])

    def _pipeline(self, src: str):
        t0 = time.time()
        wav, owns = None, False
        try:
            ext = os.path.splitext(src)[1].lower()

            if ext in _VIDEO_EXT:
                self._set_progress(0.05, t("extracting_audio"), _C["yellow"])
                wav, owns = self._to_wav(src)
            elif ext != ".wav":
                self._set_progress(0.05, t("converting_audio"), _C["yellow"])
                wav, owns = self._to_wav(src)
            else:
                wav = src

            if wav is None or not self._processing:
                return

            self._set_progress(0.20, t("generating_transcript"), _C["blue"])
            res    = self.app._orchestrator.transcriber.transcribe(wav)
            txt_tr = res.get("text", "").strip()

            if not txt_tr:
                self._set_progress(1.0, t("speech_not_recognized"), _C["red"])
                return

            self._set_text(self._tr_box, txt_tr)
            self._set_progress(0.55, t("translating"), _C["blue"])

            if not self._processing:
                return

            txt_en = self._translate_chunked(
                self.app._orchestrator.translator, txt_tr
            )
            if not self._processing:
                return

            self._set_text(self._en_box, txt_en, _C["text"])
            elapsed = time.time() - t0
            self._set_progress(1.0, t("done_tick"), _C["green"])
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
# Kitap / Belge Ceviri View  (FAZ 4)
# ══════════════════════════════════════════════════════════════════════════════

class BookView(ctk.CTkFrame):
    def __init__(self, master, cfg: ConfigManager, app):
        super().__init__(master, fg_color=_C["bg"], corner_radius=0)
        self.cfg = cfg
        self.app = app
        self._translating = False
        self._dt = None          # aktif DocumentTranslator ornegi
        self._build()

    # ── Arayuz Insasi ─────────────────────────────────────────────────────────

    def _build(self):
        _header(self, f"\U0001f4d6  {t('book_title')}",
                t("book_subtitle"))

        # Scroll alani
        body = ctk.CTkScrollableFrame(self, fg_color="transparent", corner_radius=0)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # ── 1. Dosya Secimi ────────────────────────────────────────────────
        file_card = _card(body, t("select_file"))

        file_row = ctk.CTkFrame(file_card, fg_color="transparent")
        file_row.pack(fill="x")

        self._file_entry = ctk.CTkEntry(
            file_row,
            placeholder_text=t("select_doc"),
            font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            height=36, corner_radius=10
        )
        self._file_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(
            file_row, text=t("browse"), width=76, height=36,
            fg_color=_C["surface2"], hover_color=_C["border"],
            corner_radius=10, font=ctk.CTkFont(size=11),
            command=self._browse
        ).pack(side="left")

        # ── 2. Ceviri Ayarlari ─────────────────────────────────────────────
        opt_card = _card(body, t("settings"))
        opt_card.columnconfigure(0, weight=1)
        opt_card.columnconfigure(1, weight=1)
        opt_card.columnconfigure(2, weight=1)

        # Kaynak dil
        ctk.CTkLabel(
            opt_card, text=t("source_lang"),
            font=ctk.CTkFont(size=10, weight="bold"), text_color=_C["muted"]
        ).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 4))

        self._src_lang_combo = ctk.CTkComboBox(
            opt_card, values=_LANG_NAMES, height=34,
            fg_color=_C["surface2"], border_color=_C["border"],
            font=ctk.CTkFont(size=11), corner_radius=10, state="readonly"
        )
        saved_src = self.cfg.get("language", "source_name", default="Turkish")
        self._src_lang_combo.set(saved_src if saved_src in _LANG_NAMES else "Turkish")
        self._src_lang_combo.grid(row=1, column=0, sticky="ew", padx=(0, 8))

        # Hedef dil
        ctk.CTkLabel(
            opt_card, text=t("target_lang"),
            font=ctk.CTkFont(size=10, weight="bold"), text_color=_C["muted"]
        ).grid(row=0, column=1, sticky="w", padx=(0, 8), pady=(0, 4))

        self._tgt_lang_combo = ctk.CTkComboBox(
            opt_card, values=_LANG_NAMES, height=34,
            fg_color=_C["surface2"], border_color=_C["border"],
            font=ctk.CTkFont(size=11), corner_radius=10, state="readonly"
        )
        saved_tgt = self.cfg.get("language", "target_name", default="English")
        self._tgt_lang_combo.set(saved_tgt if saved_tgt in _LANG_NAMES else "English")
        self._tgt_lang_combo.grid(row=1, column=1, sticky="ew", padx=(0, 8))

        # Chunk boyutu
        ctk.CTkLabel(
            opt_card, text=t("chunk_size"),
            font=ctk.CTkFont(size=10, weight="bold"), text_color=_C["muted"]
        ).grid(row=0, column=2, sticky="w", pady=(0, 4))

        if get_language() == "tr":
            chunk_vals = ["400 (Yerel GGUF)", "800 (API - Önerilen)", "1200 (API - Büyük)"]
            chunk_def  = "800 (API - Önerilen)"
        else:
            chunk_vals = ["400 (Local GGUF)", "800 (API - Recommended)", "1200 (API - Large)"]
            chunk_def  = "800 (API - Recommended)"

        self._chunk_combo = ctk.CTkComboBox(
            opt_card, values=chunk_vals,
            height=34, fg_color=_C["surface2"], border_color=_C["border"],
            font=ctk.CTkFont(size=11), corner_radius=10, state="readonly"
        )
        self._chunk_combo.set(chunk_def)
        self._chunk_combo.grid(row=1, column=2, sticky="ew")

        # ── 3. Eylem Cubugu ───────────────────────────────────────────────
        act_card = _card(body, t("translate"))
        act_row = ctk.CTkFrame(act_card, fg_color="transparent")
        act_row.pack(fill="x")

        self._btn_start = ctk.CTkButton(
            act_row, text=f"\u25b6  {t('start_translation')}",
            height=36, corner_radius=10,
            fg_color=_C["blue"], hover_color="#4080d0",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._start
        )
        self._btn_start.pack(side="left", padx=(0, 8))

        self._btn_cancel = ctk.CTkButton(
            act_row, text=f"\u25a0 {t('cancel')}",
            height=36, width=80, corner_radius=10,
            fg_color=_C["red_bg"], hover_color=_C["red"],
            text_color=_C["red"],
            state="disabled",
            command=self._cancel
        )
        self._btn_cancel.pack(side="left", padx=(0, 8))

        self._btn_save = ctk.CTkButton(
            act_row, text=f"\U0001f4be {t('save_translation')}",
            height=36, width=100, corner_radius=10,
            fg_color=_C["surface2"], hover_color=_C["border"],
            font=ctk.CTkFont(size=11),
            state="disabled",
            command=self._save
        )
        self._btn_save.pack(side="left")

        # ── 4. Ilerleme ────────────────────────────────────────────────────
        prog_card = _card(body, t("progress"))

        self._progress = ctk.CTkProgressBar(
            prog_card, height=8, mode="determinate",
            progress_color=_C["blue"], fg_color=_C["surface2"]
        )
        self._progress.set(0)
        self._progress.pack(fill="x", pady=(0, 6))

        self._prog_lbl = ctk.CTkLabel(
            prog_card, text=t("ready"),
            font=ctk.CTkFont(size=10), text_color=_C["dim"], anchor="w"
        )
        self._prog_lbl.pack(anchor="w")

        # ── 5. Cikti Metin Kutusu ──────────────────────────────────────────
        out_card = _card(body, t("en_translation"))

        self._out_box = ctk.CTkTextbox(
            out_card,
            font=ctk.CTkFont(size=12),
            fg_color=_C["surface2"], border_color=_C["border"], border_width=1,
            text_color=_C["text"], wrap="word", corner_radius=10,
            height=340
        )
        self._out_box.pack(fill="both", expand=True)
        self._out_box.configure(state="disabled")

        ctk.CTkLabel(
            out_card,
            text="Ceviri tamamlaninca metin burada gorunur.",
            font=ctk.CTkFont(size=9), text_color=_C["dim"]
        ).pack(anchor="w", pady=(4, 0))

    # ── Dosya Secimi ──────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title=t("browse"),
            filetypes=[
                ("Desteklenen belgeler", "*.txt *.pdf *.docx"),
                ("Metin",   "*.txt"),
                ("PDF",     "*.pdf"),
                ("Word",    "*.docx"),
                ("Hepsi",   "*.*"),
            ]
        )
        if path:
            self._file_entry.delete(0, "end")
            self._file_entry.insert(0, path)
            self._reset_output()

    def _reset_output(self):
        self._progress.set(0)
        self._prog_lbl.configure(text=t("ready"), text_color=_C["dim"])
        self._out_box.configure(state="normal")
        self._out_box.delete("0.0", "end")
        self._out_box.configure(state="disabled")
        self._btn_save.configure(state="disabled")

    # ── Baslat / Iptal ────────────────────────────────────────────────────────

    def _start(self):
        path = self._file_entry.get().strip()
        if not path:
            messagebox.showwarning(t("file_not_selected"), t("select_file_first"))
            return
        if not os.path.exists(path):
            messagebox.showerror(t("file_not_found"), t("file_exists_error", path))
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in _DOC_EXT:
            messagebox.showwarning(
                t("unsupported_format"),
                t("unsupported_ext", ext, "TXT, PDF, DOCX")
            )
            return
        if not getattr(self.app, "_backend_ready", False):
            messagebox.showwarning(t("backend_not_ready_models"),
                                   t("models_still_loading"))
            return

        src_lang = self._src_lang_combo.get()
        tgt_lang = self._tgt_lang_combo.get()
        chunk_words = int(self._chunk_combo.get().split()[0])

        self._translating = True
        self._btn_start.configure(state="disabled")
        self._btn_cancel.configure(state="normal")
        self._btn_save.configure(state="disabled")
        self._reset_output()

        threading.Thread(
            target=self._translation_pipeline,
            args=(path, src_lang, tgt_lang, chunk_words),
            daemon=True
        ).start()

    def _cancel(self):
        if self._dt is not None:
            self._dt.cancel()
        self._translating = False
        self._set_progress(0, "Iptal edildi.", _C["yellow"])
        self._btn_start.configure(state="normal")
        self._btn_cancel.configure(state="disabled")

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _translation_pipeline(self, file_path: str, src_lang: str,
                               tgt_lang: str, chunk_words: int):
        from pipeline.document_translator import DocumentTranslator

        translator = self.app._orchestrator.translator

        self._dt = DocumentTranslator(
            translator=translator,
            chunk_words=chunk_words,
            overlap_paragraphs=3
        )

        def on_progress(frac, msg):
            self._set_progress(frac, msg)
            # Her chunk tamamlandikca ciktiye ekle (streaming hissi)

        try:
            result = self._dt.translate_file(
                file_path=file_path,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                output_path=None,          # GUI kaydetme dugmesini kullanir
                progress_cb=on_progress
            )

            if self._translating:          # iptal edilmediyse
                self._append_output(result)
                self.after(0, lambda: self._btn_save.configure(state="normal"))

        except Exception as e:
            self._set_progress(0, f"Hata: {e}", _C["red"])
        finally:
            self._translating = False
            self._dt = None
            self.after(0, lambda: [
                self._btn_start.configure(state="normal"),
                self._btn_cancel.configure(state="disabled"),
            ])

    # ── Kaydet ────────────────────────────────────────────────────────────────

    def _save(self):
        text = self._out_box.get("0.0", "end").strip()
        if not text:
            messagebox.showinfo("Bos", "Kaydedilecek metin yok.")
            return
        src = self._file_entry.get().strip()
        base = os.path.splitext(os.path.basename(src))[0] if src else "ceviri"
        tgt = self._tgt_lang_combo.get().lower()[:2]
        path = filedialog.asksaveasfilename(
            title=t("save_translation"),
            initialfile=f"{base}_{tgt}.txt",
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
            messagebox.showinfo("Kaydedildi", f"Dosya kaydedildi:\n{path}")

    # ── Thread-safe yardimcilar ───────────────────────────────────────────────

    def _set_progress(self, val: float, msg: str, color: str = None):
        def _u():
            self._progress.set(max(0.0, min(1.0, val)))
            self._prog_lbl.configure(text=msg, text_color=color or _C["dim"])
        self.after(0, _u)

    def _append_output(self, text: str):
        def _u():
            self._out_box.configure(state="normal")
            self._out_box.delete("0.0", "end")
            self._out_box.insert("0.0", text)
            self._out_box.configure(state="disabled")
        self.after(0, _u)


# ══════════════════════════════════════════════════════════════════════════════
# Metin Ceviri View
# ══════════════════════════════════════════════════════════════════════════════

class TextView(ctk.CTkFrame):
    def __init__(self, master, cfg: ConfigManager, app):
        super().__init__(master, fg_color=_C["bg"], corner_radius=0)
        self.cfg = cfg
        self.app = app
        self._mic_recording  = False
        self._mic_stop_event = None
        self._build()

    def _build(self):
        _header(self, f"\U0001f4dd  {t('text_title')}",
                t("text_subtitle"))

        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 12))

        # ── Giris ─────────────────────────────────────────────────────
        in_card = _card(body, t("source_text"))
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
            br, text=f"\u25b6  {t('translate')}", height=38, width=110,
            fg_color=_C["blue"], hover_color="#4080d0",
            corner_radius=10, font=ctk.CTkFont(size=12, weight="bold"),
            command=self._translate
        )
        self._btn_tr.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            br, text=t("clear"), height=38, width=90,
            fg_color=_C["surface2"], hover_color=_C["border"],
            corner_radius=10, font=ctk.CTkFont(size=11),
            command=self._clear
        ).pack(side="left", padx=(0, 8))

        self._btn_mic = ctk.CTkButton(
            br, text=f"\U0001f3a4  {t('listening')}", height=38, width=110,
            fg_color=_C["surface2"], hover_color=_C["border"],
            corner_radius=10, font=ctk.CTkFont(size=12),
            command=self._toggle_mic
        )
        self._btn_mic.pack(side="left")

        self._mic_lbl = ctk.CTkLabel(
            br, text="",
            font=ctk.CTkFont(size=10),
            text_color=_C["muted"]
        )
        self._mic_lbl.pack(side="left", padx=(10, 0))

        # ── Cikis ─────────────────────────────────────────────────────
        out_card = _card(body, t("target_text"))
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
            out_card, text=t("copy"), height=32, width=90,
            fg_color=_C["surface2"], hover_color=_C["border"],
            corner_radius=8, font=ctk.CTkFont(size=10),
            command=self._copy
        ).pack(anchor="e", pady=(8, 0))

    # ── Mikrofon ──────────────────────────────────────────────────────────────

    def _toggle_mic(self):
        if self._mic_recording:
            if self._mic_stop_event:
                self._mic_stop_event.set()
        else:
            self._start_mic()

    def _start_mic(self):
        if not self.app._backend_ready:
            messagebox.showinfo("Bekleyin", "Modeller henuz yukleniyor.")
            return
        self._mic_recording  = True
        self._mic_stop_event = threading.Event()
        self._btn_mic.configure(
            text=f"\u23f9  {t('stop')}", fg_color=_C["red"], hover_color="#c03030"
        )
        self._mic_lbl.configure(text=t("recording"), text_color=_C["red"])
        threading.Thread(target=self._mic_capture, daemon=True).start()

    def _mic_capture(self):
        import sounddevice as sd
        import wave, tempfile, numpy as np

        SAMPLE_RATE = 16000
        frames      = []

        def _cb(indata, frame_count, time_info, status):
            frames.append(bytes(indata))

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=_cb
        ):
            self._mic_stop_event.wait()   # Durdur butonuna basilana kadar bekle

        # WAV yaz (RMS normalizasyon)
        audio_bytes = b"".join(frames)
        if audio_bytes:
            import numpy as _np
            samples = _np.frombuffer(audio_bytes, dtype=_np.int16).astype(_np.float32)
            rms = _np.sqrt(_np.mean(samples ** 2))
            if rms > 50:
                gain = min(3000.0 / rms, 10.0)
                samples = _np.clip(samples * gain, -32767, 32767)
            audio_bytes = samples.astype(_np.int16).tobytes()

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wav_path = tmp.name
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)

        # Tanıma aşaması
        self.after(0, lambda: self._btn_mic.configure(
            text="\U0001f504  Taniniyor...", fg_color=_C["yellow"],
            hover_color=_C["yellow"], state="disabled"
        ))
        self.after(0, lambda: self._mic_lbl.configure(
            text=t("processing"), text_color=_C["yellow"]
        ))

        try:
            result = self.app._orchestrator.transcriber.transcribe(wav_path)
            text   = result.get("text", "").strip()

            def _insert():
                if text:
                    self._in.delete("0.0", "end")
                    self._in.insert("0.0", text)
                    self._mic_lbl.configure(
                        text=f"{len(text.split())} kelime tanindi", text_color=_C["green"]
                    )
                else:
                    self._mic_lbl.configure(
                        text=t("speech_not_recognized"), text_color=_C["red"]
                    )
            self.after(0, _insert)
        except Exception as e:
            self.after(0, lambda: self._mic_lbl.configure(
                text=f"Hata: {e}", text_color=_C["red"]
            ))
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass
            self._mic_recording = False

            def _reset_btn():
                self._btn_mic.configure(
                    text=f"\U0001f3a4  {t('listening')}",
                    fg_color=_C["surface2"], hover_color=_C["border"],
                    state="normal"
                )
            self.after(0, _reset_btn)

    # ── Ceviri ───────────────────────────────────────────────────────────────

    def _translate(self):
        text = self._in.get("0.0", "end").strip()
        if not text:
            return
        if not self.app._backend_ready:
            messagebox.showinfo(t("backend_not_ready_models"), t("models_still_loading"))
            return
        self._set_out(t("translating"))
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
        # Bekleyen gosterimi iptal et
        if self._show_after_id:
            self.after_cancel(self._show_after_id)
        # Baska bir InfoIcon'un acik tooltip'ini kapat
        if _InfoIcon._active and _InfoIcon._active is not self:
            _InfoIcon._active._do_hide()
        # Kisa gecikmeyle goster (hizli gecislerde gereksiz acilmasin)
        self._show_after_id = self.after(150, self._do_show)

    def _on_leave(self, event=None):
        # Bekleyen gosterimi iptal et
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
        # Ana pencere odagi kaybedince tooltip'i kapat
        try:
            root = self.winfo_toplevel()
            self._focus_bind_id = root.bind("<FocusOut>", self._on_leave, add="+")
        except Exception:
            pass

    def _do_hide(self):
        # FocusOut binding'ini temizle
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


class SettingsView(ctk.CTkFrame):
    _MODES = [
        ("interactive",      "mode_interactive"),
        ("interactive_hq",   "mode_interactive_hq"),
        ("online",           "mode_online"),
        ("online_xtts",      "mode_online_xtts"),
        ("online_local_stt", "mode_online_local_stt"),
        ("offline_gpu",      "mode_offline_gpu"),
        ("offline",          "mode_offline"),
        ("hybrid_cloud_io",  "mode_hybrid_io"),
        ("hybrid_cloud_stt", "mode_hybrid_stt"),
        ("custom",           "mode_custom"),
    ]

    def __init__(self, master, cfg: ConfigManager, app):
        super().__init__(master, fg_color=_C["bg"], corner_radius=0)
        self.cfg = cfg
        self.app = app
        self._build()

    def _build(self):
        _header(self, f"\u2699  {t('settings_title')}",
                t("settings_subtitle"))

        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 12))

        # ── Calisma Modu ──────────────────────────────────────────────
        mode_card = _card(body, t("modes"))

        mode_hdr = ctk.CTkFrame(mode_card, fg_color="transparent")
        mode_hdr.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(
            mode_hdr, text=f"{t('preset_profile')}:",
            font=ctk.CTkFont(size=11), text_color=_C["muted"]
        ).pack(side="left")
        _InfoIcon(mode_hdr, t("tip_preset_profile")).pack(side="left", padx=(4, 0))

        mode_vals = [t(m[1]) for m in self._MODES]
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

        # ── Ozel Mod Secimi ───────────────────────────────────────────
        custom_card = _card(body, t("settings"))

        # STT satiri
        stt_hdr = ctk.CTkFrame(custom_card, fg_color="transparent")
        stt_hdr.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            stt_hdr, text=f"{t('stt_settings')}:",
            font=ctk.CTkFont(size=11), text_color=_C["muted"], width=160, anchor="w"
        ).pack(side="left")
        _InfoIcon(stt_hdr, t("tip_stt")).pack(side="left", padx=(4, 0))

        stt_cur = self.cfg.get("mode", "stt", "backend", default="local_gpu")
        self._stt_seg = ctk.CTkSegmentedButton(
            custom_card,
            values=["local_gpu", "local_cpu", "cloud_auto"],
            selected_color=_C["blue"], selected_hover_color="#4080d0",
            unselected_color=_C["surface2"],
            font=ctk.CTkFont(size=11)
        )
        self._stt_seg.set(stt_cur)
        self._stt_seg.pack(fill="x", pady=(0, 8))

        # LLM satiri
        llm_hdr = ctk.CTkFrame(custom_card, fg_color="transparent")
        llm_hdr.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            llm_hdr, text=f"{t('llm_settings')}:",
            font=ctk.CTkFont(size=11), text_color=_C["muted"], width=160, anchor="w"
        ).pack(side="left")
        _InfoIcon(llm_hdr, t("tip_llm")).pack(side="left", padx=(4, 0))

        llm_cur = self.cfg.get("mode", "llm", "backend", default="online")
        self._llm_seg = ctk.CTkSegmentedButton(
            custom_card,
            values=["online", "offline"],
            selected_color=_C["blue"], selected_hover_color="#4080d0",
            unselected_color=_C["surface2"],
            font=ctk.CTkFont(size=11)
        )
        self._llm_seg.set(llm_cur)
        self._llm_seg.pack(fill="x", pady=(0, 8))

        # TTS satiri
        tts_hdr = ctk.CTkFrame(custom_card, fg_color="transparent")
        tts_hdr.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            tts_hdr, text=f"{t('tts_settings')}:",
            font=ctk.CTkFont(size=11), text_color=_C["muted"], width=160, anchor="w"
        ).pack(side="left")
        _InfoIcon(tts_hdr, t("tip_tts")).pack(side="left", padx=(4, 0))

        tts_cur = self.cfg.get("mode", "tts", "backend", default="online")
        self._tts_seg = ctk.CTkSegmentedButton(
            custom_card,
            values=["online", "gpu", "offline"],
            selected_color=_C["blue"], selected_hover_color="#4080d0",
            unselected_color=_C["surface2"],
            font=ctk.CTkFont(size=11)
        )
        self._tts_seg.set(tts_cur)
        self._tts_seg.pack(fill="x", pady=(0, 8))

        # Uygula butonu
        ctk.CTkButton(
            custom_card, text=t("apply"),
            height=34, corner_radius=10,
            fg_color=_C["blue"], hover_color="#4080d0",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._apply_custom_mode
        ).pack(fill="x")

        # ── Yayıncı / İçerik Üretici Modu ──────────────────────────────
        broad_card = _card(body, t("nav_media"))
        
        broad_hdr = ctk.CTkFrame(broad_card, fg_color="transparent")
        broad_hdr.pack(fill="x", pady=(0, 6))
        
        ctk.CTkLabel(
            broad_hdr, text=f"{t('broadcaster_mode')}:",
            font=ctk.CTkFont(size=11), text_color=_C["muted"]
        ).pack(side="left")
        _InfoIcon(broad_hdr, t("tip_broadcaster")).pack(side="left", padx=(4, 0))

        # Toggle ve Dropdown satırı
        broad_row = ctk.CTkFrame(broad_card, fg_color="transparent")
        broad_row.pack(fill="x", pady=(4, 0))

        self._broad_switch = ctk.CTkSwitch(
            broad_row, text=t("active"),
            command=self._on_broadcaster_toggle,
            progress_color=_C["blue"]
        )
        if self.cfg.get("broadcaster", "enabled", default=False):
            self._broad_switch.select()
        self._broad_switch.pack(side="left", padx=(0, 20))

        # Cihaz listesi
        devices = self._get_output_devices()
        device_names = [d[1] for d in devices]
        
        self._device_combo = ctk.CTkComboBox(
            broad_row, values=device_names,
            height=34, corner_radius=10, font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            width=300,
            command=self._on_device_select
        )
        
        cur_dev_name = self.cfg.get("broadcaster", "output_device_name", default="Default")
        self._device_combo.set(cur_dev_name)
        self._device_combo.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            broad_card,
            text=t("broadcaster_hint"),
            font=ctk.CTkFont(size=9), text_color=_C["dim"]
        ).pack(anchor="w", pady=(8, 0))

        # ── Donanim bilgisi ───────────────────────────────────────────
        hw_card = _card(body, t("hardware_profile"))
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
        api_card = _card(body, t("api_keys"))
        for svc, lbl, url in [
            ("gemini",     "Gemini",     "https://aistudio.google.com/apikey"),
            ("groq",       "Groq",       "https://console.groq.com/keys"),
            ("elevenlabs", "ElevenLabs", "https://elevenlabs.io/app/settings/api-keys"),
        ]:
            self._api_row(api_card, svc, lbl, url)

        # ── ElevenLabs Ses ────────────────────────────────────────────
        voice_card = _card(body, t("elevenlabs_voice_id_label"))
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
            vr, text=t("voice_library"), width=130, height=34,
            fg_color=_C["surface2"], corner_radius=10,
            command=lambda: webbrowser.open("https://elevenlabs.io/app/voice-library")
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            vr, text=t("save_settings"), width=72, height=34,
            fg_color=_C["blue"], corner_radius=10,
            command=self._save_voice
        ).pack(side="left")

        # ── Overlay Opakligi ──────────────────────────────────────────
        ovl_card = _card(body, t("overlay_title"))
        op_row = ctk.CTkFrame(ovl_card, fg_color="transparent")
        op_row.pack(fill="x")

        ctk.CTkLabel(
            op_row, text=f"{t('opacity')}:",
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
            vad_row, text=f"{t('vad_settings')} (0-3):",
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

        # ── Çeviri Karakteri (Persona) ────────────────────────────────────────
        persona_card = _card(body, t("persona_title"))

        persona_hdr = ctk.CTkFrame(persona_card, fg_color="transparent")
        persona_hdr.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(
            persona_hdr, text=f"{t('persona_style')}:",
            font=ctk.CTkFont(size=11), text_color=_C["muted"]
        ).pack(side="left")
        _InfoIcon(persona_hdr, t("tip_persona")).pack(side="left", padx=(4, 0))

        _PERSONA_OPTIONS = [
            ("none",     "persona_none"),
            ("official", "persona_official"),
            ("streamer", "persona_streamer"),
            ("casual",   "persona_casual"),
            ("literary", "persona_literary"),
        ]
        persona_vals = [t(p[1]) for p in _PERSONA_OPTIONS]
        cur_persona  = self.cfg.get("persona", default="none")
        cur_persona_idx = next(
            (i for i, p in enumerate(_PERSONA_OPTIONS) if p[0] == cur_persona), 0
        )
        self._persona_combo = ctk.CTkComboBox(
            persona_card, values=persona_vals,
            height=36, corner_radius=10, font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            command=lambda display: self._on_persona(display, _PERSONA_OPTIONS)
        )
        self._persona_combo.set(persona_vals[cur_persona_idx])
        self._persona_combo.pack(fill="x")

        # ── UI Language Switcher ──────────────────────────────────────
        lang_card = _card(body, t("ui_language_setting"))
        lang_row = ctk.CTkFrame(lang_card, fg_color="transparent")
        lang_row.pack(fill="x")

        self._lang_seg = ctk.CTkSegmentedButton(
            lang_row, values=["tr", "en"],
            command=self._on_ui_lang,
            selected_color=_C["blue"], selected_hover_color="#4080d0",
            unselected_color=_C["surface2"],
            font=ctk.CTkFont(size=11)
        )
        self._lang_seg.set(self.cfg.get("language", "ui_language", default="tr"))
        self._lang_seg.pack(fill="x")

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
            row, text=t("save_settings"), width=72, height=32, corner_radius=8,
            fg_color=_C["blue"],
            command=lambda s=svc, e=entry: self.cfg.set_api_key(s, e.get().strip())
        ).pack(side="left")

    # ── Olaylar ───────────────────────────────────────────────────────────────

    def _on_mode(self, display: str):
        key = next((m[0] for m in self._MODES if t(m[1]) == display), None)
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

    def _apply_custom_mode(self):
        stt = self._stt_seg.get()
        llm = self._llm_seg.get()
        tts = self._tts_seg.get()

        self.cfg.set("mode", "stt", "backend", stt)
        self.cfg.set("mode", "llm", "backend", llm)
        self.cfg.set("mode", "tts", "backend", tts)
        self.cfg.save()

        # Combo'yu "custom" olarak guncelle
        custom_display = next(
            (t(m[1]) for m in self._MODES if m[0] == "custom"), None
        )
        if custom_display:
            self._mode_combo.set(custom_display)

        self.app.switch_mode("custom")

    def _save_voice(self):
        vid = self._voice_entry.get().strip()
        if vid:
            self.cfg.set_voice(vid)

    def _get_output_devices(self):
        """Sistemdeki ses çıkış cihazlarını listeler."""
        import sounddevice as sd
        try:
            devices = sd.query_devices()
            outputs = [(None, "Default")]
            for i, d in enumerate(devices):
                if d['max_output_channels'] > 0:
                    outputs.append((i, d['name']))
            return outputs
        except Exception as e:
            print(f"[HATA] Ses cihazlari listelenemedi: {e}")
            return [(None, "Default")]

    def _on_broadcaster_toggle(self):
        enabled = self._broad_switch.get()
        self.cfg.set("broadcaster", "enabled", bool(enabled))
        self.cfg.save()

        # Synthesizer'i guncelle (backend hazir degilse atla)
        if not self.app._orchestrator:
            return
        if enabled:
            self._on_device_select(self._device_combo.get())
        else:
            self.app._orchestrator.synthesizer.set_output_device(None)

    def _on_device_select(self, name: str):
        devices = self._get_output_devices()
        idx = next((d[0] for d in devices if d[1] == name), None)

        self.cfg.set("broadcaster", "output_device_index", idx)
        self.cfg.set("broadcaster", "output_device_name", name)
        self.cfg.save()

        if self._broad_switch.get() and self.app._orchestrator:
            self.app._orchestrator.synthesizer.set_output_device(idx)

    def _on_persona(self, display: str, options: list):
        key = next((p[0] for p in options if t(p[1]) == display), "none")
        self.cfg.set("persona", key)
        self.cfg.save()
        # Canlı güncelleme: orkestra hazırsa anında translator'a bildir
        if self.app._orchestrator:
            self.app._orchestrator.translator.set_persona(key)

    def _on_ui_lang(self, lang: str):
        from gui.i18n import set_language
        set_language(lang)

        self.cfg.set("language", "ui_language", lang)
        self.cfg.save()

        # Rebuild MainWindow dynamically
        main_win = self.app._main
        if main_win:
            for child in main_win.winfo_children():
                child.destroy()
            main_win.title(t("app_name"))
            main_win._build()
            main_win.switch_view("settings")
