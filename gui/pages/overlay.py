"""
Gemma Echo — Yüzen Altyazı Overlay Penceresi

Her zaman üstte, sürüklenebilir, yarı şeffaf.
result_queue üzerinden gelen STT + çeviri verilerini anlık gösterir.

Entegrasyon:
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

# ─── Tema ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")

_C = {
    "bg":       "#0d1117",   # ana arka plan
    "bar":      "#161b22",   # baslik cubugu
    "card":     "#1c2128",   # metin karti
    "border":   "#30363d",   # kenarlık
    "blue":     "#58a6ff",   # EN metni rengi
    "gray":     "#8b949e",   # TR metni rengi
    "green":    "#3fb950",   # aktif gösterge
    "yellow":   "#d29922",   # bekleme
    "red":      "#f85149",   # hata
    "white":    "#e6edf3",   # genel metin
    "dim":      "#484f58",   # buton hover
}

WIN_W   = 480
WIN_H   = 162
POLL_MS = 80     # kuyruk yoklama araligi (ms)
FADE_MS = 4000   # son mesajin solmaya baslama suresi (ms)


class Overlay(ctk.CTkToplevel):
    """
    Yüzen altyazı penceresi (CTkToplevel — HomeWindow'a bağlı).

    Parametreler:
        cfg          : ConfigManager ornegi
        result_queue : orchestrator.result_queue — None ise demo modunda calisir
    """

    def __init__(self, cfg: ConfigManager, result_queue: queue.Queue | None = None):
        super().__init__()
        self.cfg          = cfg
        self._rq          = result_queue
        self._dragging    = False
        self._drag_x      = 0
        self._drag_y      = 0
        self._last_update = 0.0   # son guncelleme zaman damgasi
        self._fade_job    = None  # solma zamanlayicisi

        self._setup_window()
        self._build_ui()
        self._start_poll()

    # ── Pencere Kurulumu ─────────────────────────────────────────────────────

    def _setup_window(self):
        self.overrideredirect(True)           # sistem baslik cubugunu kaldir
        self.wm_attributes("-topmost", True)  # her zaman ustte
        self.configure(fg_color=_C["bg"])

        opacity = self.cfg.get("overlay", "opacity", default=0.92)
        self.wm_attributes("-alpha", opacity)

        # Pencere konumu — varsayılan: üst-sağ (taskbar ve chat kutusunu kapatmaz)
        self.update_idletasks()
        x = self.cfg.get("overlay", "position_x", default=-1)
        y = self.cfg.get("overlay", "position_y", default=-1)

        if x == -1 or y == -1:
            sw = self.winfo_screenwidth()
            x  = sw - WIN_W - 24
            y  = 40   # üst-sağ köşe

        self.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

        # Başlangıçta gizli — HomeWindow'dan açılır
        self.withdraw()

    # ── UI Yapisi ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_titlebar()
        self._build_content()
        self._build_statusbar()

    def _build_titlebar(self):
        """Sürüklenebilir başlık çubuğu — uygulama adı + kontrol butonları."""
        bar = ctk.CTkFrame(self, fg_color=_C["bar"], corner_radius=0, height=30)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_propagate(False)

        # Sürükleme bağlamaları
        bar.bind("<ButtonPress-1>",   self._drag_start)
        bar.bind("<B1-Motion>",       self._drag_move)
        bar.bind("<ButtonRelease-1>", self._drag_end)

        # Sol: logo + isim
        left = ctk.CTkFrame(bar, fg_color="transparent")
        left.pack(side="left", padx=10)
        left.bind("<ButtonPress-1>",   self._drag_start)
        left.bind("<B1-Motion>",       self._drag_move)
        left.bind("<ButtonRelease-1>", self._drag_end)

        self._dot = ctk.CTkLabel(
            left, text="●", font=ctk.CTkFont(size=9),
            text_color=_C["yellow"]   # başlangıçta bekleme rengi
        )
        self._dot.pack(side="left", padx=(0, 5))
        self._dot.bind("<ButtonPress-1>", self._drag_start)
        self._dot.bind("<B1-Motion>",     self._drag_move)

        title = ctk.CTkLabel(
            left, text="GEMMA ECHO",
            font=ctk.CTkFont(family="Helvetica", size=10, weight="bold"),
            text_color=_C["dim"]
        )
        title.pack(side="left")
        title.bind("<ButtonPress-1>", self._drag_start)
        title.bind("<B1-Motion>",     self._drag_move)

        # Sağ: butonlar
        right = ctk.CTkFrame(bar, fg_color="transparent")
        right.pack(side="right", padx=6)

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
        """TR + EN metin kartı."""
        card = ctk.CTkFrame(
            self, fg_color=_C["card"], corner_radius=0,
            border_width=1, border_color=_C["border"]
        )
        card.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        card.columnconfigure(0, weight=1)

        # Türkçe satırı
        tr_row = ctk.CTkFrame(card, fg_color="transparent")
        tr_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 2))

        ctk.CTkLabel(
            tr_row, text="TR",
            font=ctk.CTkFont(size=8, weight="bold"),
            text_color=_C["dim"],
            width=20
        ).pack(side="left", anchor="n", padx=(0, 6))

        self._tr_label = ctk.CTkLabel(
            tr_row, text="Konuşmayı bekleniyor...",
            font=ctk.CTkFont(size=12), text_color=_C["gray"],
            anchor="w", justify="left", wraplength=420
        )
        self._tr_label.pack(side="left", fill="x", expand=True)

        # Ayraç
        sep = ctk.CTkFrame(card, fg_color=_C["border"], height=1)
        sep.grid(row=1, column=0, sticky="ew", padx=12, pady=4)

        # İngilizce satırı
        en_row = ctk.CTkFrame(card, fg_color="transparent")
        en_row.grid(row=2, column=0, sticky="ew", padx=12, pady=(2, 10))

        ctk.CTkLabel(
            en_row, text="EN",
            font=ctk.CTkFont(size=8, weight="bold"),
            text_color=_C["blue"],
            width=20
        ).pack(side="left", anchor="n", padx=(0, 6))

        self._en_label = ctk.CTkLabel(
            en_row, text="—",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=_C["white"],
            anchor="w", justify="left", wraplength=420
        )
        self._en_label.pack(side="left", fill="x", expand=True)

    def _build_statusbar(self):
        """Alt durum çubuğu — motor + gecikme bilgisi."""
        bar = ctk.CTkFrame(self, fg_color=_C["bar"], corner_radius=0, height=22)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)

        self._status_label = ctk.CTkLabel(
            bar, text="Hazır · VAD dinliyor",
            font=ctk.CTkFont(size=9), text_color=_C["dim"]
        )
        self._status_label.pack(side="left", padx=10)

        # Motor etiketi (sag)
        self._engine_label = ctk.CTkLabel(
            bar, text="",
            font=ctk.CTkFont(size=9), text_color=_C["dim"]
        )
        self._engine_label.pack(side="right", padx=10)

    # ── Kuyruk Yoklama ────────────────────────────────────────────────────────

    def _start_poll(self):
        self._poll()

    def _poll(self):
        """
        result_queue'yu yorarak yeni veri ceker ve UI'yi gunceller.
        Her POLL_MS milisaniyede bir kendini yeniden planlar.
        """
        if self._rq is not None:
            try:
                while True:                        # kuyrukta biriken tum mesajlari tüket
                    data = self._rq.get_nowait()
                    self._handle_result(data)
            except queue.Empty:
                pass

        self.after(POLL_MS, self._poll)

    def _handle_result(self, data: dict):
        """Gelen sozluge gore UI'yi guncelle."""
        if data.get("error"):
            self._show_error(str(data["error"]))
            return

        text_tr = data.get("text_tr", "")
        text_en = data.get("text_en", "")
        engine  = data.get("engine", "")
        e2e_ms  = data.get("latency_ms", 0)
        stt_ms  = data.get("stt_ms", 0)
        llm_ms  = data.get("llm_ms", 0)

        # Aktif çeviri — click-through kapat (sürüklenebilsin)
        self._set_clickthrough(False)

        # Metinler
        self._tr_label.configure(text=text_tr, text_color=_C["gray"])
        self._en_label.configure(text=text_en, text_color=_C["white"])

        # Durum cubugu
        e2e_s = f"{e2e_ms/1000:.1f}s"
        self._status_label.configure(
            text=f"STT {stt_ms}ms · LLM {llm_ms}ms · E2E {e2e_s}",
            text_color=_C["green"]
        )
        self._engine_label.configure(text=engine, text_color=_C["blue"])
        self._dot.configure(text_color=_C["green"])
        self._last_update = time.time()

        # Birkaç saniye sonra "dinliyor" moduna don
        if self._fade_job:
            self.after_cancel(self._fade_job)
        self._fade_job = self.after(FADE_MS, self._fade_to_idle)

    def _show_error(self, msg: str):
        self._tr_label.configure(text="Hata", text_color=_C["red"])
        self._en_label.configure(text=msg[:80], text_color=_C["red"])
        self._dot.configure(text_color=_C["red"])
        self._status_label.configure(text="Hata yakalandı", text_color=_C["red"])

    def _fade_to_idle(self):
        """Bir sure sonra göstergeyi bekleme rengine döndür + click-through aç."""
        self._dot.configure(text_color=_C["yellow"])
        self._status_label.configure(
            text="Hazır · VAD dinliyor", text_color=_C["dim"]
        )
        self._engine_label.configure(text="")
        self._set_clickthrough(True)   # boşta: fare tıklamalarını geçir

    def _set_clickthrough(self, enable: bool):
        """
        Windows: overlay'i fare tıklamalarına geçirgen yapar (WS_EX_TRANSPARENT).
        Aktif çeviri sırasında kapatılır ki sürükleme çalışsın.
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

    # ── Pencere Sürükleme ─────────────────────────────────────────────────────

    def _drag_start(self, event):
        self._dragging = True
        self._drag_x   = event.x_root - self.winfo_x()
        self._drag_y   = event.y_root - self.winfo_y()

    def _drag_move(self, event):
        if self._dragging:
            x = event.x_root - self._drag_x
            y = event.y_root - self._drag_y
            self.geometry(f"+{x}+{y}")

    def _drag_end(self, event):
        self._dragging = False
        # Konumu config'e kaydet
        self.cfg.set("overlay", "position_x", self.winfo_x())
        self.cfg.set("overlay", "position_y", self.winfo_y())
        self.cfg.save()

    # ── Buton Eylemleri ───────────────────────────────────────────────────────

    def _minimize(self):
        self.wm_attributes("-alpha", 0.0)
        self.after(200, lambda: self.wm_attributes("-alpha", 0.15))

    def _close(self):
        self.cfg.set("overlay", "position_x", self.winfo_x())
        self.cfg.set("overlay", "position_y", self.winfo_y())
        self.cfg.save()
        self.destroy()

    def _open_settings(self):
        """Ayarlar paneli acilacak (app.py tarafindan override edilir)."""
        pass   # app.py bu metodu monkey-patch edecek

    # ── Dis Arayuz ───────────────────────────────────────────────────────────

    def set_status(self, msg: str, color: str = None):
        """Arka plan thread'lerinden durum mesaji gonderme (thread-safe)."""
        self.after(0, lambda: self._status_label.configure(
            text=msg, text_color=color or _C["dim"]
        ))

    def set_listening(self):
        """Kayit basladiginda."""
        self.after(0, lambda: [
            self._dot.configure(text_color=_C["green"]),
            self._status_label.configure(
                text="Kaydediliyor...", text_color=_C["green"]
            ),
        ])

    def set_processing(self):
        """Islem suresi."""
        self.after(0, lambda: [
            self._dot.configure(text_color=_C["blue"]),
            self._status_label.configure(
                text="İşleniyor...", text_color=_C["blue"]
            ),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Demo modu (bagimsiz test)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os, threading
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from gui.config import ConfigManager

    cfg = ConfigManager()
    rq  = queue.Queue()

    overlay = Overlay(cfg, result_queue=rq)

    # 2 saniye sonra demo veri gonder
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
