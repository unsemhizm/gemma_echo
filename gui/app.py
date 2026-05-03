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
import webbrowser

import customtkinter as ctk

# Proje kokunu path'e ekle
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gui.config       import ConfigManager
from gui.hardware_scan import scan as hw_scan
from gui.pages.overlay import Overlay

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_C = {
    "bg":     "#0d1117",
    "panel":  "#161b22",
    "card":   "#1c2128",
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


# ══════════════════════════════════════════════════════════════════════════════
# Ana Uygulama
# ══════════════════════════════════════════════════════════════════════════════

class GemmaEchoApp:
    """
    Uygulamanin yasam dongusunu yoneten sinif.
    Dogrudan bir pencere degil; pencereler arasinda koordinasyon saglar.
    """

    def __init__(self):
        self.cfg      = ConfigManager()
        self._rq      = queue.Queue()   # orchestrator → overlay koprusu
        self._overlay: Overlay | None   = None
        self._backend_ready             = False
        self._recorder                  = None
        self._orchestrator              = None
        self._backend_thread            = None

    def run(self):
        if self.cfg.is_first_run():
            self._run_wizard()
        else:
            self._launch_main()

    # ── Kurulum Sihirbazi ─────────────────────────────────────────────────────

    def _run_wizard(self):
        from gui.pages.setup_wizard import SetupWizard
        wizard = SetupWizard(self.cfg, on_complete=self._on_wizard_done)
        wizard.mainloop()

    def _on_wizard_done(self, cfg: ConfigManager):
        self.cfg = cfg
        self._launch_main()

    # ── Ana Ekran ─────────────────────────────────────────────────────────────

    def _launch_main(self):
        """Kontrol panelini + overlay'i baslatir."""
        self._panel  = ControlPanel(self.cfg, app=self)
        self._overlay = Overlay(self.cfg, result_queue=self._rq)

        # Ayarlar butonunu overlay'e bagla
        self._overlay._open_settings = self._panel.show

        # Backend'i arka planda yukle
        self._backend_thread = threading.Thread(
            target=self._load_backend, daemon=True
        )
        self._backend_thread.start()

        # Overlay ana event loop'u tasiyor
        self._overlay.mainloop()

    # ── Backend Yukleme ───────────────────────────────────────────────────────

    def _load_backend(self):
        """
        STT / LLM / TTS modellerini arka planda yukler.
        Tamamlaninca orchestrator'u result_queue'ya baglar.
        """
        try:
            self._overlay.set_status("Modeller yükleniyor...", _C["yellow"])

            from stt.transcriber     import Transcriber
            from llm.translator      import Translator
            from tts.synthesizer     import Synthesizer
            from pipeline.orchestrator import Orchestrator

            transcriber  = Transcriber()
            translator   = Translator()
            synthesizer  = Synthesizer()

            mode = self.cfg.get("mode", "current", default="online")
            self._orchestrator = Orchestrator(
                transcriber, translator, synthesizer,
                initial_mode=mode
            )
            self._orchestrator.result_queue = self._rq
            self._orchestrator.warm_up()

            self._backend_ready = True
            self._overlay.set_status("Hazır · VAD dinliyor", _C["dim"])
            self._overlay.after(0, lambda: self._overlay._dot.configure(
                text_color=_C["yellow"]
            ))

        except Exception as e:
            self._overlay.set_status(f"Yükleme hatası: {e}", _C["red"])

    # ── Kayit Kontrolu ─────────────────────────────────────────────────────────

    def start_live(self):
        """VAD veya Bas-Konuş modunda kayıt başlatır."""
        if not self._backend_ready:
            self._overlay.set_status("Backend henüz hazır değil...", _C["yellow"])
            return
        if self._recorder is not None:
            return   # zaten calisiyor

        from stt.recorder import Recorder
        aggressiveness = self.cfg.get("recording", "vad_aggressiveness", default=2)
        self._recorder = Recorder(self._orchestrator, aggressiveness=aggressiveness)

        threading.Thread(target=self._recorder.run, daemon=True).start()
        self._overlay.set_status("Canlı dinleme aktif", _C["green"])

    def stop_live(self):
        """Kaydı durdurur."""
        if self._recorder:
            self._recorder._stop_event.set()
            self._recorder = None
        self._overlay.set_status("Hazır · durdu", _C["dim"])

    def switch_mode(self, mode: str):
        """Orchestrator modunu anında değiştirir."""
        if self._orchestrator:
            try:
                self._orchestrator.set_mode(mode)
                self.cfg.set_mode(mode)
                self._overlay.set_status(f"Mod: {mode.upper()}", _C["blue"])
            except Exception as e:
                self._overlay.set_status(f"Mod hatası: {e}", _C["red"])

    def process_file(self, path: str):
        """Ses/video dosyasını pipeline'a gönderir."""
        if not self._backend_ready:
            return
        threading.Thread(
            target=self._orchestrator.process, args=(path,), daemon=True
        ).start()


# ══════════════════════════════════════════════════════════════════════════════
# Kontrol Paneli (ayrı, küçük pencere)
# ══════════════════════════════════════════════════════════════════════════════

class ControlPanel(ctk.CTkToplevel):
    """
    Mod seçimi, başlat/durdur, dosya modu ve ayarlara hızlı erişim.
    Overlay'den ⚙ butonuna basılınca gösterilir.
    İstenmezse gizlenebilir — overlay bağımsız çalışır.
    """

    MODES = [
        ("interactive",       "Interactive  — STT GPU · LLM Cloud · TTS GPU"),
        ("online",            "Online       — Tümü bulut"),
        ("online_xtts",       "Online XTTS  — Bulut + Yerel TTS"),
        ("online_local_stt",  "Local STT    — STT GPU · LLM+TTS bulut"),
        ("offline_gpu",       "Offline GPU  — Tümü yerel GPU"),
        ("offline",           "Offline CPU  — Tümü yerel CPU"),
        ("hybrid_cloud_io",   "Hybrid I/O   — Bulut kulak/ağız · Yerel beyin"),
        ("hybrid_cloud_stt",  "Hybrid STT   — Bulut kulak · Yerel beyin/ses"),
    ]

    def __init__(self, cfg: ConfigManager, app: GemmaEchoApp):
        super().__init__()
        self.cfg = cfg
        self.app = app

        self.title("Gemma Echo — Kontrol")
        self.geometry("520x560")
        self.resizable(False, False)
        self.configure(fg_color=_C["bg"])
        self.protocol("WM_DELETE_WINDOW", self.hide)

        # Ekran orta-sag
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"520x560+{sw-560}+{(sh-560)//2}")

        self._build()
        self.hide()   # Baslangicta gizli

    def show(self):
        self.deiconify()
        self.lift()
        self.focus()

    def hide(self):
        self.withdraw()

    def _build(self):
        # ── Baslik ──────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=_C["accent"], corner_radius=0, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(
            hdr, text="GEMMA ECHO  Kontrol Paneli",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="white"
        ).pack(side="left", padx=20, pady=14)

        body = ctk.CTkScrollableFrame(self, fg_color=_C["bg"], corner_radius=0)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # ── Mod Secimi ───────────────────────────────────────────────
        self._section(body, "Çalışma Modu")
        mode_vals = [m[1] for m in self.MODES]
        cur_mode  = self.cfg.get("mode", "current", default="online")
        cur_idx   = next((i for i, m in enumerate(self.MODES) if m[0] == cur_mode), 0)

        self._mode_combo = ctk.CTkComboBox(
            body, values=mode_vals, width=460, height=34,
            font=ctk.CTkFont(size=11),
            command=self._on_mode_change
        )
        self._mode_combo.set(mode_vals[cur_idx])
        self._mode_combo.pack(padx=20, pady=(4, 12))

        # ── Kayit Kontrolleri ────────────────────────────────────────
        self._section(body, "Canlı Çeviri")
        btn_row = ctk.CTkFrame(body, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(4, 0))

        self._btn_start = ctk.CTkButton(
            btn_row, text="▶  Başlat", width=140, height=38,
            fg_color=_C["green"], hover_color="#2ea043",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._start
        )
        self._btn_start.pack(side="left", padx=(0, 10))

        self._btn_stop = ctk.CTkButton(
            btn_row, text="■  Durdur", width=140, height=38,
            fg_color=_C["red"], hover_color="#b91c1c",
            font=ctk.CTkFont(size=13, weight="bold"),
            state="disabled",
            command=self._stop
        )
        self._btn_stop.pack(side="left")

        # VAD / Bas-Konus toggle
        ptt_row = ctk.CTkFrame(body, fg_color="transparent")
        ptt_row.pack(fill="x", padx=20, pady=(8, 12))
        ctk.CTkLabel(
            ptt_row, text="Bas-Konuş (Push-to-Talk)",
            font=ctk.CTkFont(size=11), text_color=_C["gray"]
        ).pack(side="left")
        self._ptt_switch = ctk.CTkSwitch(
            ptt_row, text="", width=44,
            command=self._on_ptt_toggle,
            onvalue=True, offvalue=False
        )
        self._ptt_switch.pack(side="left", padx=10)
        if self.cfg.get("recording", "push_to_talk", default=False):
            self._ptt_switch.select()

        # ── Dosya Modu ───────────────────────────────────────────────
        self._section(body, "Dosya Çevirisi")
        ctk.CTkLabel(
            body,
            text="Ses veya video dosyası yükleyin. Transkript + İngilizce çeviri üretilir.",
            font=ctk.CTkFont(size=10), text_color=_C["gray"], wraplength=440
        ).pack(anchor="w", padx=20, pady=(0, 6))

        file_row = ctk.CTkFrame(body, fg_color="transparent")
        file_row.pack(fill="x", padx=20, pady=(0, 4))

        self._file_entry = ctk.CTkEntry(
            file_row, width=330, height=32, placeholder_text="Dosya seçin...",
            font=ctk.CTkFont(size=11)
        )
        self._file_entry.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            file_row, text="Gözat", width=80, height=32,
            fg_color=_C["panel"], hover_color=_C["card"],
            command=self._browse_file
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            file_row, text="Çevir ▶", width=80, height=32,
            fg_color=_C["blue"],
            command=self._process_file
        ).pack(side="left")

        # ── API Anahtarlari ──────────────────────────────────────────
        self._section(body, "API Anahtarları")
        for service, label, url in [
            ("gemini",     "Gemini",     "https://aistudio.google.com/apikey"),
            ("groq",       "Groq",       "https://console.groq.com/keys"),
            ("elevenlabs", "ElevenLabs", "https://elevenlabs.io/app/settings/api-keys"),
        ]:
            self._api_row(body, service, label, url)

        # ── ElevenLabs Ses ───────────────────────────────────────────
        self._section(body, "ElevenLabs Ses Seçimi")
        v_row = ctk.CTkFrame(body, fg_color="transparent")
        v_row.pack(fill="x", padx=20, pady=(4, 4))

        self._voice_entry = ctk.CTkEntry(
            v_row, width=280, height=32,
            placeholder_text="Voice ID",
            font=ctk.CTkFont(size=11)
        )
        vid = self.cfg.get("elevenlabs_voice_id", default="")
        if vid:
            self._voice_entry.insert(0, vid)
        self._voice_entry.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            v_row, text="Ses Kütüphanesi →", width=130, height=32,
            fg_color=_C["panel"],
            command=lambda: webbrowser.open("https://elevenlabs.io/app/voice-library")
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            v_row, text="Kaydet", width=72, height=32,
            fg_color=_C["blue"],
            command=self._save_voice
        ).pack(side="left")

        # ── Overlay Opakligi ─────────────────────────────────────────
        self._section(body, "Overlay Görünümü")
        op_row = ctk.CTkFrame(body, fg_color="transparent")
        op_row.pack(fill="x", padx=20, pady=(4, 16))

        ctk.CTkLabel(
            op_row, text="Opaklık:",
            font=ctk.CTkFont(size=11), text_color=_C["gray"]
        ).pack(side="left")

        self._opacity_slider = ctk.CTkSlider(
            op_row, from_=0.3, to=1.0, width=200,
            command=self._on_opacity_change
        )
        self._opacity_slider.set(self.cfg.get("overlay", "opacity", default=0.92))
        self._opacity_slider.pack(side="left", padx=12)

        self._opacity_label = ctk.CTkLabel(
            op_row, text=f"{self.cfg.get('overlay', 'opacity', default=0.92):.0%}",
            font=ctk.CTkFont(size=11), text_color=_C["white"], width=36
        )
        self._opacity_label.pack(side="left")

    # ── Yardimci ─────────────────────────────────────────────────────────────

    def _section(self, parent, title: str):
        ctk.CTkLabel(
            parent, text=title,
            font=ctk.CTkFont(size=12, weight="bold"), text_color=_C["blue"]
        ).pack(anchor="w", padx=20, pady=(16, 2))
        ctk.CTkFrame(parent, fg_color=_C["border"], height=1).pack(
            fill="x", padx=20, pady=(0, 4)
        )

    def _api_row(self, parent, service: str, label: str, url: str):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=3)

        ctk.CTkLabel(
            row, text=f"{label}:", width=90,
            font=ctk.CTkFont(size=11), text_color=_C["gray"], anchor="w"
        ).pack(side="left")

        entry = ctk.CTkEntry(row, width=220, height=30, show="•",
                             font=ctk.CTkFont(size=11))
        existing = self.cfg.get("api_keys", service, default="")
        if existing:
            entry.insert(0, existing)
        entry.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            row, text="Anahtar Al →", width=100, height=30,
            fg_color=_C["panel"],
            command=lambda u=url: webbrowser.open(u)
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            row, text="Kaydet", width=66, height=30,
            fg_color=_C["blue"],
            command=lambda s=service, e=entry: self._save_api(s, e)
        ).pack(side="left")

    # ── Olay Isleme ──────────────────────────────────────────────────────────

    def _on_mode_change(self, value: str):
        mode_key = next((m[0] for m in self.MODES if m[1] == value), None)
        if mode_key:
            self.app.switch_mode(mode_key)

    def _on_ptt_toggle(self):
        val = self._ptt_switch.get()
        self.cfg.set("recording", "push_to_talk", val)
        self.cfg.save()

    def _on_opacity_change(self, val: float):
        self.cfg.set("overlay", "opacity", round(val, 2))
        self.cfg.save()
        self._opacity_label.configure(text=f"{val:.0%}")
        if self.app._overlay:
            self.app._overlay.wm_attributes("-alpha", val)

    def _start(self):
        self.app.start_live()
        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")

    def _stop(self):
        self.app.stop_live()
        self._btn_start.configure(state="normal")
        self._btn_stop.configure(state="disabled")

    def _browse_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Ses/Video Dosyası Seç",
            filetypes=[
                ("Medya dosyaları", "*.wav *.mp3 *.mp4 *.mkv *.m4a *.ogg *.flac"),
                ("Tüm dosyalar",    "*.*"),
            ]
        )
        if path:
            self._file_entry.delete(0, "end")
            self._file_entry.insert(0, path)

    def _process_file(self):
        path = self._file_entry.get().strip()
        if path:
            self.app.process_file(path)

    def _save_api(self, service: str, entry: ctk.CTkEntry):
        val = entry.get().strip()
        self.cfg.set_api_key(service, val)

    def _save_voice(self):
        vid = self._voice_entry.get().strip()
        if vid:
            self.cfg.set_voice(vid)


# ══════════════════════════════════════════════════════════════════════════════
# Giris Noktasi
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = GemmaEchoApp()
    app.run()


if __name__ == "__main__":
    main()
