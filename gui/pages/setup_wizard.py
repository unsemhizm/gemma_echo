"""
Gemma Echo — Kurulum Sihirbazi (Setup Wizard)

Ilk calistirmada gosterilir. 3 adimda kurulumu tamamlar:
  Adim 1 — Donanim: Tarama sonuclari + onerilen profil onayi
  Adim 2 — API      : Groq / Gemini / ElevenLabs anahtar girisi
  Adim 3 — Ses      : ElevenLabs ses ID secimi + kurulum sonu
"""

import webbrowser
import customtkinter as ctk

from gui.config import ConfigManager
from gui.hardware_scan import scan as hw_scan

# ─── Tema ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ─── Sabitler ─────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 660, 520
STEP_COUNT   = 3

_API_LINKS = {
    "Gemini (Google AI Studio)": "https://aistudio.google.com/apikey",
    "Groq":                      "https://console.groq.com/keys",
    "ElevenLabs":                "https://elevenlabs.io/app/settings/api-keys",
}

_VOICE_LINK = "https://elevenlabs.io/app/voice-library"

_COLORS = {
    "green":  "#2ecc71",
    "yellow": "#f1c40f",
    "red":    "#e74c3c",
    "blue":   "#3498db",
    "gray":   "#7f8c8d",
    "bg":     "#1a1a2e",
    "card":   "#16213e",
    "accent": "#0f3460",
}


# ══════════════════════════════════════════════════════════════════════════════
# Ana Sihirbaz Penceresi
# ══════════════════════════════════════════════════════════════════════════════

class SetupWizard(ctk.CTk):
    """
    Bagimsiz, adim adim kurulum penceresi.
    on_complete(cfg) geri cagrisi kurulum bittiginde tetiklenir.
    """

    def __init__(self, cfg: ConfigManager, on_complete=None):
        super().__init__()
        self.cfg         = cfg
        self.on_complete = on_complete
        self._hw         = hw_scan()          # taze tarama
        self._step       = 0
        self._pages      = []                 # her adimin Frame'i

        self.title("Gemma Echo — Kurulum")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(fg_color=_COLORS["bg"])

        # Ekran ortasina yerlestir
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - WIN_W) // 2
        y = (self.winfo_screenheight() - WIN_H) // 2
        self.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

        self._build_shell()
        self._build_page_0()   # Donanim
        self._build_page_1()   # API Anahtarlari
        self._build_page_2()   # Ses + Bitis
        self._show_step(0)

    # ── Iskelet ──────────────────────────────────────────────────────────────

    def _build_shell(self):
        """Ust baslik, adim gostergesi, icerik alani, alt navigasyon."""

        # ── Baslik ──────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=_COLORS["accent"], corner_radius=0, height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr, text="GEMMA ECHO",
            font=ctk.CTkFont(family="Helvetica", size=22, weight="bold"),
            text_color="white"
        ).pack(side="left", padx=24, pady=16)

        ctk.CTkLabel(
            hdr, text="Kurulum Sihirbazi",
            font=ctk.CTkFont(size=13),
            text_color="#aabbcc"
        ).pack(side="left", padx=0, pady=20)

        # ── Adim Gostergesi ──────────────────────────────
        self._step_bar = _StepBar(self, steps=["Donanim", "API Anahtarlari", "Ses & Bitis"])
        self._step_bar.pack(fill="x", padx=0, pady=(0, 0))

        # ── Icerik alani (degisen sayfa buraya) ──────────
        self._content = ctk.CTkFrame(self, fg_color=_COLORS["bg"], corner_radius=0)
        self._content.pack(fill="both", expand=True, padx=0, pady=0)

        # ── Alt navigasyon ───────────────────────────────
        nav = ctk.CTkFrame(self, fg_color=_COLORS["accent"], corner_radius=0, height=60)
        nav.pack(fill="x", side="bottom")
        nav.pack_propagate(False)

        self._btn_back = ctk.CTkButton(
            nav, text="← Geri", width=110, height=36,
            fg_color="transparent", hover_color=_COLORS["accent"],
            border_width=1, border_color="#aabbcc",
            command=self._prev_step
        )
        self._btn_back.pack(side="left", padx=20, pady=12)

        self._btn_next = ctk.CTkButton(
            nav, text="İleri →", width=140, height=36,
            fg_color=_COLORS["blue"],
            command=self._next_step
        )
        self._btn_next.pack(side="right", padx=20, pady=12)

        self._btn_skip = ctk.CTkButton(
            nav, text="Atla", width=90, height=36,
            fg_color="transparent", hover_color=_COLORS["accent"],
            text_color="#aabbcc",
            command=self._skip
        )
        self._btn_skip.pack(side="right", padx=4, pady=12)

    # ── Adim 0: Donanim ──────────────────────────────────────────────────────

    def _build_page_0(self):
        page = ctk.CTkFrame(self._content, fg_color="transparent")
        self._pages.append(page)

        ctk.CTkLabel(
            page, text="Sisteminiz Analiz Edildi",
            font=ctk.CTkFont(size=18, weight="bold"), text_color="white"
        ).pack(anchor="w", padx=32, pady=(28, 4))

        ctk.CTkLabel(
            page, text="Donanim taraması tamamlandı. Aşağıdaki profil sizin için otomatik seçildi.",
            font=ctk.CTkFont(size=12), text_color="#aabbcc", wraplength=580
        ).pack(anchor="w", padx=32, pady=(0, 16))

        # Donanim Kart
        hw_card = ctk.CTkFrame(page, fg_color=_COLORS["card"], corner_radius=12)
        hw_card.pack(fill="x", padx=32, pady=(0, 14))

        hw = self._hw
        gpu = hw["gpu"]

        gpu_color = _COLORS["green"] if gpu["available"] else _COLORS["red"]
        if gpu["available"] and gpu["type"] == "cuda" and gpu["vram_gb"] < 4:
            gpu_color = _COLORS["yellow"]

        gpu_text = (
            f"{gpu['name']}  ({gpu['vram_gb']} GB VRAM)"
            if gpu["type"] == "cuda"
            else gpu["name"] if gpu["available"]
            else "GPU bulunamadı — CPU modu"
        )

        rows = [
            ("İşletim Sistemi", hw["os"].upper(),    "white"),
            ("RAM",             f"{hw['ram_gb']} GB", "white"),
            ("CPU Çekirdek",    str(hw["cpu_cores"]), "white"),
            ("GPU",             gpu_text,             gpu_color),
        ]
        for label, value, color in rows:
            _hw_row(hw_card, label, value, color)

        # Onerilen Profil Kart
        p = hw["recommended_profile"]
        prof_card = ctk.CTkFrame(page, fg_color="#0d2137", corner_radius=12,
                                 border_width=1, border_color=_COLORS["blue"])
        prof_card.pack(fill="x", padx=32, pady=(0, 10))

        ctk.CTkLabel(
            prof_card,
            text=f"  Önerilen Mod: {p['orchestrator_mode'].upper()}",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_COLORS["blue"]
        ).pack(anchor="w", padx=14, pady=(10, 2))

        ctk.CTkLabel(
            prof_card, text=f"  {p['reason']}",
            font=ctk.CTkFont(size=11), text_color="#aabbcc", wraplength=570, justify="left"
        ).pack(anchor="w", padx=14, pady=(0, 10))

    # ── Adim 1: API Anahtarlari ───────────────────────────────────────────────

    def _build_page_1(self):
        page = ctk.CTkFrame(self._content, fg_color="transparent")
        self._pages.append(page)

        ctk.CTkLabel(
            page, text="API Anahtarlarınızı Girin",
            font=ctk.CTkFont(size=18, weight="bold"), text_color="white"
        ).pack(anchor="w", padx=32, pady=(28, 4))

        ctk.CTkLabel(
            page,
            text="Anahtarlar config.json dosyasına kaydedilir. İstediğiniz zaman Ayarlar'dan güncelleyebilirsiniz.\n"
                 "Şimdilik doldurmak zorunda değilsiniz — atlamak için 'Atla' butonunu kullanın.",
            font=ctk.CTkFont(size=11), text_color="#aabbcc", wraplength=580, justify="left"
        ).pack(anchor="w", padx=32, pady=(0, 18))

        # Girdi satirlari
        self._api_entries = {}
        services = [
            ("gemini",     "Gemini (Google AI Studio)", "Ücretsiz kota mevcut"),
            ("groq",       "Groq",                      "Llama / Gemma için ücretsiz API"),
            ("elevenlabs", "ElevenLabs",                "TTS — ücretsiz 10k karakter/ay"),
        ]
        for key, display, hint in services:
            self._api_entries[key] = _api_row(page, display, hint, _API_LINKS[display], self.cfg)

    # ── Adim 2: Ses & Bitis ───────────────────────────────────────────────────

    def _build_page_2(self):
        page = ctk.CTkFrame(self._content, fg_color="transparent")
        self._pages.append(page)

        ctk.CTkLabel(
            page, text="ElevenLabs Ses Seçimi",
            font=ctk.CTkFont(size=18, weight="bold"), text_color="white"
        ).pack(anchor="w", padx=32, pady=(28, 4))

        ctk.CTkLabel(
            page,
            text="ElevenLabs Voice ID girin. Sesi daha sonra Ayarlar > TTS bölümünden de değiştirebilirsiniz.",
            font=ctk.CTkFont(size=11), text_color="#aabbcc", wraplength=580
        ).pack(anchor="w", padx=32, pady=(0, 18))

        # Voice ID satirı
        voice_frame = ctk.CTkFrame(page, fg_color=_COLORS["card"], corner_radius=12)
        voice_frame.pack(fill="x", padx=32, pady=(0, 14))

        ctk.CTkLabel(
            voice_frame, text="Voice ID",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="white"
        ).grid(row=0, column=0, padx=16, pady=(14, 4), sticky="w")

        ctk.CTkLabel(
            voice_frame, text="ElevenLabs ses kütüphanesinden kopyalayın",
            font=ctk.CTkFont(size=10), text_color="#aabbcc"
        ).grid(row=1, column=0, padx=16, pady=(0, 8), sticky="w")

        self._voice_entry = ctk.CTkEntry(
            voice_frame, width=320, height=36,
            placeholder_text="Örn: 21m00Tcm4TlvDq8ikWAM",
            font=ctk.CTkFont(size=12)
        )
        saved_vid = self.cfg.get("elevenlabs_voice_id", default="")
        if saved_vid:
            self._voice_entry.insert(0, saved_vid)
        self._voice_entry.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="ew")

        ctk.CTkButton(
            voice_frame, text="Sesleri Gör →", width=110, height=36,
            fg_color=_COLORS["blue"],
            command=lambda: webbrowser.open(_VOICE_LINK)
        ).grid(row=0, column=2, rowspan=2, padx=(0, 14), pady=10)

        voice_frame.columnconfigure(1, weight=1)

        # Ozet kutu
        summary_card = ctk.CTkFrame(page, fg_color=_COLORS["card"], corner_radius=12)
        summary_card.pack(fill="x", padx=32, pady=(0, 14))

        ctk.CTkLabel(
            summary_card, text="  Kurulum Özeti",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_COLORS["blue"]
        ).pack(anchor="w", padx=14, pady=(12, 6))

        self._summary_label = ctk.CTkLabel(
            summary_card, text="",
            font=ctk.CTkFont(size=11), text_color="#aabbcc",
            justify="left", wraplength=560
        )
        self._summary_label.pack(anchor="w", padx=22, pady=(0, 12))

        # Kapat notu
        ctk.CTkLabel(
            page,
            text="'Kurulumu Tamamla' butonuna tıkladıktan sonra uygulama ana ekrana geçecek.\n"
                 "Tüm ayarlara istediğiniz zaman ⚙ Ayarlar menüsünden erişebilirsiniz.",
            font=ctk.CTkFont(size=10), text_color="#666", wraplength=580, justify="left"
        ).pack(anchor="w", padx=32, pady=(4, 0))

    # ── Navigasyon ────────────────────────────────────────────────────────────

    def _show_step(self, step: int):
        for p in self._pages:
            p.pack_forget()
        self._pages[step].pack(fill="both", expand=True)
        self._step_bar.set_step(step)
        self._step = step

        # Son adima girilince ozeti guncelle
        if step == 2:
            self._refresh_summary()

        # Buton metinleri
        self._btn_back.configure(state="normal" if step > 0 else "disabled")
        if step == STEP_COUNT - 1:
            self._btn_next.configure(text="Kurulumu Tamamla ✓")
            self._btn_skip.configure(state="disabled")
        else:
            self._btn_next.configure(text="İleri →")
            self._btn_skip.configure(state="normal")

    def _next_step(self):
        self._save_current_step()
        if self._step < STEP_COUNT - 1:
            self._show_step(self._step + 1)
        else:
            self._finish()

    def _prev_step(self):
        if self._step > 0:
            self._show_step(self._step - 1)

    def _skip(self):
        """Mevcut adimi kaydetmeden atla."""
        if self._step < STEP_COUNT - 1:
            self._show_step(self._step + 1)

    # ── Kayit / Bitis ─────────────────────────────────────────────────────────

    def _save_current_step(self):
        """Mevcut adimin girislerini config'e kaydet."""
        if self._step == 1:
            # API anahtarlari
            for service, entry in self._api_entries.items():
                val = entry.get().strip()
                if val:
                    self.cfg.set_api_key(service, val)

        elif self._step == 2:
            # Voice ID
            vid = self._voice_entry.get().strip()
            if vid:
                self.cfg.set_voice(vid)

    def _finish(self):
        self._save_current_step()
        self.cfg.mark_first_run_complete()

        if self.on_complete:
            self.on_complete(self.cfg)

        self.destroy()

    def _refresh_summary(self):
        """Son adim ozet etiketini guncelle."""
        lines = []
        p = self._hw["recommended_profile"]
        lines.append(f"Mod          : {p['orchestrator_mode'].upper()}")

        for service in ("gemini", "groq", "elevenlabs"):
            status = "✓ Girildi" if self.cfg.has_api_key(service) else "— Girilmedi"
            lines.append(f"{service.capitalize():<13}: {status}")

        vid = self.cfg.get("elevenlabs_voice_id", default="")
        lines.append(f"Voice ID     : {vid if vid else '— Girilmedi (varsayılan kullanılacak)'}")
        self._summary_label.configure(text="\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# Yardimci Widget'lar
# ══════════════════════════════════════════════════════════════════════════════

class _StepBar(ctk.CTkFrame):
    """Sihirbaz adim gostergesi."""

    def __init__(self, master, steps: list[str]):
        super().__init__(master, fg_color=_COLORS["accent"], corner_radius=0, height=44)
        self.pack_propagate(False)
        self._labels = []
        self._dots   = []

        n = len(steps)
        for i, name in enumerate(steps):
            dot = ctk.CTkLabel(
                self, text="●", font=ctk.CTkFont(size=14),
                text_color=_COLORS["gray"]
            )
            dot.pack(side="left", padx=(24 if i == 0 else 4, 4))
            self._dots.append(dot)

            lbl = ctk.CTkLabel(
                self, text=name, font=ctk.CTkFont(size=11),
                text_color=_COLORS["gray"]
            )
            lbl.pack(side="left", padx=(0, 2))
            self._labels.append(lbl)

            if i < n - 1:
                ctk.CTkLabel(self, text="──", text_color=_COLORS["gray"]).pack(side="left", padx=4)

    def set_step(self, active: int):
        for i, (dot, lbl) in enumerate(zip(self._dots, self._labels)):
            if i < active:
                dot.configure(text_color=_COLORS["green"])
                lbl.configure(text_color=_COLORS["green"])
            elif i == active:
                dot.configure(text_color=_COLORS["blue"])
                lbl.configure(text_color="white")
            else:
                dot.configure(text_color=_COLORS["gray"])
                lbl.configure(text_color=_COLORS["gray"])


def _hw_row(parent, label: str, value: str, color: str):
    """Donanim bilgi satiri."""
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=16, pady=3)
    ctk.CTkLabel(
        row, text=f"{label}:", width=120, anchor="w",
        font=ctk.CTkFont(size=11), text_color="#aabbcc"
    ).pack(side="left")
    ctk.CTkLabel(
        row, text=value, anchor="w",
        font=ctk.CTkFont(size=11, weight="bold"), text_color=color
    ).pack(side="left", padx=6)


def _api_row(parent, display: str, hint: str, url: str, cfg: ConfigManager) -> ctk.CTkEntry:
    """Tek bir API servis satiri; Entry widget'i dondurur."""
    frame = ctk.CTkFrame(parent, fg_color=_COLORS["card"], corner_radius=10)
    frame.pack(fill="x", padx=32, pady=5)

    # Sol: isim + ipucu
    info = ctk.CTkFrame(frame, fg_color="transparent", width=170)
    info.pack(side="left", padx=14, pady=10)
    info.pack_propagate(False)
    ctk.CTkLabel(
        info, text=display,
        font=ctk.CTkFont(size=12, weight="bold"), text_color="white", anchor="w"
    ).pack(anchor="w")
    ctk.CTkLabel(
        info, text=hint,
        font=ctk.CTkFont(size=9), text_color="#7f8c8d", anchor="w"
    ).pack(anchor="w")

    # Orta: Entry
    service_key = display.split(" ")[0].lower()
    existing    = cfg.get("api_keys", service_key, default="")
    entry = ctk.CTkEntry(frame, width=210, height=32, show="•", font=ctk.CTkFont(size=11),
                         placeholder_text="API anahtarı...")
    if existing:
        entry.insert(0, existing)
    entry.pack(side="left", padx=8, pady=10)

    # Sag: Link butonu
    ctk.CTkButton(
        frame, text="Anahtar Al →", width=105, height=32,
        fg_color=_COLORS["blue"], font=ctk.CTkFont(size=11),
        command=lambda u=url: webbrowser.open(u)
    ).pack(side="left", padx=(0, 10), pady=10)

    return entry


# ══════════════════════════════════════════════════════════════════════════════
# Bagimsiz test calistirmasi
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from gui.config import ConfigManager

    cfg = ConfigManager()
    cfg.set("first_run", True)
    cfg.save()

    def on_done(cfg):
        print("Kurulum tamamlandi!")
        print("  first_run    :", cfg.get("first_run"))
        print("  groq key     :", cfg.get("api_keys", "groq"))
        print("  voice_id     :", cfg.get("elevenlabs_voice_id"))
        print("  mode.current :", cfg.get("mode", "current"))

    wizard = SetupWizard(cfg, on_complete=on_done)
    wizard.mainloop()
