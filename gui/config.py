"""
Konfigürasyon yöneticisi.
config.json dosyasini okur/yazar. Tum GUI bilesenlerinin
tek gercek kaynagi (single source of truth) bu siniftir.
"""

import json
import os

from gui.hardware_scan import scan as hw_scan

# config.json konumu: proje koku (main.py ile ayni dizin)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.json")


def _default_config(hw: dict) -> dict:
    """
    Donanim taramasina gore varsayilan konfigurasyonu olusturur.
    Ilk calistirmada kullanilir.
    """
    p = hw["recommended_profile"]
    return {
        "version": "1.0",
        "first_run": True,

        # ── Donanim (bilgi amacli, degistirilmez) ─────────────────
        "hardware": {
            "os":        hw["os"],
            "ram_gb":    hw["ram_gb"],
            "cpu_cores": hw["cpu_cores"],
            "gpu": {
                "available": hw["gpu"]["available"],
                "type":      hw["gpu"]["type"],
                "name":      hw["gpu"]["name"],
                "vram_gb":   hw["gpu"]["vram_gb"],
            },
        },

        # ── API Anahtarlari ────────────────────────────────────────
        "api_keys": {
            "gemini":      "",   # https://aistudio.google.com
            "groq":        "",   # https://console.groq.com
            "elevenlabs":  "",   # https://elevenlabs.io
        },

        # ── ElevenLabs Ses Secimi ──────────────────────────────────
        "elevenlabs_voice_id":   "",   # API'den secilir
        "elevenlabs_voice_name": "",   # Gosterim icin

        # ── Calisma Modu ───────────────────────────────────────────
        "mode": {
            # Orchestrator modu (VALID_MODES listesinden)
            "current": p["orchestrator_mode"],

            # Granüler bileşen ayarlari (Ayarlar panelinden degistirilebilir)
            "stt": {
                "backend": p["stt_backend"],   # "local_gpu" | "local_cpu" | "cloud_auto"
                "device":  p["stt_device"],    # "cuda" | "mps" | "cpu"
            },
            "llm": {
                "backend": p["llm_backend"],   # "online" | "offline"
                "device":  p["llm_device"],    # "cuda" | "mps" | "cpu"
                "model":   "gemma-4",          # Varsayilan: Gemma 4 (yarisma)
            },
            "tts": {
                "backend": p["tts_backend"],   # "online" | "gpu" | "offline"
                "device":  p["tts_device"],    # "cuda" | "mps" | "cpu"
            },
        },

        # ── Kayit Modu ─────────────────────────────────────────────
        "recording": {
            "push_to_talk":    False,   # True: bas-konus | False: VAD otomatik
            "vad_aggressiveness": 2,    # 0-3 arasi (webrtcvad)
            "silence_ms":      900,     # Cumle bitis sessizlik esigi (ms)
        },

        # ── Overlay Pencere ────────────────────────────────────────
        "overlay": {
            "always_on_top": True,
            "opacity":       0.92,      # 0.0 - 1.0
            "width":         480,
            "height":        160,
            "position_x":    -1,        # -1: ekran sagina yaslansın
            "position_y":    -1,        # -1: ekran altina yaslansın
        },

        # ── Dosya Modu ─────────────────────────────────────────────
        "file_mode": {
            "output_dir":        "",    # Bos: kaynak dosya konumuna yaz
            "save_transcript":   True,  # TR transkript .txt olarak kaydedilsin mi
            "save_translation":  True,  # EN ceviri .txt olarak kaydedilsin mi
        },

        # ── Onerilen Profil (bilgi amacli) ────────────────────────
        "recommended_profile": p,
    }


class ConfigManager:
    """
    config.json icin thread-safe okuma/yazma arayuzu.

    Kullanim:
        cfg = ConfigManager()
        api_key = cfg.get("api_keys", "groq")
        cfg.set("api_keys", "groq", "gsk_xxx")
        cfg.save()
    """

    def __init__(self, path: str = CONFIG_PATH):
        self._path = path
        self._data: dict = {}
        self._load()

    # ── Yukleme ────────────────────────────────────────────────────

    def _load(self):
        """
        config.json varsa yukle, yoksa donanim taramasiyla varsayilani olustur.
        """
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                return
            except (json.JSONDecodeError, OSError):
                # Bozuk dosya — yeniden olustur
                pass

        # Ilk calistirma: donanim tara, varsayilan olustur
        hw = hw_scan()
        self._data = _default_config(hw)
        self.save()

    # ── Okuma ──────────────────────────────────────────────────────

    def get(self, *keys, default=None):
        """
        Ic ice anahtarla deger oku.

        Ornekler:
            cfg.get("api_keys", "groq")          -> "gsk_xxx"
            cfg.get("mode", "stt", "backend")    -> "local_gpu"
            cfg.get("overlay", "opacity")        -> 0.92
        """
        node = self._data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def all(self) -> dict:
        """Tum konfigurasyonu dondurur (salt okunur kullanim icin)."""
        return self._data

    # ── Yazma ──────────────────────────────────────────────────────

    def set(self, *keys_and_value):
        """
        Ic ice anahtarla deger yaz. Son arguman degerdir.

        Ornekler:
            cfg.set("api_keys", "groq", "gsk_yyy")
            cfg.set("overlay", "opacity", 0.85)
            cfg.set("first_run", False)
        """
        if len(keys_and_value) < 2:
            raise ValueError("En az bir anahtar ve bir deger gereklidir.")

        *keys, value = keys_and_value
        node = self._data
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    def save(self):
        """Mevcut konfigurasyonu diske yazar."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ── Ozel Yardimcilar (GUI butonlari icin kisayollar) ──────────

    def mark_first_run_complete(self):
        self.set("first_run", False)
        self.save()

    def set_api_key(self, service: str, key: str):
        """service: 'gemini' | 'groq' | 'elevenlabs'"""
        self.set("api_keys", service, key)
        self.save()

    def set_voice(self, voice_id: str, voice_name: str = ""):
        self.set("elevenlabs_voice_id", voice_id)
        self.set("elevenlabs_voice_name", voice_name)
        self.save()

    def set_mode(self, orchestrator_mode: str):
        self.set("mode", "current", orchestrator_mode)
        self.save()

    def is_first_run(self) -> bool:
        return self.get("first_run", default=True)

    def has_api_key(self, service: str) -> bool:
        return bool(self.get("api_keys", service, default=""))
