"""
Configuration manager.

Reads and writes config.json. The single source of truth that every GUI
component consults for persisted application state.
"""

import json
import os

from gui.hardware_scan import scan as hw_scan

# config.json location: project root (alongside main.py).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.json")


def _default_config(hw: dict) -> dict:
    """
    Produce the default configuration based on the detected hardware profile.

    Used on the very first launch.
    """
    p = hw["recommended_profile"]
    return {
        "version": "1.0",
        "first_run": True,

        # ── Hardware (informational only — never mutated by the user) ─────
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

        # ── API keys ─────────────────────────────────────────────────────
        "api_keys": {
            "gemini":      "",   # https://aistudio.google.com
            "groq":        "",   # https://console.groq.com
            "elevenlabs":  "",   # https://elevenlabs.io
        },

        # ── ElevenLabs voice selection ───────────────────────────────────
        "elevenlabs_voice_id":   "",   # Chosen via the ElevenLabs API.
        "elevenlabs_voice_name": "",   # Display label.

        # ── Runtime mode ─────────────────────────────────────────────────
        "mode": {
            # Orchestrator mode (drawn from Orchestrator.VALID_MODES).
            "current": p["orchestrator_mode"],

            # Granular component settings (editable from the Settings panel).
            "stt": {
                "backend": p["stt_backend"],   # "local_gpu" | "local_cpu" | "cloud_auto"
                "device":  p["stt_device"],    # "cuda" | "mps" | "cpu"
            },
            "llm": {
                "backend": p["llm_backend"],   # "online" | "offline"
                "device":  p["llm_device"],    # "cuda" | "mps" | "cpu"
                "model":   "gemma-4",          # Default: Gemma 4 (competition baseline).
            },
            "tts": {
                "backend": p["tts_backend"],   # "online" | "gpu" | "offline"
                "device":  p["tts_device"],    # "cuda" | "mps" | "cpu"
            },
        },

        # ── Recording subsystem ──────────────────────────────────────────
        "recording": {
            "push_to_talk":         False,  # True: push-to-talk; False: VAD automatic.
            "vad_aggressiveness":   3,      # 0-3 (3 = most aggressive).
            "silence_ms":           900,    # End-of-sentence silence duration (ms).
            "streaming_enabled":    True,   # Punctuation-based early commit.
            "paragraph_silence_ms": 2000,   # End-of-paragraph silence (longer pause).
            "min_phrase_len":       15,     # Minimum frame count.
        },

        # ── Overlay window ───────────────────────────────────────────────
        "overlay": {
            "always_on_top": True,
            "opacity":       0.92,      # 0.0 - 1.0
            "width":         480,
            "height":        160,
            "position_x":    -1,        # -1: dock to the right edge of the screen.
            "position_y":    -1,        # -1: dock to the bottom edge of the screen.
        },

        # ── File mode ────────────────────────────────────────────────────
        "file_mode": {
            "output_dir":        "",    # Empty: write next to the source file.
            "save_transcript":   True,  # Persist source-language transcript as .txt.
            "save_translation":  True,  # Persist target-language translation as .txt.
        },

        # ── Language settings ────────────────────────────────────────────
        "language": {
            "source":      "tr",          # Source language code (ISO 639-1).
            "target":      "en",          # Target language code.
            "source_name": "Turkish",     # Source-language name for the LLM prompt.
            "target_name": "English",     # Target-language name for the LLM prompt.
            "ui_language": "tr",          # GUI display language.
        },

        # ── Translation persona ──────────────────────────────────────────
        # "default" — No persona (default; the user is not forced to pick one).
        # "official" | "streamer" | "casual" | "literary"
        "persona": "default",


        # ── Broadcaster / content-creator settings ───────────────────────
        "broadcaster": {
            "enabled":             False, # Route the dubbed audio to a virtual mic?
            "output_device_index": None,  # Chosen device ID (None = system default).
            "output_device_name":  "Default",
        },

        # ── Recommended profile (informational) ──────────────────────────
        "recommended_profile": p,
    }


class ConfigManager:
    """
    Thread-safe read/write façade over config.json.

    Usage:
        cfg = ConfigManager()
        api_key = cfg.get("api_keys", "groq")
        cfg.set("api_keys", "groq", "gsk_xxx")
        cfg.save()
    """

    def __init__(self, path: str = CONFIG_PATH):
        self._path = path
        self._data: dict = {}
        self._load()

    # ── Loading ──────────────────────────────────────────────────────

    def _load(self):
        """
        Load config.json when present; otherwise scan hardware and create defaults.
        """
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                return
            except (json.JSONDecodeError, OSError):
                # Corrupted file — regenerate from scratch.
                pass

        # First launch: scan hardware and emit the default profile.
        hw = hw_scan()
        self._data = _default_config(hw)
        self.save()

    # ── Reading ──────────────────────────────────────────────────────

    def get(self, *keys, default=None):
        """
        Read a value via a nested key path.

        Examples:
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
        """Return the full configuration dict (intended for read-only access)."""
        return self._data

    # ── Writing ──────────────────────────────────────────────────────

    def set(self, *keys_and_value):
        """
        Set a value at a nested key path. The final positional argument is the value.

        Examples:
            cfg.set("api_keys", "groq", "gsk_yyy")
            cfg.set("overlay", "opacity", 0.85)
            cfg.set("first_run", False)
        """
        if len(keys_and_value) < 2:
            raise ValueError("At least one key and one value are required.")

        *keys, value = keys_and_value
        node = self._data
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    def save(self):
        """Persist the current configuration to disk."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ── Convenience helpers (shortcuts for GUI buttons) ──────────────

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
