"""
Gemma Echo — Settings view.
"""

import webbrowser
import customtkinter as ctk
from gui.config import ConfigManager
from gui.i18n import t
from gui.pages._helpers import _C, _header, _card, _InfoIcon

class SettingsView(ctk.CTkFrame):
    _LANG_OPTIONS = [
        ("tr", "Turkish",  "Türkçe"),
        ("en", "English",  "English"),
        ("de", "German",   "Deutsch"),
        ("fr", "French",   "Français"),
        ("it", "Italian",  "Italiano"),
        ("ar", "Arabic",   "العربية"),
        ("es", "Spanish",  "Español"),
        ("ja", "Japanese", "日本語"),
    ]

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

        # ── Runtime mode ──────────────────────────────────────────────
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

        # ── Custom backend mix ────────────────────────────────────────
        custom_card = _card(body, t("settings"))

        # STT row.
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

        # LLM row.
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

        # TTS row.
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

        # Apply button.
        ctk.CTkButton(
            custom_card, text=t("apply"),
            height=34, corner_radius=10,
            fg_color=_C["blue"], hover_color="#4080d0",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._apply_custom_mode
        ).pack(fill="x")

        # ── Broadcaster / content-creator mode ─────────────────────────
        broad_card = _card(body, t("nav_media"))

        broad_hdr = ctk.CTkFrame(broad_card, fg_color="transparent")
        broad_hdr.pack(fill="x", pady=(0, 6))

        ctk.CTkLabel(
            broad_hdr, text=f"{t('broadcaster_mode')}:",
            font=ctk.CTkFont(size=11), text_color=_C["muted"]
        ).pack(side="left")
        _InfoIcon(broad_hdr, t("tip_broadcaster")).pack(side="left", padx=(4, 0))

        # Toggle + dropdown row.
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

        # Device list.
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

        # ── Hardware information ──────────────────────────────────────
        hw_card = _card(body, t("hardware_profile"))
        hw      = self.cfg.get("hardware") or {}
        gpu_n   = hw.get("gpu", {}).get("name", "CPU")
        ram_gb  = hw.get("ram_gb", "?")
        cpu_c   = hw.get("cpu_cores", "?")
        ctk.CTkLabel(
            hw_card,
            text=t("hardware_info_format", gpu_n, ram_gb, cpu_c),
            font=ctk.CTkFont(size=11), text_color=_C["muted"], anchor="w"
        ).pack(anchor="w")

        # ── API keys ──────────────────────────────────────────────────
        api_card = _card(body, t("api_keys"))
        for svc, lbl, url in [
            ("gemini",     "Gemini",     "https://aistudio.google.com/apikey"),
            ("groq",       "Groq",       "https://console.groq.com/keys"),
            ("elevenlabs", "ElevenLabs", "https://elevenlabs.io/app/settings/api-keys"),
        ]:
            self._api_row(api_card, svc, lbl, url)

        # ── ElevenLabs voice ──────────────────────────────────────────
        voice_card = _card(body, t("elevenlabs_voice_id_label"))
        vr = ctk.CTkFrame(voice_card, fg_color="transparent")
        vr.pack(fill="x")

        self._voice_entry = ctk.CTkEntry(
            vr, height=34, corner_radius=10, font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            placeholder_text=t("voice_id_placeholder")
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

        # ── Overlay opacity ───────────────────────────────────────────
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

        # VAD aggressiveness.
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

        # ── Translation persona ───────────────────────────────────────────────
        persona_card = _card(body, t("persona_title"))

        persona_hdr = ctk.CTkFrame(persona_card, fg_color="transparent")
        persona_hdr.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(
            persona_hdr, text=f"{t('persona_style')}:",
            font=ctk.CTkFont(size=11), text_color=_C["muted"]
        ).pack(side="left")
        _InfoIcon(persona_hdr, t("tip_persona")).pack(side="left", padx=(4, 0))

        _PERSONA_OPTIONS = [
            ("default",  "persona_none"),
            ("official", "persona_official"),
            ("streamer", "persona_streamer"),
            ("casual",   "persona_casual"),
            ("literary", "persona_literary"),
        ]
        persona_vals = [t(p[1]) for p in _PERSONA_OPTIONS]
        cur_persona  = self.cfg.get("persona", default="default")
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

        # ── Translation language ──────────────────────────────────────
        lang_card = _card(body, t("translation_language") if "translation_language" in dir() else "Translation language")

        lang_names = [f"{o[2]}  ({o[1]})" for o in self._LANG_OPTIONS]

        src_row = ctk.CTkFrame(lang_card, fg_color="transparent")
        src_row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(
            src_row, text=t("source_language_label"), width=100,
            font=ctk.CTkFont(size=11), text_color=_C["muted"], anchor="w"
        ).pack(side="left")
        cur_src = self.cfg.get("language", "source", default="tr")
        cur_src_idx = next((i for i, o in enumerate(self._LANG_OPTIONS) if o[0] == cur_src), 0)
        self._src_lang_combo = ctk.CTkComboBox(
            src_row, values=lang_names,
            height=34, corner_radius=10, font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            command=lambda v: self._on_lang_change("source", v)
        )
        self._src_lang_combo.set(lang_names[cur_src_idx])
        self._src_lang_combo.pack(side="left", fill="x", expand=True)

        tgt_row = ctk.CTkFrame(lang_card, fg_color="transparent")
        tgt_row.pack(fill="x")
        ctk.CTkLabel(
            tgt_row, text=t("target_language_label"), width=100,
            font=ctk.CTkFont(size=11), text_color=_C["muted"], anchor="w"
        ).pack(side="left")
        cur_tgt = self.cfg.get("language", "target", default="en")
        cur_tgt_idx = next((i for i, o in enumerate(self._LANG_OPTIONS) if o[0] == cur_tgt), 1)
        self._tgt_lang_combo = ctk.CTkComboBox(
            tgt_row, values=lang_names,
            height=34, corner_radius=10, font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            command=lambda v: self._on_lang_change("target", v)
        )
        self._tgt_lang_combo.set(lang_names[cur_tgt_idx])
        self._tgt_lang_combo.pack(side="left", fill="x", expand=True)

        # ── Audio devices ─────────────────────────────────────────────
        dev_card = _card(body, "Audio Devices")

        # Microphone row.
        mic_hdr = ctk.CTkFrame(dev_card, fg_color="transparent")
        mic_hdr.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            mic_hdr, text=t("microphone_label"),
            font=ctk.CTkFont(size=11), text_color=_C["muted"], anchor="w"
        ).pack(side="left")

        mic_devices   = self._get_input_devices()
        mic_names     = [d[1] for d in mic_devices]
        cur_mic_name  = self.cfg.get("recording", "mic_device_name", default=t("default_device"))
        self._mic_combo = ctk.CTkComboBox(
            dev_card, values=mic_names,
            height=34, corner_radius=10, font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            command=self._on_mic_select
        )
        self._mic_combo.set(cur_mic_name if cur_mic_name in mic_names else mic_names[0])
        self._mic_combo.pack(fill="x", pady=(0, 10))

        # Loopback (counterpart) row.
        loop_hdr = ctk.CTkFrame(dev_card, fg_color="transparent")
        loop_hdr.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            loop_hdr, text=t("loopback_device_label"),
            font=ctk.CTkFont(size=11), text_color=_C["muted"], anchor="w"
        ).pack(side="left")

        loopback_devices  = self._get_loopback_devices()
        loopback_names    = [d[1] for d in loopback_devices]
        cur_loop_name     = self.cfg.get("inbound", "loopback_device_name", default="")
        self._loop_combo  = ctk.CTkComboBox(
            dev_card, values=loopback_names if loopback_names else [t("loopback_not_found")],
            height=34, corner_radius=10, font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            command=self._on_loopback_select
        )
        matched = next((n for n in loopback_names if cur_loop_name and cur_loop_name.lower() in n.lower()), None)
        self._loop_combo.set(matched or (loopback_names[0] if loopback_names else t("loopback_not_found")))
        self._loop_combo.pack(fill="x", pady=(0, 4))

        ctk.CTkLabel(
            dev_card,
            text=t("loopback_hint_text"),
            font=ctk.CTkFont(size=9), text_color=_C["dim"]
        ).pack(anchor="w")

        # Hotkey assignment rows.
        sep = ctk.CTkFrame(dev_card, fg_color=_C["border"], height=1)
        sep.pack(fill="x", pady=(12, 10))

        ctk.CTkLabel(
            dev_card, text=t("keyboard_shortcuts_label"),
            font=ctk.CTkFont(size=11, weight="bold"), text_color=_C["text"], anchor="w"
        ).pack(anchor="w", pady=(0, 6))

        # Outbound hotkey.
        out_row = ctk.CTkFrame(dev_card, fg_color="transparent")
        out_row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(
            out_row, text=t("outbound_shortcut_label"), width=160,
            font=ctk.CTkFont(size=11), text_color=_C["muted"], anchor="w"
        ).pack(side="left")
        self._hotkey_out_entry = ctk.CTkEntry(
            out_row, height=32, corner_radius=8, width=140,
            font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            placeholder_text=t("shortcut_example_outbound")
        )
        self._hotkey_out_entry.insert(
            0, self.cfg.get("inbound", "hotkey_outbound", default="space")
        )
        self._hotkey_out_entry.pack(side="left")

        # Inbound hotkey.
        in_row = ctk.CTkFrame(dev_card, fg_color="transparent")
        in_row.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(
            in_row, text=t("inbound_shortcut_label"), width=160,
            font=ctk.CTkFont(size=11), text_color=_C["muted"], anchor="w"
        ).pack(side="left")
        self._hotkey_in_entry = ctk.CTkEntry(
            in_row, height=32, corner_radius=8, width=140,
            font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            placeholder_text=t("shortcut_example_inbound")
        )
        self._hotkey_in_entry.insert(
            0, self.cfg.get("inbound", "hotkey_inbound", default="alt")
        )
        self._hotkey_in_entry.pack(side="left")

        ctk.CTkButton(
            dev_card, text=t("save_shortcuts_button"),
            height=32, corner_radius=8,
            fg_color=_C["blue"], hover_color="#4080d0",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._save_hotkeys
        ).pack(fill="x")

        # ── UI language switcher ──────────────────────────────────────
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

    # ── Helper: a single API key row ──────────────────────────────────────────

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
            row, text=t("copy_button_text"), width=60, height=32, corner_radius=8,
            fg_color=_C["surface2"],
            command=lambda u=url: webbrowser.open(u)
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            row, text=t("save_settings"), width=72, height=32, corner_radius=8,
            fg_color=_C["blue"],
            command=lambda s=svc, e=entry: self.cfg.set_api_key(s, e.get().strip())
        ).pack(side="left")

    # ── Events ────────────────────────────────────────────────────────────────

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

        # Update the combo to "custom".
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
        """List the system's audio output devices."""
        import sounddevice as sd
        try:
            devices = sd.query_devices()
            outputs = [(None, t("default_device"))]
            for i, d in enumerate(devices):
                if d['max_output_channels'] > 0:
                    outputs.append((i, d['name']))
            return outputs
        except Exception as e:
            print(f"[ERROR] Could not enumerate audio output devices: {e}")
            return [(None, t("default_device"))]

    def _get_input_devices(self):
        """List the system's microphone (input) devices."""
        import sounddevice as sd
        try:
            devices = sd.query_devices()
            inputs = [(None, t("default_device"))]
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    try:
                        ha = sd.query_hostapis(d['hostapi'])
                        # Surface only WASAPI and MME devices (hide WDM-KS — it confuses users).
                        if ha['name'] in ('Windows WASAPI', 'MME'):
                            inputs.append((i, d['name']))
                    except Exception:
                        pass
            return inputs
        except Exception as e:
            print(f"[ERROR] Could not enumerate microphones: {e}")
            return [(None, t("default_device"))]

    def _get_loopback_devices(self):
        """List WASAPI loopback devices (uses soundcard)."""
        try:
            import soundcard as sc
            loopbacks = [m for m in sc.all_microphones(include_loopback=True) if m.isloopback]
            return [(m.id, m.name) for m in loopbacks]
        except Exception as e:
            print(f"[ERROR] Could not enumerate loopback devices: {e}")
            return []

    def _on_broadcaster_toggle(self):
        enabled = self._broad_switch.get()
        self.cfg.set("broadcaster", "enabled", bool(enabled))
        self.cfg.save()

        # Update the synthesizer (skip if the backend is not ready yet).
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

    def _on_mic_select(self, name: str):
        """Persist the selected microphone to config."""
        devices = self._get_input_devices()
        idx = next((d[0] for d in devices if d[1] == name), None)
        self.cfg.set("recording", "mic_device_index", idx)
        self.cfg.set("recording", "mic_device_name",  name)
        self.cfg.save()

    def _on_loopback_select(self, name: str):
        """Persist the selected loopback device to config."""
        self.cfg.set("inbound", "loopback_device_name", name)
        self.cfg.save()

    def _save_hotkeys(self):
        """Persist the configured hotkeys and re-register them immediately."""
        key_out = self._hotkey_out_entry.get().strip().lower()
        key_in  = self._hotkey_in_entry.get().strip().lower()

        if not key_out or not key_in:
            return
        if key_out == key_in:
            self._hotkey_out_entry.configure(border_color=_C["red"])
            self._hotkey_in_entry.configure(border_color=_C["red"])
            return

        self._hotkey_out_entry.configure(border_color=_C["border"])
        self._hotkey_in_entry.configure(border_color=_C["border"])

        self.cfg.set("inbound", "hotkey_outbound", key_out)
        self.cfg.set("inbound", "hotkey_inbound",  key_in)
        self.cfg.save()

        # Re-register the hotkeys immediately if the backend is ready.
        if self.app._backend_ready:
            self.app._reregister_hotkeys(key_out, key_in)

    def _on_persona(self, display: str, options: list):
        key = next((p[0] for p in options if t(p[1]) == display), "default")
        self.cfg.set("persona", key)
        self.cfg.save()
        # Live update: when the orchestra is ready, push the change through to the translator.
        if self.app._orchestrator:
            self.app._orchestrator.translator.set_persona(key)

    def _on_lang_change(self, direction: str, display: str):
        """Change the source or target translation language."""
        opt = next((o for o in self._LANG_OPTIONS if f"{o[2]}  ({o[1]})" == display), None)
        if not opt:
            return
        code, name = opt[0], opt[1]
        if direction == "source":
            self.cfg.set("language", "source", code)
            self.cfg.set("language", "source_name", name)
        else:
            self.cfg.set("language", "target", code)
            self.cfg.set("language", "target_name", name)
        self.cfg.save()

    def _on_ui_lang(self, lang: str):
        from gui.i18n import set_language
        set_language(lang)

        self.cfg.set("language", "ui_language", lang)
        self.cfg.save()

        # Rebuild MainWindow dynamically.
        main_win = self.app._main
        if main_win:
            for child in main_win.winfo_children():
                child.destroy()
            main_win.title(t("app_name"))
            main_win._build()
            main_win.switch_view("settings")
