"""
Gemma Echo — Live translation view.
"""

import customtkinter as ctk
from tkinter import messagebox
from gui.config import ConfigManager
from gui.i18n import t
from gui.pages._helpers import _C, _header, _card

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

        # ── Recording controls ────────────────────────────────────────
        inner = _card(scroll, t("rec_control"))

        # Status row.
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

        # Buttons.
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

        # Dual PTT buttons.
        ptt_row = ctk.CTkFrame(inner, fg_color="transparent")
        ptt_row.pack(fill="x", pady=(10, 0))

        ctk.CTkLabel(
            ptt_row, text=t("ppt_mode_label"),
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=_C["text"],
        ).pack(side="left", padx=(0, 10))

        self._btn_outbound = ctk.CTkButton(
            ptt_row, text=t("ppt_outbound_hint"),
            font=ctk.CTkFont(size=10),
            fg_color=_C["surface2"], hover_color=_C["blue_bg"],
            text_color=_C["muted"], height=30, corner_radius=8,
            command=lambda: self.app.switch_ptt("outbound"),
        )
        self._btn_outbound.pack(side="left", padx=(0, 6))

        self._btn_inbound = ctk.CTkButton(
            ptt_row, text=t("ppt_inbound_hint"),
            font=ctk.CTkFont(size=10),
            fg_color=_C["surface2"], hover_color="#1a3a1a",
            text_color=_C["muted"], height=30, corner_radius=8,
            command=lambda: self.app.switch_ptt("inbound"),
        )
        self._btn_inbound.pack(side="left")

        # ── Live telemetry (dashboard) ────────────────────────────────
        self._tel_card = _card(scroll, t("telemetry_title"))

        def _tel_row(parent, label_key):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=t(label_key), font=ctk.CTkFont(size=11, weight="bold"), text_color=_C["dim"], width=100, anchor="w").pack(side="left")
            val = ctk.CTkLabel(row, text=t("waiting_data"), font=ctk.CTkFont(size=11), text_color=_C["text"], anchor="w")
            val.pack(side="left", fill="x", expand=True)
            return val

        self._lbl_tel_engine = _tel_row(self._tel_card, "telemetry_engine")
        self._lbl_tel_persona = _tel_row(self._tel_card, "telemetry_persona")
        self._lbl_tel_stt = _tel_row(self._tel_card, "telemetry_stt")
        self._lbl_tel_llm = _tel_row(self._tel_card, "telemetry_llm")
        self._lbl_tel_tts = _tel_row(self._tel_card, "telemetry_tts")
        self._lbl_tel_e2e = _tel_row(self._tel_card, "telemetry_e2e")

        # ── Overlay controls ──────────────────────────────────────────
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

        # ── Tips ──────────────────────────────────────────────────────
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

    # ── Event handlers ────────────────────────────────────────────────────────

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
        self.app.stop_inbound()
        self.app._ptt_mode = None
        self.app._sync_ptt_hotkeys_state()
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
        self._update_ptt_state(None)

    def _update_ptt_state(self, mode):
        """Refresh the PTT button visuals on mode change (invoked from hotkey or click)."""
        if mode == "outbound":
            self._btn_outbound.configure(fg_color=_C["blue_bg"], text_color=_C["blue"])
            self._btn_inbound.configure(fg_color=_C["surface2"], text_color=_C["muted"])
            self._sdot.configure(text_color=_C["blue"])
            self._slbl.configure(text=t("ptt_outbound_active"), text_color=_C["blue"])
        elif mode == "inbound":
            self._btn_outbound.configure(fg_color=_C["surface2"], text_color=_C["muted"])
            self._btn_inbound.configure(fg_color=_C["green_bg"], text_color=_C["green"])
            self._sdot.configure(text_color=_C["green"])
            self._slbl.configure(text=t("ptt_inbound_active"), text_color=_C["green"])
        else:
            self._btn_outbound.configure(fg_color=_C["surface2"], text_color=_C["muted"])
            self._btn_inbound.configure(fg_color=_C["surface2"], text_color=_C["muted"])
            self._sdot.configure(text_color=_C["dim"])
            self._slbl.configure(text=t("live_ready_hint"), text_color=_C["muted"])

    def _show_overlay(self):
        if self.app._overlay:
            self.app._overlay.deiconify()
            self.app._overlay.lift()

    def on_leave(self):
        """Stop the microphone when the user navigates away from this page."""
        if self._recording:
            self._stop()

    def update_telemetry(self, data: dict):
        # On error keep the dashboard fresh — surface "Error" on the engine slot, "—" elsewhere.
        if data.get("error"):
            self._lbl_tel_engine.configure(text=t("telemetry_error_engine") if t("telemetry_error_engine") != "telemetry_error_engine" else "Error", text_color=_C["red"])
            for lbl in (self._lbl_tel_stt, self._lbl_tel_llm, self._lbl_tel_tts, self._lbl_tel_e2e):
                lbl.configure(text="—", text_color=_C["dim"])
            return

        engine = data.get("engine", "—")
        stt_ms = data.get("stt_ms", 0)
        llm_ms = data.get("llm_ms", 0)
        tts_ms = data.get("tts_ms", 0)
        e2e_ms = data.get("latency_ms", 0)

        eng_color = _C["blue"] if "API" in engine or "online" in engine.lower() else _C["green"]
        if "llama" in engine.lower() or "local" in engine.lower() or "offline" in engine.lower():
            eng_color = _C["green"]

        persona = self.cfg.get("persona", default="default")
        persona_str = t(f"persona_{persona}") if persona != "default" and persona != "none" else t("persona_none")

        self._lbl_tel_engine.configure(text=engine, text_color=eng_color)
        self._lbl_tel_persona.configure(text=persona_str)
        self._lbl_tel_stt.configure(text=f"{stt_ms} ms" if stt_ms else "—")
        self._lbl_tel_llm.configure(text=f"{llm_ms} ms" if llm_ms else "—")
        self._lbl_tel_tts.configure(text=f"{tts_ms} ms" if tts_ms else "—")
        self._lbl_tel_e2e.configure(text=f"{e2e_ms/1000:.2f} s" if e2e_ms else "—", text_color=_C["yellow"])
