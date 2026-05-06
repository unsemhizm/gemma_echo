"""
Gemma Echo — Metin Çeviri Ekranı (Text View)
"""

import os
import threading
import customtkinter as ctk
from tkinter import messagebox
from gui.config import ConfigManager
from gui.i18n import t
from gui.pages._helpers import _C, _header, _card

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
