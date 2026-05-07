"""
Gemma Echo — Medya Çeviri ve Dublaj Ekranı (Media View)
"""

import os
import time
import subprocess
import tempfile
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from gui.config import ConfigManager
from gui.i18n import t
from gui.pages._helpers import _C, _header, _card

_AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
_VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts"}
_ALL_EXT   = _AUDIO_EXT | _VIDEO_EXT
_LLM_CHUNK = 400

class MediaView(ctk.CTkFrame):
    def __init__(self, master, cfg: ConfigManager, app):
        super().__init__(master, fg_color=_C["bg"], corner_radius=0)
        self.cfg = cfg
        self.app = app
        self._processing  = False
        self._dubbing     = False
        self._ffmpeg_proc = None
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

        # ── Metin panelleri (Dinamik diller) ──────────────────────────
        mid = ctk.CTkFrame(self, fg_color="transparent")
        mid.pack(fill="both", expand=True, padx=24, pady=(0, 8))
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mid.rowconfigure(1, weight=1)

        self._src_lbl = ctk.CTkLabel(
            mid, text="",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_C["muted"]
        )
        self._src_lbl.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=(0, 4))

        self._tgt_lbl = ctk.CTkLabel(
            mid, text="",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_C["blue"]
        )
        self._tgt_lbl.grid(row=0, column=1, sticky="w", padx=(6, 0), pady=(0, 4))

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

        self._btn_save_src = ctk.CTkButton(
            bot, text="", height=28, width=120,
            fg_color=_C["surface2"], hover_color=_C["border"],
            corner_radius=8, font=ctk.CTkFont(size=10),
            command=lambda: self._save("source")
        )
        self._btn_save_src.pack(side="left", padx=(12, 4), pady=8)

        self._btn_save_tgt = ctk.CTkButton(
            bot, text="", height=28, width=120,
            fg_color=_C["surface2"], hover_color=_C["border"],
            corner_radius=8, font=ctk.CTkFont(size=10),
            command=lambda: self._save("target")
        )
        self._btn_save_tgt.pack(side="left", padx=(12, 4), pady=8)

        self._btn_save_both = ctk.CTkButton(
            bot, text="", height=28, width=120,
            fg_color=_C["surface2"], hover_color=_C["border"],
            corner_radius=8, font=ctk.CTkFont(size=10),
            command=lambda: (self._save("source"), self._save("target"))
        )
        self._btn_save_both.pack(side="left", padx=(12, 4), pady=8)

        self._elapsed = ctk.CTkLabel(
            bot, text="", font=ctk.CTkFont(size=10), text_color=_C["dim"]
        )
        self._elapsed.pack(side="right", padx=16)

        self._update_language_labels()

    # ── Dosya Secimi ──────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title=t("browse"),
            filetypes=[
                (t("media_filetypes_all"), "*.wav *.mp3 *.ogg *.flac *.m4a *.aac "
                                           "*.mp4 *.mkv *.avi *.mov *.webm"),
                (t("media_filetypes_audio"), "*.wav *.mp3 *.ogg *.flac *.m4a *.aac"),
                (t("media_filetypes_video"), "*.mp4 *.mkv *.avi *.mov *.webm *.ts"),
                (t("media_filetypes_any"), "*.*"),
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
        if self._ffmpeg_proc is not None:
            self._ffmpeg_proc.terminate()
            self._ffmpeg_proc = None
        self._set_progress(0, t("cancelled"), _C["yellow"])
        self._btn_process.configure(state="normal")
        self._btn_cancel.configure(state="disabled")

    # ── Dublaj ────────────────────────────────────────────────────────────────

    def _start_dubbing(self):
        if self._dubbing or self._processing:
            return
        path = self._file_entry.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning(t("media_no_file"), t("media_select_av_first"))
            return
        if os.path.splitext(path)[1].lower() not in _VIDEO_EXT | _AUDIO_EXT:
            messagebox.showwarning(
                t("media_unsupported_short"), t("media_supported_formats_hint")
            )
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
                text=t("output_path", os.path.basename(output_path)),
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
        
        # Get language settings from config
        src_lang = self.cfg.get("language", "source", default="tr")
        tgt_lang = self.cfg.get("language", "target", default="en")
        src_name = self.cfg.get("language", "source_name", default="Turkish")
        tgt_name = self.cfg.get("language", "target_name", default="English")
        
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
            res    = self.app._orchestrator.transcriber.transcribe(wav, source_lang=src_lang)
            txt_tr = res.get("text", "").strip()

            if not txt_tr:
                self._set_progress(1.0, t("speech_not_recognized"), _C["red"])
                return

            self._set_text(self._tr_box, txt_tr)
            self._set_progress(0.55, t("translating"), _C["blue"])

            if not self._processing:
                return

            txt_en = self._translate_chunked(
                self.app._orchestrator.translator, txt_tr,
                src_lang=src_lang, tgt_lang=tgt_lang,
                src_name=src_name, tgt_name=tgt_name
            )
            if not self._processing:
                return

            self._set_text(self._en_box, txt_en, _C["text"])
            elapsed = time.time() - t0
            self._set_progress(1.0, t("done_tick"), _C["green"])
            self.after(0, lambda: self._elapsed.configure(
                text=t("elapsed_seconds", f"{elapsed:.1f}"), text_color=_C["dim"]
            ))

        except Exception as e:
            self._set_progress(0, f"{t('error')}: {e}", _C["red"])
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
            self._ffmpeg_proc = subprocess.Popen(
                ["ffmpeg", "-i", src,
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                 "-y", out],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self._ffmpeg_proc.wait()
            returncode = self._ffmpeg_proc.returncode
            self._ffmpeg_proc = None
            if returncode == 0:
                return out, True
            self._set_progress(0, t("ffmpeg_error"), _C["red"])
        except FileNotFoundError:
            self._set_progress(0, t("ffmpeg_not_found"), _C["red"])
        finally:
            self._ffmpeg_proc = None
        return None, False

    def _translate_chunked(self, translator, text: str, src_lang="tr", tgt_lang="en", src_name="Turkish", tgt_name="English") -> str:
        words = text.split()
        if len(words) <= _LLM_CHUNK:
            return translator.translate(text, src_lang=src_lang, tgt_lang=tgt_lang, src_name=src_name, tgt_name=tgt_name).get("translation", text)

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
            parts.append(translator.translate(chunk, src_lang=src_lang, tgt_lang=tgt_lang, src_name=src_name, tgt_name=tgt_name).get("translation", chunk))
        return " ".join(parts)

    def _save(self, kind: str):
        box  = self._tr_box if kind == "source" else self._en_box
        text = box.get("0.0", "end").strip()
        if not text:
            messagebox.showinfo(t("save_empty_title"), t("save_empty_message"))
            return
        src  = self._file_entry.get().strip()
        base = os.path.splitext(os.path.basename(src))[0] if src else "output"
        
        lang_code = self.cfg.get("language", "source" if kind == "source" else "target", default="tr" if kind == "source" else "en")
        save_title = (
            t("save_as_transcript") if kind == "source" else t("save_as_translation")
        )
        path = filedialog.asksaveasfilename(
            title=save_title,
            initialfile=f"{base}_{lang_code}.txt",
            initialdir=self.cfg.get("file_mode", "output_dir",
                                    default=os.path.expanduser("~")),
            defaultextension=".txt",
            filetypes=[
                (t("filetype_text"), "*.txt"),
                (t("filetype_all"), "*.*"),
            ],
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.cfg.set("file_mode", "output_dir", os.path.dirname(path))
            self.cfg.save()

    def _update_language_labels(self):
        from gui.i18n import get_language
        src_name = self.cfg.get("language", "source_name", default="Turkish")
        tgt_name = self.cfg.get("language", "target_name", default="English")
        src_code = self.cfg.get("language", "source", default="tr").upper()
        tgt_code = self.cfg.get("language", "target", default="en").upper()
        
        lang_map_tr = {
            "Turkish": "Türkçe",
            "English": "İngilizce",
            "German": "Almanca",
            "French": "Fransızca",
            "Italian": "İtalyanca",
            "Spanish": "İspanyolca",
            "Arabic": "Arapça",
            "Japanese": "Japonca"
        }
        
        ui_lang = get_language()
        src_disp = lang_map_tr.get(src_name, src_name) if ui_lang == "tr" else src_name
        tgt_disp = lang_map_tr.get(tgt_name, tgt_name) if ui_lang == "tr" else tgt_name
        
        src_label_text = f"{src_disp} Transkript (STT)" if ui_lang == "tr" else f"{src_disp} Transcript (STT)"
        tgt_label_text = f"{tgt_disp} Çeviri" if ui_lang == "tr" else f"{tgt_disp} Translation"
        
        self._src_lbl.configure(text=src_label_text)
        self._tgt_lbl.configure(text=tgt_label_text)
        
        btn_src_text = f"{src_code} Kaydet" if ui_lang == "tr" else f"Save {src_code}"
        btn_tgt_text = f"{tgt_code} Kaydet" if ui_lang == "tr" else f"Save {tgt_code}"
        
        self._btn_save_src.configure(text=btn_src_text)
        self._btn_save_tgt.configure(text=btn_tgt_text)
        self._btn_save_both.configure(text=t("save_both"))

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
