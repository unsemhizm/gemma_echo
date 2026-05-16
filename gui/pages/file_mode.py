"""
Gemma Echo — File & media translation mode.

Pushes audio (.wav, .mp3, .ogg, .flac, .m4a) and video (.mp4, .mkv, .avi, .mov)
files through the STT → LLM pipeline. Displays the source transcript and the
target-language translation side by side and persists them to .txt on request.

For videos, ``ffmpeg`` is used to isolate the audio track. Long transcripts
are split into paragraphs so they never overrun the LLM context window.

Integration (app.py / ControlPanel):
    from gui.pages.file_mode import FileModeWindow
    win = FileModeWindow(cfg, app=app)
    win.show()
"""

import os
import sys
import time
import queue
import threading
import subprocess
import tempfile
from tkinter import filedialog, messagebox

import customtkinter as ctk

from gui.config import ConfigManager
from gui.i18n import t

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

_AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
_VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts"}
_ALL_EXT   = _AUDIO_EXT | _VIDEO_EXT

# Safety bound for the LLM context window (words).
_LLM_CHUNK_WORDS = 400


# ══════════════════════════════════════════════════════════════════════════════
# Main window
# ══════════════════════════════════════════════════════════════════════════════

class FileModeWindow(ctk.CTkToplevel):
    """
    Standalone file-translation window that can stay open indefinitely.

    Managed from the ControlPanel via show() / hide().
    """

    def __init__(self, cfg: ConfigManager, app=None):
        super().__init__()
        self.cfg  = cfg
        self.app  = app       # GemmaEchoApp reference (for backend access).

        self._processing  = False
        self._tmp_wav     = None    # Temporary WAV extracted from a video.

        self.title(f"{t('app_name')} — {t('file_mode_title')}")
        self.geometry("820x620")
        self.resizable(True, True)
        self.minsize(700, 500)
        self.configure(fg_color=_C["bg"])
        self.protocol("WM_DELETE_WINDOW", self.hide)

        self._center()
        self._build()
        self.withdraw()   # Start hidden.

    def _center(self):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"820x620+{(sw-820)//2}+{(sh-620)//2}")

    def show(self, filter_type: str = "all"):
        """
        filter_type: 'all' | 'video' | 'audio'

        Open the file picker with the matching filter preselected.
        """
        self._filter_type = filter_type
        self.deiconify()
        self.lift()
        self.focus()

    def hide(self):
        self.withdraw()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        # Title bar.
        hdr = ctk.CTkFrame(self, fg_color=_C["accent"], corner_radius=0, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(
            hdr, text=t("file_mode_title"),
            font=ctk.CTkFont(size=15, weight="bold"), text_color="white"
        ).pack(side="left", padx=20, pady=12)
        ctk.CTkLabel(
            hdr, text=t("file_mode_subtitle"),
            font=ctk.CTkFont(size=10), text_color="#aabbcc"
        ).pack(side="left", padx=4, pady=16)

        # File selection bar.
        self._build_file_bar()

        # Progress bar.
        self._build_progress()

        # Source / target text panels.
        self._build_text_panels()

        # Bottom button bar.
        self._build_bottom_bar()

    def _build_file_bar(self):
        bar = ctk.CTkFrame(self, fg_color=_C["panel"], corner_radius=0, height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        self._file_entry = ctk.CTkEntry(
            bar, width=440, height=32,
            placeholder_text=t("select_file"),
            font=ctk.CTkFont(size=11)
        )
        self._file_entry.pack(side="left", padx=(16, 8), pady=10)

        ctk.CTkButton(
            bar, text=t("browse"), width=80, height=32,
            fg_color=_C["card"], hover_color=_C["border"],
            command=self._browse
        ).pack(side="left", padx=(0, 8))

        self._btn_process = ctk.CTkButton(
            bar, text=t("translate"), width=100, height=32,
            fg_color=_C["blue"],
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._start_processing
        )
        self._btn_process.pack(side="left", padx=(0, 8))

        self._btn_cancel = ctk.CTkButton(
            bar, text=f"■ {t('cancel')}", width=80, height=32,
            fg_color=_C["red"], hover_color="#7f1d1d",
            state="disabled",
            command=self._cancel
        )
        self._btn_cancel.pack(side="left")

    def _build_progress(self):
        prog_frame = ctk.CTkFrame(self, fg_color=_C["bg"], height=36)
        prog_frame.pack(fill="x", padx=16, pady=(8, 0))
        prog_frame.pack_propagate(False)

        self._progress = ctk.CTkProgressBar(prog_frame, height=8, mode="determinate")
        self._progress.set(0)
        self._progress.pack(side="left", fill="x", expand=True, pady=14)

        self._prog_label = ctk.CTkLabel(
            prog_frame, text=t("ready"),
            font=ctk.CTkFont(size=10), text_color=_C["gray"], width=160, anchor="e"
        )
        self._prog_label.pack(side="left", padx=(10, 0))

    def _build_text_panels(self):
        mid = ctk.CTkFrame(self, fg_color=_C["bg"])
        mid.pack(fill="both", expand=True, padx=16, pady=8)
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mid.rowconfigure(1, weight=1)

        # Source transcript header.
        ctk.CTkLabel(
            mid, text=t("tr_transcript"),
            font=ctk.CTkFont(size=11, weight="bold"), text_color=_C["gray"]
        ).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 4))

        # Target translation header.
        ctk.CTkLabel(
            mid, text=t("en_translation"),
            font=ctk.CTkFont(size=11, weight="bold"), text_color=_C["blue"]
        ).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 4))

        # Source text box.
        self._tr_box = ctk.CTkTextbox(
            mid, font=ctk.CTkFont(size=12), text_color=_C["gray"],
            fg_color=_C["card"], border_color=_C["border"], border_width=1,
            wrap="word", state="disabled"
        )
        self._tr_box.grid(row=1, column=0, sticky="nsew", padx=(0, 6))

        # Target text box.
        self._en_box = ctk.CTkTextbox(
            mid, font=ctk.CTkFont(size=12), text_color=_C["white"],
            fg_color=_C["card"], border_color=_C["border"], border_width=1,
            wrap="word", state="disabled"
        )
        self._en_box.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

    def _build_bottom_bar(self):
        bot = ctk.CTkFrame(self, fg_color=_C["panel"], corner_radius=0, height=44)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)

        ctk.CTkButton(
            bot, text=t("save_tr"), width=110, height=30,
            fg_color=_C["card"], hover_color=_C["border"],
            command=lambda: self._save_text("tr")
        ).pack(side="left", padx=(16, 6), pady=7)

        ctk.CTkButton(
            bot, text=t("save_en"), width=110, height=30,
            fg_color=_C["card"], hover_color=_C["border"],
            command=lambda: self._save_text("en")
        ).pack(side="left", padx=(0, 6), pady=7)

        ctk.CTkButton(
            bot, text=t("save_both"), width=140, height=30,
            fg_color=_C["blue"],
            command=lambda: (self._save_text("tr"), self._save_text("en"))
        ).pack(side="left", pady=7)

        self._elapsed_label = ctk.CTkLabel(
            bot, text="",
            font=ctk.CTkFont(size=10), text_color=_C["dim"]
        )
        self._elapsed_label.pack(side="right", padx=16)

    # ── File selection ────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title=t("select_file"),
            filetypes=[
                (t("media_filetypes_all"),   "*.wav *.mp3 *.ogg *.flac *.m4a *.aac *.mp4 *.mkv *.avi *.mov *.webm"),
                (t("media_filetypes_audio"), "*.wav *.mp3 *.ogg *.flac *.m4a *.aac"),
                (t("media_filetypes_video"), "*.mp4 *.mkv *.avi *.mov *.webm *.ts"),
                (t("media_filetypes_any"),  "*.*"),
            ]
        )
        if path:
            self._file_entry.delete(0, "end")
            self._file_entry.insert(0, path)
            self._clear_results()

    def _clear_results(self):
        for box in (self._tr_box, self._en_box):
            box.configure(state="normal")
            box.delete("0.0", "end")
            box.configure(state="disabled")
        self._progress.set(0)
        self._prog_label.configure(text=t("ready"), text_color=_C["gray"])
        self._elapsed_label.configure(text="")

    # ── Start processing ──────────────────────────────────────────────────────

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

        if not self._backend_available():
            messagebox.showwarning(
                t("backend_not_ready_models"),
                t("models_still_loading")
            )
            return

        self._processing = True
        self._btn_process.configure(state="disabled")
        self._btn_cancel.configure(state="normal")
        self._clear_results()

        threading.Thread(
            target=self._pipeline, args=(path,), daemon=True
        ).start()

    def _cancel(self):
        self._processing = False
        self._set_progress(0, t("cancelled"), _C["yellow"])
        self._btn_process.configure(state="normal")
        self._btn_cancel.configure(state="disabled")

    def _backend_available(self) -> bool:
        if self.app is None:
            return False
        return getattr(self.app, "_backend_ready", False)

    # ── Pipeline (background thread) ──────────────────────────────────────────

    def _pipeline(self, src_path: str):
        """
        1. Video → extract a WAV.
        2. STT (Whisper) → full source transcript.
        3. LLM (Gemma / Flash) → target translation (chunked).
        4. Update the UI.
        """
        t0 = time.time()
        wav_path  = None
        owns_temp = False

        try:
            ext = os.path.splitext(src_path)[1].lower()

            # ── 1. Video → WAV ────────────────────────────────────────
            if ext in _VIDEO_EXT:
                self._set_progress(0.05, t("extracting_audio"), _C["yellow"])
                wav_path, owns_temp = self._extract_audio(src_path)
                if wav_path is None or not self._processing:
                    return
            elif ext != ".wav":
                # MP3 / OGG / FLAC → WAV transcoding.
                self._set_progress(0.05, t("converting_audio"), _C["yellow"])
                wav_path, owns_temp = self._convert_to_wav(src_path)
                if wav_path is None or not self._processing:
                    return
            else:
                wav_path = src_path

            # ── 2. STT (Whisper) ──────────────────────────────────────
            if not self._processing:
                return
            self._set_progress(0.20, t("generating_transcript"), _C["blue"])

            transcriber = self.app._orchestrator.transcriber
            stt_result  = transcriber.transcribe(wav_path)
            text_tr     = stt_result.get("text", "").strip()

            if not text_tr:
                self._set_progress(1.0, t("speech_not_recognized"), _C["red"])
                return

            # Surface the source transcript immediately.
            self._set_text(self._tr_box, text_tr, _C["gray"])
            self._set_progress(0.55, t("translating"), _C["blue"])

            # ── 3. LLM — long input is sent in chunks ─────────────────
            if not self._processing:
                return

            translator = self.app._orchestrator.translator
            text_en    = self._translate_chunked(translator, text_tr)

            if not self._processing:
                return

            # Display the translation.
            self._set_text(self._en_box, text_en, _C["white"])

            elapsed = time.time() - t0
            self._set_progress(1.0, t("done_tick"), _C["green"])
            self.after(0, lambda: self._elapsed_label.configure(
                text=t("elapsed_seconds", f"{elapsed:.1f}"), text_color=_C["dim"]
            ))

        except Exception as e:
            self._set_progress(0, t("error_message", e), _C["red"])

        finally:
            if owns_temp and wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
            self._processing = False
            self.after(0, lambda: [
                self._btn_process.configure(state="normal"),
                self._btn_cancel.configure(state="disabled"),
            ])

    # ── ffmpeg helpers ────────────────────────────────────────────────────────

    def _extract_audio(self, video_path: str) -> tuple[str | None, bool]:
        """Extract a 16 kHz mono WAV from a video file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        out_path = tmp.name

        try:
            import ffmpeg as ff
            (
                ff.input(video_path)
                  .output(out_path, ar=16000, ac=1, acodec="pcm_s16le")
                  .overwrite_output()
                  .run(quiet=True)
            )
            return out_path, True

        except Exception:
            # ffmpeg-python failed → fall back to the ffmpeg subprocess.
            try:
                result = subprocess.run(
                    ["ffmpeg", "-i", video_path,
                     "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                     "-y", out_path],
                    capture_output=True, timeout=600
                )
                if result.returncode == 0:
                    return out_path, True
                self._set_progress(0, f"{t('ffmpeg_error')}: {result.stderr[-200:]}", _C["red"])
                return None, False
            except FileNotFoundError:
                self._set_progress(0, t("ffmpeg_not_found"), _C["red"])
                return None, False

    def _convert_to_wav(self, audio_path: str) -> tuple[str | None, bool]:
        """Transcode MP3 / OGG / FLAC-style audio to a 16 kHz mono WAV."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        out_path = tmp.name

        try:
            import ffmpeg as ff
            (
                ff.input(audio_path)
                  .output(out_path, ar=16000, ac=1, acodec="pcm_s16le")
                  .overwrite_output()
                  .run(quiet=True)
            )
            return out_path, True

        except Exception:
            try:
                result = subprocess.run(
                    ["ffmpeg", "-i", audio_path,
                     "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                     "-y", out_path],
                    capture_output=True, timeout=300
                )
                if result.returncode == 0:
                    return out_path, True
                return None, False
            except FileNotFoundError:
                return None, False

    # ── LLM chunking ─────────────────────────────────────────────────────────

    def _translate_chunked(self, translator, full_text: str) -> str:
        """
        Split a long input into paragraphs of ``_LLM_CHUNK_WORDS`` words,
        translate each segment separately and concatenate the results.

        Short inputs (≤ chunk limit) are translated in a single call.
        """
        words = full_text.split()

        if len(words) <= _LLM_CHUNK_WORDS:
            result = translator.translate(full_text)
            return result.get("translation", full_text)

        # Split into paragraphs.
        chunks = []
        buf    = []
        for word in words:
            buf.append(word)
            if len(buf) >= _LLM_CHUNK_WORDS:
                chunks.append(" ".join(buf))
                buf = []
        if buf:
            chunks.append(" ".join(buf))

        translations = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            if not self._processing:
                break
            progress = 0.55 + (0.40 * (i / total))
            self._set_progress(
                progress,
                t("media_translate_progress", i+1, total),
                _C["blue"]
            )
            result = translator.translate(chunk)
            translations.append(result.get("translation", chunk))

        return " ".join(translations)

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_text(self, lang: str):
        box = self._tr_box if lang == "tr" else self._en_box
        text = box.get("0.0", "end").strip()

        if not text:
            messagebox.showinfo(t("save_empty_title"), t("save_empty_message"))
            return

        src_path = self._file_entry.get().strip()
        default_name = ""
        if src_path:
            base = os.path.splitext(os.path.basename(src_path))[0]
            default_name = f"{base}_{lang}.txt"

        out_dir = self.cfg.get("file_mode", "output_dir", default="")
        save_path = filedialog.asksaveasfilename(
            title=t("save_as_transcript") if lang == "tr" else t("save_as_translation"),
            initialfile=default_name,
            initialdir=out_dir or os.path.expanduser("~"),
            defaultextension=".txt",
            filetypes=[(t("filetype_text"), "*.txt"), (t("filetype_all"), "*.*")]
        )

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)
            # Remember the last-used directory.
            self.cfg.set("file_mode", "output_dir", os.path.dirname(save_path))
            self.cfg.save()

    # ── UI helpers (thread-safe) ──────────────────────────────────────────────

    def _set_progress(self, val: float, msg: str, color: str = None):
        """Thread-safe progress bar + label update."""
        def _update():
            self._progress.set(max(0.0, min(1.0, val)))
            self._prog_label.configure(
                text=msg, text_color=color or _C["gray"]
            )
        self.after(0, _update)

    def _set_text(self, box: ctk.CTkTextbox, text: str, color: str = None):
        """Thread-safe text box update."""
        def _update():
            box.configure(state="normal")
            box.delete("0.0", "end")
            box.insert("0.0", text)
            if color:
                box.configure(text_color=color)
            box.configure(state="disabled")
        self.after(0, _update)


# ══════════════════════════════════════════════════════════════════════════════
# Standalone test (mock backend)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from gui.config import ConfigManager

    class _MockTranscriber:
        def transcribe(self, path):
            time.sleep(1.5)
            return {"text": "Bu bir test transkriptidir. Gemma Echo dosya modu çalışıyor."}

    class _MockTranslator:
        def translate(self, text, context=None):
            time.sleep(1.0)
            return {"translation": "This is a test transcript. Gemma Echo file mode is working.", "engine": "mock"}

    class _MockOrchestrator:
        transcriber = _MockTranscriber()
        translator  = _MockTranslator()

    class _MockApp:
        _backend_ready  = True
        _orchestrator   = _MockOrchestrator()

    cfg = ConfigManager()
    root = ctk.CTk()
    root.withdraw()

    win = FileModeWindow(cfg, app=_MockApp())
    win.show()
    root.mainloop()
