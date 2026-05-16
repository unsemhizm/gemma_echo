"""
Gemma Echo — Book and document translation view.
"""

import os
import sys
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from gui.config import ConfigManager
from gui.i18n import t, get_language
from gui.pages._helpers import _C, _header, _card, _show_toast

# Supported language options.
_LANGS = [
    ("Turkish",  "tr"),
    ("English",  "en"),
    ("German",   "de"),
    ("French",   "fr"),
    ("Italian",  "it"),
    ("Spanish",  "es"),
    ("Arabic",   "ar"),
    ("Japanese", "ja"),
]
_LANG_NAMES = [l[0] for l in _LANGS]
_DOC_EXT     = {".txt", ".pdf", ".docx"}

class BookView(ctk.CTkFrame):
    def __init__(self, master, cfg: ConfigManager, app):
        super().__init__(master, fg_color=_C["bg"], corner_radius=0)
        self.cfg = cfg
        self.app = app
        self._translating = False
        self._dt = None          # Active DocumentTranslator instance.
        self._was_cancelled = False  # Set when _cancel was invoked (used by the finally toast logic).
        self._last_error = None  # Most recent exception message (for the toast).
        self._build()

    def _build(self):
        _header(self, f"\U0001f4d6  {t('book_title')}",
                t("book_subtitle"))

        # Scrollable surface.
        body = ctk.CTkScrollableFrame(self, fg_color="transparent", corner_radius=0)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # ── 1. File selection ─────────────────────────────────────────────
        file_card = _card(body, t("select_file"))

        file_row = ctk.CTkFrame(file_card, fg_color="transparent")
        file_row.pack(fill="x")

        self._file_entry = ctk.CTkEntry(
            file_row,
            placeholder_text=t("select_doc"),
            font=ctk.CTkFont(size=11),
            fg_color=_C["surface2"], border_color=_C["border"],
            height=36, corner_radius=10
        )
        self._file_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(
            file_row, text=t("browse"), width=76, height=36,
            fg_color=_C["surface2"], hover_color=_C["border"],
            corner_radius=10, font=ctk.CTkFont(size=11),
            command=self._browse
        ).pack(side="left")

        # ── 2. Translation settings ───────────────────────────────────────
        opt_card = _card(body, t("settings"))
        opt_card.columnconfigure(0, weight=1)
        opt_card.columnconfigure(1, weight=1)
        opt_card.columnconfigure(2, weight=1)

        # Source language.
        ctk.CTkLabel(
            opt_card, text=t("source_lang"),
            font=ctk.CTkFont(size=10, weight="bold"), text_color=_C["muted"]
        ).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 4))

        self._src_lang_combo = ctk.CTkComboBox(
            opt_card, values=_LANG_NAMES, height=34,
            fg_color=_C["surface2"], border_color=_C["border"],
            font=ctk.CTkFont(size=11), corner_radius=10, state="readonly"
        )
        saved_src = self.cfg.get("language", "source_name", default="Turkish")
        self._src_lang_combo.set(saved_src if saved_src in _LANG_NAMES else "Turkish")
        self._src_lang_combo.grid(row=1, column=0, sticky="ew", padx=(0, 8))

        # Target language.
        ctk.CTkLabel(
            opt_card, text=t("target_lang"),
            font=ctk.CTkFont(size=10, weight="bold"), text_color=_C["muted"]
        ).grid(row=0, column=1, sticky="w", padx=(0, 8), pady=(0, 4))

        self._tgt_lang_combo = ctk.CTkComboBox(
            opt_card, values=_LANG_NAMES, height=34,
            fg_color=_C["surface2"], border_color=_C["border"],
            font=ctk.CTkFont(size=11), corner_radius=10, state="readonly"
        )
        saved_tgt = self.cfg.get("language", "target_name", default="English")
        self._tgt_lang_combo.set(saved_tgt if saved_tgt in _LANG_NAMES else "English")
        self._tgt_lang_combo.grid(row=1, column=1, sticky="ew", padx=(0, 8))

        # Chunk size.
        ctk.CTkLabel(
            opt_card, text=t("chunk_size"),
            font=ctk.CTkFont(size=10, weight="bold"), text_color=_C["muted"]
        ).grid(row=0, column=2, sticky="w", pady=(0, 4))

        if get_language() == "tr":
            chunk_vals = ["400 (Yerel GGUF)", "800 (API - Önerilen)", "1200 (API - Büyük)"]
            chunk_def  = "800 (API - Önerilen)"
        else:
            chunk_vals = ["400 (Local GGUF)", "800 (API - Recommended)", "1200 (API - Large)"]
            chunk_def  = "800 (API - Recommended)"

        self._chunk_combo = ctk.CTkComboBox(
            opt_card, values=chunk_vals,
            height=34, fg_color=_C["surface2"], border_color=_C["border"],
            font=ctk.CTkFont(size=11), corner_radius=10, state="readonly"
        )
        self._chunk_combo.set(chunk_def)
        self._chunk_combo.grid(row=1, column=2, sticky="ew")

        # ── 3. Action bar ─────────────────────────────────────────────────
        act_card = _card(body, t("translate"))
        act_row = ctk.CTkFrame(act_card, fg_color="transparent")
        act_row.pack(fill="x")

        self._btn_start = ctk.CTkButton(
            act_row, text=f"\u25b6  {t('start_translation')}",
            height=36, corner_radius=10,
            fg_color=_C["blue"], hover_color="#4080d0",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._start
        )
        self._btn_start.pack(side="left", padx=(0, 8))

        self._btn_cancel = ctk.CTkButton(
            act_row, text=f"\u25a0 {t('cancel')}",
            height=36, width=80, corner_radius=10,
            fg_color=_C["red_bg"], hover_color=_C["red"],
            text_color=_C["red"],
            state="disabled",
            command=self._cancel
        )
        self._btn_cancel.pack(side="left", padx=(0, 8))

        self._btn_save = ctk.CTkButton(
            act_row, text=t("book_save_full"),
            height=36, width=140, corner_radius=10,
            fg_color=_C["surface2"], hover_color=_C["border"],
            font=ctk.CTkFont(size=11),
            state="disabled",
            command=self._save
        )
        self._btn_save.pack(side="left")

        # ── 4. Progress ───────────────────────────────────────────────────
        prog_card = _card(body, t("progress"))

        self._progress = ctk.CTkProgressBar(
            prog_card, height=8, mode="determinate",
            progress_color=_C["blue"], fg_color=_C["surface2"]
        )
        self._progress.set(0)
        self._progress.pack(fill="x", pady=(0, 6))

        self._prog_lbl = ctk.CTkLabel(
            prog_card, text=t("ready"),
            font=ctk.CTkFont(size=10), text_color=_C["dim"], anchor="w"
        )
        self._prog_lbl.pack(anchor="w")

        # ── 5. Output text box ────────────────────────────────────────────
        out_card = _card(body, t("en_translation"))

        self._out_box = ctk.CTkTextbox(
            out_card,
            font=ctk.CTkFont(size=12),
            fg_color=_C["surface2"], border_color=_C["border"], border_width=1,
            text_color=_C["text"], wrap="word", corner_radius=10,
            height=340
        )
        self._out_box.pack(fill="both", expand=True)
        self._out_box.configure(state="disabled")

        ctk.CTkLabel(
            out_card,
            text=t("book_result_hint"),
            font=ctk.CTkFont(size=9), text_color=_C["dim"]
        ).pack(anchor="w", pady=(4, 0))

    # ── File selection ────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title=t("browse"),
            filetypes=[
                (t("supported_documents_label"), "*.txt *.pdf *.docx"),
                (t("text_filetype_label"),   "*.txt"),
                (t("pdf_filetype_label"),     "*.pdf"),
                (t("word_filetype_label"),    "*.docx"),
                (t("all_files_label"),   "*.*"),
            ]
        )
        if path:
            self._file_entry.delete(0, "end")
            self._file_entry.insert(0, path)
            self._reset_output()

    def _reset_output(self):
        self._progress.set(0)
        self._prog_lbl.configure(text=t("ready"), text_color=_C["dim"])
        self._out_box.configure(state="normal")
        self._out_box.delete("0.0", "end")
        self._out_box.configure(state="disabled")
        self._out_box.configure(border_color=_C["border"])
        self._btn_save.configure(
            state="disabled",
            text=t("book_save_full"),
            fg_color=_C["surface2"],
            text_color=_C["text"],
        )

    # ── Start / cancel ────────────────────────────────────────────────────────

    def _start(self):
        path = self._file_entry.get().strip()
        if not path:
            messagebox.showwarning(t("file_not_selected"), t("select_file_first"))
            return
        if not os.path.exists(path):
            messagebox.showerror(t("file_not_found"), t("file_exists_error", path))
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in _DOC_EXT:
            messagebox.showwarning(
                t("unsupported_format"),
                t("unsupported_ext", ext, "TXT, PDF, DOCX")
            )
            return
        if not getattr(self.app, "_backend_ready", False):
            messagebox.showwarning(t("backend_not_ready_models"),
                                   t("models_still_loading"))
            return
        # D10: surface a clear error when the orchestrator / translator is not yet ready —
        # guards against AttributeError.
        orch = getattr(self.app, "_orchestrator", None)
        if orch is None or getattr(orch, "translator", None) is None:
            messagebox.showwarning(t("backend_not_ready_models"),
                                   t("models_still_loading"))
            return

        src_lang = self._src_lang_combo.get()
        tgt_lang = self._tgt_lang_combo.get()
        chunk_words = int(self._chunk_combo.get().split()[0])

        self._translating = True
        self._was_cancelled = False
        self._last_error = None
        self._btn_start.configure(state="disabled")
        self._btn_cancel.configure(state="normal")
        self._reset_output()

        threading.Thread(
            target=self._translation_pipeline,
            args=(path, src_lang, tgt_lang, chunk_words),
            daemon=True
        ).start()

    def _cancel(self):
        # B9: race fix — the pipeline finally clause may set ``self._dt = None``;
        # snapshot it into a local first so the check + call sequence is not exposed
        # to a torn-down reference.
        dt = self._dt
        if dt is not None:
            dt.cancel()
        self._was_cancelled = True
        # B1: do NOT flip ``_translating`` to False here — it is cleared in the
        # pipeline's finally block. Otherwise the 'if self._translating: append_output'
        # branch right after the click is skipped and the partial result is not
        # rendered to the textbox.
        # UX fix: do not reset the progress bar to 0 — the user wants to see how
        # far the translation reached. Only repaint the label in yellow; leave the
        # progress bar where it stopped.
        self.after(0, lambda: self._prog_lbl.configure(
            text=t("cancelled_message"), text_color=_C["yellow"]
        ))
        self._btn_cancel.configure(state="disabled")

    def on_leave(self):
        """Cancel in-flight work when the user navigates away from this page."""
        if self._translating:
            self._cancel()

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _translation_pipeline(self, file_path: str, src_lang: str,
                               tgt_lang: str, chunk_words: int):
        from pipeline.document_translator import DocumentTranslator

        translator = self.app._orchestrator.translator

        self._dt = DocumentTranslator(
            translator=translator,
            chunk_words=chunk_words,
            overlap_paragraphs=3
        )

        def on_progress(frac, msg):
            self._set_progress(frac, msg)

        def on_preview(text):
            # Live preview — flush to the textbox (called after every batch).
            self._live_update_output(text)

        # B4: hand localized progress messages to the pipeline. Missing keys
        # fall back to the pipeline's _DEFAULT_MESSAGES (English); fully backward compatible.
        messages = self._build_progress_messages()

        result = ""
        had_error = False
        try:
            # Layout-preserving translation: preserve formatting for DOCX / PDF.
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".docx", ".pdf"]:
                result = self._dt.translate_file_layout(
                    file_path=file_path,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    progress_cb=on_progress,
                    messages=messages,
                    preview_cb=on_preview,
                )
            else:
                result = self._dt.translate_file(
                    file_path=file_path,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    output_path=None,          # The GUI's "Save" button handles disk I/O.
                    progress_cb=on_progress,
                    messages=messages,
                )
        except Exception as e:
            had_error = True
            self._last_error = e
            # B3: preserve the partial result accumulated so far even on exception.
            try:
                result = self._dt.partial_result() if self._dt else ""
            except Exception:
                result = ""
        finally:
            final_text = result or ""

            # Capture preconditions inside the finally clause (before _dt is cleared).
            done = len(self._dt.translated_parts) if self._dt else 0
            total = self._dt.total_chunks if self._dt else 0
            was_cancelled = self._was_cancelled

            # Refresh the output textbox (B1: in every code path).
            if final_text:
                self._append_output(final_text)

            # Save button label + state.
            self.after(0, lambda: self._post_run_button_state(
                bool(final_text), had_error or was_cancelled
            ))

            # Surface the toast notification (UX polish).
            self.after(0, lambda: self._post_run_toast(
                final_text, had_error, was_cancelled, done, total
            ))

            self._translating = False
            # NOTE: _dt is NOT cleared — the user may click "Save" afterwards, and
            # _save() needs access to ``_dt._layout_source_kind`` /
            # ``_dt._docx_staging_path`` / ``_dt._pdf_blocks``. Otherwise every save
            # would bypass the layout preservation path and fall through to plain
            # ``_save_docx``. Starting a new translation reassigns a fresh
            # DocumentTranslator in _translation_pipeline — there is no race since
            # the ``_translating`` flag gates re-entry.
            self.after(0, lambda: [
                self._btn_start.configure(state="normal"),
                self._btn_cancel.configure(state="disabled"),
            ])

    # ── Localized progress messages ─────────────────────────────────────────

    def _build_progress_messages(self) -> dict:
        """Localized progress-message templates passed to DocumentTranslator.

        Uses the matching i18n.py keys when available; otherwise falls back to
        the English defaults defined in the pipeline (backward compatible).
        """
        # t() typically returns the key itself on a miss; spot that and return
        # None so the pipeline applies its own defaults.
        def _opt(key: str) -> str | None:
            try:
                val = t(key)
            except Exception:
                return None
            if not val or val == key:
                return None
            return val

        # Mapping of optional i18n keys — pass None for any that are missing.
        return {
            "reading":        _opt("book_reading_file"),
            "chunked":        _opt("book_chunked"),
            "translating":    _opt("book_translating"),
            "summary_update": _opt("book_summary_update"),
            "done":           _opt("book_done"),
            "cancelled":      _opt("book_cancelled"),
            "empty_pdf":      _opt("book_empty_pdf"),
            "empty_file":     _opt("book_empty_file"),
            "unsupported":    _opt("book_unsupported"),
        }

    # ── Post-run helpers (UX polish) ───────────────────────────────────────

    def _post_run_button_state(self, has_result: bool, is_partial: bool):
        """Configure the "Save" button label and color after translation completes."""
        if not has_result:
            self._btn_save.configure(
                state="disabled",
                text=t("book_save_full"),
                fg_color=_C["surface2"],
                text_color=_C["text"],
            )
            return
        if is_partial:
            # Partial result: attention-grabbing yellow / orange.
            self._btn_save.configure(
                state="normal",
                text=t("book_save_partial"),
                fg_color=_C["yellow"],
                text_color=_C["bg"],
            )
        else:
            # Full result: standard blue.
            self._btn_save.configure(
                state="normal",
                text=t("book_save_full"),
                fg_color=_C["blue"],
                text_color=_C["text"],
            )

    def _post_run_toast(self, final_text: str, had_error: bool,
                        was_cancelled: bool, done: int, total: int):
        """Surface a toast notification appropriate to the translation outcome."""
        if had_error:
            err_str = str(self._last_error) if self._last_error else ""
            low = err_str.lower()
            # Scanned PDF dedicated toast (red + explanatory).
            if ("ocr" in low or "taranmış" in low or "scanned" in low
                    or "metin içermiyor" in low or "contains no text" in low):
                _show_toast(self, t("book_toast_pdf_scan"), level="error",
                            duration_ms=7000)
                return
            if not final_text:
                # Empty / unreadable document.
                _show_toast(self, t("book_toast_empty"), level="warning")
                return
            # Generic error — but a partial result exists.
            short = err_str if len(err_str) < 140 else err_str[:140] + "..."
            _show_toast(self, t("book_toast_error", short), level="error",
                        duration_ms=6000)
            return

        if was_cancelled:
            if final_text and total:
                _show_toast(self, t("book_toast_partial_ready", done, total),
                            level="warning", duration_ms=6000)
                # Surface the short summary in the status bar as well.
                self._prog_lbl.configure(
                    text=t("book_partial_status", done, total),
                    text_color=_C["yellow"],
                )
            return

        # Complete success.
        if final_text:
            sec = total if total else (final_text.count("\n\n") + 1)
            _show_toast(self, t("book_toast_completed", sec), level="success")
        else:
            _show_toast(self, t("book_toast_empty"), level="warning")

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save(self):
        text = self._out_box.get("0.0", "end").strip()
        if not text:
            messagebox.showinfo(t("save_empty_title"), t("save_empty_message"))
            return
        src = self._file_entry.get().strip()
        base = os.path.splitext(os.path.basename(src))[0] if src else "translation"
        tgt = self._tgt_lang_combo.get().lower()[:2]
        path = filedialog.asksaveasfilename(
            title=t("save_translation"),
            initialfile=f"{base}_{tgt}.docx",
            initialdir=self.cfg.get("file_mode", "output_dir",
                                    default=os.path.expanduser("~")),
            defaultextension=".docx",
            filetypes=[
                (t("word_filetype_label"), "*.docx"),
                (t("text_filetype_label"), "*.txt"),
                (t("all_files_label"),     "*.*"),
            ],
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".docx":
                if hasattr(self, '_dt') and self._dt and self._dt._layout_source_kind:
                    try:
                        self._dt.save_layout_docx(path)
                    except Exception:
                        self._save_docx(path, text)
                else:
                    self._save_docx(path, text)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as e:
            messagebox.showerror(t("saved_title"), str(e))
            return

        # Force Windows Explorer to refresh its folder view immediately (cache workaround).
        # Without this, a file written to the desktop may not appear until the next
        # shell event fires.
        self._notify_shell(path)

        self.cfg.set("file_mode", "output_dir", os.path.dirname(path))
        self.cfg.save()
        messagebox.showinfo(t("saved_title"), t("file_saved_message", path))

    def _notify_shell(self, path: str):
        """Notify Windows Explorer of the new file (SHChangeNotify).

        On other operating systems it is a silent no-op.
        """
        if sys.platform != "win32":
            return
        try:
            import ctypes
            SHCNE_CREATE = 0x00000002
            SHCNF_PATHW  = 0x0005
            ctypes.windll.shell32.SHChangeNotify(
                SHCNE_CREATE, SHCNF_PATHW, ctypes.c_wchar_p(path), None
            )
        except Exception:
            pass  # Notification failure must not disturb the user.

    def _save_docx(self, path: str, text: str):
        """Write the translation as a clean Word document.

        Does not clone the source layout — produces an independent, easily
        readable e-book-style document with no inherited PDF / DOCX formatting.
        python-docx is already a project dependency (required for PDF parsing).
        """
        from docx import Document
        from docx.shared import Pt, Cm

        doc = Document()

        # Page margins — comfortable book-style layout.
        for section in doc.sections:
            section.top_margin    = Cm(2.2)
            section.bottom_margin = Cm(2.2)
            section.left_margin   = Cm(2.5)
            section.right_margin  = Cm(2.5)

        # Default style — Calibri 11 pt, generous line spacing.
        style = doc.styles["Normal"]
        style.font.name = "Calibri"
        style.font.size = Pt(11)

        # Paragraphs are separated by blank lines (pipeline output format).
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            p = doc.add_paragraph(para)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing = 1.35

        doc.save(path)

    # ── Thread-safe helpers ───────────────────────────────────────────────────

    def _set_progress(self, val: float, msg: str, color: str = None):
        def _u():
            self._progress.set(max(0.0, min(1.0, val)))
            self._prog_lbl.configure(text=msg, text_color=color or _C["dim"])
        self.after(0, _u)

    def _append_output(self, text: str):
        def _u():
            self._out_box.configure(state="normal")
            self._out_box.delete("0.0", "end")
            self._out_box.insert("0.0", text)
            self._out_box.configure(state="disabled")
            # "Flashy reveal" — highlight the textbox border for 1.5 s, then revert.
            highlight = _C["yellow"] if self._was_cancelled else _C["green"]
            self._out_box.configure(border_color=highlight)
            self._out_box.see("0.0")
            self.after(1500, lambda: self._out_box.configure(border_color=_C["border"]))
        self.after(0, _u)

    def _live_update_output(self, text: str):
        """Update the textbox with the in-progress translation stream.

        Differences from _append_output:
          - No border highlight (no flashing).
          - Preserves scroll position (the user's viewport must not jump).

        Passed as the ``preview_cb`` argument to DocumentTranslator; invoked
        after every batch.
        """
        def _u():
            try:
                # Snapshot the current scroll position (yview tuple: top_frac, bottom_frac).
                yview = self._out_box.yview()
            except Exception:
                yview = (0.0, 1.0)
            self._out_box.configure(state="normal")
            self._out_box.delete("0.0", "end")
            self._out_box.insert("0.0", text or "")
            self._out_box.configure(state="disabled")
            # If the user was at the bottom, follow along; otherwise restore the previous position.
            try:
                if yview[1] >= 0.98:
                    self._out_box.see("end")
                else:
                    self._out_box.yview_moveto(yview[0])
            except Exception:
                pass
        self.after(0, _u)
