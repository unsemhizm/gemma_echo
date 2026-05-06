"""
Gemma Echo — Kitap ve Belge Çeviri Ekranı (Book / Document View)
"""

import os
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from gui.config import ConfigManager
from gui.i18n import t, get_language
from gui.pages._helpers import _C, _header, _card

# Desteklenen dil secenekleri
_LANGS = [
    ("Turkish",  "tr"),
    ("English",  "en"),
    ("German",   "de"),
    ("Spanish",  "es"),
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
        self._dt = None          # aktif DocumentTranslator ornegi
        self._build()

    def _build(self):
        _header(self, f"\U0001f4d6  {t('book_title')}",
                t("book_subtitle"))

        # Scroll alani
        body = ctk.CTkScrollableFrame(self, fg_color="transparent", corner_radius=0)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # ── 1. Dosya Secimi ────────────────────────────────────────────────
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

        # ── 2. Ceviri Ayarlari ─────────────────────────────────────────────
        opt_card = _card(body, t("settings"))
        opt_card.columnconfigure(0, weight=1)
        opt_card.columnconfigure(1, weight=1)
        opt_card.columnconfigure(2, weight=1)

        # Kaynak dil
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

        # Hedef dil
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

        # Chunk boyutu
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

        # ── 3. Eylem Cubugu ───────────────────────────────────────────────
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
            act_row, text=f"\U0001f4be {t('save_translation')}",
            height=36, width=100, corner_radius=10,
            fg_color=_C["surface2"], hover_color=_C["border"],
            font=ctk.CTkFont(size=11),
            state="disabled",
            command=self._save
        )
        self._btn_save.pack(side="left")

        # ── 4. Ilerleme ────────────────────────────────────────────────────
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

        # ── 5. Cikti Metin Kutusu ──────────────────────────────────────────
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
            text="Ceviri tamamlaninca metin burada gorunur.",
            font=ctk.CTkFont(size=9), text_color=_C["dim"]
        ).pack(anchor="w", pady=(4, 0))

    # ── Dosya Secimi ──────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title=t("browse"),
            filetypes=[
                ("Desteklenen belgeler", "*.txt *.pdf *.docx"),
                ("Metin",   "*.txt"),
                ("PDF",     "*.pdf"),
                ("Word",    "*.docx"),
                ("Hepsi",   "*.*"),
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
        self._btn_save.configure(state="disabled")

    # ── Baslat / Iptal ────────────────────────────────────────────────────────

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

        src_lang = self._src_lang_combo.get()
        tgt_lang = self._tgt_lang_combo.get()
        chunk_words = int(self._chunk_combo.get().split()[0])

        self._translating = True
        self._btn_start.configure(state="disabled")
        self._btn_cancel.configure(state="normal")
        self._btn_save.configure(state="disabled")
        self._reset_output()

        threading.Thread(
            target=self._translation_pipeline,
            args=(path, src_lang, tgt_lang, chunk_words),
            daemon=True
        ).start()

    def _cancel(self):
        if self._dt is not None:
            self._dt.cancel()
        self._translating = False
        self._set_progress(0, "Iptal edildi.", _C["yellow"])
        self._btn_start.configure(state="normal")
        self._btn_cancel.configure(state="disabled")

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

        try:
            result = self._dt.translate_file(
                file_path=file_path,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                output_path=None,          # GUI kaydetme dugmesini kullanir
                progress_cb=on_progress
            )

            if self._translating:          # iptal edilmediyse
                self._append_output(result)
                self.after(0, lambda: self._btn_save.configure(state="normal"))

        except Exception as e:
            self._set_progress(0, f"Hata: {e}", _C["red"])
        finally:
            self._translating = False
            self._dt = None
            self.after(0, lambda: [
                self._btn_start.configure(state="normal"),
                self._btn_cancel.configure(state="disabled"),
            ])

    # ── Kaydet ────────────────────────────────────────────────────────────────

    def _save(self):
        text = self._out_box.get("0.0", "end").strip()
        if not text:
            messagebox.showinfo("Bos", "Kaydedilecek metin yok.")
            return
        src = self._file_entry.get().strip()
        base = os.path.splitext(os.path.basename(src))[0] if src else "ceviri"
        tgt = self._tgt_lang_combo.get().lower()[:2]
        path = filedialog.asksaveasfilename(
            title=t("save_translation"),
            initialfile=f"{base}_{tgt}.txt",
            initialdir=self.cfg.get("file_mode", "output_dir",
                                    default=os.path.expanduser("~")),
            defaultextension=".txt",
            filetypes=[("Metin", "*.txt"), ("Hepsi", "*.*")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.cfg.set("file_mode", "output_dir", os.path.dirname(path))
            self.cfg.save()
            messagebox.showinfo("Kaydedildi", f"Dosya kaydedildi:\n{path}")

    # ── Thread-safe yardimcilar ───────────────────────────────────────────────

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
        self.after(0, _u)
