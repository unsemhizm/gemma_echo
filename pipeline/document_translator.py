"""
Gemma Echo — Long-document translation pipeline (sliding window).

Supported formats: TXT, PDF, DOCX.
Algorithm: overlapping sliding window + term glossary.

Usage:
    dt = DocumentTranslator(translator, chunk_words=800, overlap_paragraphs=3)
    dt.translate_file("book.pdf", src_lang="Turkish", tgt_lang="English",
                      output_path="book_en.txt", progress_cb=None)

Layout-preserving usage:
    dt.translate_file_layout("form.docx", layout_output_path="form_en.docx",
                              src_lang="Turkish", tgt_lang="English")
    # DOCX -> DOCX: run-level in-place translation; fonts / tables / headings preserved.
    # PDF  -> DOCX: font-size-based heading detection + Heading styles.
"""

import os
import shutil
import tempfile
from dataclasses import dataclass, field

# ===========================================================
# Constants — PDF column-analysis thresholds (D1: magic numbers in one place).
# ===========================================================
_TWO_COLUMN_GUTTER_LEFT_RATIO  = 0.44
_TWO_COLUMN_GUTTER_RIGHT_RATIO = 0.56
_TWO_COLUMN_OVERLAP_TOLERANCE  = 0.04   # Max fraction of words allowed to cross the gutter.
_TWO_COLUMN_MIN_WORDS_PER_SIDE = 15     # Minimum word count required on each side.

# Turkish abbreviations that end with a period but do NOT mark a sentence boundary
# (D3 — reduces false-positive sentence splits).
_TR_ABBREVS = {
    "md.", "bkz.", "sn.", "prof.", "doç.", "dr.", "av.", "vs.", "vb.",
    "örn.", "yay.", "bas.", "çev.", "haz.", "ed.", "a.g.e.", "i.ö.", "i.s.",
    "tc.", "t.c.", "öğr.", "gör.", "aş.", "ltd.",
    "mr.", "mrs.", "ms.", "jr.", "sr.", "e.g.", "i.e.", "etc.", "st.", "no.",
}

# Sentence-initial connectives filtered from glossary candidate harvesting (D6).
_GLOSSARY_STOPWORDS = {
    "Bu", "Şu", "O", "Bunlar", "Şunlar", "Onlar",
    "Şimdi", "Sonra", "Önce", "Bugün", "Dün", "Yarın",
    "Fakat", "Ancak", "Ama", "Yine", "Aynı", "Henüz", "Aslında",
    "Çünkü", "Yani", "Belki", "Gerçekten", "İşte",
    "The", "This", "That", "These", "Those", "There", "Here",
    "However", "Therefore", "Moreover", "Although", "Because",
}

# PDF heading-detection thresholds (font-size ratio relative to body size).
# body_size = mode of font sizes seen across the document. Headings sit above it.
_PDF_H1_RATIO = 1.40   # body * 1.40+ -> H1 (e.g., body 11 pt → H1 ≥ 15 pt).
_PDF_H2_RATIO = 1.18   # body * 1.18+ -> H2 (e.g., body 11 pt → H2 13-14 pt).
# H3 is intentionally not separated — the bold mid-grey zone between H2 and body
# was being lost in LLM translation; an H1/H2/body 3-class scheme is sufficient.

# Maximum word count for a block to still be classified as a heading
# (very long "H1" paragraphs are almost certainly body text — protects against misclassification).
_PDF_HEADING_MAX_WORDS = 20


@dataclass
class Block:
    """Structured PDF block — text + classification (heading / body).

    Not used for DOCX sources because DOCX already exposes ``paragraph.style.name``;
    DOCX is translated in-place at the run level (full layout preservation).

    For PDF we infer the label heuristically from font sizes and then apply
    Heading 1/2/Normal styles when writing to Word.
    """
    text: str
    kind: str = "body"   # "h1" | "h2" | "body"


# Default progress messages (B4: GUI may override via the 'messages' parameter).
_DEFAULT_MESSAGES = {
    "reading":        "Reading file...",
    "chunked":        "{paragraphs} paragraphs — {total} chunk(s) prepared.",
    "translating":    "Translating chunk {i}/{total}...",
    "summary_update": "Chunk {i}/{total} — Updating document summary...",
    "done":           "Complete — {total} chunk(s) translated.",
    "cancelled":      "Cancelled — {done}/{total} chunk(s) translated.",
    "empty_pdf":      (
        "PDF contains no extractable text (likely a scanned image / no OCR). "
        "OCR is not supported in this release."
    ),
    "empty_file":     "File is empty or could not be read.",
    "unsupported":    "Unsupported format: .{ext}\nSupported: TXT, PDF, DOCX",
}


class DocumentTranslator:
    def __init__(self, translator, chunk_words: int = 800, overlap_paragraphs: int = 3):
        """
        Args:
            translator:           llm/translator.py Translator instance.
            chunk_words:          Approximate word budget per chunk.
                                  Recommended: 800 for the API, 400 for the local GGUF.
            overlap_paragraphs:   Number of preceding paragraphs carried into the
                                  next chunk for context continuity.
        """
        self.translator = translator
        self.chunk_words = chunk_words
        self.overlap_paragraphs = overlap_paragraphs
        self._cancel = False
        # B1/B3: incremental store of translated chunks; partial_result() lets the
        # caller retrieve everything translated so far after a cancellation / exception.
        self.translated_parts: list[str] = []
        # Used by the GUI for "done/total" indicators (toasts, status bar).
        self.total_chunks: int = 0

        # ── Layout-preserving translation (DOCX/PDF -> styled DOCX) ──
        # Populated by `translate_file_layout`, consumed by `save_layout_docx`.
        # DOCX source: path to the staged file (the in-place translation result).
        # PDF source: style-tagged Block list — rendered as Heading/Normal DOCX on save.
        self._docx_staging_path: str | None = None
        self._pdf_blocks: list[Block] | None = None
        self._layout_source_kind: str | None = None   # "docx" | "pdf" | None.

    def partial_result(self) -> str:
        """Return every chunk translated so far, joined back into a single string.

        Used by the GUI after a cancellation / exception so the user can still see
        the partial result and enable the "Save" button.
        """
        return "\n\n".join(self.translated_parts)

    # ═══════════════════════════════════════════════════════════
    # CANCELLATION
    # ═══════════════════════════════════════════════════════════

    def cancel(self):
        """Cancel the in-progress translation (thread-safe)."""
        self._cancel = True

    # ═══════════════════════════════════════════════════════════
    # PRIMARY ENTRYPOINT
    # ═══════════════════════════════════════════════════════════

    def translate_file(self, file_path: str, src_lang: str = "Turkish",
                       tgt_lang: str = "English", output_path: str = None,
                       progress_cb=None, user_glossary: dict = None,
                       messages: dict = None) -> str:
        """
        1. Read the file (PDF / DOCX / TXT).
        2. Split into paragraphs (smart preprocessing reconstructs true paragraphs).
        3. Group paragraphs into chunk_words-sized chunks.
        4. Translate each chunk with overlap, rolling summary and active glossary context.
        5. Concatenate the results and write to output_path (partial output on cancellation).
        6. Continuously update the term_glossary during translation.

        Args:
            file_path:     Source file (.txt / .pdf / .docx).
            src_lang:      Source language name passed to the LLM prompt (e.g., "Turkish").
            tgt_lang:      Target language name (e.g., "English").
            output_path:   Output file path (None → no disk write).
            progress_cb:   progress_cb(fraction: float, msg: str) — GUI progress updates.
            user_glossary: User's custom term dictionary (e.g., {"Yapay Zeka": "AI"}).
            messages:      Localized progress-message templates (B4 — i18n).
                           When None, _DEFAULT_MESSAGES is used. The GUI may supply
                           a t()-resolved dict to localize the strings.

        Returns:
            The translated text as a single string. On cancellation, returns the
            partial result accumulated up to that point.
        """
        self._cancel = False
        self.translated_parts = []

        msg = dict(_DEFAULT_MESSAGES)
        if messages:
            msg.update({k: v for k, v in messages.items() if v})

        if progress_cb:
            progress_cb(0.0, msg["reading"])

        paragraphs = self._read_file(file_path, messages=msg)
        if not paragraphs:
            raise ValueError(msg["empty_file"])

        chunks = self._chunk_paragraphs(paragraphs)
        total = len(chunks)
        self.total_chunks = total

        if progress_cb:
            progress_cb(
                0.05,
                msg["chunked"].format(paragraphs=len(paragraphs), total=total),
            )

        # D5: the default glossary is only useful when the source is Turkish;
        # leave it empty for other source languages — injecting "Yapay Zeka -> AI"
        # into a non-Turkish prompt is just noise that degrades the LLM context.
        if (src_lang or "").strip().lower() in ("turkish", "türkçe", "tr"):
            default_glossary = {
                "Gemma Echo": "Gemma Echo",
                "TÜBİTAK": "TÜBİTAK",
                "Yapay Zeka": "Artificial Intelligence",
                "Derin Öğrenme": "Deep Learning",
                "Makine Öğrenmesi": "Machine Learning",
                "Yapay Sinir Ağları": "Neural Networks",
            }
        else:
            default_glossary = {}

        term_glossary = dict(default_glossary)
        if user_glossary:
            term_glossary.update(user_glossary)

        # B2: default + user-defined terms are protected from eviction.
        protected_terms = set(default_glossary) | set(user_glossary or {})

        context_paras = []
        rolling_summary = ""
        prev_translation = ""
        completed = 0

        for i, chunk in enumerate(chunks):
            if self._cancel:
                break

            frac = 0.05 + 0.90 * (i / total)
            if progress_cb:
                progress_cb(
                    frac,
                    msg["translating"].format(i=i + 1, total=total),
                )

            chunk_text = "\n\n".join(chunk)
            translated = self._translate_chunk_with_context(
                chunk, context_paras, term_glossary, src_lang, tgt_lang,
                prev_translation=prev_translation, rolling_summary=rolling_summary,
                chunk_text=chunk_text,
            )
            self.translated_parts.append(translated)
            completed = i + 1

            # Harvest dynamic proper-noun candidates; protected terms are preserved.
            term_glossary = self._update_glossary(
                chunk_text, term_glossary, protected_terms=protected_terms
            )

            # Overlap: pass the last N paragraphs as context into the next chunk.
            context_paras = chunk[-self.overlap_paragraphs:]
            prev_translation = translated

            # D4: the rolling summary is an extra LLM call (once per 3 chunks).
            # ~10 extra calls for a 31-chunk book — small but non-zero API cost.
            if (i + 1) % 3 == 0 or i == 0:
                if progress_cb:
                    progress_cb(
                        frac,
                        msg["summary_update"].format(i=i + 1, total=total),
                    )
                try:
                    rolling_summary = self.translator.generate_summary(
                        translated, rolling_summary
                    )
                except Exception:
                    # Summary refresh failed — the main translation flow must continue.
                    pass

        result = self.partial_result()

        # B1: even on cancellation, the partial result is persisted to disk and returned.
        if output_path and result:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result)
            except Exception:
                # A disk-write failure must not discard the translation — return the result to the caller.
                pass

        if progress_cb:
            if self._cancel:
                progress_cb(
                    min(1.0, 0.05 + 0.90 * (completed / total)) if total else 0.0,
                    msg["cancelled"].format(done=completed, total=total),
                )
            else:
                progress_cb(1.0, msg["done"].format(total=total))

        return result

    # ═══════════════════════════════════════════════════════════
    # LAYOUT-PRESERVING TRANSLATION (DOCX -> DOCX, PDF -> styled DOCX)
    # ═══════════════════════════════════════════════════════════

    def translate_file_layout(self, file_path: str, src_lang: str = "Turkish",
                              tgt_lang: str = "English", progress_cb=None,
                              user_glossary: dict = None,
                              messages: dict = None,
                              preview_cb=None) -> str:
        """Public entrypoint for layout-preserving translation.

        Flow:
          - DOCX source : opens the document and translates paragraph + table
                          runs in place; fonts, tables, headings and margins are
                          preserved. The result is saved to a temporary
                          ``_docx_staging_path``.
          - PDF source  : performs font-size-based heading detection, produces a
                          Block list, translates each block. Stored in ``_pdf_blocks``.
          - TXT source  : layout has no meaning here; falls through to the classic
                          ``translate_file`` path.

        The GUI must call ``save_layout_docx(path)`` at the end of translation;
        the staged file is then copied (DOCX) or a styled DOCX is rendered from
        the blocks (PDF).

        Returns: plain preview text (for the on-screen text box).
        """
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        self._reset_layout_staging()

        if ext == "docx":
            self._layout_source_kind = "docx"
            return self._translate_docx_inplace(
                file_path, src_lang=src_lang, tgt_lang=tgt_lang,
                progress_cb=progress_cb, user_glossary=user_glossary,
                messages=messages, preview_cb=preview_cb,
            )
        elif ext == "pdf":
            self._layout_source_kind = "pdf"
            return self._translate_pdf_to_blocks(
                file_path, src_lang=src_lang, tgt_lang=tgt_lang,
                progress_cb=progress_cb, user_glossary=user_glossary,
                messages=messages, preview_cb=preview_cb,
            )
        else:
            self._layout_source_kind = None
            return self.translate_file(
                file_path, src_lang=src_lang, tgt_lang=tgt_lang,
                progress_cb=progress_cb, user_glossary=user_glossary,
                messages=messages,
            )

    def save_layout_docx(self, output_path: str):
        """Persist the layout-preserving translation result as .docx.

        - DOCX source : copy the staged file to ``output_path`` (full layout).
        - PDF source  : render the Block list into a fresh Heading/Normal-styled DOCX.
        - Other       : ValueError (caller should fall back to plain `_save_docx`).

        Partial results from a cancelled translation can also be persisted;
        untranslated paragraphs are kept as their source-language text.
        """
        if self._layout_source_kind == "docx" and self._docx_staging_path:
            if not os.path.exists(self._docx_staging_path):
                raise FileNotFoundError("Staged DOCX not found.")
            shutil.copyfile(self._docx_staging_path, output_path)
            return

        if self._layout_source_kind == "pdf" and self._pdf_blocks:
            self._render_blocks_to_docx(self._pdf_blocks, output_path)
            return

        raise ValueError(
            "No layout-preserving output available. translate_file_layout must "
            "be called first, or the source is not DOCX/PDF."
        )

    def _reset_layout_staging(self):
        """Wipe previous staging artifacts (idempotent)."""
        if self._docx_staging_path and os.path.exists(self._docx_staging_path):
            try:
                os.remove(self._docx_staging_path)
            except OSError:
                pass
        self._docx_staging_path = None
        self._pdf_blocks = None
        self._layout_source_kind = None

    # ═══════════════════════════════════════════════════════════
    # SMART TEXT AND PAGE-LAYOUT ANALYSIS (Stage 1)
    # ═══════════════════════════════════════════════════════════

    def _extract_page_text_layout_aware(self, page) -> str:
        """Pull text from a pdfplumber page with two-column awareness.

        Falls back to the default vertical-flow extraction for single-column pages.

        D1: thresholds are read from module-level constants for easier testing/tuning.
        """
        try:
            width = page.width
            height = page.height
            words = page.extract_words()
            if not words:
                return page.extract_text() or ""

            mid_start = width * _TWO_COLUMN_GUTTER_LEFT_RATIO
            mid_end   = width * _TWO_COLUMN_GUTTER_RIGHT_RATIO

            overlapping_words = [w for w in words if w['x0'] < mid_end and w['x1'] > mid_start]
            left_words  = [w for w in words if w['x1'] <= mid_start]
            right_words = [w for w in words if w['x0'] >= mid_end]

            if (
                len(overlapping_words) < len(words) * _TWO_COLUMN_OVERLAP_TOLERANCE
                and len(left_words)  > _TWO_COLUMN_MIN_WORDS_PER_SIDE
                and len(right_words) > _TWO_COLUMN_MIN_WORDS_PER_SIDE
            ):
                left_area  = page.crop((0, 0, width * 0.49, height))
                right_area = page.crop((width * 0.51, 0, width, height))
                left_text  = left_area.extract_text() or ""
                right_text = right_area.extract_text() or ""
                return left_text + "\n\n" + right_text
            return page.extract_text() or ""
        except Exception:
            return page.extract_text() or ""

    def _reconstruct_paragraphs(self, raw_text: str) -> list:
        """Intelligently merge spurious line breaks, repair hyphenated splits and
        reconstruct real paragraphs from the raw OCR-style line stream.
        """
        import re
        text = raw_text.replace("\r\n", "\n")
        lines = text.split("\n")

        paragraphs = []
        current_para = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
                continue

            # Drop page numbers and short headers / footers (noise filter).
            if line.isdigit() or (len(line) < 6 and any(k in line.lower() for k in ["page", "sayfa", "ch.", "bölüm"])):
                continue

            # Stitch hyphenated line breaks.
            has_hyphen = False
            if line.endswith("-") and len(line) > 1:
                if line[-2].isalpha():
                    line = line[:-1].rstrip()
                    has_hyphen = True

            if current_para:
                prev_line = current_para[-1]
                # Did the previous line end with a sentence-terminating punctuation?
                ends_sentence = prev_line[-1] in {".", "?", "!", ":"} if prev_line else False
                # D3: a trailing period on an abbreviation ("Dr.", "Prof.", "Md." ...) is NOT a sentence boundary.
                if ends_sentence and prev_line.endswith("."):
                    last_token = prev_line.rsplit(None, 1)[-1].lower()
                    if last_token in _TR_ABBREVS:
                        ends_sentence = False
                # Does the current line start with a lowercase letter?
                starts_lowercase = line[0].islower() if line else False

                if has_hyphen:
                    # Hyphen junction: concatenate without inserting a space.
                    current_para[-1] = prev_line + line
                elif not ends_sentence or starts_lowercase:
                    # Same paragraph continuation: join with a space.
                    current_para.append(line)
                else:
                    # New paragraph boundary.
                    paragraphs.append(" ".join(current_para))
                    current_para = [line]
            else:
                current_para = [line]

        if current_para:
            paragraphs.append(" ".join(current_para))

        # Clean the paragraphs, collapse repeated whitespace and drop very short noise lines.
        cleaned_paras = []
        for para in paragraphs:
            para = re.sub(r'\s+', ' ', para).strip()
            if para and len(para) > 8:
                cleaned_paras.append(para)

        return cleaned_paras

    # ═══════════════════════════════════════════════════════════
    # FILE READING
    # ═══════════════════════════════════════════════════════════

    def _read_file(self, path: str, messages: dict = None) -> list:
        """Read the file and return a list of paragraphs.

        - .txt  -> Smart line-break cleanup + paragraph reconstruction.
        - .pdf  -> pdfplumber with column-aware extraction + paragraph reconstruction.
        - .docx -> python-docx, paragraphs + table cell text.
        """
        msg = dict(_DEFAULT_MESSAGES)
        if messages:
            msg.update({k: v for k, v in messages.items() if v})

        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""

        if ext == "txt":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            return self._reconstruct_paragraphs(text)

        elif ext == "pdf":
            try:
                import pdfplumber
            except ImportError:
                raise ImportError(
                    "Reading PDFs requires the 'pdfplumber' package.\n"
                    "Install: pip install pdfplumber"
                )
            paragraphs = []
            raw_chars = 0
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = self._extract_page_text_layout_aware(page)
                    raw_chars += len(page_text or "")
                    page_paras = self._reconstruct_paragraphs(page_text)
                    paragraphs.extend(page_paras)
            # B6: when pdfplumber yields no text it is almost certainly a scanned PDF.
            # Surface a clear OCR explanation instead of the misleading "empty file" message.
            if not paragraphs and raw_chars < 20:
                raise ValueError(msg["empty_pdf"])
            return paragraphs

        elif ext == "docx":
            try:
                from docx import Document
            except ImportError:
                raise ImportError(
                    "Reading DOCX requires the 'python-docx' package.\n"
                    "Install: pip install python-docx"
                )
            doc = Document(path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            # B8: include table-cell text in the translation pass.
            for table in getattr(doc, "tables", []):
                for row in table.rows:
                    for cell in row.cells:
                        cell_text = (cell.text or "").strip()
                        if cell_text:
                            paragraphs.append(cell_text)
            return paragraphs

        else:
            raise ValueError(msg["unsupported"].format(ext=ext))

    # ═══════════════════════════════════════════════════════════
    # PARAGRAPH GROUPING (sliding window)
    # ═══════════════════════════════════════════════════════════

    def _chunk_paragraphs(self, paragraphs: list) -> list:
        """Group paragraphs into chunks not exceeding ``chunk_words``.

        Paragraph boundaries are respected — never split a paragraph mid-text.
        """
        chunks = []
        current_chunk = []
        current_words = 0

        for para in paragraphs:
            para_words = len(para.split())

            if current_words + para_words > self.chunk_words and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [para]
                current_words = para_words
            else:
                current_chunk.append(para)
                current_words += para_words

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # ═══════════════════════════════════════════════════════════
    # CHUNK TRANSLATION (overlap + glossary)
    # ═══════════════════════════════════════════════════════════

    def _translate_chunk_with_context(self, chunk: list, context_paras: list,
                                      term_glossary: dict,
                                      src_lang: str, tgt_lang: str,
                                      prev_translation: str = "",
                                      rolling_summary: str = "",
                                      chunk_text: str = None) -> str:
        """Translate a single chunk with overlap, prev_translation, rolling_summary
        and active glossary context.

        ``chunk_text`` is optional: callers that have already computed
        ``"\\n\\n".join(chunk)`` can pass it back to avoid duplicate work (B7).
        """
        if chunk_text is None:
            chunk_text = "\n\n".join(chunk)

        # TERM RULES: translate the listed terms exactly as specified (Stage 3).
        if term_glossary:
            rules = []
            for k, v in term_glossary.items():
                if v:
                    rules.append(f"{k} -> {v}")
                else:
                    rules.append(k)
            terms_str = ", ".join(rules[:30])  # Cap at 30 terms to keep the prompt lean.
            text_to_translate = (
                f"[STRICT GLOSSARY RULES — translate these terms exactly as specified: "
                f"{terms_str}]\n\n{chunk_text}"
            )
        else:
            text_to_translate = chunk_text

        # Forward overlap context and advanced parameters to the translator.
        result = self.translator.translate(
            text_to_translate,
            context=context_paras,   # Overlap: previous N paragraphs supplied as reference.
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_name=src_lang,
            tgt_name=tgt_lang,
            prev_translation=prev_translation,
            rolling_summary=rolling_summary
        )
        return result.get("translation", chunk_text)

    # ═══════════════════════════════════════════════════════════
    # TERM GLOSSARY
    # ═══════════════════════════════════════════════════════════

    def _update_glossary(self, chunk_src: str, glossary: dict,
                         protected_terms: set = None) -> dict:
        """Harvest proper-noun candidates from the source chunk.

        ``protected_terms`` (default + user-defined glossary) are preserved
        against the 50-entry eviction budget.
        """
        import re

        protected = protected_terms or set()

        # Skip tokens following sentence-final punctuation — they are sentence onsets.
        candidates = re.findall(
            r'(?<![.!?]\s)\b([A-ZÇĞİÖŞÜ][a-zA-ZçğışöüÇĞİŞÖÜ]{2,}'
            r'(?:\s+[A-ZÇĞİÖŞÜ][a-zA-ZçğışöüÇĞİŞÖÜ]{2,}){0,2})\b',
            chunk_src
        )

        for term in candidates:
            term = term.strip().rstrip(".,;:!?()")
            if not term or len(term) <= 2:
                continue
            # D6: drop single-token candidates that are common connectives.
            head = term.split(" ", 1)[0]
            if " " not in term and head in _GLOSSARY_STOPWORDS:
                continue
            if term not in glossary:
                glossary[term] = None  # No value — let the LLM enforce consistency.

        # B2: 50-entry cap — protected terms are always kept; only dynamically
        # harvested candidates are evicted.
        if len(glossary) > 50:
            keep_protected = {k: v for k, v in glossary.items() if k in protected}
            dynamic_items  = [(k, v) for k, v in glossary.items() if k not in protected]
            slots = max(0, 50 - len(keep_protected))
            keep_protected.update(dict(dynamic_items[-slots:]) if slots else {})
            glossary = keep_protected

        return glossary

    # ═══════════════════════════════════════════════════════════
    # LAYOUT-PRESERVING TRANSLATION — BATCH TRANSLATION HELPER
    # ═══════════════════════════════════════════════════════════

    def _iter_translated_paragraphs_batched(self, paragraphs: list,
                                            src_lang: str, tgt_lang: str,
                                            term_glossary: dict,
                                            protected_terms: set,
                                            progress_cb=None,
                                            preview_cb=None,
                                            msg: dict = None,
                                            total: int = None):
        """Group paragraphs by ``chunk_words`` and translate each group in a batch.

        Batched translation rather than per-paragraph — 36 paragraphs become
        ~5 API calls instead of 36. Paragraphs in a batch are joined with
        ``\\n\\n``, translated as one prompt, then split back. If the boundary
        count does not match, falls back to per-paragraph translation only for
        the offending batch.

        Yields: (paragraph_idx, translated_text) tuples in order.

        progress_cb: (frac, message) — fired at the start and end of each batch.
        preview_cb : (full_text_so_far) — fired after each batch for textbox flushing.
        """
        if msg is None:
            msg = _DEFAULT_MESSAGES
        if total is None:
            total = len(paragraphs)

        # 1) Group translatable paragraphs into batches (chunk_words limit).
        # Each batch: [(global_idx, text), ...].
        batches = []
        cur_batch = []
        cur_words = 0
        for idx, text in enumerate(paragraphs):
            text = text.strip()
            if not text or text.isdigit() or len(text) <= 2:
                # Skip — caller keeps the original; do not include in the batch.
                continue
            words = len(text.split())
            if cur_words + words > self.chunk_words and cur_batch:
                batches.append(cur_batch)
                cur_batch = [(idx, text)]
                cur_words = words
            else:
                cur_batch.append((idx, text))
                cur_words += words
        if cur_batch:
            batches.append(cur_batch)

        if not batches:
            return

        prev_translation = ""
        rolling_summary = ""
        completed = 0

        for bi, batch in enumerate(batches):
            if self._cancel:
                break

            frac = 0.05 + 0.90 * (completed / total) if total else 0.05
            if progress_cb:
                progress_cb(
                    frac,
                    msg["translating"].format(i=completed + 1, total=total),
                )

            batch_texts = [t for _idx, t in batch]
            ctx_paras = []
            # Context: trailing paragraphs from the previous batch.
            if bi > 0:
                prev_batch = batches[bi - 1]
                ctx_paras = [t for _idx, t in prev_batch[-self.overlap_paragraphs:]]

            translated_list = self._translate_batch_with_split(
                batch_texts, ctx_paras, term_glossary,
                src_lang, tgt_lang,
                prev_translation, rolling_summary,
            )

            # translated_list is guaranteed to match batch_texts in length (fallback enforced).
            for (idx, _src), translated in zip(batch, translated_list):
                yield (idx, translated)
                if translated:
                    prev_translation = translated
                completed += 1

            # Glossary update — over the whole batch.
            try:
                joined_src = "\n\n".join(batch_texts)
                term_glossary = self._update_glossary(
                    joined_src, term_glossary, protected_terms=protected_terms
                )
            except Exception:
                pass

            # Rolling summary — driven off the most recent translation.
            if translated_list and translated_list[-1]:
                if progress_cb:
                    progress_cb(
                        frac,
                        msg["summary_update"].format(i=completed, total=total),
                    )
                try:
                    rolling_summary = self.translator.generate_summary(
                        translated_list[-1], rolling_summary
                    )
                except Exception:
                    pass

            # Preview callback — surface live progress to the user.
            if preview_cb:
                try:
                    preview_cb(self._snapshot_preview())
                except Exception:
                    pass

    def _translate_batch_with_split(self, batch_texts: list,
                                     ctx_paras: list, term_glossary: dict,
                                     src_lang: str, tgt_lang: str,
                                     prev_translation: str,
                                     rolling_summary: str) -> list:
        """Translate a batch in a single API call and split the result on ``\\n\\n``.

        When the boundary count is not preserved (the LLM merged or split
        paragraphs), retranslate paragraph-by-paragraph. This keeps a batch
        failure isolated rather than corrupting subsequent batches.

        Returns: list of translations with the SAME length as ``batch_texts``.
        """
        if not batch_texts:
            return []

        # Single paragraph — no split concerns, translate directly.
        if len(batch_texts) == 1:
            try:
                translated = self._translate_chunk_with_context(
                    batch_texts, ctx_paras, term_glossary,
                    src_lang, tgt_lang,
                    prev_translation=prev_translation,
                    rolling_summary=rolling_summary,
                )
                return [(translated or "").strip() or batch_texts[0]]
            except Exception:
                return [batch_texts[0]]

        # Multi-paragraph — join with \n\n and translate in one call.
        try:
            joined_translation = self._translate_chunk_with_context(
                batch_texts, ctx_paras, term_glossary,
                src_lang, tgt_lang,
                prev_translation=prev_translation,
                rolling_summary=rolling_summary,
            )
        except Exception:
            joined_translation = ""

        # Split the result on the boundaries.
        parts = [p.strip() for p in (joined_translation or "").split("\n\n") if p.strip()]

        # Boundary count matches — accept.
        if len(parts) == len(batch_texts):
            return parts

        # Boundary count differs — the LLM failed to preserve the paragraph count.
        # Fallback: retranslate one paragraph at a time. Expensive but rare
        # (the LLM occasionally merges or splits short paragraphs).
        out = []
        local_prev = prev_translation
        for txt in batch_texts:
            try:
                t = self._translate_chunk_with_context(
                    [txt], ctx_paras, term_glossary,
                    src_lang, tgt_lang,
                    prev_translation=local_prev,
                    rolling_summary=rolling_summary,
                )
                t = (t or "").strip() or txt
            except Exception:
                t = txt
            out.append(t)
            local_prev = t
        return out

    def _snapshot_preview(self) -> str:
        """Render the current ``translated_parts`` list as a preview string."""
        return "\n\n".join(p for p in self.translated_parts if p)

    # ═══════════════════════════════════════════════════════════
    # LAYOUT-PRESERVING TRANSLATION — DOCX IN-PLACE
    # ═══════════════════════════════════════════════════════════

    def _translate_docx_inplace(self, file_path: str, src_lang: str, tgt_lang: str,
                                progress_cb=None, user_glossary: dict = None,
                                messages: dict = None, preview_cb=None) -> str:
        """Open the DOCX, translate paragraph + table text in BATCHES, write
        the translations back at the run level, and save to a staging file.

        IMPORTANT: ``Document.paragraphs`` does NOT include paragraphs nested
        inside table cells; tables must be traversed separately.

        Performance: paragraphs are grouped by ``chunk_words``; each group is
        translated in a single API call. 36 paragraphs ≈ 5 calls (not 36).
        """
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "DOCX layout translation requires 'python-docx'.\n"
                "Install: pip install python-docx"
            )

        self._cancel = False

        msg = dict(_DEFAULT_MESSAGES)
        if messages:
            msg.update({k: v for k, v in messages.items() if v})

        if progress_cb:
            progress_cb(0.0, msg["reading"])

        doc = Document(file_path)

        # Collect target paragraph objects + their source text.
        target_paragraphs = []
        for p in doc.paragraphs:
            if p.text.strip():
                target_paragraphs.append(p)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        if p.text.strip():
                            target_paragraphs.append(p)

        if not target_paragraphs:
            raise ValueError(msg["empty_file"])

        total = len(target_paragraphs)
        self.total_chunks = total
        # Pre-allocate translated_parts — supports out-of-order writes.
        self.translated_parts = [p.text.strip() for p in target_paragraphs]
        paragraph_texts = list(self.translated_parts)  # Source-text snapshot.

        if progress_cb:
            progress_cb(
                0.05,
                msg["chunked"].format(paragraphs=total, total=total),
            )

        # Glossary setup.
        if (src_lang or "").strip().lower() in ("turkish", "türkçe", "tr"):
            default_glossary = {
                "Gemma Echo": "Gemma Echo",
                "TÜBİTAK": "TÜBİTAK",
                "Yapay Zeka": "Artificial Intelligence",
                "Derin Öğrenme": "Deep Learning",
                "Makine Öğrenmesi": "Machine Learning",
                "Yapay Sinir Ağları": "Neural Networks",
            }
        else:
            default_glossary = {}
        term_glossary = dict(default_glossary)
        if user_glossary:
            term_glossary.update(user_glossary)
        protected_terms = set(default_glossary) | set(user_glossary or {})

        # Staging file — even on a mid-run cancellation the last persisted state can be saved.
        staging_dir = os.path.join(tempfile.gettempdir(), "gemma_echo_layout")
        os.makedirs(staging_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(file_path))[0]
        self._docx_staging_path = os.path.join(
            staging_dir, f"{base}_translated_{os.getpid()}.docx"
        )

        completed = 0
        save_counter = 0

        # Batch translate — yields arrive after each batch.
        for idx, translated in self._iter_translated_paragraphs_batched(
            paragraph_texts, src_lang, tgt_lang,
            term_glossary, protected_terms,
            progress_cb=progress_cb, preview_cb=preview_cb,
            msg=msg, total=total,
        ):
            if not translated:
                continue
            # Run-level in-place write.
            self._set_paragraph_text_preserve_first_run(target_paragraphs[idx], translated)
            self.translated_parts[idx] = translated
            completed += 1
            save_counter += 1

            # Periodic staging save (every 10 paragraphs).
            if save_counter >= 10:
                try:
                    doc.save(self._docx_staging_path)
                    save_counter = 0
                except Exception:
                    pass

        # Final save.
        try:
            doc.save(self._docx_staging_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save DOCX: {e}")

        if progress_cb:
            if self._cancel:
                progress_cb(
                    min(1.0, 0.05 + 0.90 * (completed / total)) if total else 0.0,
                    msg["cancelled"].format(done=completed, total=total),
                )
            else:
                progress_cb(1.0, msg["done"].format(total=total))

        return self._snapshot_preview()

    @staticmethod
    def _set_paragraph_text_preserve_first_run(paragraph, new_text: str):
        """Replace the paragraph text while preserving the first run's style.

        In DOCX a paragraph can have N runs (each with its own font / bold /
        italic / color). Translation returns a single string; we cannot
        reliably preserve run boundaries because TR<->EN word counts and order
        differ.

        Pragmatic strategy: write the entire translation to the first run and
        empty the text of every subsequent run. This preserves the **dominant
        style** (first run) of the paragraph.

        For paragraphs without runs (rare), a new run is appended.
        """
        if not paragraph.runs:
            paragraph.add_run(new_text)
            return
        first = paragraph.runs[0]
        first.text = new_text
        for r in paragraph.runs[1:]:
            r.text = ""

    # ═══════════════════════════════════════════════════════════
    # LAYOUT-PRESERVING TRANSLATION — PDF TO BLOCKS
    # ═══════════════════════════════════════════════════════════

    def _translate_pdf_to_blocks(self, file_path: str, src_lang: str, tgt_lang: str,
                                 progress_cb=None, user_glossary: dict = None,
                                 messages: dict = None, preview_cb=None) -> str:
        """Split the PDF into blocks via font-size heading detection and BATCH-translate them.

        The previous implementation translated each block individually
        (per-paragraph) — 36 blocks meant 36 API calls. The new approach groups
        blocks by ``chunk_words`` so a single API call handles many — ~5 calls
        per 36 blocks.

        Returns: plain preview text.
        """
        try:
            import pdfplumber  # noqa: F401 — early import for fail-fast behavior.
        except ImportError:
            raise ImportError(
                "Reading PDFs requires the 'pdfplumber' package.\n"
                "Install: pip install pdfplumber"
            )

        self._cancel = False

        msg = dict(_DEFAULT_MESSAGES)
        if messages:
            msg.update({k: v for k, v in messages.items() if v})

        if progress_cb:
            progress_cb(0.0, msg["reading"])

        # 1) Extract structure.
        raw_blocks = self._read_pdf_blocks_structured(file_path)
        if not raw_blocks:
            raise ValueError(msg["empty_pdf"])

        total = len(raw_blocks)
        self.total_chunks = total
        # Pre-allocate: translated_parts and translated_blocks are filled by index.
        self.translated_parts = [b.text for b in raw_blocks]
        translated_blocks: list = [Block(text=b.text, kind=b.kind) for b in raw_blocks]
        paragraph_texts = [b.text for b in raw_blocks]

        if progress_cb:
            progress_cb(
                0.05,
                msg["chunked"].format(paragraphs=total, total=total),
            )

        # 2) Glossary setup.
        if (src_lang or "").strip().lower() in ("turkish", "türkçe", "tr"):
            default_glossary = {
                "Gemma Echo": "Gemma Echo",
                "TÜBİTAK": "TÜBİTAK",
                "Yapay Zeka": "Artificial Intelligence",
                "Derin Öğrenme": "Deep Learning",
                "Makine Öğrenmesi": "Machine Learning",
                "Yapay Sinir Ağları": "Neural Networks",
            }
        else:
            default_glossary = {}
        term_glossary = dict(default_glossary)
        if user_glossary:
            term_glossary.update(user_glossary)
        protected_terms = set(default_glossary) | set(user_glossary or {})

        # 3) Batch translate.
        completed = 0
        for idx, translated in self._iter_translated_paragraphs_batched(
            paragraph_texts, src_lang, tgt_lang,
            term_glossary, protected_terms,
            progress_cb=progress_cb, preview_cb=preview_cb,
            msg=msg, total=total,
        ):
            if not translated:
                continue
            self.translated_parts[idx] = translated
            translated_blocks[idx] = Block(text=translated, kind=raw_blocks[idx].kind)
            completed += 1

        self._pdf_blocks = translated_blocks

        if progress_cb:
            if self._cancel:
                progress_cb(
                    min(1.0, 0.05 + 0.90 * (completed / total)) if total else 0.0,
                    msg["cancelled"].format(done=completed, total=total),
                )
            else:
                progress_cb(1.0, msg["done"].format(total=total))

        return self._snapshot_preview()

    def _read_pdf_blocks_structured(self, file_path: str) -> list:
        """Read the PDF word-by-word, group into line / paragraph blocks and label
        them as h1 / h2 / body based on font size.

        Returns: list[Block] with text + kind fields.
        """
        import pdfplumber
        from collections import Counter

        # 1) Collect every word (with the size attribute) in page order, column-aware.
        all_pages_lines = []   # [[(text, size), ...], ...] — one list-per-page of lines.
        all_sizes = []         # Global pool for body_size estimation.

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # Word-level extraction with size attribute.
                try:
                    words = page.extract_words(extra_attrs=["size"]) or []
                except Exception:
                    words = page.extract_words() or []
                if not words:
                    continue

                # Column split: group left / right separately when applicable.
                width = page.width
                mid_left = width * _TWO_COLUMN_GUTTER_LEFT_RATIO
                mid_right = width * _TWO_COLUMN_GUTTER_RIGHT_RATIO
                overlapping = [w for w in words if w['x0'] < mid_right and w['x1'] > mid_left]
                left = [w for w in words if w['x1'] <= mid_left]
                right = [w for w in words if w['x0'] >= mid_right]

                if (
                    len(overlapping) < len(words) * _TWO_COLUMN_OVERLAP_TOLERANCE
                    and len(left) > _TWO_COLUMN_MIN_WORDS_PER_SIDE
                    and len(right) > _TWO_COLUMN_MIN_WORDS_PER_SIDE
                ):
                    columns = [left, right]
                else:
                    columns = [words]

                page_lines = []
                for col_words in columns:
                    page_lines.extend(self._group_words_into_lines(col_words))

                all_pages_lines.append(page_lines)
                for line in page_lines:
                    for _t, sz in line:
                        if sz and sz > 0:
                            all_sizes.append(round(sz, 1))

        if not all_sizes:
            return []

        # 2) body_size = mode of the font sizes.
        body_size = Counter(all_sizes).most_common(1)[0][0]

        # 3) Group lines into paragraphs within each page based on font-size similarity and vertical proximity.
        blocks: list[Block] = []
        for page_lines in all_pages_lines:
            page_blocks = self._group_lines_into_blocks(page_lines, body_size)
            blocks.extend(page_blocks)

        return blocks

    @staticmethod
    def _group_words_into_lines(words: list) -> list:
        """Group words into lines by Y-coordinate.

        ``words`` is the pdfplumber.extract_words output:
        dict {x0, x1, top, bottom, text, size?}.
        """
        if not words:
            return []
        # Sort by Y-coordinate (top), then by x0 within a line.
        sorted_words = sorted(words, key=lambda w: (round(w['top'], 1), w['x0']))

        lines = []   # [[(text, size), ...], ...]
        current_line = []
        current_top = None
        Y_TOLERANCE = 3.0   # In pixels — two words are "on the same line" when their top values differ by less than this.

        for w in sorted_words:
            top = w['top']
            text = w.get('text', '')
            size = float(w.get('size', 0) or 0)
            if current_top is None or abs(top - current_top) <= Y_TOLERANCE:
                current_line.append((text, size))
                if current_top is None:
                    current_top = top
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [(text, size)]
                current_top = top

        if current_line:
            lines.append(current_line)
        return lines

    @staticmethod
    def _line_dominant_size(line: list) -> float:
        """Return the dominant (modal) font size for a line, word-weighted."""
        from collections import Counter
        sizes = [round(sz, 1) for _t, sz in line if sz and sz > 0]
        if not sizes:
            return 0.0
        return Counter(sizes).most_common(1)[0][0]

    def _group_lines_into_blocks(self, lines: list, body_size: float) -> list:
        """Group consecutive lines into paragraph blocks based on font-size similarity.

        Lines belong to the same block when:
          - Their dominant font size is within ~5% of each other.
          - They share the same "kind" (h1 / h2 / body) as the previous line.
        """
        if not lines:
            return []

        blocks: list[Block] = []
        cur_words: list[str] = []
        cur_kind: str | None = None
        cur_size: float | None = None

        def _close():
            nonlocal cur_words, cur_kind, cur_size
            if cur_words:
                text = " ".join(cur_words).strip()
                # Drop very short page-number-like lines.
                if text and not (len(text) < 4 and text.replace(" ", "").isdigit()):
                    blocks.append(Block(text=text, kind=cur_kind or "body"))
            cur_words = []
            cur_kind = None
            cur_size = None

        for line in lines:
            text = " ".join(t for t, _s in line).strip()
            if not text:
                continue
            size = self._line_dominant_size(line)
            kind = self._classify_block_kind(size, body_size)
            # Long heading lines are almost certainly body — guard against misclassification.
            if kind != "body" and len(text.split()) > _PDF_HEADING_MAX_WORDS:
                kind = "body"

            if cur_kind is None:
                cur_kind = kind
                cur_size = size
                cur_words.append(text)
                continue

            same_kind = (kind == cur_kind)
            size_close = (cur_size and size and abs(size - cur_size) / cur_size < 0.05)

            if same_kind and size_close and kind == "body":
                # Merge body paragraph.
                cur_words.append(text)
            else:
                # Close the current block and open a new one.
                _close()
                cur_kind = kind
                cur_size = size
                cur_words.append(text)

        _close()
        return blocks

    def _classify_block_kind(self, font_size: float, body_size: float) -> str:
        """Classify a block by font size relative to the body size."""
        if not font_size or not body_size:
            return "body"
        if font_size >= body_size * _PDF_H1_RATIO:
            return "h1"
        elif font_size >= body_size * _PDF_H2_RATIO:
            return "h2"
        else:
            return "body"

    # ═══════════════════════════════════════════════════════════
    # LAYOUT-PRESERVING TRANSLATION — BLOCKS TO DOCX
    # ═══════════════════════════════════════════════════════════

    def _render_blocks_to_docx(self, blocks: list[Block], output_path: str):
        """Render the Block list into a styled DOCX (Heading 1/2 / Normal)."""
        try:
            from docx import Document
            from docx.shared import Inches
        except ImportError:
            raise ImportError("Writing DOCX requires the 'python-docx' package.")

        doc = Document()
        # Document margins.
        sections = doc.sections
        if sections:
            section = sections[0]
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1.25)
            section.right_margin = Inches(1.25)

        for block in blocks:
            if block.kind == "h1":
                para = doc.add_heading(block.text, level=1)
            elif block.kind == "h2":
                para = doc.add_heading(block.text, level=2)
            else:
                para = doc.add_paragraph(block.text)
                para.style = "Normal"

        doc.save(output_path)
