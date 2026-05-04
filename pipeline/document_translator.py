"""
Gemma Echo — Uzun Belge Ceviri Pipeline'i (Sliding Window)

Desteklenen formatlar: TXT, PDF, DOCX
Algoritma: Overlap'li Kayan Pencere + Term Glossary

Kullanim:
    dt = DocumentTranslator(translator, chunk_words=800, overlap_paragraphs=3)
    dt.translate_file("kitap.pdf", src_lang="Turkish", tgt_lang="English",
                      output_path="kitap_en.txt", progress_cb=None)
"""


class DocumentTranslator:
    def __init__(self, translator, chunk_words: int = 800, overlap_paragraphs: int = 3):
        """
        Args:
            translator:           llm/translator.py Translator nesnesi
            chunk_words:          Her parcasin yaklasik kelime sayisi.
                                  API icin 800, local GGUF icin 400 onerilir.
            overlap_paragraphs:   Bir sonraki chunk'a tasinan onceki paragraf sayisi
                                  (anlam surekliligi icin).
        """
        self.translator = translator
        self.chunk_words = chunk_words
        self.overlap_paragraphs = overlap_paragraphs
        self._cancel = False

    # ═══════════════════════════════════════════════════════════
    # IPTAL
    # ═══════════════════════════════════════════════════════════

    def cancel(self):
        """Devam eden ceviriyi iptal eder (thread-safe)."""
        self._cancel = True

    # ═══════════════════════════════════════════════════════════
    # ANA GIRIS NOKTASI
    # ═══════════════════════════════════════════════════════════

    def translate_file(self, file_path: str, src_lang: str = "Turkish",
                       tgt_lang: str = "English", output_path: str = None,
                       progress_cb=None) -> str:
        """
        1. Dosyayi oku (PDF / DOCX / TXT)
        2. Paragraflara bol
        3. chunk_words kelimelik bloklara grupla
        4. Her blok icin overlap baglamiyla cevir
        5. Sonuclari birlestir, output_path'e yaz
        6. Ceviri boyunca term_glossary guncelle

        Args:
            file_path:   Kaynak dosya yolu (.txt / .pdf / .docx)
            src_lang:    Kaynak dil adi (LLM prompt icin, ornek: "Turkish")
            tgt_lang:    Hedef dil adi (ornek: "English")
            output_path: Cikti dosyasi yolu (None = diske yazmaz)
            progress_cb: progress_cb(fraction: float, msg: str) — GUI guncelleme

        Returns:
            Cevirilmis metin (str)
        """
        self._cancel = False

        if progress_cb:
            progress_cb(0.0, "Dosya okunuyor...")

        paragraphs = self._read_file(file_path)
        if not paragraphs:
            raise ValueError("Dosya bos veya okunamadi.")

        chunks = self._chunk_paragraphs(paragraphs)
        total = len(chunks)

        if progress_cb:
            progress_cb(0.05, f"{len(paragraphs)} paragraf — {total} bolum olusturuldu.")

        translated_parts = []
        term_glossary = {}
        context_paras = []

        for i, chunk in enumerate(chunks):
            if self._cancel:
                break

            frac = 0.05 + 0.90 * (i / total)
            if progress_cb:
                progress_cb(frac, f"Bolum {i + 1}/{total} cevriliyor...")

            chunk_text = "\n\n".join(chunk)
            translated = self._translate_chunk_with_context(
                chunk, context_paras, term_glossary, src_lang, tgt_lang
            )
            translated_parts.append(translated)

            # Glossary guncelle — sadece kaynak metinden ozel isim adaylari topla
            term_glossary = self._update_glossary(chunk_text, term_glossary)

            # Overlap: bir sonraki chunk'a son N paragraf baglamini tasI
            context_paras = chunk[-self.overlap_paragraphs:]

        result = "\n\n".join(translated_parts)

        if output_path and not self._cancel:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)

        if progress_cb and not self._cancel:
            progress_cb(1.0, f"Tamamlandi — {total} bolum cevirildi.")

        return result

    # ═══════════════════════════════════════════════════════════
    # DOSYA OKUMA
    # ═══════════════════════════════════════════════════════════

    def _read_file(self, path: str) -> list:
        """Dosyayi paragraf listesi olarak doner.

        - .txt  -> "\\n\\n" ile bol
        - .pdf  -> pdfplumber ile sayfa -> paragraf
        - .docx -> python-docx ile paragraph.text
        """
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""

        if ext == "txt":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            # Hem \n\n hem de tek \n ile ayrilmis paragraflar
            raw = text.replace("\r\n", "\n")
            paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
            return paragraphs

        elif ext == "pdf":
            try:
                import pdfplumber
            except ImportError:
                raise ImportError(
                    "PDF okuma icin 'pdfplumber' paketi gereklidir.\n"
                    "Kurulum: pip install pdfplumber"
                )
            paragraphs = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    for para in page_text.split("\n\n"):
                        para = para.strip()
                        if para:
                            paragraphs.append(para)
            return paragraphs

        elif ext == "docx":
            try:
                from docx import Document
            except ImportError:
                raise ImportError(
                    "DOCX okuma icin 'python-docx' paketi gereklidir.\n"
                    "Kurulum: pip install python-docx"
                )
            doc = Document(path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return paragraphs

        else:
            raise ValueError(
                f"Desteklenmeyen format: .{ext}\n"
                "Desteklenenler: TXT, PDF, DOCX"
            )

    # ═══════════════════════════════════════════════════════════
    # PARAGRAF GRUPLAMA (Kayan Pencere)
    # ═══════════════════════════════════════════════════════════

    def _chunk_paragraphs(self, paragraphs: list) -> list:
        """Paragraflari chunk_words'u gecmeyecek sekilde grupla.

        Bir paragrafi ortadan kesmez — tam paragraf sinirinda keser.
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
    # CHUNK CEVIRİSİ (Overlap + Glossary)
    # ═══════════════════════════════════════════════════════════

    def _translate_chunk_with_context(self, chunk: list, context_paras: list,
                                      term_glossary: dict,
                                      src_lang: str, tgt_lang: str) -> str:
        """Tek chunk'i overlap baglamiyla cevirir.

        Duzeltme (Fix 1): Onceki yaklasimdaki "kaba kuvvet" hatasi giderildi.
        - context_paras -> translator'in yerlesik context= parametresine gecildi.
          Boylece overlap metni cevrilecek metne karismiyor; LLM bunu "bellek"
          olarak kullanir, dogrudan ceviri yapmaz.
        - Sadece TERM RULES (tutarlilik listesi) chunk metnine on-ek olarak
          eklenir. Bunu LLM sistem talimatinin parcasi olarak goruyor.
        """
        chunk_text = "\n\n".join(chunk)

        # TERM RULES: sadece kaynak dildeki ozel isimler — kisa ve net
        if term_glossary:
            terms_str = ", ".join(list(term_glossary.keys())[:20])
            text_to_translate = (
                f"[CONSISTENCY TERMS — translate these proper nouns consistently throughout: "
                f"{terms_str}]\n\n{chunk_text}"
            )
        else:
            text_to_translate = chunk_text

        # Overlap baglamini translator'in kendi context mekanizmasina ver.
        # Bu, LLM'e "onceki cumleleri HATIRLAT" seklinde iletilir — cevirme.
        result = self.translator.translate(
            text_to_translate,
            context=context_paras,   # overlap: onceki N paragraf, referans icin
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_name=src_lang,
            tgt_name=tgt_lang,
        )
        return result.get("translation", chunk_text)

    # ═══════════════════════════════════════════════════════════
    # TERIM SOZLUGU (Term Glossary)
    # ═══════════════════════════════════════════════════════════

    def _update_glossary(self, chunk_src: str, glossary: dict) -> dict:
        """Kaynak metinden ozel isim adaylarini toplar.

        Duzeltme (Fix 2): Onceki zip() tabanli indeks eslestirmesi tamamen
        kaldirildi. Turkce sondan eklemeli, Ingilizce on-yüklemli bir dil;
        kelime sirasi ceviri sonrasi korunmaz. zip ile yapilan "3. kelime -> 3.
        kelime" eslemesi tamamen yanlis terimler uretiyordu.

        Yeni yaklasim:
          - Sadece KAYNAK metni tara (hedef metin kullanilmaz).
          - Regex ile cumle basi olmayan buyuk harfli kelime oklerini topla.
          - Bunlar "tutarli cevrilmesi istenen adaylar" olarak glossary'de
            saklanir (deger=None — LLM kendi tutarliligini saglar).
          - LLM'e bir sonraki chunk'ta "bunlari tutarli cevir" olarak iletilir.
        """
        import re

        # Cumle sonu noktalamalarindan SONRA gelen kelimeler cumle baslangiclari
        # (bunlari atla). Geri kalan buyuk harfli 1-3 kelimelik obekler aday.
        # Turkce buyuk harfleri de kapsayacak sekilde Unicode aware pattern.
        candidates = re.findall(
            r'(?<![.!?]\s)\b([A-ZÇĞİÖŞÜ][a-zA-ZçğışöüÇĞİŞÖÜ]{2,}'
            r'(?:\s+[A-ZÇĞİÖŞÜ][a-zA-ZçğışöüÇĞİŞÖÜ]{2,}){0,2})\b',
            chunk_src
        )

        for term in candidates:
            term = term.strip().rstrip(".,;:!?()")
            if term and len(term) > 2 and term not in glossary:
                glossary[term] = None  # Deger yok — LLM tutarlilik saglar

        # 50 entry siniri
        if len(glossary) > 50:
            glossary = dict(list(glossary.items())[-50:])

        return glossary
