"""
Gemma Echo — Uzun Belge Ceviri Pipeline'i (Sliding Window)

Desteklenen formatlar: TXT, PDF, DOCX
Algoritma: Overlap'li Kayan Pencere + Term Glossary

Kullanim:
    dt = DocumentTranslator(translator, chunk_words=800, overlap_paragraphs=3)
    dt.translate_file("kitap.pdf", src_lang="Turkish", tgt_lang="English",
                      output_path="kitap_en.txt", progress_cb=None)
"""

# ===========================================================
# Sabitler — PDF sütun analizi eşikleri (D1: sihirli sayılar tek yerde)
# ===========================================================
_TWO_COLUMN_GUTTER_LEFT_RATIO  = 0.44
_TWO_COLUMN_GUTTER_RIGHT_RATIO = 0.56
_TWO_COLUMN_OVERLAP_TOLERANCE  = 0.04   # gutter'a taşan kelime oranı max
_TWO_COLUMN_MIN_WORDS_PER_SIDE = 15     # her iki sütunda min. kelime

# Türkçe "nokta" ile biten ama cümle sonu olmayan kısaltmalar (D3 false-positive azaltıcı)
_TR_ABBREVS = {
    "md.", "bkz.", "sn.", "prof.", "doç.", "dr.", "av.", "vs.", "vb.",
    "örn.", "yay.", "bas.", "çev.", "haz.", "ed.", "a.g.e.", "i.ö.", "i.s.",
    "tc.", "t.c.", "öğr.", "gör.", "aş.", "ltd.",
    "mr.", "mrs.", "ms.", "jr.", "sr.", "e.g.", "i.e.", "etc.", "st.", "no.",
}

# Glossary'de özel isim adayı toplarken filtrelenmesi gereken sık cümle başı sözcükleri (D6)
_GLOSSARY_STOPWORDS = {
    "Bu", "Şu", "O", "Bunlar", "Şunlar", "Onlar",
    "Şimdi", "Sonra", "Önce", "Bugün", "Dün", "Yarın",
    "Fakat", "Ancak", "Ama", "Yine", "Aynı", "Henüz", "Aslında",
    "Çünkü", "Yani", "Belki", "Gerçekten", "İşte",
    "The", "This", "That", "These", "Those", "There", "Here",
    "However", "Therefore", "Moreover", "Although", "Because",
}

# Varsayılan progress mesajları (B4: GUI 'messages' parametresiyle override edebilir)
_DEFAULT_MESSAGES = {
    "reading":        "Dosya okunuyor...",
    "chunked":        "{paragraphs} paragraf — {total} bolum olusturuldu.",
    "translating":    "Bolum {i}/{total} cevriliyor...",
    "summary_update": "Bolum {i}/{total} — Doküman özeti güncelleniyor...",
    "done":           "Tamamlandi — {total} bolum cevirildi.",
    "cancelled":      "İptal edildi — {done}/{total} bolum cevirilmişti.",
    "empty_pdf":      (
        "PDF metin içermiyor (taranmış görüntü / OCR'siz olabilir). "
        "Bu sürümde OCR desteklenmiyor."
    ),
    "empty_file":     "Dosya bos veya okunamadi.",
    "unsupported":    "Desteklenmeyen format: .{ext}\nDesteklenenler: TXT, PDF, DOCX",
}


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
        # B1/B3: çeviri ilerledikçe toplandığı yer; iptal/exception sonrası partial_result()
        # ile dışarıdan erişilebilir.
        self.translated_parts: list[str] = []
        # GUI tarafının "done/total" gösterimi için (toast, status bar)
        self.total_chunks: int = 0

    def partial_result(self) -> str:
        """O ana kadar tamamlanmış chunk çevirilerini birleştirip döner.

        İptal veya exception durumunda GUI tarafının kullanıcıya kısmi sonuç göstermesi
        ve 'Kaydet' butonunu etkinleştirmesi için kullanılır.
        """
        return "\n\n".join(self.translated_parts)

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
                       progress_cb=None, user_glossary: dict = None,
                       messages: dict = None) -> str:
        """
        1. Dosyayi oku (PDF / DOCX / TXT)
        2. Paragraflara bol (Smart Preprocessing ile birleştirilmiş gerçek paragraflar)
        3. chunk_words kelimelik bloklara grupla
        4. Her blok icin dual-aspect, rolling summary ve active glossary baglamlariyla cevir
        5. Sonuclari birlestir, output_path'e yaz (iptal halinde de partial yazilir)
        6. Ceviri boyunca term_glossary guncelle

        Args:
            file_path:     Kaynak dosya yolu (.txt / .pdf / .docx)
            src_lang:      Kaynak dil adi (LLM prompt icin, ornek: "Turkish")
            tgt_lang:      Hedef dil adi (ornek: "English")
            output_path:   Cikti dosyasi yolu (None = diske yazmaz)
            progress_cb:   progress_cb(fraction: float, msg: str) — GUI guncelleme
            user_glossary: Kullanıcının özel terim sözlüğü (örn: {"Yapay Zeka": "AI"})
            messages:      Lokalize progress mesaj template'leri (B4 — i18n).
                           None ise _DEFAULT_MESSAGES kullanılır. GUI tarafı t() ile
                           doldurulmuş bir dict geçebilir.

        Returns:
            Cevirilmis metin (str). İptal halinde o ana kadar çevirilen partial sonuç.
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

        # D5: default_glossary kaynağı TR ise yararı var; diğer dillerde boş başlat.
        # Kaynağı TR olmayan bir dokümanda "Yapay Zeka -> AI" kuralı prompt'u kirletir.
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

        # B2: default + user terimleri eviction'dan korunur.
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

            # Dinamik özel isim adaylarını topla; protected terimleri evict etme.
            term_glossary = self._update_glossary(
                chunk_text, term_glossary, protected_terms=protected_terms
            )

            # Overlap: bir sonraki chunk'a son N paragraf baglamini tasi.
            context_paras = chunk[-self.overlap_paragraphs:]
            prev_translation = translated

            # D4: Rolling summary ekstra LLM çağrısıdır (her 3 chunk'ta bir).
            # 31 chunk'lık bir kitap için ~10 ekstra çağrı = ek API maliyeti.
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
                    # Özet güncellenemedi — ana çeviri akışı bozulmasın.
                    pass

        result = self.partial_result()

        # B1: İptal edilse bile o ana kadarki kısmi sonuç hem disk'e hem caller'a döner.
        if output_path and result:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result)
            except Exception:
                # Disk yazma hatası çeviri sonucunu çöpe atmamalı — caller'a result dön.
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
    # AKILLI METİN VE SAYFA DÜZENİ ANALİZİ (Aşama 1)
    # ═══════════════════════════════════════════════════════════

    def _extract_page_text_layout_aware(self, page) -> str:
        """pdfplumber sayfasından iki sütunlu düzeni algılayarak dikey akışta metin çeker.
        Eğer tek sütunluysa varsayılan şekilde metni çeker.

        D1: Eşikler modul-seviyesi sabitlerden okunuyor (test/tuning kolaylığı).
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
        """Satır sonlarındaki gereksiz yeni satır (\\n) karakterlerini akıllıca birleştirir,
        satır sonu heceleme tirelerini (-) temizler ve gerçek paragraflar inşa eder.
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

            # Sayfa numarası veya kısa tekrarlı üstbilgi/altbilgi filtreleme (gürültü engelleme)
            if line.isdigit() or (len(line) < 6 and any(k in line.lower() for k in ["page", "sayfa", "ch.", "bölüm"])):
                continue

            # Satır sonu tire birleştirme (heceleme)
            has_hyphen = False
            if line.endswith("-") and len(line) > 1:
                if line[-2].isalpha():
                    line = line[:-1].rstrip()
                    has_hyphen = True

            if current_para:
                prev_line = current_para[-1]
                # Önceki satır cümle bitirici bir karakterle mi bitti?
                ends_sentence = prev_line[-1] in {".", "?", "!", ":"} if prev_line else False
                # D3: Kısaltma noktası ("Dr.", "Prof.", "Md." vb.) cümle sonu sayılmaz.
                if ends_sentence and prev_line.endswith("."):
                    last_token = prev_line.rsplit(None, 1)[-1].lower()
                    if last_token in _TR_ABBREVS:
                        ends_sentence = False
                # Şu anki satır küçük harfle mi başlıyor?
                starts_lowercase = line[0].islower() if line else False

                if has_hyphen:
                    # Tire birleşimi: Boşluk bırakmadan birleştir
                    current_para[-1] = prev_line + line
                elif not ends_sentence or starts_lowercase:
                    # Aynı paragrafın devamı: Boşlukla birleştir
                    current_para.append(line)
                else:
                    # Yeni bir paragraf başlangıcı
                    paragraphs.append(" ".join(current_para))
                    current_para = [line]
            else:
                current_para = [line]

        if current_para:
            paragraphs.append(" ".join(current_para))

        # Paragrafları temizle, çoklu boşlukları erit ve çok kısa gürültü satırlarını ele
        cleaned_paras = []
        for para in paragraphs:
            para = re.sub(r'\s+', ' ', para).strip()
            if para and len(para) > 8:
                cleaned_paras.append(para)

        return cleaned_paras

    # ═══════════════════════════════════════════════════════════
    # DOSYA OKUMA
    # ═══════════════════════════════════════════════════════════

    def _read_file(self, path: str, messages: dict = None) -> list:
        """Dosyayi paragraf listesi olarak doner.

        - .txt  -> Satır sonu temizliği ve akıllı birleştirme ile paragraflara böl
        - .pdf  -> pdfplumber ile sütun duyarlı ve akıllı paragraf birleştirmeli okuma
        - .docx -> python-docx ile paragraflar + tablo hücreleri
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
                    "PDF okuma icin 'pdfplumber' paketi gereklidir.\n"
                    "Kurulum: pip install pdfplumber"
                )
            paragraphs = []
            raw_chars = 0
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = self._extract_page_text_layout_aware(page)
                    raw_chars += len(page_text or "")
                    page_paras = self._reconstruct_paragraphs(page_text)
                    paragraphs.extend(page_paras)
            # B6: pdfplumber metin çıkaramadıysa büyük olasılıkla taranmış PDF.
            # "Dosya bos" gibi yanıltıcı mesaj yerine OCR açıklaması ver.
            if not paragraphs and raw_chars < 20:
                raise ValueError(msg["empty_pdf"])
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
            # B8: Tablo hücrelerindeki metin de çeviriye dahil edilsin.
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
                                      src_lang: str, tgt_lang: str,
                                      prev_translation: str = "",
                                      rolling_summary: str = "",
                                      chunk_text: str = None) -> str:
        """Tek chunk'i overlap, prev_translation, rolling_summary ve active glossary baglamlariyla cevirir.

        chunk_text opsiyonel: caller "\\n\\n".join(chunk) zaten yaptıysa tekrar etmemek için (B7).
        """
        if chunk_text is None:
            chunk_text = "\n\n".join(chunk)

        # TERM RULES: translate terms exactly as specified (Aşama 3)
        if term_glossary:
            rules = []
            for k, v in term_glossary.items():
                if v:
                    rules.append(f"{k} -> {v}")
                else:
                    rules.append(k)
            terms_str = ", ".join(rules[:30])  # limit to 30 terms to keep prompt clean
            text_to_translate = (
                f"[STRICT GLOSSARY RULES — translate these terms exactly as specified: "
                f"{terms_str}]\n\n{chunk_text}"
            )
        else:
            text_to_translate = chunk_text

        # Overlap baglamini ve diger gelismis parametreleri translator'a ilet
        result = self.translator.translate(
            text_to_translate,
            context=context_paras,   # overlap: onceki N paragraf, referans icin
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_name=src_lang,
            tgt_name=tgt_lang,
            prev_translation=prev_translation,
            rolling_summary=rolling_summary
        )
        return result.get("translation", chunk_text)

    # ═══════════════════════════════════════════════════════════
    # TERIM SOZLUGU (Term Glossary)
    # ═══════════════════════════════════════════════════════════

    def _update_glossary(self, chunk_src: str, glossary: dict,
                         protected_terms: set = None) -> dict:
        """Kaynak metinden ozel isim adaylarini toplar.

        protected_terms: 50-entry eviction'ında korunacak terimler (default + user glossary).
        """
        import re

        protected = protected_terms or set()

        # Cumle sonu noktalamalarindan SONRA gelen kelimeler cumle baslangiclari (atla)
        candidates = re.findall(
            r'(?<![.!?]\s)\b([A-ZÇĞİÖŞÜ][a-zA-ZçğışöüÇĞİŞÖÜ]{2,}'
            r'(?:\s+[A-ZÇĞİÖŞÜ][a-zA-ZçğışöüÇĞİŞÖÜ]{2,}){0,2})\b',
            chunk_src
        )

        for term in candidates:
            term = term.strip().rstrip(".,;:!?()")
            if not term or len(term) <= 2:
                continue
            # D6: tek-kelime adayı stopword listesindeyse atla.
            head = term.split(" ", 1)[0]
            if " " not in term and head in _GLOSSARY_STOPWORDS:
                continue
            if term not in glossary:
                glossary[term] = None  # Deger yok — LLM tutarlilik saglar

        # B2: 50 entry sınırı — protected terimleri her zaman koru, sadece dinamik
        # toplanan adayları evict et.
        if len(glossary) > 50:
            keep_protected = {k: v for k, v in glossary.items() if k in protected}
            dynamic_items  = [(k, v) for k, v in glossary.items() if k not in protected]
            slots = max(0, 50 - len(keep_protected))
            keep_protected.update(dict(dynamic_items[-slots:]) if slots else {})
            glossary = keep_protected

        return glossary
