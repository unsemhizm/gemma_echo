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
                       progress_cb=None, user_glossary: dict = None) -> str:
        """
        1. Dosyayi oku (PDF / DOCX / TXT)
        2. Paragraflara bol (Smart Preprocessing ile birleştirilmiş gerçek paragraflar)
        3. chunk_words kelimelik bloklara grupla
        4. Her blok icin dual-aspect, rolling summary ve active glossary baglamlariyla cevir
        5. Sonuclari birlestir, output_path'e yaz
        6. Ceviri boyunca term_glossary guncelle

        Args:
            file_path:     Kaynak dosya yolu (.txt / .pdf / .docx)
            src_lang:      Kaynak dil adi (LLM prompt icin, ornek: "Turkish")
            tgt_lang:      Hedef dil adi (ornek: "English")
            output_path:   Cikti dosyasi yolu (None = diske yazmaz)
            progress_cb:   progress_cb(fraction: float, msg: str) — GUI guncelleme
            user_glossary: Kullanıcının özel terim sözlüğü eşlemeleri (örn: {"Yapay Zeka": "AI"}) (Aşama 3)

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

        # Projeye özel ön tanımlı terimler sözlüğü (Active Glossary)
        default_glossary = {
            "Gemma Echo": "Gemma Echo",
            "TÜBİTAK": "TÜBİTAK",
            "Yapay Zeka": "Artificial Intelligence",
            "Derin Öğrenme": "Deep Learning",
            "Makine Öğrenmesi": "Machine Learning",
            "Yapay Sinir Ağları": "Neural Networks"
        }

        # Kullanıcı sözlüğünü ön tanımlı terimler ile harmanla
        term_glossary = dict(default_glossary)
        if user_glossary:
            term_glossary.update(user_glossary)

        translated_parts = []
        context_paras = []
        rolling_summary = ""
        prev_translation = ""

        for i, chunk in enumerate(chunks):
            if self._cancel:
                break

            frac = 0.05 + 0.90 * (i / total)
            if progress_cb:
                progress_cb(frac, f"Bolum {i + 1}/{total} cevriliyor...")

            chunk_text = "\n\n".join(chunk)
            translated = self._translate_chunk_with_context(
                chunk, context_paras, term_glossary, src_lang, tgt_lang,
                prev_translation=prev_translation, rolling_summary=rolling_summary
            )
            translated_parts.append(translated)

            # Dinamik olarak yeni özel isim adayları topla ve sözlüğe ekle (değer=None)
            term_glossary = self._update_glossary(chunk_text, term_glossary)

            # Overlap: bir sonraki chunk'a son N paragraf baglamini tasI
            context_paras = chunk[-self.overlap_paragraphs:]

            # Önceki hedef çeviriyi bir sonraki chunk için kaydet (Coherence)
            prev_translation = translated

            # İlk bölümde ve her 3 bölümde bir yürüyen özeti arka planda güncelle (Overarching Summary)
            if (i + 1) % 3 == 0 or i == 0:
                if progress_cb:
                    progress_cb(frac, f"Bolum {i + 1}/{total} — Doküman özeti güncelleniyor...")
                rolling_summary = self.translator.generate_summary(translated, rolling_summary)

        result = "\n\n".join(translated_parts)

        if output_path and not self._cancel:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)

        if progress_cb and not self._cancel:
            progress_cb(1.0, f"Tamamlandi — {total} bolum cevirildi.")

        return result

    # ═══════════════════════════════════════════════════════════
    # AKILLI METİN VE SAYFA DÜZENİ ANALİZİ (Aşama 1)
    # ═══════════════════════════════════════════════════════════

    def _extract_page_text_layout_aware(self, page) -> str:
        """pdfplumber sayfasından iki sütunlu düzeni algılayarak dikey akışta metin çeker.
        Eğer tek sütunluysa varsayılan şekilde metni çeker.
        """
        try:
            width = page.width
            height = page.height
            words = page.extract_words()
            if not words:
                return page.extract_text() or ""

            # Orta dikey oluk (gutter) analizi [0.44 * width, 0.56 * width]
            mid_start = width * 0.44
            mid_end = width * 0.56

            # Orta dikey oluğa taşan kelime sayısı
            overlapping_words = [w for w in words if w['x0'] < mid_end and w['x1'] > mid_start]

            # Sol ve sağ tarafta kalan kelimeler
            left_words = [w for w in words if w['x1'] <= mid_start]
            right_words = [w for w in words if w['x0'] >= mid_end]

            # Eğer orta bölge temizse ve her iki tarafta da yeterli kelime varsa dikey iki sütun vardır.
            if len(overlapping_words) < len(words) * 0.04 and len(left_words) > 15 and len(right_words) > 15:
                # Sayfayı dikeyde sol ve sağ olarak ikiye kırpıp ayrı ayrı okuyoruz
                left_area = page.crop((0, 0, width * 0.49, height))
                right_area = page.crop((width * 0.51, 0, width, height))

                left_text = left_area.extract_text() or ""
                right_text = right_area.extract_text() or ""
                return left_text + "\n\n" + right_text
            else:
                return page.extract_text() or ""
        except Exception:
            # Hata durumunda varsayılan güvenli okumaya dön
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

    def _read_file(self, path: str) -> list:
        """Dosyayi paragraf listesi olarak doner.

        - .txt  -> Satır sonu temizliği ve akıllı birleştirme ile paragraflara böl
        - .pdf  -> pdfplumber ile sütun duyarlı ve akıllı paragraf birleştirmeli okuma
        - .docx -> python-docx ile paragraph.text
        """
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
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    # Sütun analizli akıllı metin ayıklama
                    page_text = self._extract_page_text_layout_aware(page)
                    # Akıllı satır birleştirme ve paragraf ayrıştırma
                    page_paras = self._reconstruct_paragraphs(page_text)
                    paragraphs.extend(page_paras)
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
                                      src_lang: str, tgt_lang: str,
                                      prev_translation: str = "",
                                      rolling_summary: str = "") -> str:
        """Tek chunk'i overlap, prev_translation, rolling_summary ve active glossary baglamlariyla cevirir."""
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

    def _update_glossary(self, chunk_src: str, glossary: dict) -> dict:
        """Kaynak metinden ozel isim adaylarini toplar."""
        import re

        # Cumle sonu noktalamalarindan SONRA gelen kelimeler cumle baslangiclari (atla)
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
