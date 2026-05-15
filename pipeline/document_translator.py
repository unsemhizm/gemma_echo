"""
Gemma Echo — Uzun Belge Ceviri Pipeline'i (Sliding Window)

Desteklenen formatlar: TXT, PDF, DOCX
Algoritma: Overlap'li Kayan Pencere + Term Glossary

Kullanim:
    dt = DocumentTranslator(translator, chunk_words=800, overlap_paragraphs=3)
    dt.translate_file("kitap.pdf", src_lang="Turkish", tgt_lang="English",
                      output_path="kitap_en.txt", progress_cb=None)

Layout korumali kullanim:
    dt.translate_file_layout("form.docx", layout_output_path="form_en.docx",
                              src_lang="Turkish", tgt_lang="English")
    # DOCX -> DOCX: run-level in-place ceviri, font/tablo/baslik korunur
    # PDF  -> DOCX: font-size bazli baslik tespiti + Heading stilleri
"""

import os
import shutil
import tempfile
from dataclasses import dataclass, field

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

# PDF basligi tespit esikleri (font_size oran tabanli — sayfa basina rolatif).
# Body size = doc'un ana metin font'u (mode of sizes). Heading'ler bunun ustunde olur.
_PDF_H1_RATIO = 1.40   # body * 1.40+ -> H1 (ornek: body 11pt, h1 15pt+)
_PDF_H2_RATIO = 1.18   # body * 1.18+ -> H2 (ornek: body 11pt, h2 13-14pt)
# H3'u ayri yapmiyoruz — bold + h2-arasi orta gri bolge LLM cevirisinde kayboluyordu;
# pratikte h1/h2/body 3'lu yeterli.

# Bir bloku "baslik" kabul etmek icin maksimum kelime sayisi
# (cok uzun H1 paragrafi muhtemelen body — yanlis siniflandirmadan korur)
_PDF_HEADING_MAX_WORDS = 20


@dataclass
class Block:
    """PDF okumasinda yapilandirilmis blok — text + ne tur (heading/body) oldugu.

    DOCX kaynaklarda kullanilmaz cunku DOCX'te zaten paragraph.style.name var;
    DOCX dogrudan in-place run-level cevrilir (full layout korumasi).

    PDF'te ise font-size'a bakarak heuristik etiketleme yapariz; sonra
    Word'e yazarken Heading 1/2/Normal stilleri uygulanir.
    """
    text: str
    kind: str = "body"   # "h1" | "h2" | "body"


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

        # ── Layout korumali ceviri (DOCX/PDF -> stilli DOCX) ──
        # `translate_file_layout` cagrildiginda doldurulur, save_layout_docx ile tuketilir.
        # DOCX kaynak: gecici dosya yolu (in-place ceviri sonucu)
        # PDF kaynak: stil etiketli bloklar — kayit aninda Heading/Normal docx'e cevrilir.
        self._docx_staging_path: str | None = None
        self._pdf_blocks: list[Block] | None = None
        self._layout_source_kind: str | None = None   # "docx" | "pdf" | None

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
    # LAYOUT KORUMALI CEVIRI (DOCX -> DOCX, PDF -> stilli DOCX)
    # ═══════════════════════════════════════════════════════════

    def translate_file_layout(self, file_path: str, src_lang: str = "Turkish",
                              tgt_lang: str = "English", progress_cb=None,
                              user_glossary: dict = None,
                              messages: dict = None,
                              preview_cb=None) -> str:
        """Layout korumali ceviri — public entrypoint.

        Akis:
          - DOCX kaynak  : Source doc'u acar, paragraf+tablo run'larini in-place
                           cevirir; font/tablo/baslik/margin korunur. Sonuc gecici
                           bir .docx dosyasina kaydedilir (`_docx_staging_path`).
          - PDF kaynak   : Font-size ile baslik tespiti yapar, blok listesi
                           uretir, her bloku cevirir. `_pdf_blocks`'a yazilir.
          - TXT kaynak   : Layout kavrami yok; klasik translate_file'a duser.

        GUI cevirinin sonunda `save_layout_docx(path)` cagirmali; staged dosya
        kopyalanir veya bloklardan stilli docx insa edilir.

        Returns: Duz onizleme metni (textbox icin).
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
        """Layout korumali ceviri sonucunu .docx olarak yazar.

        - DOCX kaynak  : staged dosyayi output_path'e kopyala (full layout)
        - PDF kaynak   : bloklari Heading/Normal stilli yeni docx'e insa et
        - Diger        : ValueError (caller duz `_save_docx`'e dusmeli)

        Iptal edilmis cevirinin partial sonucu da yazilabilir; cevrilmemis
        paragraflar Turkce kalir.
        """
        if self._layout_source_kind == "docx" and self._docx_staging_path:
            if not os.path.exists(self._docx_staging_path):
                raise FileNotFoundError("Staged DOCX bulunamadi.")
            shutil.copyfile(self._docx_staging_path, output_path)
            return

        if self._layout_source_kind == "pdf" and self._pdf_blocks:
            self._render_blocks_to_docx(self._pdf_blocks, output_path)
            return

        raise ValueError(
            "Layout korumali cikti yok. translate_file_layout once cagrilmali "
            "veya kaynak DOCX/PDF degil."
        )

    def _reset_layout_staging(self):
        """Onceki staging dosyalarini temizler (idempotent)."""
        if self._docx_staging_path and os.path.exists(self._docx_staging_path):
            try:
                os.remove(self._docx_staging_path)
            except OSError:
                pass
        self._docx_staging_path = None
        self._pdf_blocks = None
        self._layout_source_kind = None

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

    # ═══════════════════════════════════════════════════════════
    # LAYOUT KORUMALI CEVIRI — BATCH TRANSLATION HELPER
    # ═══════════════════════════════════════════════════════════

    def _iter_translated_paragraphs_batched(self, paragraphs: list,
                                            src_lang: str, tgt_lang: str,
                                            term_glossary: dict,
                                            protected_terms: set,
                                            progress_cb=None,
                                            preview_cb=None,
                                            msg: dict = None,
                                            total: int = None):
        """Paragraflari chunk_words'a gore gruplandirip toplu cevirir.

        Per-paragraf cevirisi yerine batch — 36 paragraf = 36 API cagrisi yerine
        ~5 cagri. \\n\\n delimiter ile birlestir, cevir, geri bol. Sayi uyusmazsa
        problemli batch icin tek tek paragrafa fallback.

        Yields: (paragraph_idx, translated_text) tuplelari sirayla.

        progress_cb: (frac, message) — her batch basinda + sonunda
        preview_cb : (full_text_so_far) — her batch sonra textbox'a flush icin
        """
        if msg is None:
            msg = _DEFAULT_MESSAGES
        if total is None:
            total = len(paragraphs)

        # 1) Cevrilebilir paragraflari batch'lere grupla (chunk_words limiti)
        # Her batch: [(global_idx, text), ...]
        batches = []
        cur_batch = []
        cur_words = 0
        for idx, text in enumerate(paragraphs):
            text = text.strip()
            if not text or text.isdigit() or len(text) <= 2:
                # Skip — caller orijinali kullanir; batch'e koyma
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
            # Onceki batch'in son paragraflari context
            if bi > 0:
                prev_batch = batches[bi - 1]
                ctx_paras = [t for _idx, t in prev_batch[-self.overlap_paragraphs:]]

            translated_list = self._translate_batch_with_split(
                batch_texts, ctx_paras, term_glossary,
                src_lang, tgt_lang,
                prev_translation, rolling_summary,
            )

            # translated_list batch_texts ile ayni uzunlukta garanti (fallback yapildi)
            for (idx, _src), translated in zip(batch, translated_list):
                yield (idx, translated)
                if translated:
                    prev_translation = translated
                completed += 1

            # Glossary update — batch'in tamamindan
            try:
                joined_src = "\n\n".join(batch_texts)
                term_glossary = self._update_glossary(
                    joined_src, term_glossary, protected_terms=protected_terms
                )
            except Exception:
                pass

            # Rolling summary — son cevirilen ile
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

            # Preview callback — kullaniciya canli ilerleme goster
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
        """Bir batch'i tek API cagrisinda cevirir, \\n\\n ile boler.

        Boundary tutmazsa (LLM paragraf sayisini koruyamadi) — paragraflari
        tek tek tekrar cevirir. Bu sekilde bir batch'in toplu basarisizligi
        diger batch'leri etkilemez.

        Returns: batch_texts ile AYNI uzunlukta cevirilmis liste.
        """
        if not batch_texts:
            return []

        # Tek paragraf — split sorunu yok, direkt cevir
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

        # Coklu paragraf — \n\n ile birlestir, tek API cagrisinda cevir
        try:
            joined_translation = self._translate_chunk_with_context(
                batch_texts, ctx_paras, term_glossary,
                src_lang, tgt_lang,
                prev_translation=prev_translation,
                rolling_summary=rolling_summary,
            )
        except Exception:
            joined_translation = ""

        # Sonucu boundary'lere bol
        parts = [p.strip() for p in (joined_translation or "").split("\n\n") if p.strip()]

        # Sayi tutuyorsa OK
        if len(parts) == len(batch_texts):
            return parts

        # Sayi tutmuyor — LLM paragraf sayisini koruyamadi.
        # Fallback: paragraflari tek tek tekrar cevir. Bu pahalidir ama nadir
        # olur (kisa paragraflarda LLM bazen birlestiriyor/boluyor).
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
        """Mevcut translated_parts listesini onizleme metnine cevirir."""
        return "\n\n".join(p for p in self.translated_parts if p)

    # ═══════════════════════════════════════════════════════════
    # LAYOUT KORUMALI CEVIRI — DOCX IN-PLACE
    # ═══════════════════════════════════════════════════════════

    def _translate_docx_inplace(self, file_path: str, src_lang: str, tgt_lang: str,
                                progress_cb=None, user_glossary: dict = None,
                                messages: dict = None, preview_cb=None) -> str:
        """DOCX'i acar, paragraf+tablo metnini BATCH bazinda cevirir, run-level
        in-place yazar, gecici dosyaya kaydeder.

        DIKKAT: `Document.paragraphs` tablo hucrelerindeki paragraflari ICERMEZ;
        tablolari ayri dolasmak gerekir.

        Performance: Paragraflar chunk_words limitine gore gruplandirilir; her
        grup tek API cagrisinda cevrilir. 36 paragraf ~= 5 cagri (yerine 36).
        """
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "DOCX layout cevirisi icin 'python-docx' gerekli.\n"
                "Kurulum: pip install python-docx"
            )

        self._cancel = False

        msg = dict(_DEFAULT_MESSAGES)
        if messages:
            msg.update({k: v for k, v in messages.items() if v})

        if progress_cb:
            progress_cb(0.0, msg["reading"])

        doc = Document(file_path)

        # Cevrilecek paragraf objelerini ve metinlerini topla
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
        # Pre-allocate translated_parts — out-of-order yazim icin
        self.translated_parts = [p.text.strip() for p in target_paragraphs]
        paragraph_texts = list(self.translated_parts)  # kaynak metin snapshot

        if progress_cb:
            progress_cb(
                0.05,
                msg["chunked"].format(paragraphs=total, total=total),
            )

        # Glossary kurulum
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

        # Staging dosyasi — yarim iptal edilirse de son hali kaydedilebilir.
        staging_dir = os.path.join(tempfile.gettempdir(), "gemma_echo_layout")
        os.makedirs(staging_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(file_path))[0]
        self._docx_staging_path = os.path.join(
            staging_dir, f"{base}_translated_{os.getpid()}.docx"
        )

        completed = 0
        save_counter = 0

        # Batch translate — her batch sonra yields gelir
        for idx, translated in self._iter_translated_paragraphs_batched(
            paragraph_texts, src_lang, tgt_lang,
            term_glossary, protected_terms,
            progress_cb=progress_cb, preview_cb=preview_cb,
            msg=msg, total=total,
        ):
            if not translated:
                continue
            # Run-level in-place yazim
            self._set_paragraph_text_preserve_first_run(target_paragraphs[idx], translated)
            self.translated_parts[idx] = translated
            completed += 1
            save_counter += 1

            # Periodik staging save (her 10 paragrafda bir)
            if save_counter >= 10:
                try:
                    doc.save(self._docx_staging_path)
                    save_counter = 0
                except Exception:
                    pass

        # Final save
        try:
            doc.save(self._docx_staging_path)
        except Exception as e:
            raise RuntimeError(f"DOCX kaydedilemedi: {e}")

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
        """Paragrafin metnini degistirir, ilk run'in stilini korur.

        DOCX'te bir paragrafin N tane run'i olabilir (her run farkli font/bold/
        italic/color). Ceviri tek metin geri verir; run sinirlarini elden gelse
        bile koruyamayiz (TR<->EN kelime sayisi/sirasi farkli).

        Pragmatik strateji: ilk run'a tum cevirilen metni yaz, kalan run'larin
        text'ini bosalt. Boylece paragrafin **en yaygin stili** (ilk run) korunur.

        Run'siz paragraflar (nadiren) icin yeni run eklenir.
        """
        if not paragraph.runs:
            paragraph.add_run(new_text)
            return
        first = paragraph.runs[0]
        first.text = new_text
        for r in paragraph.runs[1:]:
            r.text = ""

    # ═══════════════════════════════════════════════════════════
    # LAYOUT KORUMALI CEVIRI — PDF TO BLOCKS
    # ═══════════════════════════════════════════════════════════

    def _translate_pdf_to_blocks(self, file_path: str, src_lang: str, tgt_lang: str,
                                 progress_cb=None, user_glossary: dict = None,
                                 messages: dict = None, preview_cb=None) -> str:
        """PDF'yi font-size ile baslik tespiti yaparak bloklara boler, BATCH cevirir.

        Onceki implementasyon her bloku tek tek (per-paragraf) ceviriyordu —
        36 blok = 36 API cagrisi. Yeni: chunk_words limitine gore gruplandir,
        her grup tek API cagrisinda cevrilsin. ~5 cagri / 36 blok.

        Returns: Duz onizleme metni.
        """
        try:
            import pdfplumber  # noqa: F401  — sadece import erken hata icin
        except ImportError:
            raise ImportError(
                "PDF okuma icin 'pdfplumber' paketi gereklidir.\n"
                "Kurulum: pip install pdfplumber"
            )

        self._cancel = False

        msg = dict(_DEFAULT_MESSAGES)
        if messages:
            msg.update({k: v for k, v in messages.items() if v})

        if progress_cb:
            progress_cb(0.0, msg["reading"])

        # 1) Yapi cikar
        raw_blocks = self._read_pdf_blocks_structured(file_path)
        if not raw_blocks:
            raise ValueError(msg["empty_pdf"])

        total = len(raw_blocks)
        self.total_chunks = total
        # Pre-allocate: translated_parts ve translated_blocks idx'le doldurulacak
        self.translated_parts = [b.text for b in raw_blocks]
        translated_blocks: list = [Block(text=b.text, kind=b.kind) for b in raw_blocks]
        paragraph_texts = [b.text for b in raw_blocks]

        if progress_cb:
            progress_cb(
                0.05,
                msg["chunked"].format(paragraphs=total, total=total),
            )

        # 2) Glossary kurulum
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

        # 3) Batch cevir
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
        """PDF'i kelime-bazli okur, satir-paragraf grupla, font-size ile h1/h2/body etiketler.

        Returns: list[Block] - text + kind alanlariyla
        """
        import pdfplumber
        from collections import Counter

        # 1) Tum kelimeleri (font_size attr'siyla) topla — sayfa sirasiyla, sutun-duyarli
        all_pages_lines = []   # [[(text, size), ...], ...] - her sayfa icin liste-satir
        all_sizes = []         # body_size hesabi icin global

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # extract_words size attribute ile birlikte kelime-bazli okuma
                try:
                    words = page.extract_words(extra_attrs=["size"]) or []
                except Exception:
                    words = page.extract_words() or []
                if not words:
                    continue

                # Sutun ayrimi: gerekirse sol-sag ayri ayri grup yap
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

        # 2) Body size = en sik gorulen font boyutu
        body_size = Counter(all_sizes).most_common(1)[0][0]

        # 3) Satirlari paragraflara grupla — ayni page icinde, yakin font_size ve dusey mesafe
        blocks: list[Block] = []
        for page_lines in all_pages_lines:
            page_blocks = self._group_lines_into_blocks(page_lines, body_size)
            blocks.extend(page_blocks)

        return blocks

    @staticmethod
    def _group_words_into_lines(words: list) -> list:
        """Kelimeleri y-koordinatina gore satirlara grupla.

        Words pdfplumber.extract_words ciktisi: dict {x0, x1, top, bottom, text, size?}
        """
        if not words:
            return []
        # Y-koordinatina gore sirala (top), sonra ayni satirsa x0
        sorted_words = sorted(words, key=lambda w: (round(w['top'], 1), w['x0']))

        lines = []   # [[(text, size), ...], ...]
        current_line = []
        current_top = None
        Y_TOLERANCE = 3.0   # piksel — ayni satirsa top farki bu kadarin altinda

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
        """Bir satirdaki en sik gorulen font boyutu (kelime-tabanli)."""
        from collections import Counter
        sizes = [round(sz, 1) for _t, sz in line if sz and sz > 0]
        if not sizes:
            return 0.0
        return Counter(sizes).most_common(1)[0][0]

    def _group_lines_into_blocks(self, lines: list, body_size: float) -> list:
        """Ardisik satirlari font-size benzerligine gore paragraflara grupla.

        Aynı blok icindeki satirlar:
          - Dominant font size birbirine ~%5 yakin
          - Bir onceki satirla ayni "kind" (h1/h2/body)
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
                # Sayfa numarasi/header benzeri cok kisa satirlar — at
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
            # Heading'lerde cok uzun satirlari yanlis siniflandirmadan koru
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
                # Body paragrafini birlestir
                cur_words.append(text)
            else:
                # Yeni blok
                _close()
                cur_kind = kind
                cur_size = size
                cur_words.append(text)

        _close()
        return blocks

    def _classify_block_kind(self, font_size: float, body_size: float) -> str:
        """Font size'a göre blok türünü belirler."""
        if not font_size or not body_size:
            return "body"
        if font_size >= body_size * _PDF_H1_RATIO:
            return "h1"
        elif font_size >= body_size * _PDF_H2_RATIO:
            return "h2"
        else:
            return "body"

    # ═══════════════════════════════════════════════════════════
    # LAYOUT KORUMALI CEVIRI — BLOCKS TO DOCX
    # ═══════════════════════════════════════════════════════════

    def _render_blocks_to_docx(self, blocks: list[Block], output_path: str):
        """Blokları Heading/Normal stilli DOCX'e yazar."""
        try:
            from docx import Document
            from docx.shared import Inches
        except ImportError:
            raise ImportError("DOCX yazma icin 'python-docx' paketi gereklidir.")

        doc = Document()
        # Resmi belge ayarları
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
