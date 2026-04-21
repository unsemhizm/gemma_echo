import sys
import os

# Proje kök dizinini Python path'e ekle — böylece 'llm.translator' modülü bulunabilir.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.translator import Translator

# ─────────────────────────────────────────────────────────────────────────────
# TEST_SUITE — Test cümleleri listesi
#
# Her satır bir 3'lü demetidir: (türkçe_girdi, beklenen_keyword, açıklama)
#
# beklenen_keyword kuralları:
#   - Tek kelime  → İngilizce çeviride o kelime geçmeli.        Örn: "loss"
#   - "A|B" formu → A VEYA B geçmesi yeterli.                   Örn: "He|She"
#
# Test kategorileri:
#   1-10  : Gün 2 temel DoD paketi (kültürel deyimler + düz cümleler)
#   11-12 : Cinsiyet bağlamı — "O" zamiri Ayşe bağlamıyla "She" olmalı
#   13-14 : Mesleki önyargı — Doktor bağlamıyla tutarlı zamir kontrolü
#   15-16 : Joker fiil testi — "çekmek" fiiline göre doğru anlam seçimi
#   17    : Devrik cümle    — STT'den gelebilecek alışılmadık sözdizimi
#   18-19 : Cansız bağlam   — "Bilgisayar" bağlamıyla "it" zamir bağlaması
#   20    : Teknik terim    — Yazılım jargonunun bozulmadan geçmesi
# ─────────────────────────────────────────────────────────────────────────────
TEST_SUITE = [
    ("Başınız sağ olsun.",              "loss",         "Taziye deyimi"),
    ("Geçmiş olsun.",                   "well",         "İyileşme dileği"),
    ("Kolay gelsin.",                   "easy",         "Çalışma dileği"),
    ("Afiyet olsun.",                   "meal",         "Yemek dileği"),
    ("Allah razı olsun.",               "bless",        "Teşekkür deyimi"),
    ("Ellerine sağlık.",                "done",         "Takdir deyimi"),
    ("Ahmet projeyi bitirdi.",          "project",      "Düz cümle"),
    ("Yarın toplantı var.",             "meeting",      "Gündelik cümle"),
    ("Bu raporu okumak zor.",           "difficult",    "Sıfat içeren cümle"),
    ("Hava bugün çok güzel.",           "weather",      "Hava durumu"),
    ("Ayşe çok yorgun.",                "Ayşe",         "Bağlam hazırlığı"),
    ("O şimdi uyuyor.",                 "She",          "Context: Ayşe -> She eşleşmesi"),
    ("Doktor odaya girdi.",             "Doctor",       "Bağlam hazırlığı"),
    ("O hastayı muayene etti.",         "He|She",       "Bağlam: Doktor (Nötr veya tutarlı zamir)"),
    ("Bu fotoğrafı ben çektim.",        "took",         "Joker fiil: Fotoğraf çekmek"),
    ("Çok acı çekti.",                  "suffered",     "Joker fiil: Acı çekmek"),
    ("Görmedim ben onu dün.",           "see",          "Devrik yapı: Fiil başta"),
    ("Bilgisayar bozulmuş.",            "computer",     "Bağlam hazırlığı"),
    ("Onu tamir etmem lazım.",          "repair|it",    "Context: Computer -> It eşleşmesi"),
    ("Backend tarafında sıkıntı var.",  "backend",      "Yazılım terminolojisi testi"),
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT_CHAIN — Bağlam zinciri
#
# Anahtar: TEST_SUITE'deki cümle numarası (1-indexed)
# Değer   : O cümleye gönderilecek önceki cümleler listesi
#
# Nasıl çalışır:
#   translator.translate(text, context=[...]) çağrısıyla LLM'e "önceki konuşma"
#   bilgisi iletilir. Bu sayede "O" zamiri doğru kişi/nesneye bağlanabilir.
#
# Bu sözlükte bulunmayan cümle numaraları boş bağlamla (context=[]) gönderilir.
# ─────────────────────────────────────────────────────────────────────────────
CONTEXT_CHAIN = {
    12: ["Ayşe çok yorgun."],       # "O" → Ayşe → She
    14: ["Doktor odaya girdi."],     # "O" → Doktor → He/She
    19: ["Bilgisayar bozulmuş."],   # "Onu" → Bilgisayar → it
}


def _passes(keyword: str, text_en: str) -> bool:
    """
    Keyword kontrolünü yapar. '|' ile ayrılmış alternatif keywordleri destekler.

    Örnekler:
      _passes("loss",   "I am sorry for your loss.") → True
      _passes("He|She", "She examined the patient.")  → True
      _passes("He|She", "The doctor examined.")       → False
    """
    return any(k.lower() in text_en.lower() for k in keyword.split("|"))


def test_translation_quality():
    """
    Gün 2 genişletilmiş çeviri kalite testi.

    DoD (Definition of Done) kriterleri:
      1. Hiçbir çeviride 'source == error' olmamalı.
      2. LLM kaynaklı çevirilerin ortalama gecikmesi < 700ms olmalı.
      3. Her cümlede beklenen keyword İngilizce çeviride geçmeli (anlamsal kayma yok).
    """
    tr = Translator()

    # İlk çağrı GPU/model warm-up için kullanılır; latency ölçümüne dahil edilmez.
    print("\nWarm-up turu (latency'e dahil değil)...")
    tr.translate("Merhaba")

    print("\n" + "=" * 65)
    print(f"Gemma Echo — Genişletilmiş Test Paketi ({len(TEST_SUITE)} Cümle)")
    print("=" * 65)

    latencies = []  # Sadece LLM kaynaklı süreler (cultural_map anında döndüğü için ölçülmez)
    failures  = []  # Keyword kontrolünden geçemeyen cümleler
    results   = []  # Tüm çeviri sonuçları (assert'lar bu listeden okur, tekrar LLM çağrısı yapılmaz)

    for i, (text_tr, keyword, aciklama) in enumerate(TEST_SUITE, 1):
        # Bu cümle için önceden tanımlanmış bağlam varsa al, yoksa boş liste kullan.
        context = CONTEXT_CHAIN.get(i, [])
        result  = tr.translate(text_tr, context=context)

        text_en  = result["text_en"]
        duration = result["duration_ms"]
        source   = result["source"]

        # Görsel formatlama: cultural_map'ten gelen anlık çeviriler "--" gösterir.
        lat_str  = f"{duration:6.1f}ms" if duration > 0 else "  --  "
        tag      = f"[{source.upper():<14}]"
        passed   = _passes(keyword, text_en)
        status   = "✅" if passed else "❌"

        print(f"  {status} {i:02d}. {tag} {lat_str} | {text_tr}")
        if context:
            # Bağlam gönderildiyse ekrana yansıt — debugging için faydalı.
            print(f"         ⬆ bağlam : {context}")
        print(f"         → {text_en}")

        results.append(result)

        # cultural_map ve empty kaynakları LLM latency'e dahil değil.
        if source not in ("cultural_map", "empty"):
            latencies.append(duration)

        if not passed:
            failures.append((text_tr, text_en, keyword, aciklama))

    # ── DoD Assert'ları ───────────────────────────────────────────────────────
    # Tüm veriler yukarıda toplanan listelerden okunur — sıfır ekstra LLM çağrısı.
    print("\n" + "─" * 65)

    # 1. Hata kontrolü: hiçbir çeviri "error" durumunda olmamalı.
    error_count = sum(1 for r in results if r["source"] == "error")
    assert error_count == 0, f"❌ {error_count} cümlede çeviri hatası var!"

    # 2. Latency kontrolü: LLM ortalama gecikmesi Gün 2 DoD hedefinin altında olmalı.
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"  Ortalama LLM Latency : {avg_latency:.1f}ms  (hedef: <700ms)")
        assert avg_latency < 700, (
            f"❌ Latency çok yüksek! Ortalama: {avg_latency:.1f}ms (hedef: <700ms)"
        )
    else:
        print("  Tüm cümleler cultural_map'ten geldi — LLM latency ölçülemedi.")

    # 3. Doğruluk özeti: kaç cümle keyword kontrolünden geçti?
    passed_count = len(TEST_SUITE) - len(failures)
    print(f"  Doğruluk              : {passed_count}/{len(TEST_SUITE)} cümle")

    # Başarısız cümlelerin detaylı raporu — neyin neden geçmediğini gösterir.
    if failures:
        print("\n  Başarısız cümleler:")
        for t, e, k, a in failures:
            print(f"    ❌ [{a}]")
            print(f"       TR : {t}")
            print(f"       EN : {e}")
            print(f"       Beklenen keyword: '{k}'")

    # Anlamsal kayma veya bağlam hataları varsa testi başarısız say.
    assert len(failures) == 0, (
        f"❌ {len(failures)} cümlede anlamsal kayma veya bağlam hatası!"
    )

    print(f"\n✅ TÜM TESTLER GEÇTİ! ({len(TEST_SUITE)}/{len(TEST_SUITE)} cümle)")


if __name__ == "__main__":
    test_translation_quality()
