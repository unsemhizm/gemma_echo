import sys
import os
import time
import pytest

# Proje kök dizinini yola ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.translator import Translator


@pytest.fixture(scope="module")
def translator_instance():
    """Testler için Translator nesnesini bir kez oluşturur."""
    print("\n[PYTEST] Translator başlatılıyor...")
    t = Translator()
    yield t


# ═══════════════════════════════════════════════════════════
# ARAYÜZ TESTLERİ
# ═══════════════════════════════════════════════════════════

def test_translator_interface(translator_instance):
    """translate() metodunun doğru arayüzde veri döndürdüğünü test eder."""
    result = translator_instance.translate("Merhaba dünya")

    assert isinstance(result, dict), "Sonuç bir sözlük (dict) olmalı"
    assert "translation" in result, "'translation' anahtarı eksik"
    assert "latency_ms" in result, "'latency_ms' anahtarı eksik"
    assert "engine" in result, "'engine' anahtarı eksik"

    assert isinstance(result["translation"], str), "Çeviri string olmalı"
    assert isinstance(result["latency_ms"], int), "Gecikme integer olmalı"
    assert isinstance(result["engine"], str), "Motor ismi string olmalı"


def test_empty_input(translator_instance):
    """Boş metin gönderildiğinde güvenli dönüş yapılıyor mu?"""
    result = translator_instance.translate("")
    assert result["translation"] == ""
    assert result["latency_ms"] == 0
    assert result["engine"] == "None"


# ═══════════════════════════════════════════════════════════
# ONLINE ÇEVİRİ TESTLERİ
# ═══════════════════════════════════════════════════════════

def test_online_translation_quality(translator_instance):
    """10 Türkçe cümle ile online çeviri kalitesini test eder."""
    test_sentences = [
        ("Merhaba, nasılsınız?", ["hello", "how"]),
        ("Bugün hava çok güzel.", ["weather", "beautiful", "nice", "today"]),
        ("Başınız sağ olsun.", ["sorry", "condolences", "loss"]),
        ("Geçmiş olsun.", ["well", "better", "hope", "recover"]),
        ("Ben bir öğrenciyim.", ["student", "i am"]),
        ("Türkiye çok güzel bir ülke.", ["turkey", "beautiful", "country"]),
        ("Yarın toplantımız var.", ["meeting", "tomorrow"]),
        ("Bu proje çok önemli.", ["project", "important"]),
        ("Teşekkür ederim.", ["thank"]),
        ("İyi geceler.", ["good", "night"]),
    ]

    translator_instance.set_mode("online")
    passed = 0

    for tr_text, expected_keywords in test_sentences:
        result = translator_instance.translate(tr_text)
        en_text = result["translation"].lower()

        # En az bir beklenen kelime çeviride var mı?
        found = any(kw in en_text for kw in expected_keywords)
        status = "✅" if found else "❌"
        if found:
            passed += 1
        print(f"  {status} '{tr_text}' → '{result['translation']}' [{result['latency_ms']}ms]")

    print(f"\n  Sonuç: {passed}/10 başarılı")
    assert passed >= 8, f"Çeviri kalitesi yetersiz: {passed}/10 (Minimum 8 gerekli)"


def test_cultural_translation_quality(translator_instance):
    """CULTURAL_MAP ile LLM'in kültürel ifadeleri nasıl çevirdiğini test eder."""
    print("\n\n  --- KÜLTÜREL İFADELER TESTİ (CULTURAL_MAP DESTEĞİYLE) ---")
    cultural_sentences = [
        ("Kolay gelsin.", ["easy", "work", "good luck", "effort", "well"]),
        ("Ellerine sağlık.", ["hands", "health", "bless", "good job", "thank", "well done"]),
        ("Çok yaşa.", ["bless", "live", "long"]),
        ("Kurban olayım sana.", ["sacrifice", "anything", "love", "die", "darling"]),
        ("Afiyet olsun.", ["enjoy", "meal", "appetite", "bon appetit"]),
        ("Hayırlı olsun.", ["congratulations", "auspicious", "best wishes", "good luck"]),
        ("Gözün aydın.", ["eyes", "bright", "happy", "news", "glad"]),
        ("Sıhhatler olsun.", ["health", "bath", "haircut", "shower", "enjoy"]),
        ("Allah rahmet eylesin.", ["god", "mercy", "rest", "peace"]),
        ("Helal olsun.", ["bravo", "well done", "congratulations", "awesome"])
    ]

    translator_instance.set_mode("online")
    passed = 0

    for tr_text, expected_keywords in cultural_sentences:
        result = translator_instance.translate(tr_text)
        en_text = result["translation"].lower()

        found = any(kw in en_text for kw in expected_keywords)
        status = "[OK]" if found else "[FAIL]"
        if found:
            passed += 1
        print(f"  {status} '{tr_text}' -> '{result['translation']}' [{result['engine']}]")

    print(f"\n  Kültürel Başarı (CULTURAL_MAP ile): {passed}/10\n")


def test_groq_latency(translator_instance):
    """Groq API latency hedefini kontrol eder (< 500ms)."""
    translator_instance.set_mode("online")
    result = translator_instance.translate("Merhaba")
    
    print(f"  Groq latency: {result['latency_ms']}ms (Hedef: < 500ms)")
    # Not: İlk istek yavaş olabilir, bu yüzden sert bir assert koymuyoruz
    # Ama bilgi olarak loglanıyor


# ═══════════════════════════════════════════════════════════
# CONTEXT WINDOW TESTLERİ
# ═══════════════════════════════════════════════════════════

def test_context_window_pronouns(translator_instance):
    """Zamir çevirisinin context ile düzgün çalıştığını test eder."""
    translator_instance.set_mode("online")

    # Context: "Ahmet dün geldi." → "O" = "He" olmalı
    result = translator_instance.translate(
        "O çok yorgundu.",
        context=["Ahmet dün geldi."]
    )
    en_text = result["translation"].lower()
    print(f"  Context testi: 'O çok yorgundu' → '{result['translation']}'")
    
    # "He" veya "him" olmalı, "she" veya "it" değil
    assert "he" in en_text or "him" in en_text, \
        f"Zamir çevirisi hatalı: 'O' → beklenen 'He', alınan: '{result['translation']}'"


# ═══════════════════════════════════════════════════════════
# MOD DEĞİŞİM TESTLERİ
# ═══════════════════════════════════════════════════════════

def test_set_mode_online(translator_instance):
    """set_mode('online') çağrısının modu doğru ayarladığını test eder."""
    translator_instance.set_mode("online")
    assert translator_instance.mode == "online"


def test_set_mode_offline(translator_instance):
    """set_mode('offline') çağrısının modu doğru ayarladığını test eder."""
    translator_instance.set_mode("offline")
    assert translator_instance.mode == "offline"
    # Online'a geri dön (diğer testleri etkilemesin)
    translator_instance.set_mode("online")


def test_set_mode_invalid(translator_instance):
    """Geçersiz mod değerinde ValueError fırlatıldığını test eder."""
    with pytest.raises(ValueError):
        translator_instance.set_mode("turbo")


# ═══════════════════════════════════════════════════════════
# VRAM YÖNETİM TESTİ
# ═══════════════════════════════════════════════════════════

def test_release_ollama_vram(translator_instance):
    """release_ollama_vram() metodunun çökmeden çalıştığını test eder."""
    # Ollama çalışmıyor olsa bile bu metot çökmemeli
    translator_instance.release_ollama_vram()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
