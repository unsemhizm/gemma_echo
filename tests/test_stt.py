import sys
import os
import pytest

# Proje kök dizinini yola ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stt.transcriber import Transcriber

@pytest.fixture(scope="module")
def transcriber_instance():
    """Testler için Transcriber nesnesini bir kez oluşturur."""
    print("\n[PYTEST] Transcriber başlatılıyor...")
    ts = Transcriber()
    yield ts
    # Testler bitince VRAM'i temizle
    ts.switch_to_cpu()

def test_transcriber_interface(transcriber_instance):
    """transcribe() metodunun doğru arayüzde veri döndürdüğünü test eder."""
    audio_path = "audio/kayıt (3).wav" # Kendi dosya ismine göre güncelle
    
    if not os.path.exists(audio_path):
        pytest.skip(f"Test dosyası bulunamadı: {audio_path}")

    result = transcriber_instance.transcribe(audio_path)

    # 15:30-16:30 bloğundaki arayüz kontrolü
    assert isinstance(result, dict), "Sonuç bir sözlük (dict) olmalı"
    assert "text" in result, "'text' anahtarı eksik"
    assert "duration_ms" in result, "'duration_ms' anahtarı eksik"
    assert "no_speech_prob" in result, "'no_speech_prob' anahtarı eksik"
    
    assert isinstance(result["text"], str), "Metin string olmalı"
    assert isinstance(result["duration_ms"], int), "Süre integer olmalı"
    assert isinstance(result["no_speech_prob"], float), "Olasılık float olmalı"

def test_confidence_filtering_logic(transcriber_instance):
    """DoD kriterlerindeki no_speech_prob sınırlarını test eder."""
    audio_path = "audio/kayıt (2).wav"
    if not os.path.exists(audio_path):
        pytest.skip(f"Test dosyası bulunamadı: {audio_path}")

    result = transcriber_instance.transcribe(audio_path)
    
    # İnsan sesi olduğu için no_speech_prob düşük olmalı (< 0.6)
    assert result["no_speech_prob"] < 0.6, "Gürültü filtresi çok yüksek değer döndürdü!"