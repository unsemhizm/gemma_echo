import sys
import os
import time
import pytest

# Proje kök dizinini yola ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.orchestrator import Orchestrator

@pytest.fixture(scope="module")
def orchestrator_instance():
    """Testler için Orchestrator nesnesini bir kez oluşturur."""
    print("\n[PYTEST] Orchestrator başlatılıyor...")
    orch = Orchestrator()
    yield orch

def test_online_pipeline_e2e(orchestrator_instance):
    """
    Sistemi uçtan uca online modda test eder.
    audio/kayıt (3).wav dosyasını kullanarak baştan sona işler.
    """
    audio_path = "audio/kayıt (3).wav"
    if not os.path.exists(audio_path):
        pytest.skip(f"Test dosyası bulunamadı: {audio_path}")
        
    print("\n--- ONLINE E2E TEST ---")
    start_time = time.time()
    
    # Process audio will run STT, LLM, TTS in online mode
    orchestrator_instance.process_audio(audio_path)
    
    total_time = int((time.time() - start_time) * 1000)
    print(f"\n[TEST SONUCU] Online Pipeline Toplam Süre: {total_time}ms")

def test_offline_fallback_e2e(orchestrator_instance):
    """
    Offline/Fallback modunu test eder.
    Sistemi bilerek offline moda çekerek XTTS'in çalışmasını sağlar.
    """
    audio_path = "audio/kayıt (3).wav"
    if not os.path.exists(audio_path):
        pytest.skip(f"Test dosyası bulunamadı: {audio_path}")
        
    print("\n--- OFFLINE E2E TEST (FALLBACK SIMULATION) ---")
    
    # Fallback mode metodunu direkt çağırarak internet kesilmiş gibi davranıyoruz
    start_time = time.time()
    
    orchestrator_instance.fallback_mode(audio_path)
    
    total_time = int((time.time() - start_time) * 1000)
    print(f"\n[TEST SONUCU] Offline Pipeline Toplam Süre (Generation + Playback): {total_time}ms")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
