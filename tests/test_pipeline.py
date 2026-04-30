import sys
import os
import time
import pytest

# Proje kök dizinini yola ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stt.transcriber import Transcriber
from llm.translator import Translator
from tts.synthesizer import Synthesizer
from pipeline.orchestrator import Orchestrator

@pytest.fixture(scope="module")
def orchestrator_instance():
    """Testler icin Orchestrator nesnesini bir kez olusturur."""
    print("\n[PYTEST] Orchestrator baslatiliyor...")
    transcriber = Transcriber()
    translator = Translator()
    synthesizer = Synthesizer()
    orch = Orchestrator(transcriber, translator, synthesizer)
    yield orch

def test_online_pipeline_e2e(orchestrator_instance):
    """
    Sistemi uctan uca online modda test eder.
    audio/Kayıt (3).wav dosyasini kullanarak bastan sona isler.
    """
    audio_path = "audio/Kayıt (3).wav"
    if not os.path.exists(audio_path):
        pytest.skip(f"Test dosyasi bulunamadi: {audio_path}")
        
    print("\n--- ONLINE E2E TEST ---")
    start_time = time.time()
    
    orchestrator_instance.process(audio_path)
    
    total_time = int((time.time() - start_time) * 1000)
    print(f"\n[TEST SONUCU] Online Pipeline Toplam Sure: {total_time}ms")

def test_offline_fallback_e2e(orchestrator_instance):
    """
    Offline/Fallback modunu test eder.
    Sistemi bilerek offline moda cekerek XTTS'in calismasini saglar.
    """
    audio_path = "audio/Kayıt (3).wav"
    if not os.path.exists(audio_path):
        pytest.skip(f"Test dosyasi bulunamadi: {audio_path}")
        
    print("\n--- OFFLINE E2E TEST (FALLBACK SIMULATION) ---")
    
    start_time = time.time()
    
    orchestrator_instance._fallback(audio_path)
    
    total_time = int((time.time() - start_time) * 1000)
    print(f"\n[TEST SONUCU] Offline Pipeline Toplam Sure (Generation + Playback): {total_time}ms")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
