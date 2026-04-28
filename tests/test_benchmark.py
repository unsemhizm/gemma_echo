import os
import time
import pytest
import torch # Cihaz kontrolü için eklendi
from stt.transcriber import Transcriber
from llm.translator import Translator
from tts.synthesizer import Synthesizer

@pytest.fixture(scope="module")
def components():
    print("\n[BENCHMARK] Bilesenler yukleniyor...")
    transcriber = Transcriber()
    translator = Translator()
    synthesizer = Synthesizer()
    return transcriber, translator, synthesizer

def run_pipeline_test(transcriber, translator, synthesizer, audio_path, iteration):
    if not os.path.exists(audio_path):
        print(f"[HATA] Dosya bulunamadi: {audio_path}")
        return None

    print(f"\n[{iteration}. TUR] Ses isleniyor: {audio_path}")
    total_start = time.time()

    # 1. STT (Whisper)
    stt_start = time.time()
    stt_result = transcriber.transcribe(audio_path)
    text_tr = stt_result.get("text", "")
    stt_latency = int((time.time() - stt_start) * 1000)

    # 2. LLM (Ceviri)
    llm_result = translator.translate(text_tr)
    text_en = llm_result.get("translation", "")
    llm_latency = llm_result.get("latency_ms", 0)
    engine = llm_result.get("engine", "Bilinmiyor")

    # 3. TTS (Seslendirme)
    tts_start = time.time()
    # synthesizer.speak artik latency donuyor olmali
    tts_latency = synthesizer.speak(text_en)
    if tts_latency is None:
        tts_latency = int((time.time() - tts_start) * 1000)

    total_latency = int((time.time() - total_start) * 1000)

    # Donanim bilgisini guvenli sekilde al
    # WhisperModel icinde modelin nerede oldugunu Transcriber icindeki self.device'dan alabiliriz
    # Eger Transcriber'da yoksa varsayilan olarak "Device" yazalim
    dev_info = getattr(transcriber, "device", "Unknown")

    print("-" * 60)
    print(f"[{'STT (' + dev_info + ')':<20}] -> Sure: {stt_latency:>5} ms | Metin: '{text_tr}'")
    print(f"[{'LLM (' + engine + ')':<20}] -> Sure: {llm_latency:>5} ms | Ceviri: '{text_en}'")
    print(f"[{'TTS':<20}] -> Sure: {tts_latency:>5} ms | Durum: Tamamlandi")
    print(f"TOPLAM E2E SURESI        : {total_latency} ms")
    print("-" * 60)
    
    return total_latency

def test_benchmark_online(components):
    transcriber, translator, synthesizer = components
    # Dosya isimlerinin dogrulugundan emin ol (Kucuk/Buyuk harf duyarlidir)
    audio_files = ["audio/kayıt.wav", "audio/kayıt (2).wav", "audio/kayıt (3).wav"]

    print("\n" + "="*70)
    print("   ONLINE PIPELINE LATENCY BENCHMARK (BULUT MODU - 3 ORNEK)")
    print("="*70)

    translator.set_mode("online")
    synthesizer.set_mode("online")
    translator.release_ollama_vram()
    # Transcriber'in online'da GPU'da oldugundan emin olalim (Eger offline'dan gelmissek)
    if hasattr(transcriber, "load_model_gpu"):
        transcriber.load_model_gpu()
    transcriber.device = "GPU"

    for i, audio_path in enumerate(audio_files, 1):
        run_pipeline_test(transcriber, translator, synthesizer, audio_path, i)

def test_benchmark_offline(components):
    transcriber, translator, synthesizer = components
    audio_files = ["audio/kayıt.wav", "audio/kayıt (2).wav", "audio/kayıt (3).wav"]

    print("\n" + "="*70)
    print("   OFFLINE PIPELINE LATENCY BENCHMARK (YEREL MOD - 3 ORNEK)")
    print("="*70)

    translator.set_mode("offline")
    synthesizer.set_mode("offline")
    
    # Hot-Swap tetikle
    if hasattr(transcriber, "switch_to_cpu"):
        transcriber.switch_to_cpu()
        transcriber.device = "CPU"
    else:
        print("[BILGI] switch_to_cpu henuz eklenmemis.")
        transcriber.device = "GPU"

    for i, audio_path in enumerate(audio_files, 1):
        run_pipeline_test(transcriber, translator, synthesizer, audio_path, i)