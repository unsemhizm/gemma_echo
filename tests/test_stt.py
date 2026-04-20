import sys
import os
# Proje kök dizinine erişim sağlamak için path ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stt.transcriber import Transcriber

def test_batch_transcription():
    ts = Transcriber()
    # audio/ klasöründeki dosyaların listesi (Kayıt.wav, Kayıt (2).wav vb.)
    test_files = [f for f in os.listdir("audio") if f.endswith(".wav")]
    
    print(f"\nToplam {len(test_files)} dosya test ediliyor...")
    
    latencies = []
    for file in test_files:
        path = os.path.join("audio", file)
        result = ts.transcribe(path)
        latencies.append(result["duration_ms"])
        print(f"✓ {file}: {result['duration_ms']:.2f} ms")

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nOrtalama Latency: {avg_latency:.2f} ms")
    
    # DoD (Kabul Kriterleri) Kontrolü
    assert avg_latency < 400, f"Latency çok yüksek! Ortalama: {avg_latency:.2f}ms (hedef: <400ms)"
    assert len(test_files) >= 5, f"En az 5 test dosyası olmalı! Mevcut: {len(test_files)}"
    print("\n✅ GÜN 1 TESTLERİ BAŞARIYLA GEÇTİ!")

if __name__ == "__main__":
    test_batch_transcription()