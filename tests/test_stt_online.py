import sys
import os
import time

# Proje kök dizinini yola ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stt.transcriber import Transcriber

def test_multiple_audio():
    print("[TEST] Transcriber başlatılıyor (GPU)...")
    ts = Transcriber()
    
    # Test edilecek 5 dosya 
    audio_files = [
        "audio/Kayıt.wav",
        "audio/Kayıt (2).wav",
        "audio/Kayıt (3).wav",
        "audio/Kayıt (4).wav",
        "audio/Kayıt (5).wav"
    ]

    print("\n" + "="*50)
    print("   STT ONLINE STRES VE HIZ TESTİ BAŞLIYOR")
    print("="*50 + "\n")

    for file_path in audio_files:
        if not os.path.exists(file_path):
            print(f"[UYARI] {file_path} bulunamadı, lütfen dosyayı oluşturun.")
            continue

        print(f"> İşleniyor: {file_path}")
        start_time = time.time()
        
        # Transkripsiyonu çalıştır
        result = ts.transcribe(file_path)
        
        end_time = time.time()

        latency_ms = int((end_time - start_time) * 1000)
        duration_ms = result['duration_ms']
        
        # Real-Time Factor (RTF): İşlem süresinin sesin uzunluğuna oranı. 
        # 1.0'dan küçük olması gerçek zamanlı çalıştığını gösterir.
        rtf = latency_ms / duration_ms if duration_ms > 0 else 0

        print(f"  Metin: {result['text']}")
        print(f"  Ses Süresi: {duration_ms} ms | Gecikme (Latency): {latency_ms} ms")
        print(f"  RTF: {rtf:.2f} (Hedef: < 1.0) | Güven: {1 - result['no_speech_prob']:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    test_multiple_audio()