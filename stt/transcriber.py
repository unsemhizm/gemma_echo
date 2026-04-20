import time
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size="small", device="cuda", compute_type="int8"):
        """
        ekran kartının gücünü kullanmak için 'cuda' ve hız için 'int8' seçildi.
        """
        print(f"Whisper modeli yükleniyor ({model_size})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path):
        """
        Sesi metne çevirir ve geçen süreyi hesaplar.
        """
        start_time = time.time()
        
        # 'language="tr"' ekleyerek modelin dili tahmin etme yükünü alıyoruz.

        segments, info = self.model.transcribe(audio_path, beam_size=3, language="tr")
        
        text = ""
        for segment in segments:
            text += segment.text

        duration_ms = (time.time() - start_time) * 1000
        return {"text": text.strip(), "duration_ms": duration_ms}

# Bağımsız test bloğu
# if __name__ == "__main__":
#     ts = Transcriber()
    
#     # audio/ klasöründeki ses dosyanın adını buraya yaz.
#     test_file = "audio/Kayıt.wav" 
    
#     try:
#         print(f"İşleniyor: {test_file}")
#         result = ts.transcribe(test_file)
#         print(f"\n--- SONUÇ ---")
#         print(f"Metin: {result['text']}")
#         print(f"Gecikme (Latency): {result['duration_ms']:.2f} ms")
#     except Exception as e:
#         print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    ts = Transcriber()
    
    test_files = [
        "audio/Kayıt.wav",
        "audio/Kayıt (2).wav",
        "audio/Kayıt (3).wav",
        "audio/Kayıt (4).wav",
        "audio/Kayıt (5).wav"
    ]

    for file in test_files:
        try:
            print(f"\nİşleniyor: {file}")
            result = ts.transcribe(file)
            
            print(f"--- SONUÇ ---")
            print(f"Metin: {result['text']}")
            print(f"Gecikme: {result['duration_ms']:.2f} ms")

        except Exception as e:
            print(f"{file} için hata: {e}")