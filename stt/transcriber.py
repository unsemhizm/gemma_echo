import os
import sys

# DLL yollarını PATH'e en başta ekle — importlardan ÖNCE
venv_path = sys.prefix
dll_paths = [
    os.path.join(venv_path, "Lib", "site-packages", "nvidia", "cublas", "bin"),
    os.path.join(venv_path, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
    os.path.join(venv_path, "Lib", "site-packages", "ctranslate2"),
]

for path in dll_paths:
    if os.path.exists(path):
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]

# DLL yolları ayarlandıktan SONRA import et
import time
import torch
import gc
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self):
        # Başlangıç: Online Mod (GPU)
        self.device = "cuda"
        self.model_size = "small"
        self.compute_type = "int8"
        self.model = self._load_model()
        print(f"[SİSTEM] STT: {self.model_size} modeli GPU üzerinde başlatıldı.")

    def _load_model(self):
        """Modeli mevcut konfigürasyona göre yükler."""
        return WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )

    def switch_to_cpu(self):
        """
        Hot-Swap Mekanizması:
        VRAM sızıntısını önlemek için GPU'yu temizler ve CPU modeline geçer.
        """
        start_time = time.time()
        print("[SİSTEM] STT: Hot-Swap tetiklendi. CPU'ya geçiliyor...")

        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        self.device = "cpu"
        self.model_size = "base"
        self.model = self._load_model()

        end_time = time.time()
        print(f"[SİSTEM] STT: CPU moduna geçiş tamamlandı. Süre: {end_time - start_time:.2f}sn")

    def transcribe(self, audio_path):
        """Ses dosyasını metne çevirir ve v7 standartlarında veri döner."""
        
        # 1. DOSYA GÜVENLİK KONTROLLERİ
        if not os.path.exists(audio_path):
            print(f"[HATA] STT: Ses dosyası bulunamadı -> {audio_path}")
            # Sistem çökmesin diye boş ve no_speech_prob=1.0 (kesin sessizlik) dönüyoruz
            return {"text": "", "duration_ms": 0, "latency_ms": 0, "no_speech_prob": 1.0}

        if os.path.getsize(audio_path) == 0:
            print(f"[HATA] STT: Ses dosyası boş (0 bytes) -> {audio_path}")
            return {"text": "", "duration_ms": 0, "latency_ms": 0, "no_speech_prob": 1.0}

        start_time = time.time()

        # 2. GÜVENLİ TRANSKRİPSİYON BLOĞU
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language="tr",
                beam_size=2,
                best_of=2,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt="Merhaba, bu bir Türkçe ses kaydıdır."
            )

            full_text = ""
            max_no_speech_prob = 0.0

            for segment in segments:
                full_text += segment.text + " "
                if segment.no_speech_prob > max_no_speech_prob:
                    max_no_speech_prob = segment.no_speech_prob

            latency_ms = (time.time() - start_time) * 1000

            return {
                "text": full_text.strip(),
                "duration_ms": int(info.duration * 1000),
                "latency_ms": int(latency_ms),
                "no_speech_prob": max_no_speech_prob
            }

        except Exception as e:
            # Modelin içsel çökmelerini (örn. bozuk wav formatı) yakala
            print(f"[CRITICAL HATA] STT İşlemi Başarısız Oldu: {e}")
            return {"text": "", "duration_ms": 0, "latency_ms": 0, "no_speech_prob": 1.0}