import os
import sys
import time
import gc

# DLL yollarini PATH'e en basta ekle — importlardan ONCE
venv_path = sys.prefix
dll_paths = [
    os.path.join(venv_path, "Lib", "site-packages", "nvidia", "cublas", "bin"),
    os.path.join(venv_path, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
    os.path.join(venv_path, "Lib", "site-packages", "ctranslate2"),
]

for path in dll_paths:
    if os.path.exists(path):
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]

# DLL yollari ayarlandiktan SONRA import et
import torch
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from groq import Groq
import requests

load_dotenv()


class Transcriber:
    # ═══════════════════════════════════════════════════════════
    # DESTEKLENEN MODLAR
    # ═══════════════════════════════════════════════════════════
    # local_gpu      : Yerel Whisper small, GPU (online mod)
    # local_cpu      : Yerel Whisper base, CPU (offline mod)
    # cloud_auto     : Groq Whisper -> Deepgram -> Local (Fallback)

    VALID_MODES = ("local_gpu", "local_cpu", "cloud_auto")

    def __init__(self):
        # Baslangic: Online Mod (GPU)
        self.mode = "local_gpu"
        self.device = "cuda"
        self.model_size = "small"
        self.compute_type = "int8"
        self.model = self._load_local_model()
        print(f"[SISTEM] STT: {self.model_size} modeli GPU uzerinde baslatildi.")

        # Groq STT client
        groq_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_key) if groq_key else None

        # Deepgram API Key
        self.deepgram_key = os.getenv("DEEPGRAM_API_KEY")

    # ═══════════════════════════════════════════════════════════
    # MODEL YUKLEME
    # ═══════════════════════════════════════════════════════════

    def _load_local_model(self):
        """Yerel Whisper modelini mevcut konfigurasyona gore yukler."""
        return WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )

    # ═══════════════════════════════════════════════════════════
    # MOD YONETIMI
    # ═══════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        """
        STT modunu degistirir.
        'local_gpu'  -> Yerel Whisper GPU
        'local_cpu'  -> Yerel Whisper CPU
        'cloud_auto' -> Groq -> Deepgram -> Yerel Whisper
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Gecersiz STT modu: {mode}. Gecerli: {self.VALID_MODES}")

        old_mode = self.mode
        if old_mode == mode:
            return  # Zaten bu modda

        print(f"[SISTEM] STT modu degisiyor: {old_mode} -> {mode}")

        # Mod gecis mantigi
        if mode == "local_cpu":
            self._switch_to_cpu()
        elif mode == "local_gpu":
            self._switch_to_gpu()
        elif mode.startswith("cloud"):
            # Cloud modlarinda yerel modeli bellekte tutuyoruz (fallback icin)
            # Ama GPU VRAM bosaltilmali
            if self.device == "cuda":
                self._offload_gpu_model()

        self.mode = mode

    def _switch_to_cpu(self):
        """Hot-Swap: GPU modelini sil, CPU modeline gec."""
        start_time = time.time()
        print("[SISTEM] STT: Hot-Swap tetiklendi. CPU'ya geciliyor...")

        if self.model is not None:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

        self.device = "cpu"
        self.model_size = "base"
        self.compute_type = "int8"
        self.model = self._load_local_model()

        elapsed = time.time() - start_time
        print(f"[SISTEM] STT: CPU moduna gecis tamamlandi. Sure: {elapsed:.2f}sn")

    def _switch_to_gpu(self):
        """CPU modelini sil, GPU modeline gec."""
        start_time = time.time()
        print("[SISTEM] STT: GPU moduna geciliyor...")

        if self.model is not None:
            del self.model
            gc.collect()

        self.device = "cuda"
        self.model_size = "small"
        self.compute_type = "int8"
        self.model = self._load_local_model()

        elapsed = time.time() - start_time
        print(f"[SISTEM] STT: GPU moduna gecis tamamlandi. Sure: {elapsed:.2f}sn")

    def _offload_gpu_model(self):
        """GPU modelini VRAM'den bosalt, CPU modeline dusur.
        Cloud modu icin — yerel model fallback olarak RAM'de kalir."""
        print("[SISTEM] STT: VRAM bosaltiliyor (cloud modu icin)...")

        if self.model is not None:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

        # Fallback icin CPU'da base modeli hazir tut
        self.device = "cpu"
        self.model_size = "base"
        self.compute_type = "int8"
        self.model = self._load_local_model()
        print("[SISTEM] STT: VRAM bosaltildi. Fallback Whisper (base/CPU) hazir.")

    # Eski API uyumlulugu (orchestrator.py v1 uyumu)
    def switch_to_cpu(self):
        """Geriye donuk uyumluluk icin wrapper."""
        self.set_mode("local_cpu")

    # ═══════════════════════════════════════════════════════════
    # ANA TRANSKRIPSIYON METODU (Yonlendirici)
    # ═══════════════════════════════════════════════════════════

    def transcribe(self, audio_path):
        """Ses dosyasini metne cevirir.
        Aktif moda gore yerel veya bulut motora yonlendirir."""

        # Dosya guvenlik kontrolleri
        if not os.path.exists(audio_path):
            print(f"[HATA] STT: Ses dosyasi bulunamadi -> {audio_path}")
            return {"text": "", "duration_ms": 0, "latency_ms": 0, "no_speech_prob": 1.0}

        if os.path.getsize(audio_path) == 0:
            print(f"[HATA] STT: Ses dosyasi bos (0 bytes) -> {audio_path}")
            return {"text": "", "duration_ms": 0, "latency_ms": 0, "no_speech_prob": 1.0}

        if self.mode == "cloud_auto":
            return self._transcribe_cloud_auto(audio_path)
        return self._transcribe_local(audio_path)

    # ═══════════════════════════════════════════════════════════
    # YEREL TRANSKRIPSIYON — Whisper (GPU veya CPU)
    # ═══════════════════════════════════════════════════════════

    def _transcribe_local(self, audio_path):
        """Yerel Whisper modeli ile transkripsiyon."""
        start_time = time.time()

        try:
            segments, info = self.model.transcribe(
                audio_path,
                language="tr",
                beam_size=2,
                best_of=2,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt="Merhaba, bu bir Turkce ses kaydidir."
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
            print(f"[CRITICAL HATA] STT Yerel Islemi Basarisiz: {e}")
            return {"text": "", "duration_ms": 0, "latency_ms": 0, "no_speech_prob": 1.0}

    # ═══════════════════════════════════════════════════════════
    # BULUT TRANSKRIPSIYON — Şelale: Groq -> Deepgram -> Local
    # ═══════════════════════════════════════════════════════════

    def _transcribe_cloud_auto(self, audio_path):
        """Groq Whisper Large-v3 API dener.
        Basarisiz olursa Deepgram Nova-3 dener.
        O da basarisiz olursa yerel Whisper'a duser."""
        
        # 1. GROQ DENEMESI
        if self.groq_client:
            start_time = time.time()
            try:
                with open(audio_path, "rb") as audio_file:
                    result = self.groq_client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=audio_file,
                        language="tr"
                    )

                latency_ms = (time.time() - start_time) * 1000

                text = result.text.strip() if result.text else ""
                print(f"[STT] Groq Cloud basarili: '{text}' | {int(latency_ms)}ms")

                return {
                    "text": text,
                    "duration_ms": 0,
                    "latency_ms": int(latency_ms),
                    "no_speech_prob": 0.0
                }
            except Exception as e:
                print(f"[UYARI] Groq STT basarisiz: {e} -> Deepgram'a geciliyor...")
        
        # 2. DEEPGRAM DENEMESI
        if self.deepgram_key:
            start_time = time.time()
            try:
                url = "https://api.deepgram.com/v1/listen?model=nova-3&language=tr&smart_format=true"
                headers = {
                    "Authorization": f"Token {self.deepgram_key}",
                    "Content-Type": "audio/wav"
                }
                
                with open(audio_path, "rb") as audio_file:
                    response = requests.post(url, headers=headers, data=audio_file, timeout=10)
                    
                response.raise_for_status()
                data = response.json()
                
                channels = data.get('results', {}).get('channels', [])
                if channels and channels[0].get('alternatives'):
                    text = channels[0]['alternatives'][0].get('transcript', "").strip()
                else:
                    text = ""
                    
                latency_ms = (time.time() - start_time) * 1000
                print(f"[STT] Deepgram Cloud basarili: '{text}' | {int(latency_ms)}ms")

                return {
                    "text": text,
                    "duration_ms": 0,
                    "latency_ms": int(latency_ms),
                    "no_speech_prob": 0.0
                }
            except Exception as e:
                print(f"[UYARI] Deepgram STT basarisiz: {e} -> Yerel Whisper'a dusuluyor...")

        # 3. YEREL WHISPER FALLBACK
        return self._fallback_local_whisper(audio_path)

    # ═══════════════════════════════════════════════════════════
    # FALLBACK SELALESI — Cloud basarisiz olursa yerel Whisper
    # ═══════════════════════════════════════════════════════════

    def _fallback_local_whisper(self, audio_path):
        """Cloud STT cokerse yerel Whisper (CPU) ile metne dok.
        Sistem asla cokmez — hata toleransi."""
        print("[SISTEM] STT FALLBACK: Yerel Whisper'a dusuyorum...")

        # Yerel model hazir degilse yukle
        if self.model is None:
            self.device = "cpu"
            self.model_size = "base"
            self.compute_type = "int8"
            self.model = self._load_local_model()

        return self._transcribe_local(audio_path)