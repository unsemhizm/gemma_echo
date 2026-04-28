import os
import time
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import sounddevice as sd
import soundfile as sf

load_dotenv()


class Synthesizer:
    def __init__(self):
        """
        TTS katmanini baslatir. v7 Dual-State Mimarisi.

        ONLINE MOD  -> ElevenLabs streaming (eleven_turbo_v2_5)
        OFFLINE MOD -> XTTS-v2 CPU (Gun 3'te implemente edilecek)
        """
        print("[SISTEM] Ses Sentezi Modulu (Synthesizer) Baslatiliyor...")

        self.mode = "online"

        # ─── ONLINE MOTOR: ELEVENLABS ───────────────────────────
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY eksik! Lutfen .env dosyasini kontrol edin.")

        self.client = ElevenLabs(api_key=self.api_key.strip())
        self.model_id = "eleven_turbo_v2_5"
        self.voice_id = "pNInz6obpgDQGcFmaJgB"

        # ─── OFFLINE MOTOR: XTTS-v2 (Gun 3) ────────────────────
        self.xtts_model = None
        self.xtts_model_path = "tts_models/multilingual/multi-dataset/xtts_v2"

    # ═══════════════════════════════════════════════════════════
    # MOD YONETIMI
    # ═══════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        """
        TTS modunu degistirir.
        "online"  -> ElevenLabs bulut sentezi
        "offline" -> XTTS-v2 CPU sentezi
        """
        if mode not in ("online", "offline"):
            raise ValueError(f"Gecersiz mod: {mode}. 'online' veya 'offline' olmali.")

        old_mode = self.mode
        self.mode = mode
        print(f"[SISTEM] Synthesizer modu degisti: {old_mode} -> {mode}")

    # ═══════════════════════════════════════════════════════════
    # ANA SENTEZ METODU (Yonlendirici)
    # ═══════════════════════════════════════════════════════════

    def speak(self, text: str):
        """
        Metni sese donusturur ve calar.
        Aktif moda gore online veya offline motora yonlendirir.
        """
        if not text or len(text.strip()) == 0:
            return

        if self.mode == "online":
            return self.speak_online(text)
        return self.speak_offline(text)

    # ═══════════════════════════════════════════════════════════
    # ONLINE SENTEZ — ElevenLabs Streaming
    # ═══════════════════════════════════════════════════════════

    def speak_online(self, text: str):
        """
        ElevenLabs Turbo modeli ile bulut sentezi yapar.
        Hedef: ilk chunk < 600ms
        """
        if not text or len(text.strip()) == 0:
            return

        start_time = time.time()
        try:
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id
            )
            latency = int((time.time() - start_time) * 1000)
            print(f"[TTS] ElevenLabs Uretim Suresi: {latency} ms | Caliniyor...")
            play(audio)
            return latency

        except Exception as e:
            print(f"[KRITIK HATA] TTS Online Modu Basarisiz: {e}")
            return 0

    # ═══════════════════════════════════════════════════════════
    # OFFLINE SENTEZ — XTTS-v2 CPU
    # ═══════════════════════════════════════════════════════════

    def speak_offline(self, text: str):
        """
        XTTS-v2 ile yerel CPU sentezi yapar.
        Hedef: tam cumle < 4sn
        """
        if not text or len(text.strip()) == 0:
            return

        print(f"[TTS] Offline Sentez (XTTS-v2) baslatiliyor...")
        start_time = time.time()
        
        # Lazy Loading
        if self.xtts_model is None:
            print("[SISTEM] XTTS-v2 modeli yukleniyor (bu islem biraz surebilir)...")
            import torch
            _original_torch_load = torch.load
            def _safe_torch_load(*args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return _original_torch_load(*args, **kwargs)
            torch.load = _safe_torch_load
            
            from TTS.api import TTS
            # Sadece CPU kullanarak (VRAM'i isgal etmemek icin)
            self.xtts_model = TTS(self.xtts_model_path, gpu=False)
            print("[SISTEM] XTTS-v2 modeli yuklendi!")

        try:
            # Gecici ses dosyasi yolu
            output_file = "offline_output.wav"
            
            # Sentezleme
            # XTTS icin referans bir ses dosyasi gereklidir (speaker_wav).
            # Eger klonlama gerekmiyorsa TTS kütüphanesinin varsayilan yontemi kullanilabilir,
            # Ancak XTTS genelde speaker_wav zorunlu tutar. Kendi test klasöründeki seslerden birini verebiliriz.
            # Veya XTTS varsayilan speaker varsa kullanabiliriz.
            
            # Not: XTTS-v2 doğrudan tts_to_file ile calistiginda speaker_wav ve language parametreleri ister.
            # Gemma Echo klasöründeki audio/kayıt (3).wav kullanılabilir.
            speaker_wav_path = "audio/kayıt (3).wav"
            
            if not os.path.exists(speaker_wav_path):
                # Eger bulunamazsa fallback olarak sahte bos bir ses verebilir veya hata atabiliriz
                print(f"[HATA] {speaker_wav_path} bulunamadi. XTTS referans sese ihtiyac duyar.")
                return

            self.xtts_model.tts_to_file(
                text=text, 
                language="en", 
                file_path=output_file,
                speaker_wav=speaker_wav_path
            )
            
            latency = int((time.time() - start_time) * 1000)
            print(f"[TTS] XTTS-v2 Uretim Suresi: {latency} ms | Caliniyor...")
            
            # Sesi çalma
            data, fs = sf.read(output_file)
            sd.play(data, fs)
            sd.wait()  # Calma bitene kadar bekle
            return latency
            
        except Exception as e:
            print(f"[KRITIK HATA] TTS Offline Modu Basarisiz: {e}")
            return 0
