import os
import time
import shutil
import wave
import numpy as np
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import sounddevice as sd
import soundfile as sf

load_dotenv()


# ═══════════════════════════════════════════════════════════
# torchaudio.load MONKEY-PATCH
# ═══════════════════════════════════════════════════════════
# PyTorch 2.11 + torchaudio 2.11 artik torchaudio.load() icin
# torchcodec'i zorunlu tutuyor. XTTS-v2 ise torchaudio.load()
# ile ses dosyasi yukluyor. torchcodec kurmak yerine stdlib
# wave modulu ile kendi load fonksiyonumuzu yaziyoruz.
# Bu patch XTTS import edilmeden ONCE uygulanmalidir.

def _apply_torchaudio_patch():
    """torchaudio.load'u stdlib wave ile degistirir.
    Turkce karakterli dosya isimlerini temp ASCII kopyasi ile destekler."""
    import torch
    import torchaudio

    # Proje kok dizini (temp dosyalar icin)
    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _wave_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                   channels_first=True, format=None, buffer_size=4096, backend=None):
        file_path = str(uri) if not isinstance(uri, str) else uri

        # Non-ASCII karakter varsa temp dosyaya kopyala (Turkce karakter fix)
        try:
            file_path.encode('ascii')
            actual_path = os.path.abspath(file_path)
            temp_path = None
        except UnicodeEncodeError:
            temp_dir = os.path.join(_project_dir, ".tmp_audio")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, "ref_audio.wav")
            shutil.copy2(os.path.abspath(file_path), temp_path)
            actual_path = temp_path

        try:
            with wave.open(actual_path, 'rb') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                total_frames = wf.getnframes()

                if frame_offset > 0:
                    wf.setpos(frame_offset)
                    total_frames -= frame_offset

                if num_frames > 0:
                    total_frames = min(total_frames, num_frames)

                raw_data = wf.readframes(total_frames)

            # Raw bytes -> numpy float32
            if sample_width == 2:
                dtype = np.int16
                max_val = 32768.0
            elif sample_width == 4:
                dtype = np.int32
                max_val = 2147483648.0
            else:
                dtype = np.uint8
                max_val = 128.0

            data = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
            if normalize and sample_width > 1:
                data = data / max_val

            # Channel reshape
            if n_channels > 1:
                data = data.reshape(-1, n_channels)

            waveform = torch.from_numpy(data)

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # [time] -> [1, time]
            elif channels_first and waveform.dim() == 2:
                waveform = waveform.t()  # [time, channel] -> [channel, time]

            return waveform, sample_rate
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    torchaudio.load = _wave_load


class Synthesizer:
    def __init__(self):
        """
        TTS katmanini baslatir. v7 Dual-State Mimarisi.

        ONLINE MOD  -> ElevenLabs streaming (eleven_turbo_v2_5)
        OFFLINE MOD -> XTTS-v2 CPU (ses klonlama destekli)
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

        # ─── OFFLINE MOTOR: XTTS-v2 ────────────────────────────
        self.xtts_model = None
        self.xtts_model_path = "tts_models/multilingual/multi-dataset/xtts_v2"
        self._torchaudio_patched = False

        # Referans ses dosyasi (ses klonlama icin)
        # Absolute path kullaniyoruz — relative path Turkce karakterlerle sorunlu
        self._project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.speaker_wav_path = os.path.join(self._project_dir, "audio", "Kayıt (3).wav")

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
        ElevenLabs Turbo modeli ile bulut streaming sentezi yapar.
        stream() kullanarak ilk chunk'i mumkun olan en erken anda alir.
        Hedef: ilk chunk < 600ms
        """
        if not text or len(text.strip()) == 0:
            return

        start_time = time.time()
        try:
            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id
            )
            generation_latency = int((time.time() - start_time) * 1000)
            print(f"[TTS] ElevenLabs Uretim Suresi: {generation_latency} ms | Caliniyor...")
            play(audio_stream)
            total_latency = int((time.time() - start_time) * 1000)
            print(f"[TTS] ElevenLabs Toplam (Uretim + Calma): {total_latency} ms")
            return generation_latency

        except Exception as e:
            print(f"[KRITIK HATA] TTS Online Modu Basarisiz: {e}")
            return 0

    # ═══════════════════════════════════════════════════════════
    # OFFLINE SENTEZ — XTTS-v2 CPU (Ses Klonlama Destekli)
    # ═══════════════════════════════════════════════════════════

    def speak_offline(self, text: str):
        """
        XTTS-v2 ile yerel CPU sentezi yapar.
        Kullanicinin kendi sesini klonlayarak Ingilizce sentez uretir.
        Hedef: tam cumle < 4sn (warm start)
        """
        if not text or len(text.strip()) == 0:
            return

        print(f"[TTS] Offline Sentez (XTTS-v2) baslatiliyor...")
        start_time = time.time()

        # Lazy Loading — Model sadece ilk offline cagrisinda yuklenir
        if self.xtts_model is None:
            self._load_xtts_model()

        try:
            # Gecici cikti dosyasi (ASCII isimli — libsndfile uyumlulugu icin)
            output_file = os.path.join(self._project_dir, "offline_output.wav")

            # Sentezleme — kullanicinin kendi sesiyle
            self.xtts_model.tts_to_file(
                text=text,
                language="en",
                file_path=output_file,
                speaker_wav=self.speaker_wav_path
            )

            latency = int((time.time() - start_time) * 1000)
            print(f"[TTS] XTTS-v2 Uretim Suresi: {latency} ms | Caliniyor...")

            # Sesi cal
            data, fs = sf.read(output_file)
            sd.play(data, fs)
            sd.wait()
            return latency

        except Exception as e:
            print(f"[KRITIK HATA] TTS Offline Modu Basarisiz: {e}")
            return 0

    def _load_xtts_model(self):
        """XTTS-v2 modelini CPU uzerinde yukler.
        torch.load ve torchaudio.load patch'lerini uygular."""
        print("[SISTEM] XTTS-v2 modeli yukleniyor (bu islem 20-40sn surebilir)...")
        load_start = time.time()

        # torch.load weights_only patch
        import torch
        _original_torch_load = torch.load
        def _safe_torch_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)
        torch.load = _safe_torch_load

        # torchaudio.load -> wave stdlib patch
        if not self._torchaudio_patched:
            _apply_torchaudio_patch()
            self._torchaudio_patched = True

        # Model yukleme — sadece CPU (VRAM'i isgal etmemek icin)
        from TTS.api import TTS
        self.xtts_model = TTS(self.xtts_model_path, gpu=False)

        load_time = time.time() - load_start
        print(f"[SISTEM] XTTS-v2 modeli yuklendi! Sure: {load_time:.1f}sn")
