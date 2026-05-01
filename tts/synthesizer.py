# UYARI: Bu patch torchaudio 2.11.0 icin yazilmistir.
# torchaudio guncellenirse _apply_torchaudio_patch() kontrol edilmeli.

import os
import time
import shutil
import wave
import threading
import gc
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
    # ═══════════════════════════════════════════════════════════
    # DESTEKLENEN MODLAR
    # ═══════════════════════════════════════════════════════════
    # online  : ElevenLabs bulut sentezi (online / turbo)
    # offline : XTTS-v2 CPU (offline survival mod)
    # gpu     : XTTS-v2 GPU (hybrid_plus — yuksek kalite klon)

    VALID_MODES = ("online", "offline", "gpu")

    def __init__(self):
        """
        TTS katmanini baslatir. v8 Quad-State Mimarisi.

        ONLINE MOD  -> ElevenLabs (eleven_turbo_v2_5)
        OFFLINE MOD -> XTTS-v2 CPU (ses klonlama)
        GPU MOD     -> XTTS-v2 GPU (hizli ses klonlama, hybrid_plus)
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

        # ─── OFFLINE / GPU MOTOR: XTTS-v2 ──────────────────────
        self.xtts_model = None
        self.xtts_model_path = "tts_models/multilingual/multi-dataset/xtts_v2"
        self._torchaudio_patched = False
        self._xtts_ready = threading.Event()  # Background yukleme sinyali
        self._xtts_loading = False
        self._xtts_on_gpu = False  # XTTS su an GPU'da mi?

        # Referans ses dosyasi (ses klonlama icin)
        self._project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.speaker_wav_path = os.path.join(self._project_dir, "audio", "Kayıt (3).wav")

    # ═══════════════════════════════════════════════════════════
    # BACKGROUND PRELOAD — XTTS PUSU MODU
    # ═══════════════════════════════════════════════════════════

    def preload_xtts_background(self, use_gpu=False):
        """XTTS-v2'yi arka plan daemon thread'inde yukler.
        use_gpu=False -> sistem RAM (online/offline icin pusu)
        use_gpu=True  -> VRAM (hybrid_plus icin eager load)"""
        if self._xtts_loading or self.xtts_model is not None:
            return  # Zaten yukleniyor veya yuklenmis

        self._xtts_loading = True
        self._xtts_ready.clear()

        target_str = "GPU (VRAM)" if use_gpu else "CPU (sistem RAM)"

        def _load_in_background():
            try:
                print(f"[SISTEM] XTTS-v2 arka planda yukleniyor ({target_str})...")
                self._load_xtts_model(use_gpu=use_gpu)
                self._xtts_ready.set()
                print(f"[SISTEM] XTTS-v2 pusuya yatti ({target_str})! Gecis aninda hazir.")
            except Exception as e:
                print(f"[UYARI] XTTS arka plan yuklemesi basarisiz: {e}")
                self._xtts_loading = False
                self._xtts_ready.set()  # Deadlock onleme

        thread = threading.Thread(target=_load_in_background, daemon=True)
        thread.start()

    # ═══════════════════════════════════════════════════════════
    # MOD YONETIMI
    # ═══════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        """
        TTS modunu degistirir.
        'online'  -> ElevenLabs bulut sentezi
        'offline' -> XTTS-v2 CPU sentezi
        'gpu'     -> XTTS-v2 GPU sentezi (hybrid_plus)
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Gecersiz TTS modu: {mode}. Gecerli: {self.VALID_MODES}")

        old_mode = self.mode
        self.mode = mode
        print(f"[SISTEM] Synthesizer modu degisti: {old_mode} -> {mode}")

    # ═══════════════════════════════════════════════════════════
    # ANA SENTEZ METODU (Yonlendirici)
    # ═══════════════════════════════════════════════════════════

    def speak(self, text: str):
        """
        Metni sese donusturur ve calar.
        Aktif moda gore uygun motora yonlendirir.
        """
        if not text or len(text.strip()) == 0:
            return

        if self.mode == "online":
            return self.speak_online(text)
        elif self.mode == "gpu":
            return self.speak_offline(text, expect_gpu=True)
        return self.speak_offline(text, expect_gpu=False)

    # ═══════════════════════════════════════════════════════════
    # ONLINE SENTEZ — ElevenLabs
    # ═══════════════════════════════════════════════════════════

    def speak_online(self, text: str):
        """ElevenLabs Turbo modeli ile bulut sentezi yapar."""
        if not text or len(text.strip()) == 0:
            return

        start_time = time.time()
        try:
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id
            )
            play(audio)
            total_latency = int((time.time() - start_time) * 1000)
            print(f"[TTS] ElevenLabs Toplam Sure: {total_latency} ms")
            return total_latency

        except Exception as e:
            print(f"[KRITIK HATA] TTS Online Modu Basarisiz: {e}")
            return 0

    # ═══════════════════════════════════════════════════════════
    # OFFLINE / GPU SENTEZ — XTTS-v2 (Ses Klonlama Destekli)
    # ═══════════════════════════════════════════════════════════

    def speak_offline(self, text: str, expect_gpu: bool = False):
        """
        XTTS-v2 ile yerel sentez yapar.
        expect_gpu=True  -> GPU'da olmasi beklenir (hybrid_plus)
        expect_gpu=False -> CPU'da calismasi beklenir (offline)
        """
        if not text or len(text.strip()) == 0:
            return

        device_str = "GPU" if expect_gpu else "CPU"
        print(f"[TTS] Offline Sentez (XTTS-v2 {device_str}) baslatiliyor...")
        start_time = time.time()

        # Model hazir degilse yukle veya bekle
        if self.xtts_model is None:
            if self._xtts_loading:
                print("[SISTEM] XTTS arka planda yukleniyor, bekleniyor...")
                self._xtts_ready.wait()
            else:
                self._load_xtts_model(use_gpu=expect_gpu)

        # Model yuklendi ama yanlis cihazda olabilir
        if expect_gpu and not self._xtts_on_gpu:
            print("[SISTEM] XTTS CPU'da ama GPU isteniyor, GPU'ya tasiniyor...")
            self._reload_xtts_on_device(use_gpu=True)
        elif not expect_gpu and self._xtts_on_gpu:
            print("[SISTEM] XTTS GPU'da ama CPU isteniyor, CPU'ya tasiniyor...")
            self._reload_xtts_on_device(use_gpu=False)

        try:
            output_file = os.path.join(self._project_dir, "offline_output.wav")

            self.xtts_model.tts_to_file(
                text=text,
                language="en",
                file_path=output_file,
                speaker_wav=self.speaker_wav_path
            )

            latency = int((time.time() - start_time) * 1000)
            print(f"[TTS] XTTS-v2 ({device_str}) Uretim Suresi: {latency} ms | Caliniyor...")

            data, fs = sf.read(output_file)
            sd.play(data, fs)
            sd.wait()
            return latency

        except Exception as e:
            print(f"[KRITIK HATA] TTS Offline Modu Basarisiz: {e}")
            return 0

    # ═══════════════════════════════════════════════════════════
    # XTTS MODEL YUKLEME / BOSALTMA
    # ═══════════════════════════════════════════════════════════

    def _load_xtts_model(self, use_gpu=False):
        """XTTS-v2 modelini yukler. use_gpu=True ise VRAM'e yukler."""
        device_str = "GPU" if use_gpu else "CPU"
        print(f"[SISTEM] XTTS-v2 modeli yukleniyor ({device_str}, 20-40sn surebilir)...")
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

        from TTS.api import TTS
        self.xtts_model = TTS(self.xtts_model_path, gpu=use_gpu)
        self._xtts_on_gpu = use_gpu

        load_time = time.time() - load_start
        print(f"[SISTEM] XTTS-v2 modeli yuklendi ({device_str})! Sure: {load_time:.1f}sn")

    def _reload_xtts_on_device(self, use_gpu=False):
        """XTTS modelini baska bir cihaza tasinir (GPU<->CPU)."""
        self.offload_xtts()
        self._load_xtts_model(use_gpu=use_gpu)

    def offload_xtts(self):
        """XTTS modelini tamamen bellekten siler (GPU veya CPU).
        Mod gecislerinde VRAM guvenlik icin kullanilir."""
        if self.xtts_model is not None:
            was_gpu = self._xtts_on_gpu
            print(f"[SISTEM] XTTS-v2 bellekten siliniyor ({'GPU' if was_gpu else 'CPU'})...")
            del self.xtts_model
            self.xtts_model = None
            self._xtts_on_gpu = False
            self._xtts_loading = False
            self._xtts_ready.clear()
            gc.collect()

            if was_gpu:
                import torch
                torch.cuda.empty_cache()
                print("[SISTEM] XTTS-v2 GPU VRAM bosaltildi.")
            else:
                print("[SISTEM] XTTS-v2 sistem RAM'den silindi.")

    def offload_xtts_from_gpu(self):
        """XTTS'i GPU'dan bosalt. Geriye donuk uyumluluk icin wrapper."""
        if self._xtts_on_gpu:
            self.offload_xtts()
