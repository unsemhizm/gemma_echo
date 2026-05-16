# NOTE: This patch is written against torchaudio 2.11.0.
# Re-validate _apply_torchaudio_patch() whenever torchaudio is upgraded.

import os
import time
from typing import Optional
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
# PyTorch 2.11 + torchaudio 2.11 now require torchcodec for torchaudio.load().
# XTTS-v2 loads its reference audio through torchaudio.load(). Rather than
# pulling in the heavy torchcodec dependency, we redirect torchaudio.load to
# a minimal stdlib `wave`-based reader implemented below. This patch MUST be
# applied before XTTS is imported.

def _apply_torchaudio_patch():
    """Replace torchaudio.load with a stdlib `wave`-based reader.

    Also transparently handles non-ASCII (e.g., Turkish) characters in file
    paths by copying the input to a temporary ASCII path before opening.
    """
    import torch
    import torchaudio

    # Project root (used to host temp files).
    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _wave_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                   channels_first=True, format=None, buffer_size=4096, backend=None):
        file_path = str(uri) if not isinstance(uri, str) else uri

        # If the path contains non-ASCII characters, materialize it to a temporary
        # ASCII-safe location first (workaround for Windows + Turkish characters).
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

            # Raw PCM bytes -> normalized numpy float32.
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

            # Reshape multi-channel buffers into (frames, channels).
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
    # SUPPORTED MODES
    # ═══════════════════════════════════════════════════════════
    # online  : ElevenLabs cloud synthesis (Turbo voice).
    # offline : XTTS-v2 CPU (offline survival path).
    # gpu     : XTTS-v2 GPU (hybrid_plus — high-quality voice cloning).

    VALID_MODES = ("online", "offline", "gpu")

    def __init__(self):
        """
        Initialize the TTS layer (v8 Quad-State architecture).

        ONLINE  -> ElevenLabs (eleven_turbo_v2_5)
        OFFLINE -> XTTS-v2 CPU (voice cloning)
        GPU     -> XTTS-v2 GPU (fast voice cloning, hybrid_plus)
        """
        print("[SYSTEM] Speech synthesis module (Synthesizer) initializing...")

        self.mode = "online"

        # ─── ONLINE ENGINE: ELEVENLABS ──────────────────────────
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is missing! Please check your .env file.")

        self.client = ElevenLabs(api_key=self.api_key.strip())
        self.model_id = "eleven_turbo_v2_5"
        self.voice_id = "pNInz6obpgDQGcFmaJgB"

        # ─── OFFLINE / GPU ENGINE: XTTS-v2 ─────────────────────
        self.xtts_model = None
        self.xtts_model_path = "tts_models/multilingual/multi-dataset/xtts_v2"
        self._torchaudio_patched = False
        self._xtts_ready = threading.Event()  # Background-load completion signal.
        self._xtts_loading = False
        self._xtts_on_gpu = False  # Tracks whether XTTS currently lives on the GPU.

        # Reference voice sample (used by XTTS for voice cloning).
        self._project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.speaker_wav_path = os.path.join(self._project_dir, "samples", "sample_audio_2.wav")

        # ─── OUTPUT SETTINGS ────────────────────────────────────
        self.output_device = None  # None = system default; int = sounddevice index.

        self.vram_issue_callback = None  # GUI warning hook: () -> None
        self._xtts_gpu_vram_failed = False

    def _invoke_vram_callback(self):
        cb = self.vram_issue_callback
        if cb is None:
            return
        try:
            cb()
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════
    # BACKGROUND PRELOAD — XTTS "AMBUSH MODE"
    # ═══════════════════════════════════════════════════════════

    def preload_xtts_background(self, use_gpu=False):
        """Eagerly load XTTS-v2 on a daemon thread so that mode switches are instant.

        use_gpu=False -> system RAM (ambush for an offline switch).
        use_gpu=True  -> VRAM (eager load for hybrid_plus).
        """
        if self._xtts_loading or self.xtts_model is not None:
            return  # Already loading or loaded — no-op.

        self._xtts_loading = True
        self._xtts_ready.clear()

        target_str = "GPU (VRAM)" if use_gpu else "CPU (system RAM)"

        def _load_in_background():
            try:
                print(f"[SYSTEM] XTTS-v2 background preload starting ({target_str})...")
                if not self._load_xtts_model(use_gpu=use_gpu):
                    self._xtts_loading = False
                    self._xtts_ready.set()
                    return
                self._xtts_ready.set()
                print(f"[SYSTEM] XTTS-v2 is in ambush ({target_str}) — instant switch ready.")
            except Exception as e:
                print(f"[WARN] XTTS background preload failed: {e}")
                self._xtts_loading = False
                self._xtts_ready.set()  # Release any waiting threads to avoid deadlock.

        thread = threading.Thread(target=_load_in_background, daemon=True)
        thread.start()

    # ═══════════════════════════════════════════════════════════
    # MODE MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        """
        Switch the active TTS mode.
        'online'  -> ElevenLabs cloud synthesis.
        'offline' -> XTTS-v2 CPU synthesis.
        'gpu'     -> XTTS-v2 GPU synthesis (hybrid_plus).
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid TTS mode: {mode}. Allowed: {self.VALID_MODES}")

        old_mode = self.mode
        if mode == "gpu":
            self._xtts_gpu_vram_failed = False
        self.mode = mode
        print(f"[SYSTEM] Synthesizer mode transition: {old_mode} -> {mode}")

    # ═══════════════════════════════════════════════════════════
    # PRIMARY SYNTHESIS ENTRYPOINT (dispatcher)
    # ═══════════════════════════════════════════════════════════

    def speak(self, text: str, language: str = "en"):
        """
        Synthesize text to speech and play it through the configured output device.

        Dispatches to the appropriate engine based on the active mode.
        ``language``: XTTS language code (en, tr, ar, es, ja, ...).
        """
        if not text or len(text.strip()) == 0:
            return

        if self.mode == "online":
            return self.speak_online(text)
        elif self.mode == "gpu":
            return self.speak_offline(text, expect_gpu=True, language=language)
        return self.speak_offline(text, expect_gpu=False, language=language)

    def set_output_device(self, device_index: Optional[int]):
        """None = system default speaker; int = sounddevice device index."""
        self.output_device = device_index
        print(f"[TTS] Output device set: {device_index if device_index is not None else 'Default'}")

    # ═══════════════════════════════════════════════════════════
    # ONLINE SYNTHESIS — ElevenLabs
    # ═══════════════════════════════════════════════════════════

    def speak_online(self, text: str):
        """Synthesize speech through the ElevenLabs Turbo cloud model."""
        if not text or len(text.strip()) == 0:
            return

        start_time = time.time()
        try:
            # output_format="pcm_22050" — ElevenLabs returns raw PCM (not MP3).
            # sf.read does not support MP3; raw PCM lets us parse directly with numpy.
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format="pcm_22050"
            )
            audio_bytes = b"".join(audio)
            data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(data, samplerate=22050, device=self.output_device)
            sd.wait()
            total_latency = int((time.time() - start_time) * 1000)
            print(f"[TTS] ElevenLabs total latency: {total_latency} ms")
            return total_latency

        except Exception as e:
            print(f"[CRITICAL] TTS online mode failed: {e}")
            return 0

    # ═══════════════════════════════════════════════════════════
    # OFFLINE / GPU SYNTHESIS — XTTS-v2 (voice-cloning capable)
    # ═══════════════════════════════════════════════════════════

    def speak_offline(self, text: str, expect_gpu: bool = False, language: str = "en"):
        """
        Run local synthesis through XTTS-v2.
        expect_gpu=True  -> XTTS is expected to live on the GPU (hybrid_plus).
        expect_gpu=False -> XTTS is expected to live on the CPU (offline).
        """
        if not text or len(text.strip()) == 0:
            return

        device_str = "GPU" if expect_gpu else "CPU"
        print(f"[TTS] Offline synthesis (XTTS-v2 {device_str}) starting...")
        start_time = time.time()

        # If the model is not yet loaded, either lazy-load or wait for the ambush thread.
        if self.xtts_model is None:
            if expect_gpu and self._xtts_gpu_vram_failed:
                print("[SYSTEM] XTTS GPU previously failed; falling back to ElevenLabs.")
                return self.speak_online(text)
            if self._xtts_loading:
                print("[SYSTEM] XTTS background load in progress, waiting...")
                self._xtts_ready.wait()
            else:
                if not self._load_xtts_model(use_gpu=expect_gpu):
                    if expect_gpu and self._xtts_gpu_vram_failed:
                        return self.speak_online(text)
                    return 0

        # Model is loaded but may live on the wrong device — hot-migrate if necessary.
        if expect_gpu and not self._xtts_on_gpu:
            print("[SYSTEM] XTTS is on CPU but GPU was requested — migrating to GPU...")
            self._reload_xtts_on_device(use_gpu=True)
        elif not expect_gpu and self._xtts_on_gpu:
            print("[SYSTEM] XTTS is on GPU but CPU was requested — migrating to CPU...")
            self._reload_xtts_on_device(use_gpu=False)

        if self.xtts_model is None and expect_gpu and self._xtts_gpu_vram_failed:
            return self.speak_online(text)

        try:
            output_file = os.path.join(self._project_dir, "offline_output.wav")

            self.xtts_model.tts_to_file(
                text=text,
                language=language,
                file_path=output_file,
                speaker_wav=self.speaker_wav_path
            )

            latency = int((time.time() - start_time) * 1000)
            print(f"[TTS] XTTS-v2 ({device_str}) generation latency: {latency} ms | playing...")

            data, fs = sf.read(output_file)
            sd.play(data, fs, device=self.output_device)
            sd.wait()
            return latency

        except Exception as e:
            print(f"[CRITICAL] TTS offline mode failed: {e}")
            return 0

    # ═══════════════════════════════════════════════════════════
    # XTTS MODEL LOAD / UNLOAD
    # ═══════════════════════════════════════════════════════════

    def _load_xtts_model(self, use_gpu=False) -> bool:
        """Load the XTTS-v2 model onto the requested device.

        Returns False on a failed GPU load (typically due to VRAM exhaustion),
        allowing callers to gracefully degrade to the cloud engine.
        """
        device_str = "GPU" if use_gpu else "CPU"
        print(f"[SYSTEM] XTTS-v2 model loading ({device_str}, 20-40s)...")
        load_start = time.time()

        if use_gpu:
            from gpu_memory import (
                MIN_FREE_BYTES_XTTS_GPU,
                cleanup_cuda_memory,
                is_cuda_oom_error,
                vram_sufficient_for_xtts_gpu,
            )

            ok, free = vram_sufficient_for_xtts_gpu()
            if not ok:
                print(
                    f"[WARN] XTTS GPU VRAM pre-check failed "
                    f"(free: {free} B, threshold: {MIN_FREE_BYTES_XTTS_GPU} B)"
                )
                self._xtts_gpu_vram_failed = True
                cleanup_cuda_memory()
                self._invoke_vram_callback()
                return False

        # torch.load weights_only compatibility shim.
        import torch
        _original_torch_load = torch.load
        def _safe_torch_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)
        torch.load = _safe_torch_load

        # Apply the torchaudio.load -> wave stdlib patch (idempotent).
        if not self._torchaudio_patched:
            _apply_torchaudio_patch()
            self._torchaudio_patched = True

        try:
            from TTS.api import TTS
            self.xtts_model = TTS(self.xtts_model_path, gpu=use_gpu)
            self._xtts_on_gpu = use_gpu
            if use_gpu:
                self._xtts_gpu_vram_failed = False
        except Exception as e:
            self.xtts_model = None
            self._xtts_on_gpu = False
            from gpu_memory import cleanup_cuda_memory, is_cuda_oom_error

            cleanup_cuda_memory()
            if use_gpu and is_cuda_oom_error(e):
                print(f"[WARN] XTTS GPU CUDA OOM: {e}")
                self._xtts_gpu_vram_failed = True
                self._invoke_vram_callback()
                return False
            raise

        load_time = time.time() - load_start
        print(f"[SYSTEM] XTTS-v2 model loaded ({device_str}). Elapsed: {load_time:.1f}s")
        return True

    def _reload_xtts_on_device(self, use_gpu=False):
        """Migrate the XTTS model between GPU and CPU."""
        self.offload_xtts()
        self._load_xtts_model(use_gpu=use_gpu)

    def offload_xtts(self):
        """Fully evict the XTTS model from memory (GPU or CPU).

        Invoked during mode transitions to free VRAM for the next pipeline stage.
        """
        if self.xtts_model is not None:
            was_gpu = self._xtts_on_gpu
            print(f"[SYSTEM] XTTS-v2 evicting from memory ({'GPU' if was_gpu else 'CPU'})...")
            del self.xtts_model
            self.xtts_model = None
            self._xtts_on_gpu = False
            self._xtts_loading = False
            self._xtts_ready.clear()
            gc.collect()

            if was_gpu:
                import torch
                torch.cuda.empty_cache()
                print("[SYSTEM] XTTS-v2 GPU VRAM reclaimed.")
            else:
                print("[SYSTEM] XTTS-v2 evicted from system RAM.")
            self._xtts_gpu_vram_failed = False

    def offload_xtts_from_gpu(self):
        """Evict XTTS from the GPU only (backwards-compatible wrapper)."""
        if self._xtts_on_gpu:
            self.offload_xtts()
