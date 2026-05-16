import os
import sys
import time
import gc

# Prepend the CUDA / cuDNN / CTranslate2 DLL directories to PATH *before* any
# torch / faster-whisper imports. Required on Windows so the loader can resolve
# the GPU acceleration libraries shipped inside the virtual environment.
venv_path = sys.prefix
dll_paths = [
    os.path.join(venv_path, "Lib", "site-packages", "nvidia", "cublas", "bin"),
    os.path.join(venv_path, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
    os.path.join(venv_path, "Lib", "site-packages", "ctranslate2"),
]

for path in dll_paths:
    if os.path.exists(path):
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]

# All native dependencies are resolvable on PATH — safe to import now.
import torch
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from groq import Groq
import requests

load_dotenv()


class Transcriber:
    # ═══════════════════════════════════════════════════════════
    # SUPPORTED MODES
    # ═══════════════════════════════════════════════════════════
    # local_gpu      : Local Whisper 'small', GPU-resident (online runtime)
    # local_cpu      : Local Whisper 'base',  CPU-resident (offline runtime)
    # cloud_auto     : Groq Whisper → Deepgram → Local Whisper (fallback cascade)

    VALID_MODES = ("local_gpu", "local_gpu_hq", "local_cpu", "cloud_auto")

    def __init__(self):
        # Default startup configuration: online mode on the GPU.
        self.mode = "local_gpu"
        self.device = "cuda"
        self.model_size = "small"
        self.compute_type = "int8"
        print(f"[SYSTEM] STT: Loading Whisper '{self.model_size}' onto the GPU... (5-15s on first launch)")
        self.model = self._load_local_model()
        print(f"[SYSTEM] STT: Whisper '{self.model_size}' ready.")

        # Groq STT client (cloud Whisper-large-v3 accelerator).
        groq_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_key) if groq_key else None

        # Deepgram API key (secondary cloud STT fallback).
        self.deepgram_key = os.getenv("DEEPGRAM_API_KEY")

    # ═══════════════════════════════════════════════════════════
    # MODEL LOADING
    # ═══════════════════════════════════════════════════════════

    def _load_local_model(self):
        """Instantiate the local Whisper model with the current device/size/precision settings."""
        return WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )

    # ═══════════════════════════════════════════════════════════
    # GPU WARM-UP — Pre-compile CUDA kernels
    # ═══════════════════════════════════════════════════════════

    def warm_up(self):
        """Warm up the local Whisper model with a silent dummy inference pass.

        Forces CUDA kernel compilation up-front so the first real utterance
        avoids the cold-start penalty (~800 ms saving on cold load).
        Skipped in ``cloud_auto`` mode to avoid wasting an API call.
        """
        if self.mode == "cloud_auto":
            return  # Unnecessary when running on cloud STT.

        print("[SYSTEM] STT GPU warm-up in progress (dummy inference)...")
        start = time.time()

        # Synthesize a 0.5 s silent PCM16 mono WAV (8000 samples × 2 bytes = 16 kB).
        import wave
        import struct
        _tmp_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".tmp_audio"
        )
        os.makedirs(_tmp_dir, exist_ok=True)
        _silent_wav = os.path.join(_tmp_dir, "warmup_silence.wav")

        with wave.open(_silent_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)       # int16
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<8000h", *([0] * 8000)))  # 0.5 s of silence

        try:
            self._transcribe_local(_silent_wav)
        except Exception:
            pass  # Silent input may raise — safe to ignore for the warm-up path.

        elapsed = int((time.time() - start) * 1000)
        print(f"[SYSTEM] STT GPU warm-up complete ({elapsed} ms). First utterance latency optimized.")

    # ═══════════════════════════════════════════════════════════
    # MODE MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        """
        Switch the active STT mode.
        'local_gpu'  -> Local Whisper on GPU.
        'local_cpu'  -> Local Whisper on CPU.
        'cloud_auto' -> Groq → Deepgram → local Whisper (cascading fallback).
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid STT mode: {mode}. Allowed: {self.VALID_MODES}")

        old_mode = self.mode
        if old_mode == mode:
            return  # Already in the requested mode — no-op.

        print(f"[SYSTEM] STT mode transition: {old_mode} -> {mode}")

        # Mode transition dispatch.
        if mode == "local_cpu":
            self._switch_to_cpu()
        elif mode == "local_gpu":
            self._switch_to_gpu()
        elif mode == "local_gpu_hq":
            self._switch_to_gpu_hq()
        elif mode.startswith("cloud"):
            # Cloud modes keep the local model resident in RAM as a fallback,
            # but VRAM must be reclaimed for the LLM / TTS workloads.
            if self.device == "cuda":
                self._offload_gpu_model()

        self.mode = mode

    def _switch_to_cpu(self):
        """Hot-swap path: discard the GPU model and load the CPU model."""
        start_time = time.time()
        print("[SYSTEM] STT: hot-swap triggered. Migrating to CPU...")

        if self.model is not None:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

        self.device = "cpu"
        self.model_size = "base"
        self.compute_type = "int8"
        print("[SYSTEM] STT: loading Whisper 'base' on the CPU...")
        self.model = self._load_local_model()

        elapsed = time.time() - start_time
        print(f"[SYSTEM] STT: CPU transition complete. Elapsed: {elapsed:.2f}s")

    def _switch_to_gpu(self):
        """Discard the CPU model and load the GPU model."""
        start_time = time.time()
        print("[SYSTEM] STT: migrating to GPU...")

        if self.model is not None:
            del self.model
            gc.collect()

        self.device = "cuda"
        self.model_size = "small"
        self.compute_type = "int8"
        print("[SYSTEM] STT: loading Whisper 'small' on the GPU...")
        self.model = self._load_local_model()

        elapsed = time.time() - start_time
        print(f"[SYSTEM] STT: GPU transition complete. Elapsed: {elapsed:.2f}s")

    def _switch_to_gpu_hq(self):
        """Hot-swap path: load the 'medium' model onto the GPU for higher quality."""
        start_time = time.time()
        print("[SYSTEM] STT: switching to high-quality mode (medium)...")

        if self.model is not None:
            del self.model
            gc.collect()

        self.device = "cuda"
        self.model_size = "medium"
        self.compute_type = "int8"
        print("[SYSTEM] STT: loading Whisper 'medium' on the GPU...")
        self.model = self._load_local_model()

        elapsed = time.time() - start_time
        print(f"[SYSTEM] STT: HQ transition complete. Elapsed: {elapsed:.2f}s")

    def _offload_gpu_model(self):
        """Evict the GPU model from VRAM and demote to the CPU model.

        Used when entering cloud mode — the local model stays resident in RAM
        as an emergency fallback if every cloud STT provider fails.
        """
        print("[SYSTEM] STT: reclaiming VRAM for cloud mode...")

        if self.model is not None:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

        # Keep a CPU-resident 'base' model warm for fallback transcription.
        self.device = "cpu"
        self.model_size = "base"
        self.compute_type = "int8"
        self.model = self._load_local_model()
        print("[SYSTEM] STT: VRAM reclaimed. Fallback Whisper (base/CPU) ready.")

    # Legacy API compatibility shim (orchestrator.py v1).
    def switch_to_cpu(self):
        """Backwards-compatible wrapper around set_mode('local_cpu')."""
        self.set_mode("local_cpu")

    # ═══════════════════════════════════════════════════════════
    # PRIMARY TRANSCRIPTION ENTRYPOINT (dispatcher)
    # ═══════════════════════════════════════════════════════════

    def transcribe(self, audio_path, source_lang="tr"):
        """Transcribe an audio file to text.

        Dispatches to the local or cloud transcription path based on the
        currently active mode.
        """

        # Defensive file checks.
        if not os.path.exists(audio_path):
            print(f"[ERROR] STT: audio file not found -> {audio_path}")
            return {"text": "", "duration_ms": 0, "latency_ms": 0, "no_speech_prob": 1.0}

        if os.path.getsize(audio_path) == 0:
            print(f"[ERROR] STT: audio file is empty (0 bytes) -> {audio_path}")
            return {"text": "", "duration_ms": 0, "latency_ms": 0, "no_speech_prob": 1.0}

        if self.mode == "cloud_auto":
            return self._transcribe_cloud_auto(audio_path, source_lang=source_lang)
        return self._transcribe_local(audio_path, source_lang=source_lang)

    # ═══════════════════════════════════════════════════════════
    # LOCAL TRANSCRIPTION — Whisper (GPU or CPU)
    # ═══════════════════════════════════════════════════════════

    def _transcribe_local(self, audio_path, source_lang="tr"):
        """Run transcription through the locally hosted Whisper model."""
        start_time = time.time()

        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=source_lang,
                beam_size=2,
                best_of=2,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
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
            print(f"[CRITICAL] STT local inference failed: {e}")
            return {"text": "", "duration_ms": 0, "latency_ms": 0, "no_speech_prob": 1.0}

    # ═══════════════════════════════════════════════════════════
    # CLOUD TRANSCRIPTION — Cascade: Groq → Deepgram → Local
    # ═══════════════════════════════════════════════════════════

    def _transcribe_cloud_auto(self, audio_path, source_lang="tr"):
        """Attempt the Groq-hosted Whisper-large-v3 API.

        On failure, fall back to Deepgram Nova-3. If both cloud providers fail,
        the call cascades down to the locally hosted Whisper engine.
        """

        # 1. GROQ ATTEMPT
        if self.groq_client:
            start_time = time.time()
            try:
                with open(audio_path, "rb") as audio_file:
                    result = self.groq_client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=audio_file,
                        language=source_lang
                    )

                latency_ms = (time.time() - start_time) * 1000

                text = result.text.strip() if result.text else ""
                print(f"[STT] Groq cloud success: '{text}' | {int(latency_ms)} ms")

                return {
                    "text": text,
                    "duration_ms": 0,
                    "latency_ms": int(latency_ms),
                    "no_speech_prob": 0.0
                }
            except Exception as e:
                print(f"[WARN] Groq STT failed: {e} -> falling through to Deepgram...")

        # 2. DEEPGRAM ATTEMPT
        if self.deepgram_key:
            start_time = time.time()
            try:
                url = f"https://api.deepgram.com/v1/listen?model=nova-3&language={source_lang}&smart_format=true"
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
                print(f"[STT] Deepgram cloud success: '{text}' | {int(latency_ms)} ms")

                return {
                    "text": text,
                    "duration_ms": 0,
                    "latency_ms": int(latency_ms),
                    "no_speech_prob": 0.0
                }
            except Exception as e:
                print(f"[WARN] Deepgram STT failed: {e} -> falling through to local Whisper...")

        # 3. LOCAL WHISPER FALLBACK
        return self._fallback_local_whisper(audio_path, source_lang=source_lang)

    # ═══════════════════════════════════════════════════════════
    # FALLBACK CASCADE — Last-resort local Whisper when cloud STT fails
    # ═══════════════════════════════════════════════════════════

    def _fallback_local_whisper(self, audio_path, source_lang="tr"):
        """Final fallback: transcribe on the local CPU Whisper engine when every
        cloud provider has failed. The system never goes down — fault tolerant.
        """
        print("[SYSTEM] STT FALLBACK: cascading to local Whisper...")

        # Lazy-load the local model if it was previously offloaded.
        if self.model is None:
            self.device = "cpu"
            self.model_size = "base"
            self.compute_type = "int8"
            self.model = self._load_local_model()

        return self._transcribe_local(audio_path, source_lang=source_lang)
