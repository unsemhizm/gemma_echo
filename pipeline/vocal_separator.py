"""Demucs-based vocal separator — dubbing pre-processing stage.

Motivation
==========
Two critical problems arise on videos that contain background music:

1. Whisper marks the music-dominant intervals as 'no_speech' and silently
   skips the speech regions, so they never make it into the transcript.
2. When the XTTS-v2 reference audio contains music, voice cloning degrades —
   the output becomes metallic / robotic / artifact-laden because the model
   interprets the music as a "throat tone" and synthesizes it accordingly.

Solution: separate the input audio into VOCALS + INSTRUMENTAL using Meta's
Demucs (htdemucs) model. Whisper and XTTS then run exclusively on the clean
vocals.wav, and the original instrumental is mixed back in (with auto-ducking)
during the final assembly step.

Design decision: programmatic API (not the CLI)
===============================================
The Demucs CLI relies on ``torchaudio.save``; under torchaudio 2.11+ the save
function mandates the ``torchcodec`` package, which on Windows clashes with
the installed FFmpeg DLL versions (``libtorchcodec_core4/5.dll`` load errors).
We therefore use the programmatic API instead:

  - Audio load:  ffmpeg subprocess -> WAV -> read with ``soundfile``.
  - Demucs:      ``apply_model`` (direct PyTorch, no file I/O).
  - Audio write: ``soundfile`` (bypasses torchaudio.save — no torchcodec needed).

Bonus: in programmatic mode we own the Demucs model lifecycle and can call
``unload()`` to release VRAM in order — required so the Whisper stage can
claim VRAM next.

VRAM estimate: htdemucs ~3 GB GPU inference. A ~4-minute video takes ~30-60s
on the GPU and ~5-10 min on the CPU.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time

log = logging.getLogger(__name__)


class VocalSeparator:
    """Programmatic wrapper around Demucs htdemucs.

    Lazy-loaded: the model is allocated on the first ``separate()`` call.
    ``unload()`` releases VRAM and must be called after the Demucs stage so
    that Whisper / Gemma can claim the GPU for the next pipeline stage.
    """

    MODEL_NAME = "htdemucs"     # Hybrid Transformer Demucs (good speed/quality trade-off).
    DEMUCS_SR = 44100           # Demucs training sample rate; inputs are
                                # resampled to this rate via ffmpeg.

    def __init__(self):
        self._available = None    # is_available() cache.
        self._model = None
        self._device = None

    # ─────────────────────────────────────────────────────────────────────
    # CAPABILITY CHECK
    # ─────────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check whether Demucs + soundfile can be imported. Cached after the first call."""
        if self._available is not None:
            return self._available
        try:
            import demucs.pretrained  # noqa: F401
            import demucs.apply       # noqa: F401
            import soundfile          # noqa: F401
            import torch              # noqa: F401
            self._available = True
        except Exception as e:
            log.warning(f"Demucs availability check failed: {e}")
            self._available = False
        return self._available

    # ─────────────────────────────────────────────────────────────────────
    # MODEL LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Lazy-load the Demucs model on the first separate() invocation."""
        if self._model is not None:
            return
        import torch
        from demucs.pretrained import get_model

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading Demucs ({self.MODEL_NAME}) (device={self._device})...")
        t0 = time.time()
        self._model = get_model(self.MODEL_NAME)
        self._model.to(self._device)
        self._model.eval()
        log.info(f"Demucs ready ({int((time.time() - t0) * 1000)} ms).")

    def unload(self):
        """Evict the Demucs model from VRAM. Called before the Whisper stage in the pipeline."""
        if self._model is None:
            return
        try:
            self._model.cpu()
        except Exception:
            pass
        del self._model
        self._model = None
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("Demucs evicted from VRAM.")

    # ─────────────────────────────────────────────────────────────────────
    # CORE SEPARATION (programmatic API)
    # ─────────────────────────────────────────────────────────────────────

    def separate(self, input_path: str, work_dir: str | None = None,
                 progress_cb=None) -> tuple[str, str]:
        """Split an audio/video file into vocals + instrumental.

        Steps:
          1) Convert to 44.1 kHz stereo PCM WAV via an ffmpeg subprocess
             (bypasses torchaudio.load and the torchcodec conflict).
          2) Read with soundfile -> torch tensor (channels, samples).
          3) Run demucs.apply.apply_model for a 4-stem separation
             (drums / bass / other / vocals).
          4) Write the vocals stem and the sum of the other three stems
             (instrumental) as WAV files via soundfile.

        Returns:
            (vocals_path, instrumental_path) — both inside work_dir.

        Raises:
            RuntimeError: missing dependency, ffmpeg load error or model error.
        """
        if not self.is_available():
            raise RuntimeError(
                "Demucs not available. Install with: pip install demucs"
            )

        if not os.path.exists(input_path):
            raise RuntimeError(f"Input not found: {input_path}")

        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix="ge_demucs_")
        else:
            os.makedirs(work_dir, exist_ok=True)

        # ── 1) ffmpeg → normalize input to 44.1 kHz stereo WAV ───────────
        if progress_cb:
            try:
                progress_cb(0.05, "Demucs: normalizing audio...")
            except Exception:
                pass

        temp_input = os.path.join(work_dir, "input_44k_stereo.wav")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-ar", str(self.DEMUCS_SR), "-ac", "2",
             "-c:a", "pcm_s16le", temp_input],
            capture_output=True, timeout=600,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio normalization failed:\n"
                f"{r.stderr.decode(errors='replace')[-300:]}"
            )

        # ── 2) Read via soundfile → torch tensor ─────────────────────────
        import soundfile as sf
        import torch
        import numpy as np

        audio_np, sr = sf.read(temp_input, dtype="float32", always_2d=True)
        # audio_np shape: (samples, channels). Demucs expects (channels, samples).
        wav_tensor = torch.from_numpy(np.ascontiguousarray(audio_np.T))  # (C, T)

        self._ensure_loaded()

        # ── 3) Demucs apply_model ────────────────────────────────────────
        if progress_cb:
            try:
                progress_cb(0.20, "Demucs: running stem separation...")
            except Exception:
                pass

        from demucs.apply import apply_model

        # Normalization: standard practice during Demucs training (mean/std).
        ref = wav_tensor.mean(0)
        ref_mean = ref.mean()
        ref_std = ref.std() + 1e-8
        wav_norm = (wav_tensor - ref_mean) / ref_std

        t_inf = time.time()
        with torch.no_grad():
            sources = apply_model(
                self._model, wav_norm[None],
                device=self._device,
                progress=False,
                num_workers=0,
            )[0]  # (n_stems, channels, samples)
        sources = sources * ref_std + ref_mean

        log.info(f"Demucs inference: {int((time.time() - t_inf))}s, "
                 f"{sources.shape[0]} stems")

        # ── 4) Split into vocals + instrumental and persist via soundfile ─
        if progress_cb:
            try:
                progress_cb(0.90, "Demucs: writing outputs...")
            except Exception:
                pass

        stem_names = list(self._model.sources)  # ['drums','bass','other','vocals']
        vocals_idx = stem_names.index("vocals")
        vocals = sources[vocals_idx]  # (channels, samples)

        # Instrumental = sum of every stem except vocals.
        instrumental = torch.zeros_like(vocals)
        for i, name in enumerate(stem_names):
            if name != "vocals":
                instrumental = instrumental + sources[i]

        vocals_path = os.path.join(work_dir, "vocals.wav")
        instr_path = os.path.join(work_dir, "no_vocals.wav")

        # soundfile expects (samples, channels) — transpose with .T.
        sf.write(vocals_path, vocals.cpu().numpy().T, sr, subtype="PCM_16")
        sf.write(instr_path, instrumental.cpu().numpy().T, sr, subtype="PCM_16")

        # Remove the intermediate input file.
        try:
            os.remove(temp_input)
        except OSError:
            pass

        if progress_cb:
            try:
                progress_cb(1.0, "Demucs: complete.")
            except Exception:
                pass

        log.info(f"Demucs separation complete: {vocals_path}, {instr_path}")
        return vocals_path, instr_path

    # ─────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def cleanup(work_dir: str):
        """Remove the Demucs work directory once vocals/instrumental are no longer needed."""
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            log.warning(f"Failed to remove Demucs work_dir ({work_dir}): {e}")
