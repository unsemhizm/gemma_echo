"""Demucs tabanli vokal ayristirici — dublaj on-isleme.

Amac
====
Arka plan muzigi olan videolarda iki kritik sorun olusuyor:

1. Whisper, muzik dominant oldugu anlarda 'no_speech' karari verip konusma
   bolgelerini transkripsiyona dahil etmiyor (sessizce atliyor).
2. XTTS-v2 referans seste muzik duydugunda klonlamayi bozuyor: cikti ses
   metalik / robotik / cizirti dolu hale geliyor (model muzigi 'gırtlaktan
   gelen bir dokun' sanip sentezliyor).

Cozum: Meta'nin Demucs (htdemucs) modeli ile sesi VOCALS + INSTRUMENTAL
olarak ayir. Whisper ve XTTS yalnizca tertemiz vocals.wav uzerinde calisir;
final mix asamasinda orijinal instrumental geri eklenir (auto-ducking ile).

Mimari karari: Programmatic API (CLI degil)
===========================================
Demucs CLI'yi torchaudio.save kullaniyor — torchaudio 2.11+ save fonksiyonu
artik 'torchcodec' paketini zorunlu kiliyor. torchcodec ise Windows'ta
FFmpeg surum DLL'leriyle catisip yuklenemiyor (libtorchcodec_core4/5.dll
hatasi). Bu nedenle CLI yerine programmatic API kullaniyoruz:

  - Ses yukleme:  ffmpeg subprocess -> WAV -> soundfile ile oku
  - Demucs:       apply_model (PyTorch direkt, dosya I/O yok)
  - Ses yazma:    soundfile (torchaudio.save bypass — torchcodec gereksiz)

Bonus: programmatic modda Demucs modelini biz kontrol ediyoruz, unload()
metodu ile VRAM'i sirali bosaltabiliyoruz (pipeline'da Whisper'a yer acmak
icin gerekli).

VRAM tahmini: htdemucs ~3GB GPU inference. ~4 dakikalik video icin
GPU'da ~30-60 sn, CPU'da ~5-10 dk surer.
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
    """Demucs htdemucs programmatic wrapper.

    Lazy-load: ilk separate() cagrisinda model GPU/CPU'ya yuklenir.
    unload(): VRAM'i bosaltir — pipeline'da Whisper/Gemma'ya yer acmak icin
    Demucs adimi bittikten sonra cagrilmali.
    """

    MODEL_NAME = "htdemucs"     # Hybrid Transformer Demucs (denge: hız + kalite)
    DEMUCS_SR = 44100           # Demucs egitim sample rate'i; girdi bu hiza
                                # cevrilir (ffmpeg ile)

    def __init__(self):
        self._available = None    # is_available() cache
        self._model = None
        self._device = None

    # ─────────────────────────────────────────────────────────────────────
    # YETERLILIK KONTROLU
    # ─────────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Demucs + soundfile import edilebilir mi? Tek sefer test, cache'li."""
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
    # MODEL YASAM DONGUSU
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Demucs modelini lazy-load yapar (ilk separate() cagrisinda)."""
        if self._model is not None:
            return
        import torch
        from demucs.pretrained import get_model

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Demucs ({self.MODEL_NAME}) yukleniyor (device={self._device})...")
        t0 = time.time()
        self._model = get_model(self.MODEL_NAME)
        self._model.to(self._device)
        self._model.eval()
        log.info(f"Demucs hazir ({int((time.time() - t0) * 1000)}ms).")

    def unload(self):
        """Demucs modelini VRAM'den bosalt. Pipeline'da Whisper'dan once cagrilir."""
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
        log.info("Demucs VRAM'den bosaltildi.")

    # ─────────────────────────────────────────────────────────────────────
    # ASIL AYIRMA (programmatic API)
    # ─────────────────────────────────────────────────────────────────────

    def separate(self, input_path: str, work_dir: str | None = None,
                 progress_cb=None) -> tuple[str, str]:
        """Bir audio/video dosyasini vocals + instrumental olarak ayirir.

        Adimlar:
          1) ffmpeg subprocess ile 44.1kHz stereo PCM WAV'a cevir (torchaudio
             yuklemesini bypass — torchcodec catismasi olmaz).
          2) soundfile ile oku -> torch tensor (channels, samples).
          3) demucs.apply.apply_model ile 4-stem ayrim (drums/bass/other/vocals).
          4) vocals stem'i ve diger 3 stem'in toplamini (instrumental) soundfile
             ile WAV olarak yaz.

        Returns:
            (vocals_path, instrumental_path) — ikisi de work_dir altinda.

        Raises:
            RuntimeError: bagimlilik eksik, ffmpeg yukleme hatasi veya model
                          hatasi.
        """
        if not self.is_available():
            raise RuntimeError(
                "Demucs bulunamadi. Yuklemek icin: pip install demucs"
            )

        if not os.path.exists(input_path):
            raise RuntimeError(f"Girdi bulunamadi: {input_path}")

        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix="ge_demucs_")
        else:
            os.makedirs(work_dir, exist_ok=True)

        # ── 1) ffmpeg ile 44.1kHz stereo WAV'a normalize et ──────────────
        if progress_cb:
            try:
                progress_cb(0.05, "Demucs: ses normalize ediliyor...")
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
                f"ffmpeg ses normalize hatasi:\n"
                f"{r.stderr.decode(errors='replace')[-300:]}"
            )

        # ── 2) soundfile ile yukle → torch tensor ────────────────────────
        import soundfile as sf
        import torch
        import numpy as np

        audio_np, sr = sf.read(temp_input, dtype="float32", always_2d=True)
        # audio_np shape: (samples, channels). Demucs (channels, samples) ister.
        wav_tensor = torch.from_numpy(np.ascontiguousarray(audio_np.T))  # (C, T)

        self._ensure_loaded()

        # ── 3) Demucs apply_model ────────────────────────────────────────
        if progress_cb:
            try:
                progress_cb(0.20, "Demucs: stem ayristirma calisiyor...")
            except Exception:
                pass

        from demucs.apply import apply_model

        # Normalizasyon: Demucs eğitiminde standart pratik (mean/std)
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

        # ── 4) Vocals + instrumental olarak ayir ve soundfile ile yaz ────
        if progress_cb:
            try:
                progress_cb(0.90, "Demucs: ciktilar yaziliyor...")
            except Exception:
                pass

        stem_names = list(self._model.sources)  # ['drums','bass','other','vocals']
        vocals_idx = stem_names.index("vocals")
        vocals = sources[vocals_idx]  # (channels, samples)

        # Instrumental = vocals haric tum stem'lerin toplami
        instrumental = torch.zeros_like(vocals)
        for i, name in enumerate(stem_names):
            if name != "vocals":
                instrumental = instrumental + sources[i]

        vocals_path = os.path.join(work_dir, "vocals.wav")
        instr_path = os.path.join(work_dir, "no_vocals.wav")

        # soundfile (samples, channels) bekler → .T transpose
        sf.write(vocals_path, vocals.cpu().numpy().T, sr, subtype="PCM_16")
        sf.write(instr_path, instrumental.cpu().numpy().T, sr, subtype="PCM_16")

        # Gecici giris dosyasini sil
        try:
            os.remove(temp_input)
        except OSError:
            pass

        if progress_cb:
            try:
                progress_cb(1.0, "Demucs: tamam.")
            except Exception:
                pass

        log.info(f"Demucs ayrim tamam: {vocals_path}, {instr_path}")
        return vocals_path, instr_path

    # ─────────────────────────────────────────────────────────────────────
    # TEMIZLIK
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def cleanup(work_dir: str):
        """Demucs gecici klasorunu siler (vocals/instrumental artik gerekmiyorsa)."""
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            log.warning(f"Demucs work_dir silinemedi ({work_dir}): {e}")
