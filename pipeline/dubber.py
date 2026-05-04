"""
Gemma Echo — Video Dublaj Boru Hatti (Tek Konusmaci)

Turkce video -> Ingilizce ses, orijinal konusmaci sesi korunur.

Pipeline siralari:
  1. ffmpeg      : Video'dan 16kHz mono WAV ayikla
  2. Whisper     : Zaman damgali transkript (segment start/end/text)
  3. Gemma 4 Q4  : Her segmenti yerelde Ingilizceye cevir (llama-cpp)
  4. XTTS-v2     : Referans sesten speaker latent hesapla (bir kez)
  5. XTTS-v2     : Her segment icin ses klonlama inference
  6. NumPy/sf    : Segmentleri orijinal zaman eksenine yerlestir
  7. ffmpeg      : Yeni audio track'i videoyla birlestir

Cikti: <kaynak_video>_dubbed.mp4
"""

import os
import wave
import subprocess
import numpy as np
import soundfile as sf


class DubbingPipeline:

    XTTS_SR = 24000  # XTTS-v2 cikti ornekleme hizi

    def __init__(self, transcriber, translator, synthesizer):
        self.transcriber = transcriber
        self.translator  = translator
        self.synthesizer = synthesizer

        self._project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._tmp_dir = os.path.join(self._project_dir, ".tmp_audio")
        os.makedirs(self._tmp_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # ANA PIPELINE
    # ═══════════════════════════════════════════════════════════

    def process(self, video_path: str, output_path: str, src_lang="tr", tgt_lang="en", 
                src_name="Turkish", tgt_name="English", progress_cb=None):
        """
        Tam dublaj pipeline'ini calistirir.

        Args:
            video_path:  Kaynak video (mp4, mkv, avi, ...)
            output_path: Cikti yolu (<video>_dubbed.mp4)
            progress_cb: (fraction, msg, color) -> None  [GUI koprusu]

        Returns:
            output_path (basarida)

        Raises:
            RuntimeError: Kritik hata durumunda
        """

        def prog(f, msg, color="#5b9ef9"):
            print(f"[DUBBER] ({int(f*100)}%) {msg}")
            if progress_cb:
                progress_cb(f, msg, color)

        wav_path = ref_wav = dubbed_wav = None
        seg_wavs = []

        try:
            # ── 1. Ses ayikla ─────────────────────────────────────────
            prog(0.03, "Video'dan ses ayiklaniyor...", "#f5a623")
            wav_path = self._extract_wav(video_path)
            video_duration = self._get_wav_duration(wav_path)
            prog(0.08, f"Video suresi: {video_duration:.1f}s", "#5b9ef9")

            # ── 2. Whisper transkript ──────────────────────────────────
            prog(0.10, "Whisper transkript olusturuluyor...", "#5b9ef9")
            segments = self._transcribe_segments(wav_path, language=src_lang)
            if not segments:
                raise RuntimeError(f"Videoda {src_name} konusma taninamadi.")
            prog(0.22, f"{len(segments)} segment tanindi.", "#23d05e")

            # ── 3. Yerel Gemma 4 Q4 ile ceviri ────────────────────────
            if self.translator.local_llm is None:
                prog(0.24, "Yerel Gemma 4 yukleniyor (30-60sn)...", "#f5a623")
                self.translator.load_local_model()

            translated = []
            total = len(segments)
            for i, seg in enumerate(segments):
                p = 0.24 + 0.28 * (i / total)
                prog(p, f"Ceviri: {i+1}/{total}  \"{seg['text'][:40]}\"", "#5b9ef9")
                result = self.translator.translate(
                    seg["text"], 
                    src_lang=src_lang, 
                    tgt_lang=tgt_lang,
                    src_name=src_name,
                    tgt_name=tgt_name
                )
                text_en = result.get("translation", "").strip()
                translated.append({**seg, "text_en": text_en})

            # ── 4. Referans ses (konusmaci profili) ───────────────────
            prog(0.53, "Konusmaci referans sesi seciliyor...", "#f5a623")
            ref_wav = self._extract_reference(wav_path, segments)

            # ── 5. XTTS speaker latent (bir kez hesapla) ──────────────
            prog(0.56, "XTTS-v2 hazırlaniyor...", "#f5a623")
            self._ensure_xtts_loaded()
            prog(0.60, "Konusmaci ses profili olusturuluyor...", "#f5a623")
            gpt_latent, spk_emb = self._get_speaker_latents(ref_wav)

            # ── 5b. Her segment icin ses sentezi ─────────────────────
            seg_wavs = [None] * total
            for i, seg in enumerate(translated):
                p = 0.60 + 0.28 * (i / total)
                prog(p, f"Ses sentezi: {i+1}/{total}", "#5b9ef9")

                if not seg.get("text_en"):
                    continue

                out_wav = os.path.join(self._tmp_dir, f"dub_seg_{i:04d}.wav")
                self._synthesize_segment(
                    seg["text_en"], gpt_latent, spk_emb, out_wav
                )
                seg_wavs[i] = out_wav

            # ── 6. Timeline assembly ──────────────────────────────────
            prog(0.89, "Ses parcalari timeline'a yerlestiriliyor...", "#f5a623")
            dubbed_wav = os.path.join(self._tmp_dir, "dubbed_full.wav")
            self._assemble_audio(translated, seg_wavs, video_duration, dubbed_wav)

            # ── 7. Video + audio birlestir ────────────────────────────
            prog(0.94, "Video ile birlestiriliyor...", "#f5a623")
            self._mux(video_path, dubbed_wav, output_path)

            prog(1.0, f"Tamamlandi!  →  {os.path.basename(output_path)}", "#23d05e")
            return output_path

        finally:
            # Gecici dosyalari temizle
            self._cleanup(seg_wavs + [wav_path, ref_wav, dubbed_wav])

    # ═══════════════════════════════════════════════════════════
    # ADIM 1: WAV AYIKLAMA
    # ═══════════════════════════════════════════════════════════

    def _extract_wav(self, video_path: str) -> str:
        out = os.path.join(self._tmp_dir, "dub_input.wav")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out],
            capture_output=True, timeout=600
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg ses ayiklama hatasi:\n{r.stderr.decode(errors='replace')[:300]}"
            )
        return out

    def _get_wav_duration(self, wav_path: str) -> float:
        with wave.open(wav_path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()

    # ═══════════════════════════════════════════════════════════
    # ADIM 2: ZAMAN DAMGALI TRANSKRIPT
    # ═══════════════════════════════════════════════════════════

    def _transcribe_segments(self, wav_path: str, language="tr") -> list:
        """Faster-whisper ile her segment icin start/end/text doner."""
        segments_iter, _ = self.transcriber.model.transcribe(
            wav_path,
            language=language,
            beam_size=2,
            best_of=2,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        result = []
        for seg in segments_iter:
            if seg.no_speech_prob > 0.5:
                continue
            text = seg.text.strip()
            if not text:
                continue
            result.append({"start": seg.start, "end": seg.end, "text": text})
        return result

    # ═══════════════════════════════════════════════════════════
    # ADIM 4: REFERANS SES
    # ═══════════════════════════════════════════════════════════

    def _extract_reference(self, wav_path: str, segments: list) -> str:
        """En uzun ve net segmenti (max 8sn) referans ses olarak cikarir."""
        best = max(segments, key=lambda s: s["end"] - s["start"])
        start_s = best["start"]
        end_s   = min(best["end"], start_s + 8.0)

        out = os.path.join(self._tmp_dir, "dub_reference.wav")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path,
             "-ss", str(start_s), "-to", str(end_s),
             "-ar", "22050", "-ac", "1", out],
            capture_output=True, timeout=30
        )
        if r.returncode != 0:
            # Fallback: ilk 6 saniyeyi al
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path,
                 "-ss", "0", "-to", "6",
                 "-ar", "22050", "-ac", "1", out],
                capture_output=True, timeout=30
            )
        return out

    # ═══════════════════════════════════════════════════════════
    # ADIM 5: XTTS SES KLONLAMA
    # ═══════════════════════════════════════════════════════════

    def _ensure_xtts_loaded(self):
        """XTTS modelinin GPU'da yuklü olmasini saglar."""
        if self.synthesizer.xtts_model is None:
            if self.synthesizer._xtts_loading:
                self.synthesizer._xtts_ready.wait()
            else:
                self.synthesizer._load_xtts_model(use_gpu=True)

    def _get_speaker_latents(self, ref_wav: str):
        """Referans sesten speaker kondisyonlama latent'ini hesaplar (bir kez)."""
        tts_model = self.synthesizer.xtts_model.synthesizer.tts_model
        gpt_latent, spk_emb = tts_model.get_conditioning_latents(
            audio_path=[ref_wav]
        )
        return gpt_latent, spk_emb

    def _synthesize_segment(self, text: str, gpt_latent, spk_emb,
                            out_path: str):
        """Tek segment icin XTTS inference, WAV dosyasina yazar."""
        tts_model = self.synthesizer.xtts_model.synthesizer.tts_model
        output = tts_model.inference(
            text=text,
            language="en",
            gpt_cond_latent=gpt_latent,
            speaker_embedding=spk_emb,
            temperature=0.65,
            speed=1.0,
        )
        wav_np = np.array(output["wav"], dtype=np.float32)
        sf.write(out_path, wav_np, self.XTTS_SR)

    # ═══════════════════════════════════════════════════════════
    # ADIM 6: TIMELINE ASSEMBLY
    # ═══════════════════════════════════════════════════════════

    def _assemble_audio(self, segments: list, seg_wavs: list,
                        video_duration: float, out_path: str):
        """
        Sentezlenen segmentleri orijinal zaman damgalarina yerlestir.
          - Ingilizce ses ozgunden uzunsa: ffmpeg atempo ile sikistir
          - Kisa kalirsa : sessizlik ile doldurulur (zaten sifir array)
        """
        SR = self.XTTS_SR
        total_frames = int((video_duration + 1.0) * SR)
        output = np.zeros(total_frames, dtype=np.float32)

        for seg, wav_path in zip(segments, seg_wavs):
            if wav_path is None or not os.path.exists(wav_path):
                continue

            seg_duration = seg["end"] - seg["start"]
            start_frame  = int(seg["start"] * SR)

            data, _ = sf.read(wav_path, dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]

            eng_duration = len(data) / SR

            # Ingilizce ses ozgunden >%5 uzunsa hizlandir
            if eng_duration > seg_duration * 1.05:
                ratio = min(eng_duration / seg_duration, 2.5)
                data  = self._atempo_stretch(data, SR, ratio, wav_path)

            end_frame = min(start_frame + len(data), total_frames)
            output[start_frame:end_frame] = data[:end_frame - start_frame]

        sf.write(out_path, output, SR)

    def _atempo_stretch(self, data: np.ndarray, sr: int, ratio: float,
                        orig_path: str) -> np.ndarray:
        """ffmpeg atempo filtresi ile ses hizlandirir (ratio > 1.0)."""
        tmp_out = orig_path.replace(".wav", "_tempo.wav")

        # atempo max 2.0; daha yuksek icin zincirleme
        if ratio <= 2.0:
            af = f"atempo={ratio:.4f}"
        else:
            af = f"atempo=2.0,atempo={ratio / 2.0:.4f}"

        r = subprocess.run(
            ["ffmpeg", "-y", "-i", orig_path, "-filter:a", af, tmp_out],
            capture_output=True, timeout=60
        )
        if r.returncode == 0 and os.path.exists(tmp_out):
            result, _ = sf.read(tmp_out, dtype="float32")
            try:
                os.remove(tmp_out)
            except OSError:
                pass
            return result
        return data  # Hata durumunda orijinali kullan

    # ═══════════════════════════════════════════════════════════
    # ADIM 7: VIDEO MUX
    # ═══════════════════════════════════════════════════════════

    def _mux(self, video_path: str, audio_path: str, output_path: str):
        """Orijinal video stream + yeni audio track birlestir.
        Video yeniden encode edilmez (stream copy) — hizli."""
        r = subprocess.run(
            ["ffmpeg", "-y",
             "-i", video_path,
             "-i", audio_path,
             "-c:v", "copy",
             "-map", "0:v:0",
             "-map", "1:a:0",
             "-shortest",
             output_path],
            capture_output=True, timeout=600
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg mux hatasi:\n{r.stderr.decode(errors='replace')[:300]}"
            )

    # ═══════════════════════════════════════════════════════════
    # TEMIZLIK
    # ═══════════════════════════════════════════════════════════

    def _cleanup(self, paths: list):
        for p in paths:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
