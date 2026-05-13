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

    def __init__(self, transcriber, translator, synthesizer, config=None):
        self.transcriber = transcriber
        self.translator  = translator
        self.synthesizer = synthesizer
        self.config      = config

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

        wav_path = dubbed_wav = None
        ref_wavs = []
        seg_wavs = []

        # Dublajda kullanıcı ayarları baz alınmalı; sadece varsayılan olarak local offline çalışmamalı.
        # Mevcut modu kaydet, finally'de geri yükle (kullanıcının online tercihi bozulmasın).
        prev_mode = self.translator.mode
        prev_stt_mode = self.transcriber.mode

        target_mode = self.translator.mode
        if self.config is not None:
            target_mode = self.config.get("mode", "llm", "backend", default=target_mode)

        if target_mode == "offline":
            self.translator.set_mode("offline")
        else:
            self.translator.set_mode("online")
            self.translator.unload_local_model()

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

            # XTTS, 1-3 kelimelik veya <1.5s segmentlerde garip prosody uretir.
            # Bu tip kisa segmentleri bir oncekine birlestirip daha dogal cumleler
            # olustur. Hem cevirinin baglami iyilesir, hem TTS kalitesi artar.
            raw_count = len(segments)
            segments = self._merge_short_segments(segments)
            if len(segments) < raw_count:
                prog(0.22, f"{raw_count} segment -> {len(segments)} (kisa segmentler birlestirildi).", "#23d05e")
            else:
                prog(0.22, f"{len(segments)} segment tanindi.", "#23d05e")

            # ── VRAM serbest bırak: Whisper bitti, sırayla LLM ve XTTS gelecek ──
            # 6GB GPU'da Whisper(~1GB) + Gemma 8192ctx(~4GB) + XTTS(~2GB) = ~7GB sığmaz.
            # Sadece STT GPU modeli aktifse boşalt; cloud_auto zaten VRAM kullanmaz.
            if prev_stt_mode in ("local_gpu", "local_gpu_hq"):
                try:
                    self.transcriber.set_mode("local_cpu")
                    prog(0.23, "Whisper VRAM bosaltildi (LLM icin pay aciliyor)...", "#5b9ef9")
                except Exception:
                    # Bosaltma basarisiz olsa bile dublaj devam etsin.
                    pass

            # ── 3. Yerel Gemma 4 Q4 ile ceviri ────────────────────────
            # Whisper segmentleri kisa (5-15 kelime). Her biri standalone cevirilirse
            # zamir/baglam kaybi olur. Direkt media cevirisindeki kalite icin
            # prev_translation + son N segmentin kaynagi + rolling_summary gecirilir.
            # Bu nedenle yerel modeli genis ctx ile yukluyoruz (8192).
            if self.translator.local_llm is None:
                prog(0.24, "Yerel Gemma 4 yukleniyor (30-60sn)...", "#f5a623")
                if not self.translator.load_local_model(8192):
                    raise RuntimeError(
                        "Yerel Gemma yuklenemedi (VRAM). "
                        "Ayarlar'dan LLM'i online yapin veya daha dusuk VRAM profili secin."
                    )
            elif getattr(self.translator, "loaded_n_ctx", 512) < 8192:
                # Zaten yukluyse ama kucuk ctx ile, baglam icin yeniden yukle.
                prog(0.24, "Yerel Gemma 4 baglam icin yeniden yukleniyor...", "#f5a623")
                self.translator.load_local_model(8192)

            translated = []
            total = len(segments)
            prev_translation = ""
            rolling_summary = ""
            CTX_WINDOW = 3  # son N segmentin kaynak metni baglam olarak gecirilir

            for i, seg in enumerate(segments):
                p = 0.24 + 0.28 * (i / total)
                prog(p, f"Ceviri: {i+1}/{total}  \"{seg['text'][:40]}\"", "#5b9ef9")

                # Son CTX_WINDOW segmentin Turkce metni — referans baglam
                context_segs = [s["text"] for s in segments[max(0, i - CTX_WINDOW):i]]

                result = self.translator.translate(
                    seg["text"],
                    context=context_segs,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    src_name=src_name,
                    tgt_name=tgt_name,
                    prev_translation=prev_translation,
                    rolling_summary=rolling_summary,
                )
                text_en = result.get("translation", "").strip()
                translated.append({**seg, "text_en": text_en})
                prev_translation = text_en

                # Her 5 segmentte bir rolling summary guncelle (kitap ceviri pattern'i).
                # Segmentler kisa oldugu icin 3 yerine 5 — gereksiz LLM cagrisi azaltir.
                if (i + 1) % 5 == 0 or i == 0:
                    try:
                        rolling_summary = self.translator.generate_summary(
                            text_en, rolling_summary
                        )
                    except Exception:
                        # Ozet basarisiz olsa bile ana ceviri akisi devam etsin.
                        pass

            # ── VRAM serbest birak: tum segmentler cevrildi, Gemma'ya artik gerek yok ──
            # XTTS-v2 ~2GB VRAM lazim. Gemma 8192ctx (~4GB) bosaltilmasi sonraki
            # asamayi rahatlatir, OOM riskini sifirlar.
            try:
                self.translator.unload_local_model()
                prog(0.52, "Gemma VRAM bosaltildi (XTTS icin pay aciliyor)...", "#5b9ef9")
            except Exception:
                pass

            # ── 4. Referans ses (konusmaci profili) ─────────────────────
            prog(0.53, "Konusmaci referans sesi seciliyor...", "#f5a623")
            ref_wavs = self._extract_reference(wav_path, segments)
            prog(0.55, f"{len(ref_wavs)} adet temiz referans segmenti seciliyor.", "#23d05e")

            # ── 5. XTTS speaker latent (bir kez hesapla) ──────────────
            prog(0.56, "XTTS-v2 hazırlaniyor...", "#f5a623")
            self._ensure_xtts_loaded()
            prog(0.60, "Konusmaci ses profili olusturuluyor...", "#f5a623")
            gpt_latent, spk_emb = self._get_speaker_latents(ref_wavs)

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
            # Kullanicinin orijinal modunu geri yukle (online ise online).
            try:
                self.translator.set_mode(prev_mode)
            except Exception:
                pass
            # Whisper'i de orijinal moduna dondur (canli ceviri etkilenmesin).
            try:
                self.transcriber.set_mode(prev_stt_mode)
            except Exception:
                pass
            # Gecici dosyalari temizle
            self._cleanup(seg_wavs + ref_wavs + [wav_path, dubbed_wav])

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
        """Faster-whisper ile her segment icin start/end/text doner.

        Dublaj icin **ozel bir medium model** yukler — kullanicinin
        transcriber'i etkilenmez. Bu kritik: cloud_auto modda kullanicinin
        transcriber.model'i 'base/CPU' fallback olur, Turkce icin yetersizdir.
        Dublaj batch bir is, kalite > hiz; bu yuzden medium kullaniyoruz.

        Strateji:
          1. GPU varsa medium-GPU (en iyi Turkce dogrulugu, ~1.5 GB VRAM)
          2. GPU yoksa veya yuklemede OOM olursa small-CPU'ya dus
          3. Transkript bitince modeli serbest birak (LLM icin VRAM ac)
        """
        from faster_whisper import WhisperModel
        import torch as _torch
        import gc as _gc

        local_model = None
        used_size, used_device = "small", "cpu"

        # 1. Once medium-GPU dene
        if _torch.cuda.is_available():
            try:
                print("[DUBBER] Whisper 'medium' GPU yukleniyor (yuksek kalite transkript)...")
                local_model = WhisperModel("medium", device="cuda", compute_type="int8")
                used_size, used_device = "medium", "cuda"
            except Exception as e:
                print(f"[DUBBER] medium-GPU basarisiz ({e}); small-CPU'ya dusuluyor.")
                local_model = None

        # 2. Fallback: small-CPU (base'den daha iyi, CPU'da makul hiz)
        if local_model is None:
            print("[DUBBER] Whisper 'small' CPU yukleniyor (fallback)...")
            local_model = WhisperModel("small", device="cpu", compute_type="int8")
            used_size, used_device = "small", "cpu"

        try:
            segments_iter, _info = local_model.transcribe(
                wav_path,
                language=language,
                beam_size=5,                          # 2 -> 5 (daha iyi arama, +%10 dogruluk)
                best_of=5,                            # 2 -> 5
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
                condition_on_previous_text=True,      # Baglam yardim eder, hatali metin yapismaz
                temperature=0.0,                      # Deterministic
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.5,
            )
            result = []
            for seg in segments_iter:
                if seg.no_speech_prob > 0.5:
                    continue
                text = seg.text.strip()
                if not text:
                    continue
                result.append({"start": seg.start, "end": seg.end, "text": text})
            print(f"[DUBBER] Whisper '{used_size}/{used_device}' transkript bitti, {len(result)} segment.")
            return result
        finally:
            # Modeli hemen serbest birak — Gemma'ya VRAM lazim
            try:
                del local_model
            except Exception:
                pass
            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

    def _merge_short_segments(self, segments: list) -> list:
        """Cok kisa segmentleri bir oncekine birlestir.

        XTTS-v2 < 1.5 saniye veya 1-3 kelimelik metinlerde garip prosody ve
        akustik artifakt uretir. Bu segmentleri oncekiyle birlestirerek daha
        dogal cumleler olustururuz. Yan etki: cevirinin baglami da iyilesir
        (kisa parcayi standalone cevirmek yerine biraz daha uzun context).

        Birlestirme kosullari:
          - Sure < 1.5s VEYA kelime sayisi < 4 (kisa segment kriteri)
          - Onceki segmentle arasinda < 0.6sn bosluk (akustik uzaklik)
          - Birlestikten sonra toplam sure <= 12s (XTTS sınırı asilmasin)
        """
        if not segments:
            return segments

        MIN_DURATION  = 1.5
        MIN_WORDS     = 4
        MAX_GAP       = 0.6
        MAX_MERGED    = 12.0

        merged = []
        for seg in segments:
            duration = seg["end"] - seg["start"]
            words    = len(seg["text"].split())
            is_short = duration < MIN_DURATION or words < MIN_WORDS

            if merged and is_short:
                prev = merged[-1]
                gap = seg["start"] - prev["end"]
                new_total = seg["end"] - prev["start"]
                if gap <= MAX_GAP and new_total <= MAX_MERGED:
                    prev["end"]  = seg["end"]
                    prev["text"] = (prev["text"].rstrip() + " " + seg["text"].lstrip()).strip()
                    continue

            merged.append(dict(seg))
        return merged

    # ═══════════════════════════════════════════════════════════
    # ADIM 4: REFERANS SES
    # ═══════════════════════════════════════════════════════════

    def _extract_reference(self, wav_path: str, segments: list) -> list:
        """Konusmaci klonlamasi icin coklu referans ses cikarir.

        XTTS-v2 birden fazla referans kabul eder; bu konusmaci identity'sini
        sabitler ve tek bir bozuk segmentin (muzik, gurultu) etkisini azaltir.

        Strateji:
          - 4-10 saniye araliginda olan segmentleri tercih et (cok kisa = az bilgi,
            cok uzun = arka plan gurultusu birikir)
          - En uzun 3 segmenti seç
          - Her birinde ffmpeg silenceremove ile bas/son sessizliklerini kirp
          - Sonuc < 2sn ise o referansi reddet

        Returns: Liste halinde wav dosya yollari (1-3 adet).
        """
        # Aday segmentler: 4-10 saniye araliginda
        candidates = [s for s in segments
                      if 4.0 <= (s["end"] - s["start"]) <= 10.0]

        if not candidates:
            # Fallback: en uzun 1 segment (her ihtimale karsi)
            candidates = sorted(segments, key=lambda s: s["end"] - s["start"],
                                reverse=True)[:1]
        else:
            # En uzun 3'u (XTTS multi-ref icin yeterli)
            candidates = sorted(candidates, key=lambda s: s["end"] - s["start"],
                                reverse=True)[:3]

        # silenceremove: bas tarafta 0.1sn sessizligi -40dB esikle kirp,
        # areverse ile son tarafta da ayni islemi tekrarla
        silence_filter = (
            "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,"
            "areverse,"
            "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,"
            "areverse"
        )

        ref_paths = []
        for i, seg in enumerate(candidates):
            out = os.path.join(self._tmp_dir, f"dub_ref_{i}.wav")
            end_s = min(seg["end"], seg["start"] + 10.0)
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path,
                 "-ss", str(seg["start"]), "-to", str(end_s),
                 "-af", silence_filter,
                 "-ar", "22050", "-ac", "1", out],
                capture_output=True, timeout=30
            )
            if r.returncode == 0 and os.path.exists(out):
                # Minimum 2sn — daha kisa ref XTTS icin yetersiz
                try:
                    dur = self._get_wav_duration(out)
                    if dur >= 2.0:
                        ref_paths.append(out)
                        continue
                except Exception:
                    pass
                # 2sn'den kisa veya okunamadi: dosyayi sil
                try:
                    os.remove(out)
                except OSError:
                    pass

        # Hicbir aday yeterli olmazsa: ilk 8sn'yi cig al (son care)
        if not ref_paths:
            fallback = os.path.join(self._tmp_dir, "dub_ref_fallback.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path,
                 "-ss", "0", "-to", "8",
                 "-ar", "22050", "-ac", "1", fallback],
                capture_output=True, timeout=30
            )
            if os.path.exists(fallback):
                ref_paths.append(fallback)

        return ref_paths

    # ═══════════════════════════════════════════════════════════
    # ADIM 5: XTTS SES KLONLAMA
    # ═══════════════════════════════════════════════════════════

    def _ensure_xtts_loaded(self):
        """XTTS modelinin GPU'da yuklü olmasini saglar."""
        if self.synthesizer.xtts_model is None:
            if self.synthesizer._xtts_loading:
                self.synthesizer._xtts_ready.wait()
            elif not self.synthesizer._load_xtts_model(use_gpu=True):
                raise RuntimeError(
                    "XTTS GPU yuklenemedi (VRAM). Dublaj icin yeterli VRAM gerekir."
                )
        if self.synthesizer.xtts_model is None:
            raise RuntimeError("XTTS modeli yuklenemedi.")

    def _get_speaker_latents(self, ref_wavs):
        """Referans seslerden speaker kondisyonlama latent'ini hesaplar (bir kez).

        ref_wavs: tekil str veya str listesi (XTTS multi-ref destegi).
        Coklu ref kullanildiginda XTTS GPT bunlari ortalar -> konusmaci
        identity'si daha sabit, tek bir bozuk segmentin etkisi azalir.
        """
        if isinstance(ref_wavs, str):
            ref_wavs = [ref_wavs]
        tts_model = self.synthesizer.xtts_model.synthesizer.tts_model
        gpt_latent, spk_emb = tts_model.get_conditioning_latents(
            audio_path=ref_wavs
        )
        return gpt_latent, spk_emb

    def _synthesize_segment(self, text: str, gpt_latent, spk_emb,
                            out_path: str):
        """Tek segment icin XTTS inference, WAV dosyasina yazar.

        Konservatif/stabil parametreler — dublaj kalitesi icin secildi.
        XTTS-v2 default'lari: temperature=0.75, length_penalty=1.0,
        repetition_penalty=10.0, top_k=50, top_p=0.85, speed=1.0.

          - temperature=0.50 (↓0.75): daha tutarli prosody, daha az 'dogaclama'.
            Belgesel anlatim icin idealdir; tek konusmaci tonunu sabitler.
          - repetition_penalty=10.0 (default): XTTS halusinasyon egilimine karsi
            agresif. Default'u koruyoruz — kelime tekrari/takilmayi onler.
          - length_penalty=1.0 (default): konusma suresine notr.
          - top_k=50, top_p=0.85 (default): asiri sampling yok.
          - speed=1.0: hiz timeline assembly'de atempo ile ayarlanir.
          - enable_text_splitting=False: cumle bolme isini disarida segment
            bazinda zaten yapiyoruz; XTTS'e tek cumle veriyoruz.
        """
        tts_model = self.synthesizer.xtts_model.synthesizer.tts_model
        output = tts_model.inference(
            text=text,
            language="en",
            gpt_cond_latent=gpt_latent,
            speaker_embedding=spk_emb,
            temperature=0.50,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=50,
            top_p=0.85,
            speed=1.0,
            enable_text_splitting=False,
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

            # Ingilizce ses ozgunden >%5 uzunsa hizlandir.
            # ANCAK 1.3x uzeri atempo cizirti/robotluk yapar — bu seviyenin uzerine
            # cikma. Asiri tasma durumunda segment ya sonraki segmente hafifce
            # tasacak (kabul edilebilir overlap) ya da onceki versiyondaki gibi
            # cizirti yapacakti — daha temiz ses tercih edilir.
            if eng_duration > seg_duration * 1.05:
                ratio = eng_duration / seg_duration
                if ratio <= 1.30:
                    data = self._atempo_stretch(data, SR, ratio, wav_path)
                # ratio > 1.30: hizlandirmayi atla, sesi temiz birak
                # (kullanici hafif zaman kaymasi gorebilir ama ses cizirti yapmaz)

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
