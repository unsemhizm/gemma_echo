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
import re
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
                src_name="Turkish", tgt_name="English", progress_cb=None,
                transcript_cb=None, translation_cb=None):
        """
        Tam dublaj pipeline'ini calistirir.

        Args:
            video_path:    Kaynak video (mp4, mkv, avi, ...)
            output_path:   Cikti yolu (<video>_dubbed.mp4)
            progress_cb:   (fraction, msg, color) -> None  [GUI ilerleme cubugu]
            transcript_cb: (segments_list) -> None  [Whisper biter bitmez tum
                           segmentleri verir; her segment dict: start/end/text]
            translation_cb:(idx, total, english_text) -> None  [her segment
                           cevrildiginde anlik bilgi]

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
            # Proaktif konsolidasyon: 6-14sn'lik bloklara grupla.
            # LLM cagrisi ~%50 azalir, XTTS daha dogal prosody uretir.
            after_short = len(segments)
            segments = self._consolidate_segments(segments)
            final_count = len(segments)
            if final_count < raw_count:
                prog(0.22, f"{raw_count} segment -> {after_short} (kisa) -> {final_count} (konsolide bloklar).", "#23d05e")
            else:
                prog(0.22, f"{final_count} segment tanindi.", "#23d05e")

            # GUI: Whisper transkript hazir, kaynak metni hemen kullaniciya goster.
            # Boylece kullanici cevirinin gelmesini beklemeden tanimanin dogrulugunu
            # gozden gecirebilir.
            if transcript_cb:
                try:
                    transcript_cb([dict(s) for s in segments])
                except Exception:
                    pass  # GUI hatasi pipeline'i dursurmasin

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

                # GUI: bu segmentin cevirisi hazir; anlik panele yansisin.
                if translation_cb:
                    try:
                        translation_cb(i, total, text_en)
                    except Exception:
                        pass  # GUI hatasi pipeline'i dursurmasin

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

                # XTTS-v2 emoji/markdown/cok kisa metinlerde halusinasyon yapar.
                # Burada budayip, gerekirse segmenti atliyoruz (sessizlik kalir).
                clean_text = self._clean_text_for_tts(seg["text_en"])
                if not clean_text:
                    continue

                out_wav = os.path.join(self._tmp_dir, f"dub_seg_{i:04d}.wav")
                self._synthesize_segment(
                    clean_text, gpt_latent, spk_emb, out_wav
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

    def _consolidate_segments(self, segments: list) -> list:
        """Whisper segmentlerini 6-14sn'lik mantiksal bloklara grupla.

        `_merge_short_segments`'ten farkli: o sadece kritik kisa parcalari
        kurtariyor. Bu fonksiyon ise **tum segmentleri** proaktif olarak
        daha buyuk bloklara birlestirir.

        Faydalari:
          - LLM cagri sayisi yaklasik yariya iner (kota dostu, hizli)
          - XTTS daha uzun ve dogal cumleler alir -> prosody kalitesi artar,
            robotik his azalir
          - Cumleler arasi baglam butunlugu korunur

        Birlestirme kurallari (oncelik sirasiyla):
          1. Blok suresi TARGET_MIN'i (6s) gectiyse VE mevcut metin cumle sonu
             noktalama ile bitiyorsa -> blogu kapat (dogal cumle siniri)
          2. Iki segment arasi bosluk > MAX_GAP (0.8s) -> blogu kapat
             (uzun durus, muhtemel cumle/konu degisimi)
          3. Birlestiginde TARGET_MAX'i (14s) asiyorsa -> blogu kapat
             (XTTS uzun text'te kalitesini kaybeder)
          4. Yukaridakilerin hicbiri degilse -> birlestir
        """
        if not segments:
            return segments

        TARGET_MIN = 6.0     # blok bu suresinin altindaysa cumle sonu olsa bile kapatma
        TARGET_MAX = 14.0    # blok bu sureyi gecemez
        MAX_GAP    = 0.8     # ardisik segmentler arasi tolere edilen bosluk
        SENT_END   = (".", "!", "?")

        consolidated = []
        current = None

        for seg in segments:
            if current is None:
                current = dict(seg)
                continue

            gap = seg["start"] - current["end"]
            merged_dur = seg["end"] - current["start"]
            current_dur = current["end"] - current["start"]
            current_text = current["text"].rstrip()
            ends_sentence = current_text.endswith(SENT_END)

            # Blok kapatma kararlari
            close_block = False
            if merged_dur > TARGET_MAX:
                close_block = True
            elif gap > MAX_GAP:
                close_block = True
            elif ends_sentence and current_dur >= TARGET_MIN:
                close_block = True

            if close_block:
                consolidated.append(current)
                current = dict(seg)
            else:
                # Birlestir
                current["end"] = seg["end"]
                sep = " " if not current["text"].rstrip().endswith("-") else ""
                current["text"] = (current["text"].rstrip() + sep + seg["text"].lstrip()).strip()

        if current is not None:
            consolidated.append(current)

        return consolidated

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
        # Aday segmentler: 4-10 saniye araliginda — burada **bol** topla,
        # sonra kalite skoruyla en iyi 3'u secelim. Ne kadar cok aday, o kadar
        # secici olabiliriz (gurultulu/muzikli segmentleri eleriz).
        candidates = [s for s in segments
                      if 4.0 <= (s["end"] - s["start"]) <= 10.0]

        if not candidates:
            # Fallback: en uzun 1 segment (her ihtimale karsi)
            candidates = sorted(segments, key=lambda s: s["end"] - s["start"],
                                reverse=True)[:1]
        else:
            # En uzun 10 adayi al — 3'unu skor ile sececegiz
            candidates = sorted(candidates, key=lambda s: s["end"] - s["start"],
                                reverse=True)[:10]

        # silenceremove: bas tarafta 0.1sn sessizligi -40dB esikle kirp,
        # areverse ile son tarafta da ayni islemi tekrarla
        silence_filter = (
            "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,"
            "areverse,"
            "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,"
            "areverse"
        )

        # Her adayi cikar + skorla. Skorlama mantigi `_score_reference_wav`'da:
        #   yuksek mean_rms (cok kisik degil) +
        #   dusuk rms varyansi (konusma tonu kararli, muzik degil) +
        #   dusuk silence ratio (yarisi sessizlik degil) -> yuksek skor.
        scored = []  # [(score, wav_path, seg)]
        for i, seg in enumerate(candidates):
            out = os.path.join(self._tmp_dir, f"dub_ref_cand_{i}.wav")
            end_s = min(seg["end"], seg["start"] + 10.0)
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path,
                 "-ss", str(seg["start"]), "-to", str(end_s),
                 "-af", silence_filter,
                 "-ar", "22050", "-ac", "1", out],
                capture_output=True, timeout=30
            )
            if r.returncode != 0 or not os.path.exists(out):
                continue
            try:
                dur = self._get_wav_duration(out)
                if dur < 2.0:
                    os.remove(out)
                    continue
            except Exception:
                continue
            score = self._score_reference_wav(out)
            if score <= 0.0:
                # Sessizlik ya da bozuk, ele
                try: os.remove(out)
                except OSError: pass
                continue
            scored.append((score, out, seg))

        # En iyi 3 skoru sec, gerisini sil
        scored.sort(key=lambda x: x[0], reverse=True)
        ref_paths = [p for (_, p, _) in scored[:3]]
        for _, p, _ in scored[3:]:
            try: os.remove(p)
            except OSError: pass

        if ref_paths:
            top_scores = [f"{s:.3f}" for (s, _, _) in scored[:len(ref_paths)]]
            print(f"[DUBBER] Referans secimi: {len(ref_paths)} aday "
                  f"(skorlar: {', '.join(top_scores)})")

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

    def _score_reference_wav(self, wav_path: str) -> float:
        """Bir referans wav'inin XTTS icin uygunlugunu skorlar.

        Skorlama mantigi:
          - mean_rms: ortalama enerji. Cok kisik (uzak/yankili) sesi cezalandirir.
          - rms_kararliligi (1 / (1 + std/mean)): muzik veya degisken arka plan
            yuksek varyans uretir; konusma tonu daha kararlidir.
          - silence_ratio: 100ms penceredeki sessiz oranlari. Yuksek olmasi
            referans icindeki bos zamanlari isaretler; kotu sinyal.
          - clipping_ratio: |x| > 0.99 olan ornek orani. Distorted ses XTTS'i
            sapitir.

        Returns: 0.0 = kotu/atilmali, ~0.10-0.30 = normal konusma, daha yuksek = ideal.
        """
        try:
            with wave.open(wav_path, "rb") as wf:
                sw = wf.getsampwidth()
                sr = wf.getframerate()
                n = wf.getnframes()
                raw = wf.readframes(n)
            if sw == 2:
                x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sw == 4:
                x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                return 0.0
            if len(x) < sr * 1.5:
                return 0.0

            # 100ms pencere RMS
            w = max(1, sr // 10)
            n_w = len(x) // w
            if n_w < 5:
                return 0.0
            frames = x[:n_w * w].reshape(n_w, w)
            rms_pw = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
            mean_rms = float(rms_pw.mean())
            std_rms = float(rms_pw.std())
            silence_ratio = float((rms_pw < 0.01).mean())
            clipping_ratio = float((np.abs(x) > 0.99).mean())

            # Cok kisik ya da cok bos: ele
            if mean_rms < 0.015 or silence_ratio > 0.5:
                return 0.0
            # Asiri clipping: distorted, ele
            if clipping_ratio > 0.02:
                return 0.0

            consistency = 1.0 / (1.0 + std_rms / (mean_rms + 1e-6))
            score = mean_rms * consistency * (1.0 - silence_ratio) * (1.0 - clipping_ratio)
            return float(score)
        except Exception:
            return 0.0

    @staticmethod
    def _clean_text_for_tts(text: str) -> str:
        """XTTS'e gondermeden once metni temizle.

        XTTS-v2 hassas: emoji, fazla noktalama, markdown isaretleri, parantez
        icindeki yan aciklamalar ("(laughs)", "[music]") ile karsilasinca
        halusinasyon yapar/sapitir. Burada bunlari budariz.

        Returns: temiz metin, ya da cok kisa/anlamsizsa "" (segment atlanir).
        """
        if not text:
            return ""
        s = text.strip()
        # Markdown ve tirnak/asteriks
        s = re.sub(r"[*_~`#]+", "", s)
        s = s.replace("\u201c", "").replace("\u201d", "").replace("\u2018", "").replace("\u2019", "'")
        s = s.replace('"', "")
        # Parantez/koseli parantez icindeki yan aciklamalar (laughs, music vb.)
        s = re.sub(r"\([^)]{0,40}\)", "", s)
        s = re.sub(r"\[[^\]]{0,40}\]", "", s)
        # Emoji ve cogu non-BMP sembol
        s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
        # Cok arda gelen ayni noktalama: "..." -> "...", "!!!" -> "!"
        s = re.sub(r"([.!?,;:])\1{2,}", r"\1\1\1", s)
        # Cok bosluk
        s = re.sub(r"\s+", " ", s).strip()
        # Cok kisa veya sadece noktalama: segment atla
        alnum = re.sub(r"[^A-Za-z0-9\u00C0-\u017F]", "", s)
        if len(alnum) < 4:
            return ""
        return s

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
