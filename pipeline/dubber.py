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

        # Vokal ayristirici (Demucs CLI wrapper). HER video icin default calisir;
        # is_available() False donerse graceful fallback (ham ses) ile devam edilir.
        # Mimari karari: kullaniciya "muzik var mi?" diye sorulMAZ — kurumsal
        # uretim icin tam otomatik on-isleme.
        from pipeline.vocal_separator import VocalSeparator
        self.separator = VocalSeparator()

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

        # DUBLAJ MODU: system_prompt'a uzunluk + context hard rule'lari ekler.
        # EN cevirisini TR uzunlugunda tutup snowball/desync birikimini sifirlar.
        # Belge cevirisinde asla aktiflesmez; finally'de mutlaka kapatilir.
        try:
            self.translator.set_dubbing_mode(True)
        except Exception:
            pass

        # Demucs gecici klasoru — finally'de temizlenir.
        demucs_work_dir = None
        instrumental_path = None
        try:
            # ── 1. Ses ayikla (16kHz mono — fallback / Whisper raw) ────
            prog(0.02, "Video'dan ses ayiklaniyor...", "#f5a623")
            wav_path = self._extract_wav(video_path)
            video_duration = self._get_wav_duration(wav_path)
            prog(0.04, f"Video suresi: {video_duration:.1f}s", "#5b9ef9")

            # ── 1.5. Vokal/enstrumental ayristirma (Demucs) ────────────
            # KURUMSAL: Demucs HER videoda otomatik calisir. Muzik yoksa bile
            # vocals.wav daha temiz olur (gurultu/yanki azalir), instrumental
            # bos/sessiz ciksa bile mix asamasinda sorun olusmaz.
            # Kurulu degilse graceful fallback (uyari + ham ses).
            if self.separator.is_available():
                prog(0.05, "Demucs vokal ayristirma basliyor (~30-60sn)...", "#f5a623")
                demucs_work_dir = os.path.join(self._tmp_dir, "demucs_work")
                try:
                    vocals_hq, instrumental_hq = self.separator.separate(
                        video_path,  # Demucs CLI ffmpeg ile video'yu dogrudan kabul eder
                        work_dir=demucs_work_dir,
                        progress_cb=lambda f, m="": prog(0.05 + 0.04 * f, m or "Vokal ayristiriliyor...", "#f5a623"),
                    )
                    # Whisper / _extract_reference 16kHz mono bekliyor → downsample.
                    clean_wav = os.path.join(self._tmp_dir, "dub_vocals_16k.wav")
                    self._resample_mono(vocals_hq, clean_wav, 16000)
                    wav_path = clean_wav  # Whisper ve referans bunu kullanacak
                    instrumental_path = instrumental_hq
                    # KRITIK: Demucs bittikten hemen sonra VRAM'i bosalt.
                    # htdemucs ~3GB VRAM tutuyor; Whisper-medium (~1.5GB) ve
                    # Gemma 8192ctx (~4GB) icin sirali pay acilmasi gerek.
                    try:
                        self.separator.unload()
                    except Exception:
                        pass
                    prog(0.09, "Vokal ayristirma tamam — Whisper/XTTS temiz ses uzerinde.", "#23d05e")
                except Exception as e:
                    print(f"[DUBBER] Demucs basarisiz, ham ses ile devam: {e}")
                    prog(0.09, "Demucs basarisiz, ham ses ile devam.", "#f5a623")
                    instrumental_path = None
                    # Hata olsa bile model VRAM'de kaldiysa bosalt
                    try:
                        self.separator.unload()
                    except Exception:
                        pass
            else:
                prog(0.09, "Demucs kurulu degil → ham ses ile devam (kalite dusebilir).", "#f5a623")

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

                # SURE-BAZLI KELEPCE: dile-duyarli (CHARS_PER_SEC tablosu translator
                # icinde). Eski "kelime sayisi <= TR" kurali sadece TR<->EN icin
                # dogru calisirdi; bu yaklasim AR/JA/ZH dahil tum dillerde dogrudur.
                seg_duration = max(0.5, seg["end"] - seg["start"])

                result = self.translator.translate(
                    seg["text"],
                    context=context_segs,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    src_name=src_name,
                    tgt_name=tgt_name,
                    prev_translation=prev_translation,
                    rolling_summary=rolling_summary,
                    target_duration_sec=seg_duration,
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

            # ── 7. Final mix: dublaj + (varsa) instrumental + video birlesimi ─
            # instrumental_path varsa: ffmpeg sidechain compression ile auto-ducking
            # uygulanir — orijinal muzik/ortam sesi kisik sekilde dublajin altina
            # yerlesir, konusma sirasinda otomatik kisilir.
            # Yoksa: klasik mux (sadece dublaj sesi).
            prog(0.94, "Video ile birlestiriliyor (sidechain mix)...", "#f5a623")
            self._mix_with_instrumental(
                video_path, dubbed_wav, instrumental_path, output_path
            )

            prog(1.0, f"Tamamlandi!  →  {os.path.basename(output_path)}", "#23d05e")
            return output_path

        finally:
            # Dublaj modunu MUTLAKA kapat — belge cevirisine sizmasin.
            try:
                self.translator.set_dubbing_mode(False)
            except Exception:
                pass
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
            # Demucs gecici klasorunu sil (vocals + instrumental dahil)
            if demucs_work_dir is not None:
                try:
                    self.separator.cleanup(demucs_work_dir)
                except Exception:
                    pass

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
                # VAD KAPALI: Faster-Whisper VAD agresif davranip uzun konusma
                # parcalarini "sessizlik" sayip atiyor. VAD sadece XTTS referans
                # ses cikarmada (gurultu/muzik filtreleme icin) kullanilmali —
                # transkripsiyonda hicbir konusma kaybi olmasin.
                vad_filter=False,
                condition_on_previous_text=True,      # Baglam yardim eder, hatali metin yapismaz
                temperature=0.0,                      # Deterministic

                # ── WHISPER BEKCILERI TAMAMEN GEVSETILDI (DEMUCS UYUMU) ─────
                # Pipeline akisi: Demucs once vokal/instrumental ayriyor, Whisper
                # sadece TEMIZ vocals.wav uzerinde calisiyor. Halusinasyon kaynagi
                # olan muzik/gurultu zaten yok. Bu yuzden Whisper'in kendi koruma
                # filtreleri arti-katki vermiyor, tam tersine bogulan/sulu vokal
                # bolgelerinde gercek konusmayi siliyor.
                #
                # Onceki ayarlar (0.5 default -> 0.85) bazi 30sn'lik konusma
                # bloklarini hala yutuyordu. Final tasarim:
                #   - compression_ratio_threshold=None: kendini-tekrar koruma
                #     kapali (Demucs sonrasi nadir; gercek konusmayi silmesin)
                #   - log_prob_threshold=None: Whisper bogulan sese %50 emin
                #     bile olsa metne donsun, atmasin
                #   - no_speech_threshold=0.95: yalnizca %95 'kesin sessizlik'
                #     sayilan segmentler atilir
                compression_ratio_threshold=None,
                log_prob_threshold=None,
                no_speech_threshold=0.95,
            )
            result = []
            for seg in segments_iter:
                # Manuel post-filter de %95'e cekildi — Whisper'a yetki ver,
                # sadece kesin gurultu durumunda ele.
                if seg.no_speech_prob > 0.95:
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
        """Whisper segmentlerini 6-10sn'lik mantiksal bloklara grupla.

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
        TARGET_MAX = 10.0    # XTTS-v2 attention budget: 10sn ustu prosody bozulur,
                             # halusinasyon/metalik tonlama. Eskiden 14sn idi -> dusuruldu.
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
          - speed=1.0: KRITIK. Once speed=1.15 + assembly'de atempo<=1.15 vardi,
            ama bu CIFTE SIKISTIRMA — XTTS'in zaten %15 hizli/deforme sinyali
            ustune ffmpeg phase-vocoder bininca ses metaliklesir. speed=1.0'a
            geri donduk: dogal timbre korunur. Uzunluk sorunu artik (a) translator
            uzunluk kelepcesi (concise EN), (b) _assemble_audio icindeki snowball
            guard + emergency atempo ile cozuluyor — bu yapi yalnizca DESYNC
            kritik oldugunda atempo'yu tetikler, normal segmentte hicbir
            post-processing yok.
          - enable_text_splitting=True: KRITIK. _consolidate_segments 6-14sn
            mantiksal bloklar uretiyor, icinde 2-3 cumle olabiliyor. False iken
            XTTS uzun multi-sentence metni tek pass'te isleyip cumlenin
            ortasinda <eos> uretiyor (multi-sentence dropout) ve robotik
            tonlama yapiyor. True iken XTTS'in kendi sentence splitter'i her
            cumleyi ayri prompt olarak isleyip dogal pause ile birlestiriyor —
            atlama ve robotluk hissi kaybolur.
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
            enable_text_splitting=True,
        )
        wav_np = np.array(output["wav"], dtype=np.float32)
        sf.write(out_path, wav_np, self.XTTS_SR)

    # ═══════════════════════════════════════════════════════════
    # ADIM 6: TIMELINE ASSEMBLY
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _trim_dead_air(data: np.ndarray, sr: int,
                       threshold_db: float = -40.0,
                       frame_ms: int = 20,
                       margin_ms: int = 20) -> np.ndarray:
        """RMS-tabanli olu sessizlik (dead air) budama.

        Eski |amplitude|>0.01 yontemi XTTS noise floor'una yetersizdi (noise
        ~-30dBFS, threshold ~-40dBFS); fonksiyon sessizligi gercekten budayamiyor,
        atempo gereksiz tetikleniyordu. Yeni mantik:
          - 20ms pencerelerde RMS hesapla, dB'ye cevir
          - threshold_db'nin (default -40dBFS) ustundeki ilk/son pencereyi bul
          - ±margin_ms (default 20ms) tampon birak (konusma basi kesilmesin)

        threshold_db = -40dBFS pratik: XTTS pure silence < -55dB, noise floor
        ~-35dB, voiced > -20dB. -40dB ikisinin arasinda guvenli orta yol.
        """
        if data.size == 0:
            return data
        frame_len = max(1, int((frame_ms / 1000.0) * sr))
        n_frames = len(data) // frame_len
        if n_frames < 2:
            return data
        windows = data[:n_frames * frame_len].reshape(n_frames, frame_len)
        rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-12)
        voiced = db > threshold_db
        if not voiced.any():
            return data  # tamamen sessiz — capsule kararini yukariya birak
        first_f = int(np.argmax(voiced))
        last_f = int(n_frames - np.argmax(voiced[::-1]))
        margin_frames = max(1, int((margin_ms / 1000.0) * sr / frame_len))
        first_f = max(0, first_f - margin_frames)
        last_f = min(n_frames, last_f + margin_frames)
        return data[first_f * frame_len: last_f * frame_len]

    @staticmethod
    def _fade_out(data: np.ndarray, sr: int, fade_ms: int = 80) -> np.ndarray:
        """Sesin sonuna lineer fade-out uygular (sert kesim cıkkardamasin)."""
        if data.size == 0:
            return data
        fade_len = min(len(data), int((fade_ms / 1000.0) * sr))
        if fade_len <= 1:
            return data
        out = data.copy()
        out[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        return out

    def _assemble_audio(self, segments: list, seg_wavs: list,
                        video_duration: float, out_path: str):
        """
        Sentezlenen segmentleri orijinal zaman damgalarina yerlestir.

        Mimari kurallar (4 mühendislik notu):
          1. XTTS olu sessizligini once bud (RMS-tabanli) → uzunluk dogru
          2. Normalde atempo YOK (XTTS speed=1.0); ses olduğu gibi yerlesir
          3. Cumleler arasi minimum 0.2sn nefes pay (corba onleme)
          4. SNOWBALL GUARD: birikmis desync > 1.5sn ise acil sync devreye girer:
             a) Nefes pay 0 (sigdir)
             b) Emergency atempo (max ratio 1.25) — sadece bu durumda phase-vocoder kabul
             c) Hala uzunsa fade-out ile sert kirp (videoya yetis)
        """
        SR = self.XTTS_SR
        BREATH_FRAMES = int(0.20 * SR)             # cumleler arasi min 200ms
        MAX_DESYNC_SEC = 1.5                       # birikebilir maksimum kayma
        MAX_DESYNC_FRAMES = int(MAX_DESYNC_SEC * SR)
        EMERGENCY_RATIO_CAP = 1.25                 # acil durumda atempo'ya izin
        # Tampon (mux sonda kirpiyor)
        max_overflow = max(5.0, video_duration * 0.20)
        total_frames = int((video_duration + max_overflow) * SR)
        output = np.zeros(total_frames, dtype=np.float32)

        last_end_frame = 0
        emergency_count = 0

        for seg, wav_path in zip(segments, seg_wavs):
            if wav_path is None or not os.path.exists(wav_path):
                continue

            seg_duration = seg["end"] - seg["start"]
            orig_start_frame = int(seg["start"] * SR)

            data, _ = sf.read(wav_path, dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]

            # 1) DEAD-AIR TRIM (RMS-tabanli) — uzunluk hesabi temizlensin.
            data = self._trim_dead_air(data, SR)

            # 4) DESYNC ALGI: onceki segment, bu segmentin orijinal baslangicini
            # ne kadar gectiyse o kadar birikmis kayma var.
            drift_frames = max(0, last_end_frame - orig_start_frame)
            emergency = drift_frames > MAX_DESYNC_FRAMES

            # 2+4) Normal mod: hicbir post-processing YOK (XTTS speed=1.0 dogal).
            #      Acil mod: emergency atempo (max 1.25) + hala uzunsa fade-out trim.
            if emergency:
                eng_duration = len(data) / SR
                if eng_duration > seg_duration * 1.05:
                    ratio = eng_duration / seg_duration
                    apply_ratio = min(ratio, EMERGENCY_RATIO_CAP)
                    data = self._atempo_stretch(data, SR, apply_ratio, wav_path)
                    # Hala fazla uzunsa: sert trim + fade-out
                    eng_after = len(data) / SR
                    if eng_after > seg_duration * 1.20:
                        max_len = int(seg_duration * 1.15 * SR)
                        data = self._fade_out(data[:max_len], SR)
                emergency_count += 1

            # 3) NEFES PAYI: normalde 200ms; acil modda 0 (sigdir).
            breath = 0 if emergency else (BREATH_FRAMES if last_end_frame > 0 else 0)
            min_start = last_end_frame + breath
            start_frame = max(orig_start_frame, min_start)
            end_frame = min(start_frame + len(data), total_frames)
            if end_frame <= start_frame:
                continue
            output[start_frame:end_frame] = data[:end_frame - start_frame]
            last_end_frame = end_frame

        if emergency_count > 0:
            print(f"[DUBBER] Snowball guard {emergency_count} kez devreye girdi "
                  f"(desync > {MAX_DESYNC_SEC}sn).")

        # Sondaki bos kuyrugu kirp.
        output = output[:max(last_end_frame, int(video_duration * SR))]
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

    def _resample_mono(self, in_path: str, out_path: str, sr: int = 16000):
        """WAV/audio dosyasini ffmpeg ile mono + hedef sample_rate'e cevirir.
        Demucs ciktisi (vocals_hq, 44.1kHz stereo) -> Whisper girdisi
        (16kHz mono) donusumunde kullanilir."""
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", in_path,
             "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", out_path],
            capture_output=True, timeout=300
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg resample hatasi:\n{r.stderr.decode(errors='replace')[:300]}"
            )

    def _mix_with_instrumental(self, video_path: str, dubbed_wav: str,
                               instrumental_path: str | None,
                               output_path: str):
        """Final mix: dublaj sesi + (varsa) orijinal instrumental + video.

        instrumental_path None ise: klasik mux (stream-copy, sadece dublaj sesi).

        instrumental_path varsa: ffmpeg sidechaincompress ile auto-ducking.
        Konusma (dublaj) varken arka plan muzigi/SFX dinamik olarak kisilir;
        konusma yokken eski seviyeye geri doner. Bu, kurumsal dublajda standart
        teknik (broadcast / podcast post-production icin de ayni filtre).

        Sidechain parametre secimi:
          threshold=0.03   → konusma siddeti dustigunda bile ducking tetiklesin
          ratio=20         → ducking agresif (-10..-15dB tipik)
          attack=5ms       → konusmaya hizli tepki (kelime baslangici kisilsin)
          release=300ms    → konusma sonu yumuşak yukseliş (ani patlama olmasin)
          makeup=1         → ducked sinyalin kazanci nötr
        """
        if instrumental_path is None or not os.path.exists(instrumental_path):
            # FALLBACK: klasik stream-copy mux (eski _mux davranisi).
            r = subprocess.run(
                ["ffmpeg", "-y",
                 "-i", video_path,
                 "-i", dubbed_wav,
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
            return

        # SIDECHAIN COMPRESSION + MIX + VIDEO MUX (tek ffmpeg cagrisi)
        # Inputs: 0=video, 1=dubbed_voice, 2=instrumental
        # Filter: voice ve bg ayni sample rate (24kHz, XTTS_SR) + stereo'ya cevrilir;
        # bg (instrumental) voice'a duyarli sidechaincompress ile ducked;
        # sonra ducked_bg + voice amix ile birlestirilir; sonuc tek stereo track.
        sr = self.XTTS_SR
        filter_complex = (
            f"[1:a]aresample={sr},aformat=channel_layouts=stereo,"
            f"asplit=2[voice_out][voice_sc];"
            f"[2:a]aresample={sr},aformat=channel_layouts=stereo[bg];"
            f"[bg][voice_sc]sidechaincompress="
            f"threshold=0.03:ratio=20:attack=5:release=300:makeup=1[bg_ducked];"
            f"[voice_out][bg_ducked]amix=inputs=2:duration=first:"
            f"weights=2 1:normalize=0[a_out]"
        )
        r = subprocess.run(
            ["ffmpeg", "-y",
             "-i", video_path,
             "-i", dubbed_wav,
             "-i", instrumental_path,
             "-filter_complex", filter_complex,
             "-map", "0:v:0",
             "-map", "[a_out]",
             "-c:v", "copy",
             "-c:a", "aac", "-b:a", "192k",
             "-shortest",
             output_path],
            capture_output=True, timeout=900
        )
        if r.returncode != 0:
            err = r.stderr.decode(errors='replace')[-500:]
            print(f"[DUBBER] sidechain mix basarisiz, fallback mux deneniyor:\n{err}")
            # FALLBACK: sidechain calismadi (eski ffmpeg / filter desteksiz)
            self._mix_with_instrumental(video_path, dubbed_wav, None, output_path)

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
