"""
Gemma Echo — Video Dubbing Pipeline (single speaker)

Turkish video -> English audio, preserving the original speaker's voice.

Pipeline stages:
  1. ffmpeg      : Extract a 16 kHz mono WAV from the video.
  2. Whisper     : Produce a timestamped transcript (segment start / end / text).
  3. Gemma 4 Q4  : Translate each segment locally via the C++ inference engine.
  4. XTTS-v2     : Compute the speaker latent from the reference audio (once).
  5. XTTS-v2     : Run voice-cloned inference for each segment.
  6. NumPy/sf    : Place the synthesized segments back on the original timeline.
  7. ffmpeg      : Mux the new audio track into the source video.

Output: <source_video>_dubbed.mp4
"""

import os
import re
import wave
import subprocess
import numpy as np
import soundfile as sf


class DubbingPipeline:

    XTTS_SR = 24000  # XTTS-v2 output sample rate.

    def __init__(self, transcriber, translator, synthesizer, config=None):
        self.transcriber = transcriber
        self.translator  = translator
        self.synthesizer = synthesizer
        self.config      = config

        self._project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._tmp_dir = os.path.join(self._project_dir, ".tmp_audio")
        os.makedirs(self._tmp_dir, exist_ok=True)

        # Vocal separator (Demucs programmatic wrapper). Runs on EVERY video by
        # default; if is_available() returns False the pipeline gracefully falls
        # back to the raw audio path.
        # Design decision: the user is NOT prompted with "does this video have
        # music?" — the pre-processing is fully automated for production-grade
        # reliability.
        from pipeline.vocal_separator import VocalSeparator
        self.separator = VocalSeparator()

    # ═══════════════════════════════════════════════════════════
    # PRIMARY PIPELINE
    # ═══════════════════════════════════════════════════════════

    def process(self, video_path: str, output_path: str, src_lang="tr", tgt_lang="en",
                src_name="Turkish", tgt_name="English", progress_cb=None,
                transcript_cb=None, translation_cb=None):
        """
        Execute the full dubbing pipeline.

        Args:
            video_path:    Source video (mp4, mkv, avi, ...).
            output_path:   Output path (typically <video>_dubbed.mp4).
            progress_cb:   (fraction, msg, color) -> None  [GUI progress bar].
            transcript_cb: (segments_list) -> None  [invoked once Whisper completes;
                           each segment is a dict with start / end / text].
            translation_cb:(idx, total, english_text) -> None  [called after each
                           segment is translated].

        Returns:
            output_path on success.

        Raises:
            RuntimeError: on a critical pipeline failure.
        """

        def prog(f, msg, color="#5b9ef9"):
            print(f"[DUBBER] ({int(f*100)}%) {msg}")
            if progress_cb:
                progress_cb(f, msg, color)

        wav_path = dubbed_wav = None
        ref_wavs = []
        seg_wavs = []

        # The user's runtime settings drive the dubbing run; we must not force
        # local-offline. Snapshot the current modes and restore them in the
        # finally block so the user's online preference is preserved.
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

        # DUBBING MODE: appends the length + context "hard rules" to the system
        # prompt. Keeps EN translation length close to TR and prevents snowball
        # desync accumulation. Never activated during document translation;
        # always disabled in the finally block.
        try:
            self.translator.set_dubbing_mode(True)
        except Exception:
            pass

        # Demucs scratch directory — cleaned up in the finally block.
        demucs_work_dir = None
        instrumental_path = None
        try:
            # ── 1. Extract audio (16 kHz mono — Whisper raw / fallback) ────
            prog(0.02, "Extracting audio from video...", "#f5a623")
            wav_path = self._extract_wav(video_path)
            video_duration = self._get_wav_duration(wav_path)
            prog(0.04, f"Video duration: {video_duration:.1f}s", "#5b9ef9")

            # ── 1.5. Vocal / instrumental separation (Demucs) ──────────
            # ENTERPRISE: Demucs runs automatically on every video. Even when
            # there is no music, vocals.wav is cleaner (noise / reverb reduced)
            # and an empty instrumental track is harmless during the final mix.
            # When the dependency is missing we degrade gracefully to raw audio.
            if self.separator.is_available():
                prog(0.05, "Demucs vocal separation starting (~30-60s)...", "#f5a623")
                demucs_work_dir = os.path.join(self._tmp_dir, "demucs_work")
                try:
                    vocals_hq, instrumental_hq = self.separator.separate(
                        video_path,  # Demucs CLI accepts the video directly via ffmpeg.
                        work_dir=demucs_work_dir,
                        progress_cb=lambda f, m="": prog(0.05 + 0.04 * f, m or "Separating vocals...", "#f5a623"),
                    )
                    # Whisper / _extract_reference expect 16 kHz mono → downsample.
                    clean_wav = os.path.join(self._tmp_dir, "dub_vocals_16k.wav")
                    self._resample_mono(vocals_hq, clean_wav, 16000)
                    wav_path = clean_wav  # Whisper and the reference extractor consume this file.
                    instrumental_path = instrumental_hq
                    # CRITICAL: release Demucs VRAM immediately after this stage.
                    # htdemucs occupies ~3 GB; Whisper-medium (~1.5 GB) and
                    # Gemma 8192-ctx (~4 GB) need sequential headroom.
                    try:
                        self.separator.unload()
                    except Exception:
                        pass
                    prog(0.09, "Vocal separation complete — Whisper / XTTS will work on the clean audio.", "#23d05e")
                except Exception as e:
                    print(f"[DUBBER] Demucs failed, continuing with raw audio: {e}")
                    prog(0.09, "Demucs failed, continuing with raw audio.", "#f5a623")
                    instrumental_path = None
                    # Free any lingering VRAM even on failure.
                    try:
                        self.separator.unload()
                    except Exception:
                        pass
            else:
                prog(0.09, "Demucs not installed → continuing with raw audio (quality may degrade).", "#f5a623")

            # ── 2. Whisper transcription ───────────────────────────────
            prog(0.10, "Whisper generating transcript...", "#5b9ef9")
            segments = self._transcribe_segments(wav_path, language=src_lang)
            if not segments:
                raise RuntimeError(f"No {src_name} speech detected in the video.")

            # XTTS-v2 produces awkward prosody on 1-3-word or <1.5 s segments.
            # Merge those short segments into the preceding one so we feed XTTS
            # more natural sentence-length utterances. Side benefit: the LLM
            # also gets longer context to translate.
            raw_count = len(segments)
            segments = self._merge_short_segments(segments)
            # Proactive consolidation: group into 6-14 s logical blocks.
            # ~50% fewer LLM calls and noticeably more natural XTTS prosody.
            after_short = len(segments)
            segments = self._consolidate_segments(segments)
            final_count = len(segments)
            if final_count < raw_count:
                prog(0.22, f"{raw_count} segments -> {after_short} (short-merged) -> {final_count} (consolidated blocks).", "#23d05e")
            else:
                prog(0.22, f"{final_count} segments detected.", "#23d05e")

            # GUI: as soon as the transcript is ready, surface the source text
            # to the user so they can review recognition accuracy without
            # waiting for the translations to land.
            if transcript_cb:
                try:
                    transcript_cb([dict(s) for s in segments])
                except Exception:
                    pass  # A GUI failure must not halt the pipeline.

            # ── Release VRAM: Whisper is done, the LLM and XTTS run next ──
            # On a 6 GB GPU: Whisper (~1 GB) + Gemma 8192-ctx (~4 GB) + XTTS (~2 GB) = ~7 GB → does not fit.
            # Evict only when the GPU STT model is active; cloud_auto consumes no VRAM.
            if prev_stt_mode in ("local_gpu", "local_gpu_hq"):
                try:
                    self.transcriber.set_mode("local_cpu")
                    prog(0.23, "Whisper evicted from VRAM (making room for the LLM)...", "#5b9ef9")
                except Exception:
                    # Even if eviction fails, the dubbing run must continue.
                    pass

            # ── 3. Translate via the local Gemma 4 Q4 engine ──────────
            # Whisper segments are short (5-15 words). Translating each one in
            # isolation would lose pronoun / context. To match media-translation
            # quality we forward prev_translation + the last N source segments +
            # rolling_summary, so the local model is loaded with a wide context (8192).
            if self.translator.local_llm is None:
                prog(0.24, "Loading the local Gemma 4 engine (30-60s)...", "#f5a623")
                if not self.translator.load_local_model(8192):
                    raise RuntimeError(
                        "Local Gemma failed to load (VRAM). "
                        "Switch the LLM to online in Settings or pick a smaller VRAM profile."
                    )
            elif getattr(self.translator, "loaded_n_ctx", 512) < 8192:
                # Already loaded but with a smaller context — reload with the wide window.
                prog(0.24, "Reloading the local Gemma 4 engine with a wider context...", "#f5a623")
                self.translator.load_local_model(8192)

            translated = []
            total = len(segments)
            prev_translation = ""
            rolling_summary = ""
            CTX_WINDOW = 3  # Number of preceding source segments included in the context window.

            for i, seg in enumerate(segments):
                p = 0.24 + 0.28 * (i / total)
                prog(p, f"Translating: {i+1}/{total}  \"{seg['text'][:40]}\"", "#5b9ef9")

                # Source text of the last CTX_WINDOW segments — reference context.
                context_segs = [s["text"] for s in segments[max(0, i - CTX_WINDOW):i]]

                # DURATION-BASED CONSTRAINT: language-aware (CHARS_PER_SEC table
                # is owned by the translator). The legacy "word-count <= TR"
                # rule only worked for TR<->EN; this approach is correct for
                # AR / JA / ZH as well.
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

                # GUI: this segment's translation is ready — surface it live.
                if translation_cb:
                    try:
                        translation_cb(i, total, text_en)
                    except Exception:
                        pass  # A GUI failure must not halt the pipeline.

                # Refresh the rolling summary every 5 segments (document-translation
                # pattern). Cadence is 5 (not 3) because dubbing segments are
                # short — fewer summary calls keeps LLM load reasonable.
                if (i + 1) % 5 == 0 or i == 0:
                    try:
                        rolling_summary = self.translator.generate_summary(
                            text_en, rolling_summary
                        )
                    except Exception:
                        # The summary refresh failing must not abort translation.
                        pass

            # ── Release VRAM: all segments translated, Gemma is no longer needed ──
            # XTTS-v2 needs ~2 GB VRAM. Evicting Gemma 8192-ctx (~4 GB) gives
            # the next stage breathing room and eliminates OOM risk.
            try:
                self.translator.unload_local_model()
                prog(0.52, "Gemma evicted from VRAM (making room for XTTS)...", "#5b9ef9")
            except Exception:
                pass

            # ── 4. Reference audio (speaker profile) ──────────────────
            prog(0.53, "Selecting speaker reference audio...", "#f5a623")
            ref_wavs = self._extract_reference(wav_path, segments)
            prog(0.55, f"{len(ref_wavs)} clean reference segment(s) selected.", "#23d05e")

            # ── 5. XTTS speaker latent (computed once) ────────────────
            prog(0.56, "Preparing XTTS-v2...", "#f5a623")
            self._ensure_xtts_loaded()
            prog(0.60, "Building the speaker voice profile...", "#f5a623")
            gpt_latent, spk_emb = self._get_speaker_latents(ref_wavs)

            # ── 5b. Synthesize each segment ──────────────────────────
            seg_wavs = [None] * total
            for i, seg in enumerate(translated):
                p = 0.60 + 0.28 * (i / total)
                prog(p, f"Synthesizing: {i+1}/{total}", "#5b9ef9")

                if not seg.get("text_en"):
                    continue

                # XTTS-v2 hallucinates on emoji / markdown / very short inputs.
                # Sanitize the text here; skip the segment (leave silence) if
                # the result is empty.
                clean_text = self._clean_text_for_tts(seg["text_en"])
                if not clean_text:
                    continue

                out_wav = os.path.join(self._tmp_dir, f"dub_seg_{i:04d}.wav")
                self._synthesize_segment(
                    clean_text, gpt_latent, spk_emb, out_wav
                )
                seg_wavs[i] = out_wav

            # ── 6. Timeline assembly ──────────────────────────────────
            prog(0.89, "Placing audio segments on the timeline...", "#f5a623")
            dubbed_wav = os.path.join(self._tmp_dir, "dubbed_full.wav")
            self._assemble_audio(translated, seg_wavs, video_duration, dubbed_wav)

            # ── 7. Final mix: dubbed voice + (optional) instrumental + video mux ─
            # When instrumental_path is set, ffmpeg's sidechain compression
            # performs auto-ducking — the original music / ambience plays at a
            # reduced level under the dubbed dialogue and recovers between
            # utterances.
            # When it is None: classic mux (dubbed voice only).
            prog(0.94, "Muxing into video (sidechain mix)...", "#f5a623")
            self._mix_with_instrumental(
                video_path, dubbed_wav, instrumental_path, output_path
            )

            prog(1.0, f"Complete!  →  {os.path.basename(output_path)}", "#23d05e")
            return output_path

        finally:
            # ALWAYS disable dubbing mode — it must not leak into document translation.
            try:
                self.translator.set_dubbing_mode(False)
            except Exception:
                pass
            # Restore the user's original translator mode (online if it was online).
            try:
                self.translator.set_mode(prev_mode)
            except Exception:
                pass
            # Restore the original Whisper mode so live translation is unaffected.
            try:
                self.transcriber.set_mode(prev_stt_mode)
            except Exception:
                pass
            # Clean up the temporary files.
            self._cleanup(seg_wavs + ref_wavs + [wav_path, dubbed_wav])
            # Remove the Demucs scratch directory (vocals + instrumental included).
            if demucs_work_dir is not None:
                try:
                    self.separator.cleanup(demucs_work_dir)
                except Exception:
                    pass

    # ═══════════════════════════════════════════════════════════
    # SUBTITLE-ONLY PIPELINE (no XTTS)
    # ═══════════════════════════════════════════════════════════

    def process_subtitles(self, video_path: str, output_path: str,
                          src_lang="tr", tgt_lang="en",
                          src_name="Turkish", tgt_name="English",
                          progress_cb=None, transcript_cb=None,
                          translation_cb=None, burn_in: bool = False):
        """Generate a translated subtitle track and mux it into the video.

        Reuses stages 1-3 of the dubbing pipeline (extract audio -> Whisper ->
        per-segment LLM translation), then writes an SRT file and muxes it into
        the source video as a soft subtitle track (no XTTS, no re-encode).

        Args:
            burn_in: If True, hard-burn the subtitles into the video (re-encodes).
                     If False (default), embed as a soft subtitle track (fast, lossless).
        """

        def prog(f, msg, color="#5b9ef9"):
            print(f"[SUBS] ({int(f*100)}%) {msg}")
            if progress_cb:
                progress_cb(f, msg, color)

        wav_path = None
        srt_path = None

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
            self.translator.set_dubbing_mode(True)
        except Exception:
            pass

        try:
            # ── 1. Extract audio (16 kHz mono) ─────────────────────────
            prog(0.05, "Extracting audio from video...", "#f5a623")
            wav_path = self._extract_wav(video_path)

            # ── 2. Whisper transcription ───────────────────────────────
            prog(0.15, "Whisper generating transcript...", "#5b9ef9")
            segments = self._transcribe_segments(wav_path, language=src_lang)
            if not segments:
                raise RuntimeError(f"No {src_name} speech detected in the video.")

            # Subtitling needs SHORT, snappy segments (~3-5 s, 1-2 lines on
            # screen). The dubbing path consolidates segments into 6-14 s
            # blocks for natural XTTS prosody, but applying that here causes
            # the whole transcript to appear as one giant subtitle covering
            # the screen. Only rescue critically short fragments (<0.6 s);
            # keep Whisper's natural sentence-length segmentation otherwise.
            segments = self._merge_short_segments(segments)
            prog(0.30, f"{len(segments)} segments detected.", "#23d05e")

            if transcript_cb:
                try:
                    transcript_cb([dict(s) for s in segments])
                except Exception:
                    pass

            # Free Whisper VRAM before loading the LLM (same logic as dubbing).
            if prev_stt_mode in ("local_gpu", "local_gpu_hq"):
                try:
                    self.transcriber.set_mode("local_cpu")
                except Exception:
                    pass

            # ── 3. Translate each segment ──────────────────────────────
            if self.translator.mode == "offline" and self.translator.local_llm is None:
                prog(0.32, "Loading the local Gemma 4 engine (30-60s)...", "#f5a623")
                self.translator.load_local_model(8192)
            elif self.translator.mode == "offline" and getattr(self.translator, "_n_ctx", 4096) < 8192:
                prog(0.32, "Reloading the local Gemma 4 engine with a wider context...", "#f5a623")
                self.translator.load_local_model(8192)

            translated = []
            total = len(segments)
            prev_translation = ""
            rolling_summary = ""
            CTX_WINDOW = 3

            for i, seg in enumerate(segments):
                p = 0.32 + 0.55 * (i / total)
                prog(p, f"Translating: {i+1}/{total}  \"{seg['text'][:40]}\"", "#5b9ef9")

                context_segs = [s["text"] for s in segments[max(0, i - CTX_WINDOW):i]]
                seg_duration = max(0.5, seg["end"] - seg["start"])

                try:
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
                    text_en = result.get("translation", "").strip() or seg["text"]
                except Exception as e:
                    print(f"[SUBS] Translation failed for segment {i}: {e}")
                    text_en = seg["text"]

                translated.append({**seg, "text_en": text_en})
                prev_translation = text_en

                if translation_cb:
                    try:
                        translation_cb(i, total, text_en)
                    except Exception:
                        pass

                if (i + 1) % 5 == 0 or i == 0:
                    try:
                        rolling_summary = self.translator.generate_summary(
                            text_en, rolling_summary
                        )
                    except Exception:
                        pass

            # ── 4. Write the SRT file next to the output ───────────────
            prog(0.90, "Writing SRT file...", "#f5a623")
            srt_path = os.path.splitext(output_path)[0] + ".srt"
            self._write_srt(translated, srt_path)

            # ── 5. Mux subtitles into the video ────────────────────────
            prog(0.95, "Muxing subtitles into the video...", "#f5a623")
            if burn_in:
                self._burn_subtitles(video_path, srt_path, output_path)
            else:
                self._mux_soft_subtitles(video_path, srt_path, output_path, tgt_lang)

            prog(1.0, "Subtitles ready.", "#23d05e")
            return output_path

        finally:
            try:
                self.translator.set_mode(prev_mode)
            except Exception:
                pass
            try:
                self.translator.set_dubbing_mode(False)
            except Exception:
                pass
            try:
                self.transcriber.set_mode(prev_stt_mode)
            except Exception:
                pass
            self._cleanup([wav_path])

    @staticmethod
    def _format_srt_timestamp(seconds: float) -> str:
        """Convert a float seconds value to SRT format ``HH:MM:SS,mmm``."""
        if seconds < 0:
            seconds = 0.0
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _write_srt(self, segments: list, srt_path: str,
                   max_line_chars: int = 42, max_lines: int = 2):
        """Write translated segments to an SRT file (UTF-8, BOM-less).

        Each segment is normalised for on-screen readability:
          1. If the translated text is short enough, it is wrapped to
             ``max_lines`` lines of at most ``max_line_chars`` characters.
          2. If the text is longer than the wrap budget allows, it is split
             at sentence boundaries (``. ``, ``? ``, ``! ``) into multiple
             SRT entries whose durations are proportional to the character
             count of each piece. This prevents a single 6-line block from
             covering half the screen — exactly the bug reported by the
             user ("alt yazı tüm ekranı kaplıyor").
        """
        import textwrap, re

        def _wrap(s: str) -> str:
            """Soft-wrap a string into at most ``max_lines`` short lines."""
            s = s.strip().replace("\r\n", "\n")
            if not s:
                return ""
            wrapped = textwrap.wrap(
                s, width=max_line_chars,
                break_long_words=False, break_on_hyphens=False
            )
            return "\n".join(wrapped[:max_lines]) if wrapped else s

        def _split_pieces(s: str) -> list:
            """Split text into sentence-sized pieces that fit the wrap budget."""
            budget = max_line_chars * max_lines
            if len(s) <= budget:
                return [s]
            # Split at sentence terminators while keeping the punctuation.
            parts = re.split(r"(?<=[.!?])\s+", s.strip())
            pieces, cur = [], ""
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                candidate = (cur + " " + p).strip() if cur else p
                if len(candidate) <= budget:
                    cur = candidate
                else:
                    if cur:
                        pieces.append(cur)
                    # If a single sentence still exceeds the budget, fall
                    # back to a hard wrap by words.
                    if len(p) > budget:
                        words, buf = p.split(), ""
                        for w in words:
                            cand = (buf + " " + w).strip() if buf else w
                            if len(cand) <= budget:
                                buf = cand
                            else:
                                pieces.append(buf)
                                buf = w
                        cur = buf
                    else:
                        cur = p
            if cur:
                pieces.append(cur)
            return pieces or [s]

        idx = 1
        with open(srt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                text = (seg.get("text_en") or "").strip()
                if not text:
                    continue
                seg_start = float(seg["start"])
                seg_end   = float(seg["end"])
                seg_dur   = max(0.4, seg_end - seg_start)

                pieces = _split_pieces(text)
                total_chars = sum(len(p) for p in pieces) or 1
                cursor = seg_start
                for j, piece in enumerate(pieces):
                    if j == len(pieces) - 1:
                        piece_end = seg_end
                    else:
                        piece_end = cursor + seg_dur * (len(piece) / total_chars)
                    start_ts = self._format_srt_timestamp(cursor)
                    end_ts   = self._format_srt_timestamp(max(cursor + 0.3, piece_end))
                    f.write(f"{idx}\n{start_ts} --> {end_ts}\n{_wrap(piece)}\n\n")
                    idx += 1
                    cursor = piece_end

    def _mux_soft_subtitles(self, video_path: str, srt_path: str,
                            output_path: str, lang_code: str = "eng"):
        """Embed the SRT into the video as a soft subtitle track (no re-encode)."""
        # ISO-639-2/T 3-letter code is preferred for mov_text; map common cases.
        iso_map = {
            "en": "eng", "tr": "tur", "de": "deu", "fr": "fra",
            "it": "ita", "es": "spa", "ar": "ara", "ja": "jpn",
        }
        lang3 = iso_map.get(lang_code, lang_code[:3] or "und")

        ext = os.path.splitext(output_path)[1].lower()
        sub_codec = "mov_text" if ext in (".mp4", ".m4v", ".mov") else "srt"

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", srt_path,
            "-map", "0:v?", "-map", "0:a?", "-map", "1:0",
            "-c:v", "copy", "-c:a", "copy",
            "-c:s", sub_codec,
            f"-metadata:s:s:0", f"language={lang3}",
            "-disposition:s:0", "default",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg subtitle mux failed:\n{r.stderr.decode(errors='replace')[:400]}"
            )

    def _burn_subtitles(self, video_path: str, srt_path: str, output_path: str):
        """Hard-burn the subtitles into the video (re-encodes the video stream).

        Cinematic style (Netflix / theatrical): white sans-serif text with a
        crisp black outline and a soft drop shadow — NO opaque background
        box. ``BorderStyle=1`` draws an outline + shadow only; the previous
        ``BorderStyle=3`` (opaque box) covered too much of the frame and
        looked like 90s teletext. Colours use the ASS ``&HAABBGGRR`` byte
        order: ``&H00FFFFFF`` = opaque white text, ``&H00000000`` = opaque
        black outline/shadow. ``MarginV=40`` keeps the line inside the
        bottom safe area without hugging the edge.
        """
        # ffmpeg subtitles filter requires forward slashes and escaping on Windows.
        srt_filter = srt_path.replace("\\", "/").replace(":", "\\:")
        force_style = (
            "FontName=Arial,FontSize=16,Bold=1,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "BackColour=&H00000000,"
            "BorderStyle=1,Outline=1.5,Shadow=0.5,"
            "Alignment=2,MarginV=12"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles='{srt_filter}':force_style='{force_style}'",
            "-c:a", "copy",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg subtitle burn-in failed:\n{r.stderr.decode(errors='replace')[:400]}"
            )

    # ═══════════════════════════════════════════════════════════
    # STAGE 1: WAV EXTRACTION
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
                f"ffmpeg audio extraction failed:\n{r.stderr.decode(errors='replace')[:300]}"
            )
        return out

    def _get_wav_duration(self, wav_path: str) -> float:
        with wave.open(wav_path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()

    # ═══════════════════════════════════════════════════════════
    # STAGE 2: TIMESTAMPED TRANSCRIPTION
    # ═══════════════════════════════════════════════════════════

    def _transcribe_segments(self, wav_path: str, language="tr") -> list:
        """Run faster-whisper and return start / end / text per segment.

        Loads a dedicated **medium model** for dubbing so the user's transcriber
        is not affected. This is critical: when cloud_auto is active, the user's
        transcriber.model is the 'base/CPU' fallback, which is insufficient for
        Turkish. Dubbing is a batch workload — quality matters more than speed,
        so we deliberately use 'medium'.

        Strategy:
          1. Try medium on the GPU (best Turkish accuracy, ~1.5 GB VRAM).
          2. Fall back to small on the CPU when no GPU is available or load OOMs.
          3. Release the model once the transcript is produced (free VRAM for the LLM).
        """
        from faster_whisper import WhisperModel
        import torch as _torch
        import gc as _gc

        local_model = None
        used_size, used_device = "small", "cpu"

        # 1. Try medium-GPU first.
        if _torch.cuda.is_available():
            try:
                print("[DUBBER] Loading Whisper 'medium' on the GPU (high-quality transcript)...")
                local_model = WhisperModel("medium", device="cuda", compute_type="int8")
                used_size, used_device = "medium", "cuda"
            except Exception as e:
                print(f"[DUBBER] medium-GPU failed ({e}); falling back to small-CPU.")
                local_model = None

        # 2. Fallback: small-CPU (better than 'base', acceptable CPU latency).
        if local_model is None:
            print("[DUBBER] Loading Whisper 'small' on the CPU (fallback)...")
            local_model = WhisperModel("small", device="cpu", compute_type="int8")
            used_size, used_device = "small", "cpu"

        try:
            segments_iter, _info = local_model.transcribe(
                wav_path,
                language=language,
                beam_size=5,                          # 2 -> 5 (wider search, +~10% accuracy).
                best_of=5,                            # 2 -> 5.
                # VAD DISABLED: the faster-whisper VAD is overly aggressive
                # and labels long speech blocks as 'silence', dropping them
                # entirely. VAD is only used in the reference-audio extractor
                # (to filter noise / music) — transcription should never drop
                # speech.
                vad_filter=False,
                condition_on_previous_text=True,      # Context helps; the transcript stays coherent.
                temperature=0.0,                      # Deterministic decoding.

                # ── WHISPER GUARD RAILS FULLY RELAXED (POST-DEMUCS) ─────
                # Pipeline ordering: Demucs separates vocals / instrumental,
                # Whisper runs only on the clean vocals.wav — the hallucination
                # sources (music, noise) are already gone. Whisper's own
                # protective filters now produce zero upside and actively delete
                # real speech in low-energy / breathy regions.
                #
                # Previous settings (0.5 default -> 0.85) still swallowed some
                # 30-s speech blocks. Final design:
                #   - compression_ratio_threshold=None: self-repetition guard
                #     disabled (rare after Demucs; must not delete real speech).
                #   - log_prob_threshold=None: even 50%-confident speech is
                #     emitted (don't drop it).
                #   - no_speech_threshold=0.95: only segments classified as
                #     95%-certain silence are dropped.
                compression_ratio_threshold=None,
                log_prob_threshold=None,
                no_speech_threshold=0.95,
            )
            result = []
            for seg in segments_iter:
                # Manual post-filter also lifted to 95% — defer to Whisper and
                # only reject in the highest-confidence noise cases.
                if seg.no_speech_prob > 0.95:
                    continue
                text = seg.text.strip()
                if not text:
                    continue
                result.append({"start": seg.start, "end": seg.end, "text": text})
            print(f"[DUBBER] Whisper '{used_size}/{used_device}' transcript done; {len(result)} segments.")
            return result
        finally:
            # Release the model immediately — Gemma needs VRAM next.
            try:
                del local_model
            except Exception:
                pass
            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

    def _merge_short_segments(self, segments: list) -> list:
        """Merge very short segments into the preceding one.

        XTTS-v2 produces awkward prosody and acoustic artifacts on inputs that
        are <1.5 s or 1-3 words. Merging short fragments into their predecessor
        yields more natural sentence-length utterances. Bonus: the translator
        sees more context (instead of translating a short fragment in isolation).

        Merge conditions:
          - Duration < 1.5 s OR word count < 4 (short-segment criteria).
          - Gap to the preceding segment < 0.6 s (acoustic proximity).
          - Merged total <= 12 s (do not blow past the XTTS limit).
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
        """Group Whisper segments into 6-10 s logical blocks.

        Distinct from ``_merge_short_segments``, which only rescues critically
        short fragments. This function **proactively** consolidates every
        segment into larger blocks.

        Benefits:
          - LLM call count drops by roughly half (quota-friendly, faster).
          - XTTS sees longer, more natural sentences -> better prosody, less
            robotic delivery.
          - Cross-sentence context coherence is preserved.

        Block-closing rules (in priority order):
          1. Block duration > TARGET_MIN (6 s) AND the current text ends in
             sentence-final punctuation -> close the block (natural sentence
             boundary).
          2. Gap between adjacent segments > MAX_GAP (0.8 s) -> close
             (long pause, likely scene / topic change).
          3. Merging would exceed TARGET_MAX (14 s) -> close
             (XTTS loses quality on overly long text).
          4. None of the above -> merge.
        """
        if not segments:
            return segments

        TARGET_MIN = 6.0     # Below this duration the block stays open even on a sentence boundary.
        TARGET_MAX = 10.0    # XTTS-v2 attention budget: prosody breaks down beyond 10 s,
                             # producing hallucination / metallic tone. Originally 14 s — lowered.
        MAX_GAP    = 0.8     # Tolerated silence gap between adjacent segments.
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

            # Block-closing decisions.
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
                # Merge.
                current["end"] = seg["end"]
                sep = " " if not current["text"].rstrip().endswith("-") else ""
                current["text"] = (current["text"].rstrip() + sep + seg["text"].lstrip()).strip()

        if current is not None:
            consolidated.append(current)

        return consolidated

    # ═══════════════════════════════════════════════════════════
    # STAGE 4: REFERENCE AUDIO
    # ═══════════════════════════════════════════════════════════

    def _extract_reference(self, wav_path: str, segments: list) -> list:
        """Extract multiple reference clips for speaker cloning.

        XTTS-v2 accepts multi-reference input; supplying several clips stabilizes
        the speaker identity and dilutes the impact of any single corrupted
        clip (music, noise).

        Strategy:
          - Prefer 4-10 s segments (too short -> insufficient signal;
            too long -> background noise accumulates).
          - Pick the 3 longest segments.
          - Trim leading / trailing silence in each clip with ffmpeg
            ``silenceremove``.
          - Reject any candidate that comes out shorter than 2 s.

        Returns: a list of WAV file paths (1-3 entries).
        """
        # Candidate segments in the 4-10 s window — collect generously so the
        # quality scorer can pick the best 3. The bigger the candidate pool, the
        # more selective we can be (rejecting noisy / musical fragments).
        candidates = [s for s in segments
                      if 4.0 <= (s["end"] - s["start"]) <= 10.0]

        if not candidates:
            # Fallback: take the single longest segment.
            candidates = sorted(segments, key=lambda s: s["end"] - s["start"],
                                reverse=True)[:1]
        else:
            # Take the 10 longest candidates — the scorer will pick the top 3.
            candidates = sorted(candidates, key=lambda s: s["end"] - s["start"],
                                reverse=True)[:10]

        # silenceremove: trim leading silence below -40 dB for up to 0.1 s, then
        # use ``areverse`` to do the same at the tail.
        silence_filter = (
            "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,"
            "areverse,"
            "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,"
            "areverse"
        )

        # Extract every candidate and score it. Scoring logic lives in
        # ``_score_reference_wav``:
        #   high mean_rms (not too quiet) +
        #   low RMS variance (stable speech tone, not music) +
        #   low silence ratio (not >50% silence) -> high score.
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
                # Silence or corrupted — reject.
                try: os.remove(out)
                except OSError: pass
                continue
            scored.append((score, out, seg))

        # Keep the top-3 scores; delete the rest.
        scored.sort(key=lambda x: x[0], reverse=True)
        ref_paths = [p for (_, p, _) in scored[:3]]
        for _, p, _ in scored[3:]:
            try: os.remove(p)
            except OSError: pass

        if ref_paths:
            top_scores = [f"{s:.3f}" for (s, _, _) in scored[:len(ref_paths)]]
            print(f"[DUBBER] Reference selection: {len(ref_paths)} candidate(s) "
                  f"(scores: {', '.join(top_scores)})")

        # Last-resort fallback: take the first 8 s raw if every candidate failed.
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
    # STAGE 5: XTTS VOICE CLONING
    # ═══════════════════════════════════════════════════════════

    def _score_reference_wav(self, wav_path: str) -> float:
        """Score a reference WAV's fitness for XTTS.

        Scoring rationale:
          - mean_rms: average energy. Penalizes very quiet / distant / reverberant audio.
          - RMS stability (1 / (1 + std/mean)): music or a variable background
            produces high variance; speech tone is more consistent.
          - silence_ratio: fraction of 100 ms windows below the silence floor;
            high values indicate dead air inside the reference — bad signal.
          - clipping_ratio: fraction of samples with |x| > 0.99. Distortion
            throws XTTS off.

        Returns: 0.0 = bad / reject, ~0.10-0.30 = normal speech, higher = ideal.
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

            # 100 ms windowed RMS.
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

            # Too quiet or too sparse: reject.
            if mean_rms < 0.015 or silence_ratio > 0.5:
                return 0.0
            # Excessive clipping: distorted, reject.
            if clipping_ratio > 0.02:
                return 0.0

            consistency = 1.0 / (1.0 + std_rms / (mean_rms + 1e-6))
            score = mean_rms * consistency * (1.0 - silence_ratio) * (1.0 - clipping_ratio)
            return float(score)
        except Exception:
            return 0.0

    @staticmethod
    def _clean_text_for_tts(text: str) -> str:
        """Sanitize the text before feeding it to XTTS.

        XTTS-v2 is sensitive: emoji, excessive punctuation, markdown markers
        and parenthetical stage directions ("(laughs)", "[music]") trigger
        hallucinations and prosody breakdowns. We strip them here.

        Returns: the cleaned string, or "" when the result is too short /
        meaningless (the segment will then be skipped).
        """
        if not text:
            return ""
        s = text.strip()
        # Markdown / quotes / asterisks.
        s = re.sub(r"[*_~`#]+", "", s)
        s = s.replace("\u201c", "").replace("\u201d", "").replace("\u2018", "").replace("\u2019", "'")
        s = s.replace('"', "")
        # Parenthetical / bracketed stage directions (laughs, music, ...).
        s = re.sub(r"\([^)]{0,40}\)", "", s)
        s = re.sub(r"\[[^\]]{0,40}\]", "", s)
        # Emoji and most non-BMP symbols.
        s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
        # Collapse runaway repeated punctuation: "..." -> "...", "!!!" -> "!".
        s = re.sub(r"([.!?,;:])\1{2,}", r"\1\1\1", s)
        # Whitespace.
        s = re.sub(r"\s+", " ", s).strip()
        # Too short / pure punctuation: skip the segment.
        alnum = re.sub(r"[^A-Za-z0-9\u00C0-\u017F]", "", s)
        if len(alnum) < 4:
            return ""
        return s

    def _ensure_xtts_loaded(self):
        """Guarantee that the XTTS model is loaded on the GPU."""
        if self.synthesizer.xtts_model is None:
            if self.synthesizer._xtts_loading:
                self.synthesizer._xtts_ready.wait()
            elif not self.synthesizer._load_xtts_model(use_gpu=True):
                raise RuntimeError(
                    "XTTS failed to load on the GPU (VRAM). Dubbing requires sufficient VRAM."
                )
        if self.synthesizer.xtts_model is None:
            raise RuntimeError("XTTS model failed to load.")

    def _get_speaker_latents(self, ref_wavs):
        """Compute the speaker-conditioning latents from the reference audio (once).

        ref_wavs: a single str or a list of strs (XTTS multi-ref support).
        Multi-ref input is averaged by the XTTS GPT, stabilizing the speaker
        identity and reducing the impact of any single corrupted segment.
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
        """Run XTTS inference for a single segment and write the WAV to disk.

        Conservative / stable parameter set — chosen for dubbing quality.
        XTTS-v2 defaults: temperature=0.75, length_penalty=1.0,
        repetition_penalty=10.0, top_k=50, top_p=0.85, speed=1.0.

          - temperature=0.50 (↓0.75): more consistent prosody, less "improvisation".
            Ideal for documentary-style narration; keeps the single-speaker tone stable.
          - repetition_penalty=10.0 (default): aggressive defense against XTTS
            hallucination tendencies. Keep the default — suppresses repeated /
            stuck syllables.
          - length_penalty=1.0 (default): neutral on duration.
          - top_k=50, top_p=0.85 (default): no aggressive sampling.
          - speed=1.0: CRITICAL. The earlier design used speed=1.15 plus
            atempo<=1.15 in assembly, but this was DOUBLE COMPRESSION — XTTS'
            already 15% accelerated / deformed signal followed by an ffmpeg
            phase vocoder produced a metallic timbre. We reverted to speed=1.0:
            natural timbre is preserved. Length is now controlled by
            (a) the translator length constraint (concise EN) and
            (b) the snowball-guard + emergency atempo inside ``_assemble_audio``
            — which only activates when desync is critical. Normal segments
            see zero post-processing.
          - enable_text_splitting=True: CRITICAL. ``_consolidate_segments``
            emits 6-14 s logical blocks containing 2-3 sentences. With this
            flag False XTTS processes the multi-sentence text in a single
            pass, emits <eos> mid-sentence (multi-sentence dropout) and the
            delivery becomes robotic. With it True the XTTS internal sentence
            splitter handles each sentence as its own prompt and stitches the
            results with natural pauses — robotic / dropout artifacts disappear.
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
    # STAGE 6: TIMELINE ASSEMBLY
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _trim_dead_air(data: np.ndarray, sr: int,
                       threshold_db: float = -40.0,
                       frame_ms: int = 20,
                       margin_ms: int = 20) -> np.ndarray:
        """RMS-based dead-air trimming.

        The legacy ``|amplitude|>0.01`` approach was insufficient against the
        XTTS noise floor (noise ~-30 dBFS, threshold ~-40 dBFS); the function
        could not actually remove the silence and atempo would fire needlessly.
        New logic:
          - Compute RMS in 20 ms windows, convert to dB.
          - Find the first / last window above ``threshold_db`` (default -40 dBFS).
          - Keep a ±margin_ms (default 20 ms) buffer so utterance onsets are not clipped.

        threshold_db = -40 dBFS in practice: XTTS pure silence < -55 dB,
        noise floor ~-35 dB, voiced > -20 dB. -40 dB sits safely between them.
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
            return data  # Fully silent — defer the decision to the caller.
        first_f = int(np.argmax(voiced))
        last_f = int(n_frames - np.argmax(voiced[::-1]))
        margin_frames = max(1, int((margin_ms / 1000.0) * sr / frame_len))
        first_f = max(0, first_f - margin_frames)
        last_f = min(n_frames, last_f + margin_frames)
        return data[first_f * frame_len: last_f * frame_len]

    @staticmethod
    def _fade_out(data: np.ndarray, sr: int, fade_ms: int = 80) -> np.ndarray:
        """Apply a linear fade-out to the tail (prevents an audible hard cut)."""
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
        Place every synthesized segment back at its original timestamp.

        Engineering rules (four design notes):
          1. Trim XTTS dead air first (RMS-based) so duration math is honest.
          2. Normally NO atempo (XTTS speed=1.0); the audio lands as is.
          3. Enforce a 0.2 s minimum breathing gap between sentences (anti-clipping).
          4. SNOWBALL GUARD: when accumulated desync > 1.5 s, emergency sync kicks in:
             a) Zero breathing gap (force fit).
             b) Emergency atempo (max ratio 1.25) — only here is the phase
                vocoder accepted.
             c) If still too long, hard-clip with a fade-out to match the video.
        """
        SR = self.XTTS_SR
        BREATH_FRAMES = int(0.20 * SR)             # Minimum 200 ms between sentences.
        MAX_DESYNC_SEC = 1.5                       # Maximum accumulated drift.
        MAX_DESYNC_FRAMES = int(MAX_DESYNC_SEC * SR)
        EMERGENCY_RATIO_CAP = 1.25                 # atempo ceiling in emergency mode.
        # Output buffer (mux trims the trailing slack).
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

            # 1) DEAD-AIR TRIM (RMS-based) — clean up the duration math.
            data = self._trim_dead_air(data, SR)

            # 4) DESYNC DETECTION: by how much did the previous segment cross
            # this segment's original start — i.e. accumulated drift.
            drift_frames = max(0, last_end_frame - orig_start_frame)
            emergency = drift_frames > MAX_DESYNC_FRAMES

            # 2+4) Normal mode: zero post-processing (XTTS speed=1.0 is natural).
            #      Emergency mode: emergency atempo (max 1.25) + hard-trim + fade.
            if emergency:
                eng_duration = len(data) / SR
                if eng_duration > seg_duration * 1.05:
                    ratio = eng_duration / seg_duration
                    apply_ratio = min(ratio, EMERGENCY_RATIO_CAP)
                    data = self._atempo_stretch(data, SR, apply_ratio, wav_path)
                    # Still too long? Hard-trim + fade-out.
                    eng_after = len(data) / SR
                    if eng_after > seg_duration * 1.20:
                        max_len = int(seg_duration * 1.15 * SR)
                        data = self._fade_out(data[:max_len], SR)
                emergency_count += 1

            # 3) BREATHING GAP: 200 ms in normal mode; 0 in emergency mode (force fit).
            breath = 0 if emergency else (BREATH_FRAMES if last_end_frame > 0 else 0)
            min_start = last_end_frame + breath
            start_frame = max(orig_start_frame, min_start)
            end_frame = min(start_frame + len(data), total_frames)
            if end_frame <= start_frame:
                continue
            output[start_frame:end_frame] = data[:end_frame - start_frame]
            last_end_frame = end_frame

        if emergency_count > 0:
            print(f"[DUBBER] Snowball guard engaged {emergency_count} time(s) "
                  f"(desync > {MAX_DESYNC_SEC}s).")

        # Trim the trailing slack.
        output = output[:max(last_end_frame, int(video_duration * SR))]
        sf.write(out_path, output, SR)

    def _atempo_stretch(self, data: np.ndarray, sr: int, ratio: float,
                        orig_path: str) -> np.ndarray:
        """Speed up audio via the ffmpeg atempo filter (ratio > 1.0)."""
        tmp_out = orig_path.replace(".wav", "_tempo.wav")

        # atempo accepts at most 2.0 — chain when the ratio exceeds 2.0.
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
        return data  # Fall back to the original audio on error.

    # ═══════════════════════════════════════════════════════════
    # STAGE 7: VIDEO MUX
    # ═══════════════════════════════════════════════════════════

    def _resample_mono(self, in_path: str, out_path: str, sr: int = 16000):
        """Convert a WAV / audio file to mono at the target sample rate via ffmpeg.

        Used to bridge the Demucs output (vocals_hq, 44.1 kHz stereo) into the
        Whisper input format (16 kHz mono).
        """
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", in_path,
             "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", out_path],
            capture_output=True, timeout=300
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg resample failed:\n{r.stderr.decode(errors='replace')[:300]}"
            )

    def _mix_with_instrumental(self, video_path: str, dubbed_wav: str,
                               instrumental_path: str | None,
                               output_path: str):
        """Final mix: dubbed voice + (optional) original instrumental + video.

        When instrumental_path is None: classic stream-copy mux (dubbed voice only).

        When instrumental_path is present: ffmpeg ``sidechaincompress`` performs
        auto-ducking. The background music / SFX is dynamically attenuated under
        the dubbed dialogue and recovers between utterances — the standard
        technique in enterprise dubbing and broadcast / podcast post-production.

        Sidechain parameter choices:
          threshold=0.03   → ducking still triggers on quieter dialogue.
          ratio=20         → aggressive ducking (-10 to -15 dB typical).
          attack=5 ms      → fast response (word onsets are ducked instantly).
          release=300 ms   → smooth recovery after the dialogue ends (no popping).
          makeup=1         → neutral makeup gain on the ducked signal.
        """
        if instrumental_path is None or not os.path.exists(instrumental_path):
            # FALLBACK: classic stream-copy mux (legacy _mux behavior).
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
                    f"ffmpeg mux failed:\n{r.stderr.decode(errors='replace')[:300]}"
                )
            return

        # SIDECHAIN COMPRESSION + MIX + VIDEO MUX in a single ffmpeg call.
        # Inputs: 0=video, 1=dubbed_voice, 2=instrumental.
        # Filter graph: align voice and bg to the same sample rate (24 kHz, XTTS_SR)
        # and to stereo; duck the instrumental ('bg') against the voice via
        # sidechaincompress; mix voice + ducked_bg via amix into a single stereo
        # track.
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
            print(f"[DUBBER] sidechain mix failed, retrying with fallback mux:\n{err}")
            # FALLBACK: sidechain unsupported (older ffmpeg / missing filter).
            self._mix_with_instrumental(video_path, dubbed_wav, None, output_path)

    # ═══════════════════════════════════════════════════════════
    # CLEANUP
    # ═══════════════════════════════════════════════════════════

    def _cleanup(self, paths: list):
        for p in paths:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
