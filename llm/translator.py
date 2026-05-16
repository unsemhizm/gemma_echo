import os
import gc
import json
import time
import string
import threading
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from llama_cpp import Llama
from core.logger import get_logger

# Load environment variables from .env.
load_dotenv()

log = get_logger(__name__)


def _load_cultural_concepts() -> dict:
    """Load the cultural-concept dictionary from data/cultural_concepts.json."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "cultural_concepts.json")
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


CULTURAL_CONCEPTS = _load_cultural_concepts()


# ── LANGUAGE-AWARE CHARACTER RATE TABLE (DUBBING USE) ─────────────────────────
# Different languages encode different amounts of information per second of
# speech, so the number of characters that "fits" in a second varies. Calibrated
# from Pellegrino et al. (2011) "A cross-language perspective on speech
# information rate" combined with industry dubbing / subtitling averages:
#
#   - Latin-alphabet spoken languages: 13–17 chars/sec (avg 4–6 letters/word)
#   - CJK (Chinese / Japanese / Korean): 6–9 chars/sec (each glyph is far denser)
#   - Arabic: compact words, letter count close to Latin → 14 chars/sec
#
# Used inside translator.translate() whenever target_duration_sec is provided
# to compute a character budget. This makes the dubbing constraint *duration
# based* rather than *word-count based*, so AR / JA / ZH behave correctly.
CHARS_PER_SEC = {
    "en": 15, "es": 17, "fr": 16, "de": 13, "it": 17, "pt": 16,
    "tr": 14, "ru": 14, "pl": 14, "nl": 14, "cs": 14, "hu": 14,
    "ar": 14, "ja":  8, "zh":  6, "ko":  9, "hi": 13,
}
DEFAULT_CHARS_PER_SEC = 14  # Conservative midpoint for unknown languages.

# ── Persona templates (dynamic — adapted via {tgt_lang} at format time) ─────
#
# "default" → No persona instruction is injected; the model performs a neutral
#             literal-friendly translation. Users are not forced to pick a persona.
# "none"    → Identical alias of "default", retained for backwards compatibility.
PERSONA_TEMPLATES = {
    # ── Neutral (default) ────────────────────────────────────────────────────
    "default": "",   # No persona instruction is appended to the prompt.
    "none":    "",   # Backwards-compatible alias.

    # ── Stylistic personas ───────────────────────────────────────────────────
    "official": (
        "Tone: Act as a senior diplomat or academic professional. "
        "Use the most formal, polished, and respectful register of {tgt_lang}. "
        "Avoid all slang, contractions, and colloquialisms."
    ),
    "streamer": (
        "Tone: Act as an energetic live streamer speaking to a {tgt_lang}-speaking audience. "
        "Use popular internet slang, gaming terminology, and hype expressions that are "
        "native to {tgt_lang} internet culture. "
        "Do NOT copy English slang word-for-word — find the culturally equivalent term in {tgt_lang}."
    ),
    "casual": (
        "Tone: Act as a close friend of the speaker. "
        "Use everyday, relaxed, and casual {tgt_lang} conversational style "
        "with friendly expressions and informal greetings natural to {tgt_lang} culture."
    ),
    "literary": (
        "Tone: Act as a skilled literary translator. "
        "Use vivid, expressive, and elegant {tgt_lang}. "
        "Preserve the emotional tone and imagery of the original text."
    ),
}

class Translator:
    def __init__(self):
        """
        Initialize the translation layer (v8 Multi-State architecture).

        ONLINE MODE — two-tier fallback cascade:
          Tier 1: Gemini API (Gemma 4 26B)      → Primary translator (quality-first)
          Tier 2: Gemini API (Gemini 2.5 Flash) → Speed fallback

        OFFLINE MODE — zero-dependency local engine:
          Local C++ inference engine → ./models/gemma-4-q4.gguf
          Lazy-loaded on demand so VRAM is never wasted while online.
        """
        log.info("Translator v8 'Multi-State' module initializing...")

        # Active mode: "online" (default) or "offline".
        self.mode = "online"

        # ─── ONLINE ENGINES ─────────────────────────────────────

        # 1. PRIMARY ENGINE: GEMINI API (GEMMA 4 26B)
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY is missing!")
        self.gemini_client = genai.Client(api_key=self.gemini_key.strip())
        self.gemma4_api_model = "gemma-4-26b-a4b-it"

        # 2. SECONDARY ENGINE: GEMINI API (GEMINI 2.5 FLASH)
        self.gemini_fallback_model = "gemini-2.5-flash"

        # ─── OFFLINE ENGINE (local C++ inference / zero-dependency) ────
        # Loaded on demand — keeps VRAM free while the cloud path is active.

        self.local_model_path = "./models/gemma-4-q4.gguf"
        self.local_llm = None  # Lazy-loaded via load_local_model().
        self._llm_vram_failed = False
        self.vram_issue_callback = None  # GUI warning hook: () -> None

        # ── Online quota cooldown (429 RESOURCE_EXHAUSTED) ───────────────
        # When the free tier exhausts its quota, suspending the online tier
        # is far cheaper than spending 20-40s per chunk on doomed requests.
        # Critical for document translation where wasting 30s per chunk is
        # unacceptable — we drop straight to the local engine instead.
        self._online_cooldown_until = 0.0   # Epoch seconds; online is skipped while time.time() < this.
        self._consecutive_quota_errors = 0  # Used to back off the cooldown duration exponentially.


        # Serialize concurrent translate() calls and protect shared state
        # (system_prompt, quota counters, ...).
        self._translate_lock = threading.Lock()

        # Active persona — defaults to "default" (no style instruction injected).
        # The user may optionally choose a persona.
        # Valid values: "default" | "none" | "official" | "streamer" | "casual" | "literary"
        self.persona = "default"

        # Default system prompt — used for direct offline calls made before
        # translate() builds a language-aware prompt.
        self.system_prompt = self._build_system_prompt()

    def set_persona(self, persona: str):
        """Configure the active persona style.

        Valid values:
          "default"  — No persona instruction; pure literal-friendly translation (default).
          "none"     — Alias of "default" kept for backwards compatibility.
          "official" — Diplomatic / academic formal register.
          "streamer" — Streamer / internet-jargon register.
          "casual"   — Casual / friendly conversational register.
          "literary" — Literary / poetic translation register.

        Any invalid value falls back to "default".
        """
        valid = {"default", "none", "official", "streamer", "casual", "literary"}
        self.persona = persona if persona in valid else "default"

    def _build_system_prompt(self, src_lang="Turkish", tgt_lang="English") -> str:
        """Construct a dynamic system prompt.

        When persona is "default" or "none" (or unknown), only the base
        translation instruction is returned — no stylistic suffix is added.
        When a stylistic persona is active, its template is appended to the base.

        When ``_dubbing_mode`` is enabled the prompt is extended with a
        "hard rule" enforcing length / context constraints so the spoken
        output fits the dubbing timeline.
        """
        base = (
            f"You are a lightning-fast translator. Translate the following {src_lang} text to {tgt_lang}. "
            f"Reply ONLY with the {tgt_lang} translation. Do not add quotes, explanations, or any other text. "
            f"CRITICAL: Never translate idioms, proverbs, or cultural expressions word-for-word. "
            f"Always find the natural, culturally equivalent expression a native {tgt_lang} speaker would actually say."
        )
        # DUBBING CONSTRAINT: appended when dubber.py calls set_dubbing_mode(True).
        # This flag is never True during document translation, so TXT/DOCX/PDF
        # workflows are unaffected.
        #
        # OLD design: "EN <= TR word count" — only correct for TR<->EN.
        # NEW design: duration-based, language-agnostic. Every translate() call
        # injects a "[DUBBING TIMING] ... ≤ N characters" hint into the user
        # message (see length_hint computation in translate()). This works
        # correctly for AR / JA / ZH as well.
        if getattr(self, "_dubbing_mode", False):
            base += (
                f"\n\nDUBBING TIMING CONSTRAINT (HARD RULE):\n"
                f"- This translation will be spoken aloud over a video timeline.\n"
                f"- Each user message contains a [DUBBING TIMING] tag with a character"
                f" limit calibrated to the segment's audio duration in {tgt_lang}.\n"
                f"- Your {tgt_lang} translation MUST stay within that character limit.\n"
                f"- Be concise: drop filler words, hedges, redundant phrases.\n"
                f"- Prefer short, punchy sentences. Shorter is always safer for sync.\n"
                f"- If forced to choose between literal accuracy and timing, choose timing"
                f" — keep meaning, drop unnecessary words.\n"
                f"\nCONTEXT USE (HARD RULE):\n"
                f"- Any previous translation or rolling summary is CONTEXT ONLY.\n"
                f"- NEVER repeat, paraphrase, or re-translate previous segments.\n"
                f"- Translate ONLY the current new {src_lang} text given to you."
            )
        persona_template = PERSONA_TEMPLATES.get(self.persona, "")
        if persona_template:  # "default" / "none" map to empty strings — this branch is skipped.
            persona_instr = persona_template.format(tgt_lang=tgt_lang)
            return f"{base}\n{persona_instr}"
        return base

    def set_dubbing_mode(self, on: bool):
        """Enable or disable dubbing mode.

        When enabled, the system prompt gains the length + context "hard rules"
        defined in ``_build_system_prompt``. Document translation must always
        leave this flag False — the length constraint would degrade quality.
        """
        self._dubbing_mode = bool(on)
        # The active source/target languages are not known here; rebuild with
        # the default Turkish->English pair. The downstream translate() path
        # rebuilds the prompt with the correct languages at call time, so this
        # invocation only commits the flag transition.
        self.system_prompt = self._build_system_prompt()


    # ═══════════════════════════════════════════════════════════
    # LOCAL MODEL MANAGEMENT — Lazy Load / Unload
    # ═══════════════════════════════════════════════════════════

    def _invoke_vram_callback(self):
        cb = self.vram_issue_callback
        if cb is None:
            return
        try:
            cb()
        except Exception:
            pass

    def load_local_model(self, context_size: int = 512) -> bool:
        """Load the local GGUF model into VRAM via the C++ inference engine.

        Idempotent: if the model is already resident at the requested context
        size, the call is a no-op. If a different context size is loaded, the
        old model is evicted before reloading.

        Returns True on success, False on VRAM exhaustion / OOM.
        """
        if self.local_llm is not None:
            same_ctx  = getattr(self, "loaded_n_ctx", 512) == context_size
            if same_ctx:
                return True   # Already loaded with the correct configuration — no-op.
            log.info(f"Reloading local model (n_ctx {getattr(self, 'loaded_n_ctx', 512)} → {context_size}).")
            self.unload_local_model()

        if self._llm_vram_failed:
            return False

        from gpu_memory import (
            MIN_FREE_BYTES_LOCAL_LLM,
            cleanup_cuda_memory,
            is_cuda_oom_error,
            vram_sufficient_for_llm,
        )

        ok, free = vram_sufficient_for_llm()
        if not ok:
            log.warning(
                f"Local LLM VRAM pre-check failed "
                f"(free: {free} B, threshold: {MIN_FREE_BYTES_LOCAL_LLM} B)"
            )
            self._llm_vram_failed = True
            cleanup_cuda_memory()
            self._invoke_vram_callback()
            return False

        log.info(f"Loading local LLM (n_ctx={context_size}): {self.local_model_path}")
        start = time.time()
        try:
            self.local_llm = Llama(
                model_path=self.local_model_path,
                n_gpu_layers=-1,
                n_ctx=context_size,
                verbose=False,
            )
            self.loaded_n_ctx  = context_size
        except Exception as e:
            self.local_llm = None
            cleanup_cuda_memory()
            if is_cuda_oom_error(e):
                # │ VRAM EXHAUSTION ──────────────────────────────────────────────────────
                # exc_info=True ensures the full stack trace is persisted to the log
                # file. Without it we lose the line + timing of CUDA OOM events.
                log.error(
                    "Local LLM CUDA OOM — VRAM insufficient, model not loaded. "
                    "Falling back to the online tier.",
                    exc_info=True
                )
                self._llm_vram_failed = True
                self._invoke_vram_callback()
                return False
            raise

        self._llm_vram_failed = False
        elapsed = int((time.time() - start) * 1000)
        log.info(f"Local LLM ready ({elapsed} ms).")
        return True

    def unload_local_model(self):
        """Fully evict the local model from RAM and VRAM.

        Invoked when transitioning to online mode to prevent VRAM contention
        with the XTTS or Whisper engines.
        """
        if self.local_llm is None:
            self._llm_vram_failed = False
            return

        log.info("Evicting local LLM from VRAM...")
        del self.local_llm
        self.local_llm = None
        self._llm_vram_failed = False
        if hasattr(self, "loaded_n_ctx"):
            del self.loaded_n_ctx
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Local LLM evicted from VRAM.")

    # ═══════════════════════════════════════════════════════════
    # MODE MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        """
        Switch the active translation mode.
        "online"  → Cloud APIs (Gemini API [Gemma 4] → Gemini API [Flash])
        "offline" → Local C++ inference engine (./models/gemma-4-q4.gguf, lazy-loaded)
        """
        if mode not in ("online", "offline"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'online' or 'offline'.")

        old_mode = self.mode
        self.mode = mode
        if mode == "offline":
            self._llm_vram_failed = False
        log.info(f"Translator mode transition: {old_mode} -> {mode}")


    # ═══════════════════════════════════════════════════════════
    # PRIMARY TRANSLATION ENTRYPOINT (dispatcher)
    # ═══════════════════════════════════════════════════════════

    def translate(self, text_tr: str, context: list = None, src_lang="tr", tgt_lang="en",
                  src_name="Turkish", tgt_name="English", prev_translation: str = "",
                  rolling_summary: str = "", target_duration_sec: float = None) -> dict:
        """
        Translate the supplied text into the target language.

        Dispatches to the online or offline engine based on the active mode.

        Args:
            target_duration_sec: Used exclusively by the dubbing path
                (set_dubbing_mode=True). When provided, a character-budget hint
                is injected into the user message so the LLM keeps the
                translation within bounds.
                Formula: CHARS_PER_SEC[tgt_lang] * duration * 1.05 (margin).
        """
        if context is None:
            context = []

        with self._translate_lock:
            if not text_tr or len(text_tr.strip()) == 0:
                return {"translation": "", "latency_ms": 0, "engine": "None"}

            cultural_result, match_type = self._check_cultural(text_tr, src_lang, tgt_lang)

            if match_type == "exact_fast":
                return {"translation": cultural_result, "latency_ms": 0, "engine": "CulturalMap"}

            hint = ""
            if match_type == "exact_intent":
                idiom, intent = cultural_result
                hint = (
                    f"CULTURAL CONTEXT: The phrase '{idiom}' conveys the meaning of "
                    f"'{intent}'. Translate this naturally into {tgt_name}, matching the active style.\n\n"
                )
            elif match_type == "partial":
                idiom, intent = cultural_result
                hint = (
                    f"CULTURAL CONTEXT: The text contains '{idiom}' which conveys '{intent}'. "
                    f"Translate this expression naturally into {tgt_name} as part of the full sentence.\n\n"
                )

            # ── DUBBING DURATION CONSTRAINT ────────────────────────────
            # Compute a language-aware character budget and inject it into the
            # user message. CHARS_PER_SEC[lang] is calibrated separately for
            # Latin / CJK / Arabic scripts; the legacy word-count heuristic
            # broke down on JA/ZH whereas this approach holds uniformly.
            if getattr(self, "_dubbing_mode", False) and target_duration_sec:
                lang_code = (tgt_lang or "en").lower()[:2]
                rate = CHARS_PER_SEC.get(lang_code, DEFAULT_CHARS_PER_SEC)
                # 5% margin — fits at XTTS native speed=1.0; the emergency
                # atempo branch kicks in only if drift exceeds 1.5s (see
                # assemble_audio).
                max_chars = max(20, int(target_duration_sec * rate * 1.05))
                length_hint = (
                    f"[DUBBING TIMING] Segment duration: {target_duration_sec:.1f}s. "
                    f"Your {tgt_name} translation MUST be \u2264 {max_chars} characters. "
                    f"Be concise; drop fillers; preserve meaning over literal wording.\n\n"
                )
                hint = length_hint + hint

            # Refresh the system prompt with the current source/target languages.
            self.system_prompt = self._build_system_prompt(src_name, tgt_name)

            if self.mode == "online":
                return self.translate_online(text_tr, context, hint, prev_translation, rolling_summary)
            return self.translate_offline(text_tr, context, hint, prev_translation, rolling_summary)

    # ═══════════════════════════════════════════════════════════
    # ROLLING SUMMARY GENERATION (Stage 2)
    # ═══════════════════════════════════════════════════════════

    def generate_summary(self, text: str, current_summary: str = "") -> str:
        """Update the running document summary using the newly translated chunk."""
        prompt = (
            f"You are an assistant summarizing a document so far. "
            f"Update the following current summary with the key points from the new translated section. "
            f"Keep the total summary under 100 words. Keep it clear, factual and coherent.\n\n"
            f"Current Summary: {current_summary or 'No summary yet.'}\n\n"
            f"New Translated Section: {text}\n\n"
            f"Updated Summary (Reply ONLY with the new updated summary under 100 words):"
        )

        if self.mode == "online":
            try:
                # Competition policy: Gemma 4 is always the first-priority engine.
                resp = self.gemini_client.models.generate_content(
                    model=self.gemma4_api_model,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=150
                    ),
                    contents=prompt
                )
                return (resp.text or "").strip()
            except Exception as e:
                log.warning(f"Gemma 4 summarization failed — falling back to Gemini Flash.", exc_info=True)
                try:
                    # Fast / cheap secondary engine.
                    resp = self.gemini_client.models.generate_content(
                        model=self.gemini_fallback_model,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            max_output_tokens=150
                        ),
                        contents=prompt
                    )
                    return (resp.text or "").strip()
                except Exception:
                    return current_summary
        else:
            # 512-token context is sufficient for generate_summary (prompt + summary < 400 tokens).
            if self.local_llm is None:
                if not self.load_local_model(512):
                    return current_summary
            try:
                # Account for chat-template overhead on the offline path.
                CHAT_TEMPLATE_OVERHEAD = 35
                safe_margin = 100
                try:
                    p_tokens = len(self.local_llm.tokenize(prompt.encode('utf-8'))) + CHAT_TEMPLATE_OVERHEAD
                    ctx = getattr(self, "loaded_n_ctx", 512)
                    safe_max = max(30, ctx - p_tokens - safe_margin)
                    actual_max = min(100, safe_max)
                except Exception:
                    actual_max = 80  # Conservative fallback budget.

                if actual_max < 20:
                    log.warning("generate_summary: prompt is saturating the context — summary skipped.")
                    return current_summary

                response = self.local_llm.create_chat_completion(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=actual_max
                )
                return response["choices"][0]["message"]["content"].strip()
            except Exception:
                return current_summary

    # ═══════════════════════════════════════════════════════════
    # ONLINE TRANSLATION — 3-tier turbo fallback cascade
    # (Gemini API [Gemma 4] → Gemini API [Flash] → local GGUF)
    # ═══════════════════════════════════════════════════════════

    def _maybe_trigger_quota_cooldown(self, exc: Exception, model_name: str):
        """Trigger the online cooldown when 429 RESOURCE_EXHAUSTED is detected.

        Detection: looks for "429" or "RESOURCE_EXHAUSTED" in the exception
        string.
        """
        msg = str(exc)
        if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
            return  # Non-quota error — handled by the two-tier path in _trigger_online_failure_cooldown.
        self._trigger_online_failure_cooldown(model_name, reason="429")

    def _trigger_online_failure_cooldown(self, model_name: str, reason: str = "failure"):
        """Start the online-tier cooldown when the cloud path has completely failed.

        Triggers:
          - 429 RESOURCE_EXHAUSTED (quota exhausted).
          - Both Gemma 4 and Flash timing out (240s+ wasted per chunk).
          - Simultaneous failure on both tiers.

        Cooldown schedule:
          - First failure: 180s (3 min).
          - Each subsequent consecutive failure: 2x backoff (3 → 6 → 12 → ... capped at 60 min).
          - When the free-tier 20/min limit is hit, every subsequent chunk
            drops instantly to offline rather than burning 2 minutes per chunk
            on doomed cloud requests.
        """
        self._consecutive_quota_errors += 1
        cooldown_sec = min(3600, 180 * (2 ** (self._consecutive_quota_errors - 1)))
        self._online_cooldown_until = time.time() + cooldown_sec
        log.warning(
            f"⚠ Online cooldown ({reason}, {model_name}): "
            f"skipping cloud tiers for {cooldown_sec}s ({cooldown_sec // 60} min). "
            f"(consecutive failures: {self._consecutive_quota_errors})"
        )

    def _gemini_call(self, model_name: str, user_message: str, timeout: float = 20.0, max_output_tokens: int = 300):
        """Issue a Gemini API call on a daemon thread.

        - Uses ``generate_content`` (non-streaming) — guaranteed model compatibility.
        - Raises ``TimeoutError`` if the response does not arrive within ``timeout`` seconds.
        - Daemon thread isolation — the main thread is never blocked.
        - Raises ``ValueError`` on an empty response so the caller can cascade.
        """
        result = [None]
        error = [None]

        # The Gemini 2.5 family is a "thinking" model — it can spend the entire
        # output budget on internal reasoning and leave nothing for the actual
        # translation. Translation does not need internal reasoning, so disable it.
        cfg_kwargs = dict(
            system_instruction=self.system_prompt,
            temperature=0.2,
            max_output_tokens=max_output_tokens,
        )
        if "2.5" in model_name:
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass  # Older SDK without ThinkingConfig — fail silently.

        def _is_transient_5xx(exc) -> bool:
            """Detect transient 5xx errors that warrant a fast retry.

            429 (quota) and 4xx (bad request) are NOT retried — they are permanent.
            """
            try:
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if isinstance(code, int) and 500 <= code < 600:
                    return True
                msg = str(exc)
                return ("500 INTERNAL" in msg) or ("503 " in msg) or ("UNAVAILABLE" in msg)
            except Exception:
                return False

        def _do_request():
            return self.gemini_client.models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(**cfg_kwargs),
                contents=user_message
            )

        def _call():
            try:
                try:
                    resp = _do_request()
                except Exception as e1:
                    # 5xx -> single fast retry. Google's API is intermittently
                    # flaky; cascading straight to the fallback would reset
                    # context and max_tokens with a quality penalty. Logs show
                    # dozens of "500 INTERNAL" hits per session, most of which
                    # resolve with a single retry.
                    if _is_transient_5xx(e1):
                        log.info(f"[GEMINI] {model_name} 5xx -> retrying once after 0.8s...")
                        time.sleep(0.8)
                        resp = _do_request()
                    else:
                        raise
                result[0] = resp.text or ""
                # ── DIAGNOSTICS: surface why the response terminated ──
                try:
                    cand = resp.candidates[0] if resp.candidates else None
                    fr = getattr(cand, "finish_reason", None) if cand else None
                    um = getattr(resp, "usage_metadata", None)
                    in_tok = getattr(um, "prompt_token_count", "?") if um else "?"
                    out_tok = getattr(um, "candidates_token_count", "?") if um else "?"
                    out_chars = len(result[0])
                    log.info(
                        f"[GEMINI] {model_name} | finish_reason={fr} | "
                        f"max_out={max_output_tokens} | in_tok={in_tok} out_tok={out_tok} | "
                        f"chars={out_chars}"
                    )
                except Exception:
                    pass
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            raise TimeoutError(f"Gemini ({model_name}) did not respond within {timeout}s.")
        if error[0] is not None:
            raise error[0]
        if not result[0]:
            raise ValueError(f"Gemini ({model_name}) returned an empty response.")
        return result[0]

    def translate_online(self, text_tr: str, context: list = [], hint: str = "",
                         prev_translation: str = "", rolling_summary: str = "") -> dict:
        """
        Translate via the cloud APIs.

        Two-tier fallback: Gemini API (Gemma 4) failure → Gemini 2.5 Flash.
        If both fail, the call cascades to the offline engine.

        Quota cooldown: on a free-tier 429 the online path is temporarily
        suspended (rather than burning timeout budget on doomed requests).
        """
        # Enrich the user message with context, prev_translation, rolling_summary.
        user_message = self._build_user_message(text_tr, context, hint, prev_translation, rolling_summary)

        # ── QUOTA COOLDOWN GUARD ─────────────────────────────────────────
        # While active, skip the cloud tiers and go straight to offline.
        now = time.time()
        if now < self._online_cooldown_until:
            remaining = int(self._online_cooldown_until - now)
            log.info(
                f"Online quota cooldown active ({remaining}s remaining) — "
                f"bypassing cloud tiers, dropping to offline directly."
            )
            try:
                return self.translate_offline(text_tr, context, hint, prev_translation, rolling_summary)
            except Exception:
                log.critical("Offline path also failed (during cooldown).", exc_info=True)
                return {"translation": "[TRANSLATION ERROR]", "latency_ms": 0, "engine": "Failed"}

        # ── DYNAMIC TIMEOUT BUDGET ───────────────────────────────────────
        # Live audio (push-to-talk) demands a tight 20s fallback.
        # Book translation (long text or heavy context) deserves a 60s budget.
        input_words = len(text_tr.split())
        is_heavy_workload = bool(prev_translation or rolling_summary or input_words > 50)
        dynamic_timeout = 60.0 if is_heavy_workload else 20.0

        # Gemma 4 (primary): can return successfully after 90-150s even when
        # under quota pressure. An overly aggressive timeout (60s) effectively
        # disables it, so we grant it a 2x budget. Flash is the fast fallback
        # and uses the nominal timeout.
        gemma4_timeout = dynamic_timeout * 2  # 40s (live) / 120s (heavy)
        flash_timeout  = dynamic_timeout       # 20s (live) / 60s (heavy)

        # Adaptive output budget driven by input word count (prevents truncation).
        # TR→EN translation typically inflates output by 20-30%; Gemma 4 may also
        # burn reasoning tokens that count against the same budget. Therefore:
        #   - Floor 600  (was 300; logs showed MAX_TOKENS hits at 300).
        #   - Multiplier 4 (was 3; safety margin for reasoning + language expansion).
        # Examples: live 10 words → 600, dubbing/media 25 words → 600,
        # 100 words → 600, 200 words → 800, 1024 words → 4096 cap.
        dynamic_max_tokens = min(4096, max(600, input_words * 4))

        # Track per-tier failure so that BOTH-failing triggers the cooldown
        # (regardless of timeout vs error), ensuring subsequent chunks skip
        # the cloud path entirely.
        gemma4_failed_reason = None
        flash_failed_reason = None

        # --- TIER 1: GEMINI API (GEMMA 4) ---
        gemini_gemma_start = time.time()
        try:
            translation = self._gemini_call(self.gemma4_api_model, user_message, timeout=gemma4_timeout, max_output_tokens=dynamic_max_tokens)
            self._consecutive_quota_errors = 0
            latency = int((time.time() - gemini_gemma_start) * 1000)
            return {
                "translation": translation.strip(),
                "latency_ms": latency,
                "engine": f"Gemini API ({self.gemma4_api_model})"
            }
        except TimeoutError:
            log.warning(f"Gemini API ({self.gemma4_api_model}) timeout ({gemma4_timeout}s) -> cascading to Gemini 2.5 Flash...")
            gemma4_failed_reason = "timeout"
        except Exception as e:
            self._maybe_trigger_quota_cooldown(e, self.gemma4_api_model)
            log.warning(
                f"Gemini API ({self.gemma4_api_model}) error -> cascading to Gemini 2.5 Flash.",
                exc_info=True
            )
            gemma4_failed_reason = "error"

        # --- TIER 2: GEMINI 2.5 FLASH ---
        gemini_flash_start = time.time()
        try:
            translation = self._gemini_call(self.gemini_fallback_model, user_message, timeout=flash_timeout, max_output_tokens=dynamic_max_tokens)
            latency = int((time.time() - gemini_flash_start) * 1000)
            self._consecutive_quota_errors = 0
            return {
                "translation": translation.strip(),
                "latency_ms": latency,
                "engine": f"Gemini API ({self.gemini_fallback_model})"
            }
        except TimeoutError:
            log.warning(f"Gemini 2.5 Flash timeout ({flash_timeout}s). Dropping to local model...")
            flash_failed_reason = "timeout"
        except Exception as e:
            self._maybe_trigger_quota_cooldown(e, self.gemini_fallback_model)
            log.warning("Gemini 2.5 Flash error. Dropping to local model.", exc_info=True)
            flash_failed_reason = "error"

        # BOTH TIERS FAILED — trigger cooldown (timeout or error, indifferently).
        # _maybe_trigger_quota_cooldown only increments the counter on 429;
        # on timeouts we still need to start an aggressive cooldown here to
        # avoid wasting another 240s on the next chunk.
        if gemma4_failed_reason and flash_failed_reason:
            # Avoid double-counting when a quota cooldown is already armed.
            if self._consecutive_quota_errors == 0 or (time.time() >= self._online_cooldown_until):
                self._trigger_online_failure_cooldown(
                    "both_layers",
                    reason=f"gemma4_{gemma4_failed_reason}+flash_{flash_failed_reason}"
                )

        # --- TIER 3: LOCAL OFFLINE MODEL FALLBACK (uninterrupted service) ---
        log.warning("All cloud APIs failed — engaging the local Gemma model (GGUF)...")
        try:
            return self.translate_offline(text_tr, context, hint, prev_translation, rolling_summary)
        except Exception:
            log.critical(
                "Local offline model also failed — every tier of the cascade is down!",
                exc_info=True
            )
            return {
                "translation": "[TRANSLATION ERROR]",
                "latency_ms": 0,
                "engine": "Failed"
            }

    # ═══════════════════════════════════════════════════════════
    # OFFLINE TRANSLATION — Local C++ inference engine (zero-dependency)
    # ═══════════════════════════════════════════════════════════

    def translate_offline(self, text_tr: str, context: list = [], hint: str = "",
                          prev_translation: str = "", rolling_summary: str = "") -> dict:
        """
        Translate via the local GGUF model.

        Requires no internet connectivity. The model is lazy-loaded into VRAM
        on first use.
        """
        # Pick the context window based on workload type:
        # - Standard live translation: 512 (fast, minimum memory).
        # - Document translation (rolling_summary / prev_translation present): 8192.
        #   Gemma 4 supports 131k natively; a 800-word chunk + system_prompt +
        #   glossary commonly exceeds 2048 tokens (measured: ~2700 tokens).
        req_ctx = 8192 if (prev_translation or rolling_summary) else 512

        if self.local_llm is None:
            if self._llm_vram_failed:
                return self.translate_online(text_tr, context, hint, prev_translation, rolling_summary)
            if not self.load_local_model(req_ctx):
                return self.translate_online(text_tr, context, hint, prev_translation, rolling_summary)
        else:
            # BUG FIX: only ever GROW the context window, never shrink it.
            # Without this guard, the dubber would load 8192, then on the first
            # segment (prev_translation="") compute req_ctx=512 and the model
            # would be reloaded down to 512 — which then triggered the spurious
            # "context full" warning on short Whisper segments and bounced the
            # request between translate_online and translate_offline in an
            # infinite loop.
            loaded = getattr(self, "loaded_n_ctx", 512)
            if loaded < req_ctx:
                self.load_local_model(req_ctx)

        user_message = self._build_user_message(text_tr, context, hint, prev_translation, rolling_summary)
        start_time = time.time()

        # Accurate token accounting + chat-template overhead + larger safety margin.
        CHAT_TEMPLATE_OVERHEAD = 35   # Gemma chat-template tokens (<bos>, <start_of_turn>, ...).
        safe_margin = 150              # 50 → 150 (safety margin increased).

        try:
            sys_tokens  = len(self.local_llm.tokenize(self.system_prompt.encode('utf-8')))
            user_tokens = len(self.local_llm.tokenize(user_message.encode('utf-8')))
            prompt_tokens = sys_tokens + user_tokens + CHAT_TEMPLATE_OVERHEAD
        except Exception:
            # Fallback: word-based estimate with a +50% safety coefficient.
            word_est = len((self.system_prompt + user_message).split())
            prompt_tokens = int(word_est * 1.5) + CHAT_TEMPLATE_OVERHEAD

        context_size = getattr(self, "loaded_n_ctx", req_ctx)
        # BUG FIX: do NOT clamp with max() — a negative remainder was previously
        # masked and bypassed the overflow guard. Compute the signed remainder
        # first to surface true overflows.
        raw_remaining = context_size - prompt_tokens - safe_margin

        # OVERFLOW: the prompt itself is over budget (rolling_summary +
        # prev_translation may have bloated). Rather than recursing into the
        # online path, shed the heavy context and retry once.
        if raw_remaining < 30 and (prev_translation or rolling_summary):
            log.warning(
                f"Offline prompt overflow ({prompt_tokens}/{context_size} tokens). "
                f"Dropping prev_translation + rolling_summary and retrying with bare prompt."
            )
            user_message = self._build_user_message(text_tr, context, hint, "", "")
            try:
                sys_tokens  = len(self.local_llm.tokenize(self.system_prompt.encode('utf-8')))
                user_tokens = len(self.local_llm.tokenize(user_message.encode('utf-8')))
                prompt_tokens = sys_tokens + user_tokens + CHAT_TEMPLATE_OVERHEAD
            except Exception:
                word_est = len((self.system_prompt + user_message).split())
                prompt_tokens = int(word_est * 1.5) + CHAT_TEMPLATE_OVERHEAD
            raw_remaining = context_size - prompt_tokens - safe_margin

        remaining_space = max(30, raw_remaining)

        # Adaptive lower bound on the desired output token count.
        input_words = len(text_tr.split())
        desired_tokens = min(
            int(input_words * 2.0),
            max(60, remaining_space)  # Never exceed remaining_space.
        )

        # Clamp the output budget against the safe remaining space; for very
        # short inputs (Whisper segments of 5-9 words) reserve at least 32
        # output tokens. Without this floor, dynamic_max_tokens < 20 used to
        # trigger a spurious "context full" path.
        dynamic_max_tokens = max(
            min(32, remaining_space),  # Lower bound.
            min(desired_tokens, remaining_space)
        )

        # Safety net: a true overflow case — the prompt already exceeds the context.
        if raw_remaining < 30:
            # TOKEN OVERFLOW ─────────────────────────────────────────────────────────────
            log.warning(
                f"Offline context overflow! prompt={prompt_tokens} tokens, "
                f"n_ctx={context_size}, remaining={remaining_space}. Cascading to online."
            )
            # INFINITE-LOOP GUARD: if the online cooldown is armed, translate_online
            # will bounce right back to translate_offline forever. Return a direct
            # bypass result instead.
            if time.time() < self._online_cooldown_until:
                log.error(
                    "Both offline overflow and online cooldown active — translation skipped."
                )
                return {
                    "translation": text_tr,  # Pass the raw text through so the dubbing pipeline does not break.
                    "latency_ms": 0,
                    "engine": "Bypass (overflow+cooldown)"
                }
            return self.translate_online(text_tr, context, hint, prev_translation, rolling_summary)

        try:
            response = self.local_llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=dynamic_max_tokens
            )
            translation = response["choices"][0]["message"]["content"].strip()
            latency = int((time.time() - start_time) * 1000)

            return {
                "translation": translation,
                "latency_ms": latency,
                "engine": "llama-cpp (local)"
            }
        except Exception:
            # GENERIC OFFLINE FAILURE ────────────────────────────────────────────────────
            # exc_info=True → full traceback is persisted to the log file. Silent
            # errors such as "Requested tokens exceed context window" are now
            # surfaced rather than swallowed.
            log.error(
                "Offline translation failed — full traceback below:",
                exc_info=True
            )
            return {
                "translation": "[TRANSLATION ERROR]",
                "latency_ms": 0,
                "engine": "Failed"
            }

    # ═══════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════

    def _build_user_message(self, text_tr: str, context: list = [], hint: str = "",
                            prev_translation: str = "", rolling_summary: str = "",
                            max_ctx_chars: int = 800) -> str:
        """
        Build the user message, appending context, prev_translation and rolling_summary
        when present. Long contributions are truncated to keep the prompt within budget.
        """
        msg = hint
        if rolling_summary:
            # The rolling summary is already condensed — 300 chars is plenty.
            trimmed_summary = rolling_summary[:300]
            msg += f"[DOCUMENT SUMMARY]\n{trimmed_summary}\n\n"
        if context:
            context_str = " ".join(context)[-max_ctx_chars:]  # Keep only the last N characters.
            msg += f"[PREVIOUS PARAGRAPHS]\n{context_str}\n\n"
        if prev_translation:
            # Trim the previous translation as well.
            trimmed_prev = prev_translation[-400:]
            msg += f"[PREVIOUS TRANSLATION]\n{trimmed_prev}\n\n"
        msg += f"Translate: {text_tr}"
        return msg

    def _tr_lower(self, text: str) -> str:
        """Safely lowercases Turkish characters before standard lowering."""
        return text.replace("İ", "i").replace("I", "ı").lower()

    def _strip_punct(self, text: str) -> str:
        """Removes punctuation for clean matching."""
        return text.translate(str.maketrans('', '', string.punctuation)).strip()

    def _check_cultural(self, text: str, src_lang: str, tgt_lang: str) -> tuple:
        fixed_text = self._tr_lower(text)
        clean_input = self._strip_punct(fixed_text)

        cmap = CULTURAL_CONCEPTS.get(src_lang)
        if not cmap:
            return None, "none"

        # Stage 1: exact match.
        for key, data in cmap.items():
            clean_key = self._strip_punct(self._tr_lower(key))
            if clean_input == clean_key:
                # Fast path: EN target + no persona ("default" or "none") → bypass the LLM (0 ms).
                if tgt_lang == "en" and self.persona in ("default", "none"):
                    return data["en_default"], "exact_fast"
                # Deep path: pass the intent as a hint to the LLM.
                return (key, data["intent"]), "exact_intent"

        # Stage 2: partial match — inject the intent as a hint.
        for key, data in cmap.items():
            clean_key = self._strip_punct(self._tr_lower(key))
            if clean_key in clean_input:
                return (key, data["intent"]), "partial"

        return None, "none"
