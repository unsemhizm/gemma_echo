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

# Çevresel değişkenleri yükle
load_dotenv()

log = get_logger(__name__)


def _load_cultural_concepts() -> dict:
    """Kültürel kavramları data/cultural_concepts.json dosyasından yükler."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "cultural_concepts.json")
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


CULTURAL_CONCEPTS = _load_cultural_concepts()

# ── Persona Şablonları (Dinamik — {tgt_lang} ile hedef dile göre uyarlanır) ──
#
# "default" → Hiçbir persona talimatı eklenmez; saf, tarafsız çeviri yapılır.
#              Kullanıcı bir persona seçmek zorunda değildir.
# "none"    → "default" ile özdeş; geriye dönük uyumluluk için korunmuştur.
PERSONA_TEMPLATES = {
    # ── Tarafsız (varsayılan) ─────────────────────────────────────────────────
    "default": "",   # Prompt'a ek talimat eklenmez — saf çeviri motoru
    "none":    "",   # Geriye dönük uyumluluk takma adı

    # ── Stil personaları ─────────────────────────────────────────────────────
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
        Çeviri katmanını başlatır. v8 Multi-State Mimarisi.

        ONLINE MOD — 2 Katmanlı Fallback Zinciri:
          Katman 1: Gemini API (Gemma 4 26B)     → Ana Çevirmen (Kalite Odaklı)
          Katman 2: Gemini API (Gemini 2.5 Flash)→ Hız Yedeği

        OFFLINE MOD — Sıfır Bağımlılık (Zero-Dependency):
          llama-cpp-python → ./models/gemma-4-q4.gguf
          Talep üzerine yüklenir (lazy load), VRAM israfı olmaz.
        """
        log.info("Translator v8 'Multi-State' Modülü başlatılıyor...")

        # Aktif mod: "online" (varsayılan) veya "offline"
        self.mode = "online"

        # ─── ONLINE MOTORLAR ───────────────────────────────────

        # 1. BİRİNCİL MOTOR: GEMINI API (GEMMA 4 26B)
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY eksik!")
        self.gemini_client = genai.Client(api_key=self.gemini_key.strip())
        self.gemma4_api_model = "gemma-4-26b-a4b-it"

        # 2. İKİNCİL MOTOR: GEMINI API (GEMINI 2.5 FLASH)
        self.gemini_fallback_model = "gemini-2.5-flash"

        # ─── OFFLINE MOTOR (llama-cpp / Zero-Dependency) ───────
        # Model talep üzerine yüklenir — online modda VRAM boşa işgal etmez.

        self.local_model_path = "./models/gemma-4-q4.gguf"
        self.local_llm = None  # lazy load: load_local_model() ile yüklenir
        self._llm_vram_failed = False
        self.vram_issue_callback = None  # GUI uyarısı: () -> None

        # ── Online kota cooldown (429 RESOURCE_EXHAUSTED) ─────────────────
        # Free-tier kotası bittiğinde her chunk için 20-40s timeout beklemek
        # yerine, online katmanı geçici olarak devre dışı bırak. Doküman çevirisinde
        # kritik: chunk başına 30s ziyan etmek yerine direkt local'e in.
        self._online_cooldown_until = 0.0   # epoch sn; time.time() < bu ise online skip
        self._consecutive_quota_errors = 0   # ardışık 429 sayısı (cooldown'u büyütür)

        
        # Eşzamanlı istekleri sıraya almak ve paylaşılan state'i (system_prompt, quota vb.) korumak için kilit
        self._translate_lock = threading.Lock()

        # Aktif persona — varsayılan "default" (hiçbir stil talimatı eklenmez).
        # Kullanıcı isteğe bağlı olarak bir persona seçebilir.
        # Geçerli değerler: "default" | "none" | "official" | "streamer" | "casual" | "literary"
        self.persona = "default"

        # Varsayilan sistem promptu — translate() cagrisi oncesi direct offline cagrilari icin
        self.system_prompt = self._build_system_prompt()

    def set_persona(self, persona: str):
        """Aktif persona stilini ayarlar.

        Geçerli değerler:
          "default"  — Hiçbir persona talimatı yok; saf çeviri (varsayılan).
          "none"     — "default" ile özdeş, geriye dönük uyumluluk.
          "official" — Diplomat / akademik resmi üslup.
          "streamer" — Yayıncı / internet jargonu.
          "casual"   — Samimi / günlük konuşma.
          "literary" — Edebi / şiirsel çeviri.

        Geçersiz bir değer verilirse 'default' kullanılır.
        """
        valid = {"default", "none", "official", "streamer", "casual", "literary"}
        self.persona = persona if persona in valid else "default"

    def _build_system_prompt(self, src_lang="Turkish", tgt_lang="English") -> str:
        """Dinamik sistem promptu oluşturur.

        Persona "default" veya "none" ise (ya da bilinmiyorsa) sadece temel
        çeviri talimatı döner — prompt'a herhangi bir stil eki yapılmaz.
        Bir stil personası seçilmişse ilgili talimat base'e eklenir.
        """
        base = (
            f"You are a lightning-fast translator. Translate the following {src_lang} text to {tgt_lang}. "
            f"Reply ONLY with the {tgt_lang} translation. Do not add quotes, explanations, or any other text. "
            f"CRITICAL: Never translate idioms, proverbs, or cultural expressions word-for-word. "
            f"Always find the natural, culturally equivalent expression a native {tgt_lang} speaker would actually say."
        )
        persona_template = PERSONA_TEMPLATES.get(self.persona, "")
        if persona_template:  # "default" ve "none" boş string → bu dal çalışmaz
            persona_instr = persona_template.format(tgt_lang=tgt_lang)
            return f"{base}\n{persona_instr}"
        return base


    # ═══════════════════════════════════════════════════════════
    # YEREL MODEL YONETIMI — Lazy Load / Unload
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
        """Yerel GGUF modelini llama-cpp ile VRAM'e yükler.
        Zaten talep edilen bağlam boyutunda yüklüyse tekrar yüklemez (idempotent).
        Eğer farklı bir boyutta yüklüyse, önce eski modeli indirip yenisini yükler.
        Dönüş: başarı True; VRAM / OOM durumunda False."""
        if self.local_llm is not None:
            same_ctx  = getattr(self, "loaded_n_ctx", 512) == context_size
            if same_ctx:
                return True   # zaten doğru konfigürasyonda yüklü — hiçbir şey yapma
            log.info(f"Model yeniden yükleniyor (n_ctx {getattr(self, 'loaded_n_ctx', 512)} → {context_size}).")
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
                f"Yerel LLM VRAM ön kontrolü başarısız "
                f"(boş: {free} B, eşik: {MIN_FREE_BYTES_LOCAL_LLM} B)"
            )
            self._llm_vram_failed = True
            cleanup_cuda_memory()
            self._invoke_vram_callback()
            return False

        log.info(f"Yerel LLM yükleniyor (n_ctx={context_size}): {self.local_model_path}")
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
                # │ VRAM TAŞMAŞI ─────────────────────────────────────────────────────────────
                # exc_info=True ile tam stack trace log dosyasına yazılır.
                # Bu olmadan CUDA OOM ne zaman, hangi satırda olduğu bilinmez.
                log.error(
                    "Yerel LLM CUDA OOM — VRAM yetersiz, model yüklenemedi. "
                    "Sistem online fallback'e geçiyor.",
                    exc_info=True
                )
                self._llm_vram_failed = True
                self._invoke_vram_callback()
                return False
            raise

        self._llm_vram_failed = False
        elapsed = int((time.time() - start) * 1000)
        log.info(f"Yerel LLM hazır ({elapsed}ms).")
        return True

    def unload_local_model(self):
        """Yerel modeli bellekten ve VRAM'den tamamen bosaltir.
        Online moda geciste cagrilir — VRAM catismasini onler."""
        if self.local_llm is None:
            self._llm_vram_failed = False
            return

        log.info("Yerel LLM VRAM'den boşaltılıyor...")
        del self.local_llm
        self.local_llm = None
        self._llm_vram_failed = False
        if hasattr(self, "loaded_n_ctx"):
            del self.loaded_n_ctx
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Yerel LLM VRAM'den boşaltıldı.")

    # ═══════════════════════════════════════════════════════════
    # MOD YÖNETİMİ
    # ═══════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        """
        Çeviri modunu değiştirir.
        "online"  → Bulut API'leri (Gemini API [Gemma 4] → Gemini API [Flash])
        "offline" → Yerel llama-cpp (./models/gemma-4-q4.gguf, lazy load)
        """
        if mode not in ("online", "offline"):
            raise ValueError(f"Geçersiz mod: {mode}. 'online' veya 'offline' olmalı.")

        old_mode = self.mode
        self.mode = mode
        if mode == "offline":
            self._llm_vram_failed = False
        log.info(f"Translator modu değişti: {old_mode} -> {mode}")


    # ═══════════════════════════════════════════════════════════
    # ANA ÇEVİRİ METODU (Yönlendirici)
    # ═══════════════════════════════════════════════════════════

    def translate(self, text_tr: str, context: list = None, src_lang="tr", tgt_lang="en",
                  src_name="Turkish", tgt_name="English", prev_translation: str = "",
                  rolling_summary: str = "") -> dict:
        """
        Gelen metni hedef dile çevirir.
        Aktif moda göre online veya offline motora yönlendirir.
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
    
            # Sistem promptunu guncelle
            self.system_prompt = self._build_system_prompt(src_name, tgt_name)
    
            if self.mode == "online":
                return self.translate_online(text_tr, context, hint, prev_translation, rolling_summary)
            return self.translate_offline(text_tr, context, hint, prev_translation, rolling_summary)

    # ═══════════════════════════════════════════════════════════
    # YÜRÜYEN ÖZET OLUŞTURMA (Aşama 2)
    # ═══════════════════════════════════════════════════════════

    def generate_summary(self, text: str, current_summary: str = "") -> str:
        """Gelen yeni çevrilen parçayı ve mevcut özeti kullanarak yürüyen döküman özetini günceller."""
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
                # Yarışma gereği her zaman 1. öncelik Gemma 4 modelimizdir!
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
                log.warning(f"Gemma 4 Özetleme Hatası, Gemini Flash Fallback devreye giriyor.", exc_info=True)
                try:
                    # Fallback olarak hızlı/ekonomik yedek motoru kullanıyoruz
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
            # generate_summary için sabit 512 yeterli (prompt + özet < 400 token)
            if self.local_llm is None:
                if not self.load_local_model(512):
                    return current_summary
            try:
                # DÜZELTME: Offline dalında token hesabı ekle
                CHAT_TEMPLATE_OVERHEAD = 35
                safe_margin = 100
                try:
                    p_tokens = len(self.local_llm.tokenize(prompt.encode('utf-8'))) + CHAT_TEMPLATE_OVERHEAD
                    ctx = getattr(self, "loaded_n_ctx", 512)
                    safe_max = max(30, ctx - p_tokens - safe_margin)
                    actual_max = min(100, safe_max)
                except Exception:
                    actual_max = 80  # konservatif fallback

                if actual_max < 20:
                    log.warning("generate_summary: prompt context'i dolduruyor, özet atlanıyor.")
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
    # ONLINE ÇEVİRİ — 3 Katmanlı Turbo Fallback Zinciri
    # (Gemini API [Gemma 4] → Gemini API [Flash] )
    # ═══════════════════════════════════════════════════════════

    def _maybe_trigger_quota_cooldown(self, exc: Exception, model_name: str):
        """429 RESOURCE_EXHAUSTED algılandığında online katmanı geçici devre dışı bırakır.

        Detection: exception string'inde "429" veya "RESOURCE_EXHAUSTED" arar.
        """
        msg = str(exc)
        if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
            return  # quota dışı bir hata — _trigger_online_failure_cooldown 2-katman seviyesinde sayar
        self._trigger_online_failure_cooldown(model_name, reason="429")

    def _trigger_online_failure_cooldown(self, model_name: str, reason: str = "failure"):
        """Online katmanın 'tamamen başarısız' olduğu durumlarda cooldown başlatır.

        Tetikleyiciler:
          - 429 RESOURCE_EXHAUSTED (kota)
          - Hem Gemma-4 hem Flash zaman aşımı (her chunk için 240s+ ziyan)
          - Aynı anda her iki katmanda başarısız olma

        Cooldown formülü:
          - İlk başarısızlık: 180s (3dk)
          - Ardışık her başarısızlık: 2x (3dk → 6dk → 12dk → ... cap 60dk)
          - Free-tier 20/dk limiti bittiğinde tüm akış instant offline'a iner;
            kullanıcı her chunk için 2dk timeout beklemez.
        """
        self._consecutive_quota_errors += 1
        cooldown_sec = min(3600, 180 * (2 ** (self._consecutive_quota_errors - 1)))
        self._online_cooldown_until = time.time() + cooldown_sec
        log.warning(
            f"⚠ Online cooldown ({reason}, {model_name}): "
            f"{cooldown_sec}s ({cooldown_sec // 60}dk) süreyle bulut katmanları skip. "
            f"(ardışık başarısızlık: {self._consecutive_quota_errors})"
        )

    def _gemini_call(self, model_name: str, user_message: str, timeout: float = 20.0, max_output_tokens: int = 300):
        """Gemini API cagrisini daemon thread ile calistirir.

        - generate_content (non-streaming) kullanir — model uyumlulugu garantili.
        - timeout saniye icinde cevap gelmezse TimeoutError firlatir.
        - Daemon thread: ana thread hic bloklanmaz.
        - Bos cevap gelirse ValueError firlatir → bir sonraki katmana duser.
        """
        result = [None]
        error = [None]

        # Gemini 2.5 ailesi "thinking" model — output budget'i internal reasoning'e
        # harcayip ceviriye cok az token birakir. Ceviri gorevinde thinking gereksiz.
        cfg_kwargs = dict(
            system_instruction=self.system_prompt,
            temperature=0.2,
            max_output_tokens=max_output_tokens,
        )
        if "2.5" in model_name:
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass  # SDK eski surumde ThinkingConfig yoksa sessizce gec

        def _is_transient_5xx(exc) -> bool:
            """500/503 gibi gecici sunucu hatalari icin hizli retry yapilir.
            429 (kota), 4xx (kotu istek) retry edilmez — onlar kalici."""
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
                    # 5xx -> 1 hizli retry (Google API flaky oluyor; fallback'e zipladiginda
                    # baglam ve max_tokens reset oluyor, kalitesizlige sebep). Loglarda
                    # 500 INTERNAL onlarca kez goruldu, retry ile cogu cozulur.
                    if _is_transient_5xx(e1):
                        log.info(f"[GEMINI] {model_name} 5xx -> 0.8s sonra 1 retry...")
                        time.sleep(0.8)
                        resp = _do_request()
                    else:
                        raise
                result[0] = resp.text or ""
                # ── TANI: kesme sebebini ortaya koy ──
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
            raise TimeoutError(f"Gemini ({model_name}) {timeout}s icinde cevap vermedi.")
        if error[0] is not None:
            raise error[0]
        if not result[0]:
            raise ValueError(f"Gemini ({model_name}) bos cevap dondu.")
        return result[0]

    def translate_online(self, text_tr: str, context: list = [], hint: str = "",
                         prev_translation: str = "", rolling_summary: str = "") -> dict:
        """
        Bulut API'leri üzerinden çeviri yapar.
        2 katmanlı fallback: Gemini API (Gemma 4) başarısız → Gemini 2.5 Flash
        İkisi de başarısız olursa hata döner.

        Kota cooldown: Free-tier 429 alındığında bir süre direkt offline'a düşülür
        (her chunk için boşa timeout beklemek yerine).
        """
        # Context varsa kullanıcı mesajını zenginleştir
        user_message = self._build_user_message(text_tr, context, hint, prev_translation, rolling_summary)

        # ── KOTA COOLDOWN GUARD ──────────────────────────────────────────
        # Aktifse online katmanları atla, direkt offline'a in.
        now = time.time()
        if now < self._online_cooldown_until:
            remaining = int(self._online_cooldown_until - now)
            log.info(
                f"Online kota cooldown aktif ({remaining}s kaldı) — "
                f"bulut atlanıyor, direkt offline'a iniliyor."
            )
            try:
                return self.translate_offline(text_tr, context, hint, prev_translation, rolling_summary)
            except Exception:
                log.critical("Offline da başarısız (cooldown sırasında).", exc_info=True)
                return {"translation": "[ÇEVİRİ HATASI]", "latency_ms": 0, "engine": "Failed"}

        # ── DİNAMİK TİMEOUT HESABI ───────────────────────────────────────
        # Canlı ses (telsiz) için hızlı fallback (20s) istiyoruz.
        # Kitap çevirisinde (çok uzun metin veya geçmiş bağlam) API'ye 60s zaman tanıyoruz.
        input_words = len(text_tr.split())
        is_heavy_workload = bool(prev_translation or rolling_summary or input_words > 50)
        dynamic_timeout = 60.0 if is_heavy_workload else 20.0

        # Gemma 4 birincil motor — kota askida iken bile 90-150s sonunda cevap dondurebiliyor.
        # Cok agresif timeout (60s) onu pratik olarak pasiflestiriyor. 2x pay birakiyoruz.
        # Flash ise hizli fallback olarak normal sureyle kullanilir.
        gemma4_timeout = dynamic_timeout * 2  # 40s (live) / 120s (heavy)
        flash_timeout  = dynamic_timeout       # 20s (live) / 60s (heavy)

        # Girdi kelime sayisina gore dinamik cikti token limiti (kesme onleme).
        # TR→EN cevirisinde cikti ~%20-30 sisiyor; ayrica Gemma 4 bazen reasoning
        # token'i harciyor — bunlar da output butcesinden dusuluyor. Bu yuzden:
        #   - Floor 600 (eskiden 300; loglarda hep 300 hit ediyordu, MAX_TOKENS)
        #   - Carpan 4 (eskiden 3; reasoning + dil sismesi icin emniyet payi)
        # Canli: 10 kelime → 600, Dublaj/Medya: 25 kelime → 600, 100 kelime → 600,
        # 200 kelime → 800, 1024 kelime → 4096 cap.
        dynamic_max_tokens = min(4096, max(600, input_words * 4))

        # Bu cagrida hangi katmanlarin basarisiz oldugunu takip et — IKISI DE
        # basarisizsa cooldown tetikle (timeout fark etmeksizin), boylece sonraki
        # chunk'lar bulut katmanlarini direkt skip etsin.
        gemma4_failed_reason = None
        flash_failed_reason = None

        # --- KATMAN 1: GEMINI API (GEMMA 4) ---
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
            log.warning(f"Gemini API ({self.gemma4_api_model}) Zaman Aşımı ({gemma4_timeout}s) -> Gemini 2.5 Flash'a geçiliyor...")
            gemma4_failed_reason = "timeout"
        except Exception as e:
            self._maybe_trigger_quota_cooldown(e, self.gemma4_api_model)
            log.warning(
                f"Gemini API ({self.gemma4_api_model}) hatası -> Gemini 2.5 Flash'a geçiliyor.",
                exc_info=True
            )
            gemma4_failed_reason = "error"

        # --- KATMAN 2: GEMINI 2.5 FLASH ---
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
            log.warning(f"Gemini 2.5 Flash Zaman Aşımı ({flash_timeout}s). Yerel modele düşülüyor...")
            flash_failed_reason = "timeout"
        except Exception as e:
            self._maybe_trigger_quota_cooldown(e, self.gemini_fallback_model)
            log.warning("Gemini 2.5 Flash hatası. Yerel modele düşülüyor.", exc_info=True)
            flash_failed_reason = "error"

        # IKI KATMAN DA BASARISIZ — cooldown tetikle (timeout veya error fark etmez).
        # _maybe_trigger_quota_cooldown sadece 429'da counter artirir; timeout durumunda
        # buradan agresif cooldown baslamali ki sonraki chunk 240s daha ziyan etmesin.
        if gemma4_failed_reason and flash_failed_reason:
            # Quota cooldown zaten artmissa tekrar artirma (cift sayim olmasin)
            if self._consecutive_quota_errors == 0 or (time.time() >= self._online_cooldown_until):
                self._trigger_online_failure_cooldown(
                    "both_layers",
                    reason=f"gemma4_{gemma4_failed_reason}+flash_{flash_failed_reason}"
                )

        # --- KATMAN 3: YEREL OFFLINE MODEL FALLBACK (Kesintisiz Hizmet) ---
        log.warning("Tüm Bulut API'leri başarısız! Yerel Gemma Modeli (GGUF) devreye sokuluyor...")
        try:
            return self.translate_offline(text_tr, context, hint, prev_translation, rolling_summary)
        except Exception:
            log.critical(
                "Yerel çevrimdışı model de başarısız oldu — tüm katmanlar çöktü!",
                exc_info=True
            )
            return {
                "translation": "[ÇEVİRİ HATASI]",
                "latency_ms": 0,
                "engine": "Failed"
            }

    # ═══════════════════════════════════════════════════════════
    # OFFLINE ÇEVİRİ — Yerel llama-cpp (Zero-Dependency)
    # ═══════════════════════════════════════════════════════════

    def translate_offline(self, text_tr: str, context: list = [], hint: str = "",
                          prev_translation: str = "", rolling_summary: str = "") -> dict:
        """
        Yerel GGUF modeli üzerinden çeviri yapar.
        İnternet gerektirmez. Model lazy load ile VRAM'e alınır.
        """
        # Bağlam parametresi belirleme:
        # - Standart canlı çeviri: 512 (hızlı, minimum bellek)
        # - Doküman çevirisi (rolling_summary/prev_translation var): 8192
        #   Gemma 4 zaten 131k destekliyor; 800-kelimelik chunk + system_prompt + glossary
        #   2048'i kolay aşıyor (gerçek ölçüm: ~2700 token).
        req_ctx = 8192 if (prev_translation or rolling_summary) else 512

        if self.local_llm is None:
            if self._llm_vram_failed:
                return self.translate_online(text_tr, context, hint, prev_translation, rolling_summary)
            if not self.load_local_model(req_ctx):
                return self.translate_online(text_tr, context, hint, prev_translation, rolling_summary)
        else:
            # BUG FIX: Modeli sadece BUYUT, asla kucultme.
            # Aksi halde dubber 8192 ile yukledikten sonra ilk segmentte
            # (prev_translation="") req_ctx=512 hesaplaniyor ve model 512'ye
            # dusuruluyordu -> bu da kisa Whisper segmentlerinde "context doldu"
            # warning'i tetikleyip translate_online <-> translate_offline
            # sonsuz dongusunu acmis oluyordu.
            loaded = getattr(self, "loaded_n_ctx", 512)
            if loaded < req_ctx:
                self.load_local_model(req_ctx)

        user_message = self._build_user_message(text_tr, context, hint, prev_translation, rolling_summary)
        start_time = time.time()

        # DÜZELTME: Doğru token hesabı + chat template overhead + artırılmış güvenlik marjı
        CHAT_TEMPLATE_OVERHEAD = 35   # Gemma chat template tokenleri (<bos>, <start_of_turn>, vb.)
        safe_margin = 150              # 50 → 150 (güvenlik payı artırıldı)

        try:
            sys_tokens  = len(self.local_llm.tokenize(self.system_prompt.encode('utf-8')))
            user_tokens = len(self.local_llm.tokenize(user_message.encode('utf-8')))
            prompt_tokens = sys_tokens + user_tokens + CHAT_TEMPLATE_OVERHEAD
        except Exception:
            # Fallback: kelime-bazlı tahmin, ekstra %50 güvenlik katsayısı
            word_est = len((self.system_prompt + user_message).split())
            prompt_tokens = int(word_est * 1.5) + CHAT_TEMPLATE_OVERHEAD

        context_size = getattr(self, "loaded_n_ctx", req_ctx)
        # BUG FIX: max() ile clamp etme — negatif kalan boşluk guard'ı atlatıyordu.
        # Önce gerçek (signed) boşluğu hesapla, overflow'u yakala.
        raw_remaining = context_size - prompt_tokens - safe_margin

        # OVERFLOW: prompt zaten context'i aşıyor (rolling_summary + prev_translation şişmiş olabilir).
        # Online'a recursion yapmak yerine ağır bağlamı düşürüp tekrar dene.
        if raw_remaining < 30 and (prev_translation or rolling_summary):
            log.warning(
                f"Offline prompt overflow ({prompt_tokens}/{context_size} token). "
                f"prev_translation + rolling_summary düşürülüyor, sade çeviri deneniyor."
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

        # Kelime sayısına göre istenen token miktarı — adaptif alt sınır
        input_words = len(text_tr.split())
        desired_tokens = min(
            int(input_words * 2.0),
            max(60, remaining_space)  # asla remaining_space'i aşamaz
        )

        # Cikisi kalan guvenli bosluga kelepceliyoruz; cok kisa input'larda
        # (Whisper segmentleri 5-9 kelime) en az 32 token cikti payi birak.
        # Aksi halde dynamic_max_tokens < 20 sahte "context doldu" tetikliyordu.
        dynamic_max_tokens = max(
            min(32, remaining_space),  # alt sinir
            min(desired_tokens, remaining_space)
        )

        # Guvenlik net'i: GERCEK overflow durumu — prompt zaten context'i asti.
        if raw_remaining < 30:
            # TOKEN TASMASI ─────────────────────────────────────────────────────────────
            log.warning(
                f"Offline context overflow! prompt={prompt_tokens} token, "
                f"n_ctx={context_size}, kalan={remaining_space}. Online'a dusuluyor."
            )
            # SONSUZ DONGU GUARD: online cooldown'da ise translate_online
            # tekrar translate_offline'a dusecek -> sonsuz dongu. Direkt hata don.
            if time.time() < self._online_cooldown_until:
                log.error(
                    "Hem offline overflow hem online cooldown — ceviri atlandi."
                )
                return {
                    "translation": text_tr,  # ham metni gec, dublaj akisini kirma
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
            # GENEL OFFLINE HATASI ───────────────────────────────────────────────────────
            # exc_info=True → tam traceback log dosyasına gider.
            # "Requested tokens exceed context window" gibi sessiz hatalar
            # artık izlenebilir ve kaybolmaz.
            log.error(
                "Offline çeviri başarısız — tam hata izİ aşağıda:",
                exc_info=True
            )
            return {
                "translation": "[ÇEVİRİ HATASI]",
                "latency_ms": 0,
                "engine": "Failed"
            }

    # ═══════════════════════════════════════════════════════════
    # YARDIMCI METOTLAR
    # ═══════════════════════════════════════════════════════════

    def _build_user_message(self, text_tr: str, context: list = [], hint: str = "",
                            prev_translation: str = "", rolling_summary: str = "",
                            max_ctx_chars: int = 800) -> str:
        """
        Context, prev_translation ve rolling_summary varsa prompt'a ekler.
        Uzun içerikler için kırpma uygular.
        """
        msg = hint
        if rolling_summary:
            # Rolling summary'yi kırp — zaten özet, 300 char yeterli
            trimmed_summary = rolling_summary[:300]
            msg += f"[DOCUMENT SUMMARY]\n{trimmed_summary}\n\n"
        if context:
            context_str = " ".join(context)[-max_ctx_chars:]  # son N char
            msg += f"[PREVIOUS PARAGRAPHS]\n{context_str}\n\n"
        if prev_translation:
            # Önceki çeviriyi de kırp
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

        # Stage 1: Exact Match
        for key, data in cmap.items():
            clean_key = self._strip_punct(self._tr_lower(key))
            if clean_input == clean_key:
                # Fast-track: EN hedef + persona yok ("default" veya "none") → LLM bypass (0ms)
                if tgt_lang == "en" and self.persona in ("default", "none"):
                    return data["en_default"], "exact_fast"
                # Deep-track: intent hint ile LLM'e git
                return (key, data["intent"]), "exact_intent"

        # Stage 2: Partial Match — intent hint olarak enjekte et
        for key, data in cmap.items():
            clean_key = self._strip_punct(self._tr_lower(key))
            if clean_key in clean_input:
                return (key, data["intent"]), "partial"

        return None, "none"
