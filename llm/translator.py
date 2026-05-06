import os
import gc
import time
import string
import threading
import torch
from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types
from llama_cpp import Llama

# Çevresel değişkenleri yükle
load_dotenv()

CULTURAL_CONCEPTS = {
    "tr": {
        "hoş geldin":        {"intent": "welcoming someone who has just arrived",                                          "en_default": "Welcome."},
        "hoş bulduk":        {"intent": "responding warmly to being welcomed upon arrival",                                "en_default": "Glad to be here."},
        "görüşürüz":         {"intent": "a casual farewell indicating you will meet again soon",                          "en_default": "See you later."},
        "kendine iyi bak":   {"intent": "a warm farewell expressing care for someone's wellbeing",                        "en_default": "Take care."},
        "sağlıcakla kal":    {"intent": "a farewell wishing someone to remain healthy and well",                          "en_default": "Stay well."},
        "yolun açık olsun":  {"intent": "wishing someone safe and easy travels on their journey",                         "en_default": "Safe travels."},
        "allah'a emanet ol": {"intent": "a farewell entrusting someone to God's protection",                              "en_default": "May God protect you."},
        "hayırlı olsun":     {"intent": "congratulating someone or wishing them well on a new beginning or purchase",     "en_default": "Congratulations, best wishes."},
        "gözün aydın":       {"intent": "sharing in someone's joy over good news, expressing happiness for them",         "en_default": "I am so happy for your good news."},
        "ellerine sağlık":   {"intent": "complimenting someone on their work, craft, or food they prepared",              "en_default": "Well done, thank you."},
        "çok yaşa":          {"intent": "blessing someone after they sneeze, wishing them long life",                     "en_default": "Bless you."},
        "sen de gör":        {"intent": "responding to birthday wishes, hoping the same good things for the other person","en_default": "Thank you, same to you."},
        "iyi ki doğdun":     {"intent": "celebrating someone's birthday and expressing happiness they were born",         "en_default": "Happy birthday!"},
        "nice senelere":     {"intent": "wishing someone many more happy years, especially on birthdays or anniversaries","en_default": "Many happy returns."},
        "helal olsun":       {"intent": "expressing admiration and giving credit for someone's achievement or effort",    "en_default": "Well deserved, bravo."},
        "sıhhatler olsun":   {"intent": "wishing someone well after a haircut, shower, or personal grooming",            "en_default": "Enjoy your fresh look."},
        "geçmiş olsun":      {"intent": "wishing someone a speedy recovery from illness, hardship, or misfortune",       "en_default": "Get well soon."},
        "başınız sağ olsun": {"intent": "expressing deep condolences to someone who has lost a loved one",               "en_default": "I am so sorry for your loss."},
        "allah rahmet eylesin": {"intent": "praying for the soul of someone who has passed away",                        "en_default": "May they rest in peace."},
        "canın sağ olsun":   {"intent": "consoling someone over a loss, emphasizing their life and health matter most",  "en_default": "What matters is you are safe."},
        "üzme kendini":      {"intent": "encouraging someone not to blame themselves or feel bad",                        "en_default": "Don't be so hard on yourself."},
        "kısmet değilmiş":   {"intent": "accepting that something was not meant to be, expressing resignation with fate","en_default": "It wasn't meant to be."},
        "hayırlısı olsun":   {"intent": "hoping for the best outcome in an uncertain situation",                         "en_default": "Let's hope for the best."},
        "kolay gelsin":      {"intent": "wishing someone ease and success in their current work or task",                 "en_default": "Good luck with your work."},
        "afiyet olsun":      {"intent": "wishing someone to enjoy their meal or food",                                   "en_default": "Enjoy your meal."},
        "bereket versin":    {"intent": "wishing someone abundance, often said by a seller after receiving payment or as a blessing",  "en_default": "May it bring you abundance."},
        "ziyade olsun":      {"intent": "thanking a host after a meal, wishing them abundance",                          "en_default": "Thank you for the meal."},
        "iyi çalışmalar":    {"intent": "wishing someone a productive and pleasant work shift",                          "en_default": "Have a good shift."},
        "eyvallah":          {"intent": "expressing casual gratitude or acknowledgment",                                  "en_default": "Thanks, got it."},
        "estağfurullah":     {"intent": "humbly deflecting praise or thanks, meaning do not mention it",                 "en_default": "Not at all, don't mention it."},
        "aman diyeyim":      {"intent": "giving a friendly warning or caution to be careful",                            "en_default": "Watch out, be careful."},
        "hadi canım":        {"intent": "expressing disbelief or playful surprise",                                      "en_default": "No way, you are kidding."},
        "yok artık":         {"intent": "expressing shock or disbelief at something outrageous or unexpected",           "en_default": "Unbelievable."},
        "ne halt ettin sen": {"intent": "expressing strong disapproval or shock at something someone did wrong",         "en_default": "What have you done!"},
        "kurban olayım sana":{"intent": "expressing deep affection and devotion to someone",                             "en_default": "I would do anything for you."},
        "allah razı olsun":  {"intent": "expressing deep gratitude and blessing someone for their kindness",             "en_default": "May God bless you for this."},
        "allah korusun":     {"intent": "expressing hope that something bad will not happen, similar to God forbid",     "en_default": "God forbid."},
        "allah rahatlık versin": {"intent": "wishing someone a good night's sleep and peaceful rest",                   "en_default": "Good night, sleep well."},
        "elveda":            {"intent": "a formal and final farewell",                                                   "en_default": "Farewell."},
        "hoşça kal":         {"intent": "a warm goodbye",                                                                "en_default": "Goodbye."},
    }
}

# ── Persona Şablonları (Dinamik — {tgt_lang} ile hedef dile göre uyarlanır) ──
PERSONA_TEMPLATES = {
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

        ONLINE MOD — 3 Katmanlı Turbo Fallback Zinciri:
          Katman 1: Gemini API (Gemma 4 26B)     → Ana Çevirmen (Kalite Odaklı)
          Katman 2: Gemini API (Gemini 2.5 Flash)→ Hız Yedeği
          Katman 3: Groq (Llama 3.1 8B)          → Hız / Güvenlik Yedeği

        OFFLINE MOD — Sıfır Bağımlılık (Zero-Dependency):
          llama-cpp-python → ./models/gemma-4-q4.gguf
          Talep üzerine yüklenir (lazy load), VRAM israfı olmaz.
        """
        print("[SİSTEM] Translator v8 'Multi-State' Modülü Başlatılıyor...")
        
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

        # 3. ÜÇÜNCÜL MOTOR: GROQ (HIZ YEDEĞİ)
        self.groq_key = os.getenv("GROQ_API_KEY")
        if not self.groq_key:
            raise ValueError("GROQ_API_KEY eksik!")
        self.groq_client = Groq(api_key=self.groq_key.strip())
        self.groq_model = "llama-3.1-8b-instant"

        # ─── OFFLINE MOTOR (llama-cpp / Zero-Dependency) ───────
        # Model talep üzerine yüklenir — online modda VRAM boşa işgal etmez.

        self.local_model_path = "./models/gemma-4-q4.gguf"
        self.local_llm = None  # lazy load: load_local_model() ile yüklenir

        # Aktif persona ("none" | "official" | "streamer" | "casual" | "literary")
        self.persona = "none"

        # Varsayilan sistem promptu — translate() cagrisi oncesi direct offline cagrilari icin
        self.system_prompt = self._build_system_prompt()

    def set_persona(self, persona: str):
        """Aktif persona stilini ayarlar. Gecersiz deger verilirse 'none' kullanilir."""
        valid = {"none", "official", "streamer", "casual", "literary"}
        self.persona = persona if persona in valid else "none"

    def _build_system_prompt(self, src_lang="Turkish", tgt_lang="English") -> str:
        """Dinamik sistem promptu olusturur. Aktif persona varsa stil talimati eklenir."""
        base = (
            f"You are a lightning-fast translator. Translate the following {src_lang} text to {tgt_lang}. "
            f"Reply ONLY with the {tgt_lang} translation. Do not add quotes, explanations, or any other text. "
            f"CRITICAL: Never translate idioms, proverbs, or cultural expressions word-for-word. "
            f"Always find the natural, culturally equivalent expression a native {tgt_lang} speaker would actually say."
        )
        persona_template = PERSONA_TEMPLATES.get(self.persona, "")
        if persona_template:
            persona_instr = persona_template.format(tgt_lang=tgt_lang)
            return f"{base}\n{persona_instr}"
        return base


    # ═══════════════════════════════════════════════════════════
    # YEREL MODEL YONETIMI — Lazy Load / Unload
    # ═══════════════════════════════════════════════════════════

    def load_local_model(self):
        """Yerel GGUF modelini llama-cpp ile VRAM'e yükler.
        Zaten yüklüyse tekrar yüklemez (idempotent)."""
        if self.local_llm is not None:
            return

        print(f"[SISTEM] Yerel LLM yukleniyor: {self.local_model_path}")
        start = time.time()
        self.local_llm = Llama(
            model_path=self.local_model_path,
            n_gpu_layers=-1,   # Tum katmanlari GPU'ya yukle (-1 = tam GPU)
            n_ctx=512,
            verbose=False
        )
        elapsed = int((time.time() - start) * 1000)
        print(f"[SISTEM] Yerel LLM hazir ({elapsed}ms).")

    def unload_local_model(self):
        """Yerel modeli bellekten ve VRAM'den tamamen bosaltir.
        Online moda geciste cagrilir — VRAM catismasini onler."""
        if self.local_llm is None:
            return

        print("[SISTEM] Yerel LLM VRAM'den bosaltilyior...")
        del self.local_llm
        self.local_llm = None
        gc.collect()
        torch.cuda.empty_cache()
        print("[SISTEM] Yerel LLM VRAM'den bosaltildi.")

    # ═══════════════════════════════════════════════════════════
    # MOD YÖNETİMİ
    # ═══════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        """
        Çeviri modunu değiştirir.
        "online"  → Bulut API'leri (Gemini API [Gemma 4] → Gemini API [Flash] → Groq)
        "offline" → Yerel llama-cpp (./models/gemma-4-q4.gguf, lazy load)
        """
        if mode not in ("online", "offline"):
            raise ValueError(f"Geçersiz mod: {mode}. 'online' veya 'offline' olmalı.")
        
        old_mode = self.mode
        self.mode = mode
        print(f"[SİSTEM] Translator modu değişti: {old_mode} -> {mode}")

    # ═══════════════════════════════════════════════════════════
    # ANA ÇEVİRİ METODU (Yönlendirici)
    # ═══════════════════════════════════════════════════════════

    def translate(self, text_tr: str, context: list = [], src_lang="tr", tgt_lang="en", 
                  src_name="Turkish", tgt_name="English") -> dict:
        """
        Gelen metni hedef dile çevirir.
        Aktif moda göre online veya offline motora yönlendirir.
        
        Args:
            text_tr:   Çevrilecek metin
            context:   Zamir çevirisi için önceki cümleler (opsiyonel)
            src_lang:  Kaynak dil kodu (tr, en, ...)
            tgt_lang:  Hedef dil kodu
            src_name:  LLM promptu için kaynak dil adı
            tgt_name:  LLM promptu için hedef dil adı
        
        Returns:
            dict: {"translation": str, "latency_ms": int, "engine": str}
        """
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
            return self.translate_online(text_tr, context, hint)
        return self.translate_offline(text_tr, context, hint)

    # ═══════════════════════════════════════════════════════════
    # ONLINE ÇEVİRİ — 3 Katmanlı Turbo Fallback Zinciri
    # (Gemini API [Gemma 4] → Gemini API [Flash] → Groq)
    # ═══════════════════════════════════════════════════════════

    def _gemini_call(self, model_name: str, user_message: str, timeout: float = 8.0):
        """Gemini API cagrisini daemon thread ile calistirir.

        - generate_content (non-streaming) kullanir — model uyumlulugu garantili.
        - timeout saniye icinde cevap gelmezse TimeoutError firlatir.
        - Daemon thread: ana thread hic bloklanmaz.
        - Bos cevap gelirse ValueError firlatir → bir sonraki katmana duser.
        """
        result = [None]
        error = [None]

        def _call():
            try:
                resp = self.gemini_client.models.generate_content(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        temperature=0.2,
                        max_output_tokens=300
                    ),
                    contents=user_message
                )
                result[0] = resp.text or ""
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

    def translate_online(self, text_tr: str, context: list = [], hint: str = "") -> dict:
        """
        Bulut API'leri üzerinden çeviri yapar.
        3 katmanlı fallback: Gemini API (Gemma 4) başarısız → Gemini 2.5 Flash → Groq
        Hepsi başarısız olursa hata döner.
        """
        # Context varsa kullanıcı mesajını zenginleştir
        user_message = self._build_user_message(text_tr, context, hint)

        # --- KATMAN 1: GEMINI API (GEMMA 4) ---
        gemini_gemma_start = time.time()
        try:
            translation = self._gemini_call(self.gemma4_api_model, user_message)
            latency = int((time.time() - gemini_gemma_start) * 1000)
            return {
                "translation": translation.strip(),
                "latency_ms": latency,
                "engine": f"Gemini API ({self.gemma4_api_model})"
            }
        except TimeoutError:
            print(f"\n[UYARI] Gemini API (Gemma 4) Zaman Asimi (8s) -> Gemini 2.5 Flash'a Geciliyor...")
        except Exception as e:
            print(f"\n[UYARI] Gemini API (Gemma 4) Hatasi: {e} -> Gemini 2.5 Flash'a Geciliyor...")

        # --- KATMAN 2: GEMINI 2.5 FLASH ---
        gemini_flash_start = time.time()
        try:
            translation = self._gemini_call(self.gemini_fallback_model, user_message)
            latency = int((time.time() - gemini_flash_start) * 1000)
            return {
                "translation": translation.strip(),
                "latency_ms": latency,
                "engine": f"Gemini API ({self.gemini_fallback_model})"
            }
        except TimeoutError:
            print(f"\n[UYARI] Gemini 2.5 Flash Zaman Asimi (8s) -> Groq'a Geciliyor...")
        except Exception as e:
            print(f"\n[UYARI] Gemini 2.5 Flash Hatasi: {e} -> Groq'a Geciliyor...")

        # --- KATMAN 3: GROQ ---
        start_time = time.time()
        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=300
            )
            latency = int((time.time() - start_time) * 1000)
            return {
                "translation": response.choices[0].message.content.strip(),
                "latency_ms": latency,
                "engine": f"Groq ({self.groq_model})"
            }
        except Exception as e:
            print(f"\n[KRİTİK HATA] Tüm Online Çeviri Katmanları Çöktü: {e}")
            return {
                "translation": "[ÇEVİRİ HATASI]",
                "latency_ms": 0,
                "engine": "Failed"
            }

    # ═══════════════════════════════════════════════════════════
    # OFFLINE ÇEVİRİ — Yerel llama-cpp (Zero-Dependency)
    # ═══════════════════════════════════════════════════════════

    def translate_offline(self, text_tr: str, context: list = [], hint: str = "") -> dict:
        """
        Yerel GGUF modeli üzerinden çeviri yapar.
        İnternet gerektirmez. Model lazy load ile VRAM'e alınır.
        """
        # Güvenlik: model yüklü değilse yükle (doğrudan offline moda girilince)
        if self.local_llm is None:
            self.load_local_model()

        user_message = self._build_user_message(text_tr, context, hint)
        start_time = time.time()

        try:
            response = self.local_llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=150
            )
            translation = response["choices"][0]["message"]["content"].strip()
            latency = int((time.time() - start_time) * 1000)

            return {
                "translation": translation,
                "latency_ms": latency,
                "engine": "llama-cpp (local)"
            }
        except Exception as e:
            print(f"\n[KRİTİK HATA] Offline Çeviri Basarisiz: {e}")
            return {
                "translation": "[ÇEVİRİ HATASI]",
                "latency_ms": 0,
                "engine": "Failed"
            }

    # ═══════════════════════════════════════════════════════════
    # YARDIMCI METOTLAR
    # ═══════════════════════════════════════════════════════════

    def _build_user_message(self, text_tr: str, context: list = [], hint: str = "") -> str:
        """
        Context varsa zamir çevirisi için önceki cümleleri prompt'a ekler.
        Context yoksa sadece çevrilecek metni döner.
        
        Örnek:
            text_tr = "O çok yorgundu"
            context = ["Ahmet dün geldi."]
            → "Context: Ahmet dün geldi.\n\nTranslate: O çok yorgundu"
        """
        msg = hint
        if context:
            context_str = " ".join(context)
            msg += f"Context: {context_str}\n\n"
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
                # Fast-track: EN hedef + persona yok → LLM bypass (0ms)
                if tgt_lang == "en" and self.persona == "none":
                    return data["en_default"], "exact_fast"
                # Deep-track: intent hint ile LLM'e git
                return (key, data["intent"]), "exact_intent"

        # Stage 2: Partial Match — intent hint olarak enjekte et
        for key, data in cmap.items():
            clean_key = self._strip_punct(self._tr_lower(key))
            if clean_key in clean_input:
                return (key, data["intent"]), "partial"

        return None, "none"