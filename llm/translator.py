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

CULTURAL_MAP = {
    "hoş geldin": "Welcome.",
    "hoş bulduk": "Glad to be here.",
    "görüşürüz": "See you later.",
    "kendine iyi bak": "Take care of yourself.",
    "sağlıcakla kal": "Stay well.",
    "yolun açık olsun": "Safe travels.",
    "allah'a emanet ol": "May God protect you.",
    "hayırlı olsun": "Congratulations, best wishes.",
    "gözün aydın": "I am so happy for your good news.",
    "ellerine sağlık": "Well done, thank you.",
    "çok yaşa": "Bless you.",
    "sen de gör": "Thank you, you too.",
    "iyi ki doğdun": "Happy birthday!",
    "nice senelere": "Many happy returns.",
    "helal olsun": "Bravo, well done.",
    "sıhhatler olsun": "Enjoy your haircut/shower.",
    "geçmiş olsun": "Get well soon.",
    "başınız sağ olsun": "I am sorry for your loss.",
    "allah rahmet eylesin": "May they rest in peace.",
    "canın sağ olsun": "Don't worry about it.",
    "üzme kendini": "Don't beat yourself up.",
    "kısmet değilmiş": "It wasn't meant to be.",
    "hayırlısı olsun": "Let's hope for the best.",
    "kolay gelsin": "May it be easy for you.",
    "afiyet olsun": "Enjoy your meal.",
    "bereket versin": "Thanks, may it bring abundance.",
    "ziyade olsun": "Thank you for the meal.",
    "iyi çalışmalar": "Have a good shift/work.",
    "eyvallah": "Thanks, alright.",
    "estağfurullah": "Not at all / Don't mention it.",
    "aman diyeyim": "Watch out / Be careful.",
    "hadi canım": "No way / You're kidding.",
    "yok artık": "Unbelievable.",
    "ne halt ettin sen": "What have you done!",
    "kurban olayım sana": "I would do anything for you.",
    "allah razı olsun": "May God bless you.",
    "allah korusun": "God forbid.",
    "allah rahatlık versin": "Rest in peace.",
    "elveda": "Farewell.",
    "hoşça kal": "Goodbye."
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

        # ─── ORTAK PROMPT ──────────────────────────────────────

        # Sistem Promptu: LLM'in gevezelik yapmasını engeller
        self.system_prompt = (
            "You are a lightning-fast translator. Translate the following Turkish text to English. "
            "Reply ONLY with the English translation. Do not add quotes, explanations, or any other text."
        )


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

    def translate(self, text_tr: str, context: list = []) -> dict:
        """
        Gelen Türkçe metni İngilizceye çevirir.
        Aktif moda göre online veya offline motora yönlendirir.
        
        Args:
            text_tr:  Çevrilecek Türkçe metin
            context:  Zamir çevirisi için önceki cümleler (opsiyonel)
        
        Returns:
            dict: {"translation": str, "latency_ms": int, "engine": str}
        """
        if not text_tr or len(text_tr.strip()) == 0:
            return {"translation": "", "latency_ms": 0, "engine": "None"}

        cultural_result, match_type = self._check_cultural(text_tr)

        if match_type == "exact":
            return {"translation": cultural_result, "latency_ms": 0, "engine": "CulturalMap"}

        hint = ""
        if match_type == "partial":
            tr_idiom, en_idiom = cultural_result
            hint = f"CRITICAL RULE: Translate '{tr_idiom}' as '{en_idiom}', translate the rest naturally.\n\n"

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

    def _check_cultural(self, text_tr: str) -> tuple[str | tuple[str, str] | None, str]:
        fixed_text = self._tr_lower(text_tr)
        clean_input = self._strip_punct(fixed_text)

        # Stage 1: Exact Match
        for key, value in CULTURAL_MAP.items():
            clean_key = self._strip_punct(self._tr_lower(key))
            if clean_input == clean_key:
                return value, "exact"

        # Stage 2: Partial Match
        for key, value in CULTURAL_MAP.items():
            clean_key = self._strip_punct(self._tr_lower(key))
            if clean_key in clean_input:
                return (key, value), "partial"

        return None, "none"