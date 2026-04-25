import os
import time
import requests
import string
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from google import genai

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
        Çeviri katmanını başlatır. v7 Dual-State Mimarisi.

        ONLINE MOD — 3 Katmanlı Turbo Fallback Zinciri:
          Katman 1: Groq (Llama 3.1 8B)         → Hız Odaklı (< 500ms)
          Katman 2: OpenRouter (Gemma 4 26B)     → Model Sadakati (Fallback 1)
          Katman 3: Gemini 2.5 Flash             → Güvenlik Ağı (Fallback 2)

        OFFLINE MOD — Yerel Ollama:
          Gemma 4 E2B Q4_K_M → localhost:11434
        """
        print("[SİSTEM] Translator v7 'Dual-State' Modülü Başlatılıyor...")
        
        # Aktif mod: "online" (varsayılan) veya "offline"
        self.mode = "online"

        # ─── ONLINE MOTORLAR ───────────────────────────────────

        # 1. BİRİNCİL MOTOR: GROQ (HIZ ŞAMPİYONU)
        self.groq_key = os.getenv("GROQ_API_KEY")
        if not self.groq_key:
            raise ValueError("GROQ_API_KEY eksik!")
        self.groq_client = Groq(api_key=self.groq_key.strip())
        self.groq_model = "llama-3.1-8b-instant" 
        
        # 2. İKİNCİL MOTOR: OPENROUTER (GEMMA 4)
        self.or_key = os.getenv("OPENROUTER_API_KEY")
        if not self.or_key:
            raise ValueError("OPENROUTER_API_KEY eksik!")
        self.or_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.or_key.strip()
        )
        self.or_model = "google/gemma-4-26b-a4b-it:free"
        
        # 3. ÜÇÜNCÜL MOTOR: GEMINI 2.5 FLASH (GÜVENLİK YEDEĞİ)
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY eksik!")
        self.gemini_client = genai.Client(api_key=self.gemini_key.strip())
        self.gemini_model = "gemini-2.5-flash"

        # ─── OFFLINE MOTOR ─────────────────────────────────────

        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "gemma4-echo"  # Modelfile'dan oluşturulan isim

        # ─── ORTAK PROMPT ──────────────────────────────────────

        # Sistem Promptu: LLM'in gevezelik yapmasını engeller
        self.system_prompt = (
            "You are a lightning-fast translator. Translate the following Turkish text to English. "
            "Reply ONLY with the English translation. Do not add quotes, explanations, or any other text."
        )

        # Context Promptu: Zamir çevirisi için önceki cümleleri sağlar
        self.context_prompt = (
            "CONTEXT is provided ONLY for pronoun resolution. DO NOT translate the context. "
            "Output ONLY the English translation of the LAST Turkish sentence."
        )

    # ═══════════════════════════════════════════════════════════
    # MOD YÖNETİMİ
    # ═══════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        """
        Çeviri modunu değiştirir.
        "online"  → Bulut API'leri (Groq → OpenRouter → Gemini)
        "offline" → Yerel Ollama (Gemma 4 E2B Q4_K_M)
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
    # (Groq → OpenRouter → Gemini)
    # ═══════════════════════════════════════════════════════════

    def translate_online(self, text_tr: str, context: list = [], hint: str = "") -> dict:
        """
        Bulut API'leri üzerinden çeviri yapar.
        3 katmanlı fallback: Groq başarısız → OpenRouter → Gemini
        Hepsi başarısız olursa hata döner.
        """
        # Context varsa kullanıcı mesajını zenginleştir
        user_message = self._build_user_message(text_tr, context, hint)

        # --- KATMAN 1: GROQ ---
        start_time = time.time()
        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1, 
                max_tokens=150
            )
            latency = int((time.time() - start_time) * 1000)
            return {
                "translation": response.choices[0].message.content.strip(),
                "latency_ms": latency,
                "engine": f"Groq ({self.groq_model})"
            }
        except Exception as e:
            print(f"\n[UYARI] Groq Darboğazı: {e} -> OpenRouter'a Geçiliyor...")

        # --- KATMAN 2: OPENROUTER ---
        or_start = time.time()
        try:
            response = self.or_client.chat.completions.create(
                model=self.or_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
                max_tokens=150
            )
            latency = int((time.time() - or_start) * 1000)
            return {
                "translation": response.choices[0].message.content.strip(),
                "latency_ms": latency,
                "engine": f"OpenRouter ({self.or_model})"
            }
        except Exception as e:
            print(f"\n[UYARI] OpenRouter Hatası: {e} -> Gemini'ye Geçiliyor...")

        # --- KATMAN 3: GEMINI ---
        gemini_start = time.time()
        try:
            full_prompt = f"{self.system_prompt}\n\n"
            if context:
                full_prompt += f"{self.context_prompt}\n\n"
            full_prompt += f"Text to translate: '{text_tr}'"
            
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=full_prompt
            )
            latency = int((time.time() - gemini_start) * 1000)
            return {
                "translation": response.text.strip(),
                "latency_ms": latency,
                "engine": "Gemini 2.5 Flash"
            }
        except Exception as e:
            print(f"\n[KRİTİK HATA] Tüm Online Çeviri Katmanları Çöktü: {e}")
            return {
                "translation": "[ÇEVİRİ HATASI]",
                "latency_ms": 0,
                "engine": "Failed"
            }

    # ═══════════════════════════════════════════════════════════
    # OFFLINE ÇEVİRİ — Yerel Ollama (Gemma 4 E2B Q4_K_M)
    # ═══════════════════════════════════════════════════════════

    def translate_offline(self, text_tr: str, context: list = [], hint: str = "") -> dict:
        """
        Yerel Ollama üzerinden çeviri yapar.
        İnternet gerektirmez. GPU'da Gemma 4 E2B Q4_K_M çalışır.
        """
        user_message = self._build_user_message(text_tr, context, hint)
        
        start_time = time.time()
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": user_message,
                    "system": self.system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 150
                    }
                },
                timeout=15
            )
            response.raise_for_status()
            
            result = response.json()
            latency = int((time.time() - start_time) * 1000)
            
            return {
                "translation": result.get("response", "").strip(),
                "latency_ms": latency,
                "engine": f"Ollama ({self.ollama_model})"
            }
        except Exception as e:
            print(f"\n[KRİTİK HATA] Offline Çeviri Başarısız: {e}")
            return {
                "translation": "[ÇEVİRİ HATASI]",
                "latency_ms": 0,
                "engine": "Failed"
            }

    # ═══════════════════════════════════════════════════════════
    # OLLAMA VRAM YÖNETİMİ
    # ═══════════════════════════════════════════════════════════

    def release_ollama_vram(self):
        """
        Ollama'ya keep_alive=0 isteği göndererek Gemma'yı VRAM'den boşaltır.
        Online moda geçerken çağrılmalıdır.
        Rapordaki KRİTİK MİMARİ DÜZELTME: OLLAMA KEEP-ALIVE VRAM ÇATIŞMASI çözümü.
        """
        try:
            requests.post(
                self.ollama_url,
                json={"model": self.ollama_model, "keep_alive": 0},
                timeout=5
            )
            print("[SİSTEM] Ollama VRAM boşaltıldı (keep_alive=0)")
        except Exception as e:
            # Ollama çalışmıyorsa sorun değil — zaten VRAM'de model yok
            print(f"[BİLGİ] Ollama bağlantısı yok (beklenen davranış): {e}")

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