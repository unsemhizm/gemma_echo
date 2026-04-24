import os
import time
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from google import genai

# Çevresel değişkenleri yükle
load_dotenv()

class Translator:
    def __init__(self):
        """
        Çeviri katmanını başlatır. 3 Katmanlı Turbo Mimariyi kurar.
        Katman 1: Groq (Llama 3.1 8B) - Hız Odaklı (< 500ms)
        Katman 2: OpenRouter (Gemma 4 26B) - Model Sadakati (Fallback 1)
        Katman 3: Gemini 2.5 Flash - Güvenlik Ağı (Fallback 2)
        """
        print("[SİSTEM] Translator v7 'Turbo' Modülü Başlatılıyor...")
        
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

        # Sistem Promptu: LLM'in gevezelik yapmasını engeller
        self.system_prompt = (
            "You are a lightning-fast translator. Translate the following Turkish text to English. "
            "Reply ONLY with the English translation. Do not add quotes, explanations, or any other text."
        )

    def translate(self, text: str) -> dict:
        """
        Gelen metni İngilizceye çevirir. 3 katmanlı fallback stratejisini izler.
        """
        if not text or len(text.strip()) == 0:
            return {"translation": "", "latency_ms": 0, "engine": "None"}

        # --- KATMAN 1: GROQ ---
        start_time = time.time()
        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
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
                    {"role": "user", "content": text}
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
            full_prompt = f"{self.system_prompt}\n\nText to translate: '{text}'"
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
            print(f"\n[KRİTİK HATA] Tüm Çeviri Katmanları Çöktü: {e}")
            return {
                "translation": "[ÇEVİRİ HATASI]",
                "latency_ms": 0,
                "engine": "Failed"
            }