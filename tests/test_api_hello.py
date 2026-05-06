import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .env dosyasındaki anahtarları sisteme yükle
load_dotenv()

def test_apis():
    print("[TEST] 2 Katmanlı Gemini Mimari Kontrol Ediliyor...\n")
    print("="*50)

    # ---------------------------------------------------------
    # 1. BİRİNCİL MOTOR (ÇEVİRİDE 1. KATMAN): GEMINI API (GEMMA 4)
    # ---------------------------------------------------------
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("[HATA] GEMINI_API_KEY .env dosyasında bulunamadı!")
    else:
        try:
            print("[BEKLEYİN] 1. KATMAN: Gemini API (Gemma 4 26B) test ediliyor...")
            client = genai.Client(api_key=gemini_key.strip())
            start_time = time.time()
            
            response = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                config=types.GenerateContentConfig(
                    system_instruction="You are a lightning-fast translator.",
                    temperature=0.2,
                    max_output_tokens=150
                ),
                contents="Say 'Hello' and nothing else."
            )
            
            latency = (time.time() - start_time) * 1000
            print(f"[BAŞARILI] GEMINI (GEMMA 4)! Cevap: '{response.text.strip()}'")
            print(f"[SÜRE] Gemini (Gemma 4) Gecikmesi: {latency:.2f} ms")
        except Exception as e:
            print(f"[HATA] GEMINI (GEMMA 4) BAĞLANTI HATASI: {e}")
            
    print("-" * 50)

    # ---------------------------------------------------------
    # 2. İKİNCİL MOTOR (HIZ YEDEĞİ): GEMINI 2.5 FLASH
    # ---------------------------------------------------------
    if not gemini_key:
        print("[HATA] GEMINI_API_KEY .env dosyasında bulunamadı!")
    else:
        try:
            print("[BEKLEYİN] 2. KATMAN: Gemini 2.5 Flash test ediliyor...")
            client = genai.Client(api_key=gemini_key.strip()) 
            start_time = time.time()
            
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents="Say 'Hello' and nothing else."
            )
            
            latency = (time.time() - start_time) * 1000
            print(f"[BAŞARILI] GEMINI (FLASH)! Cevap: '{response.text.strip()}'")
            print(f"[SÜRE] Gemini Gecikmesi: {latency:.2f} ms")
        except Exception as e:
            print(f"[HATA] GEMINI BAĞLANTI HATASI: {e}")

    print("=" * 50)

if __name__ == "__main__":
    test_apis()
