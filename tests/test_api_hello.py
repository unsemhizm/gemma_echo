import os
import time
from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types

# .env dosyasındaki anahtarları sisteme yükle
load_dotenv()

def test_apis():
    print("[TEST] 3 Katmanlı Turbo Mimari Kontrol Ediliyor...\n")
    print("Hedef: Groq ile < 500ms hıza ulaşmak.\n" + "="*50)

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

    print("-" * 50)

    # ---------------------------------------------------------
    # 3. ÜÇÜNCÜL MOTOR (GÜVENLİK YEDEĞİ): GROQ
    # ---------------------------------------------------------
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("[HATA] GROQ_API_KEY .env dosyasında bulunamadı!")
    else:
        try:
            print("[BEKLEYİN] 3. KATMAN: Groq (Llama 3.1 8B) test ediliyor...")
            from groq import Groq
            client = Groq(api_key=groq_key.strip())
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Say 'Hello' and nothing else."}],
                temperature=0.1,
                max_tokens=10
            )
            
            latency = (time.time() - start_time) * 1000
            print(f"[BAŞARILI] GROQ! Cevap: '{response.choices[0].message.content.strip()}'")
            print(f"[SÜRE] Groq Gecikmesi: {latency:.2f} ms")
            if latency < 500:
                print("[HEDEF] VURULDU: 500ms'nin altındayız!")
            else:
                print("[UYARI] Hedefin biraz üstündeyiz, ağ gecikmesi olabilir.")
        except Exception as e:
            print(f"[HATA] GROQ BAĞLANTI HATASI: {e}")
            
    print("=" * 50)

if __name__ == "__main__":
    test_apis()
