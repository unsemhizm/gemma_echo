import os
import time
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from google import genai

# .env dosyasındaki anahtarları sisteme yükle
load_dotenv()

def test_apis():
    print("[TEST] 3 Katmanlı Turbo Mimari Kontrol Ediliyor...\n")
    print("Hedef: Groq ile < 500ms hıza ulaşmak.\n" + "="*50)

    # ---------------------------------------------------------
    # 1. BİRİNCİL MOTOR: GROQ (HIZ ŞAMPİYONU)
    # ---------------------------------------------------------
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ HATA: GROQ_API_KEY .env dosyasında bulunamadı!")
    else:
        try:
            print("⏳ 1. KATMAN: Groq (Llama 3.1 8B) test ediliyor...")
            client = Groq(api_key=groq_key.strip())
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Say 'Hello' and nothing else."}],
                temperature=0.1,
                max_tokens=10
            )
            
            latency = (time.time() - start_time) * 1000
            print(f"✅ GROQ BAŞARILI! Cevap: '{response.choices[0].message.content.strip()}'")
            print(f"⏱️ Groq Gecikmesi: {latency:.2f} ms")
            if latency < 500:
                print("🚀 HEDEF VURULDU: 500ms'nin altındayız!")
            else:
                print("⚠️ Hedefin biraz üstündeyiz, ağ gecikmesi olabilir.")
        except Exception as e:
            print(f"❌ GROQ BAĞLANTI HATASI: {e}")
            
    print("-" * 50)

    # ---------------------------------------------------------
    # 2. İKİNCİL MOTOR: OPENROUTER (GEMMA 4)
    # ---------------------------------------------------------
    or_key = os.getenv("OPENROUTER_API_KEY")
    if not or_key:
        print("❌ HATA: OPENROUTER_API_KEY .env dosyasında bulunamadı!")
    else:
        try:
            print("⏳ 2. KATMAN: OpenRouter (Gemma 4 26B) test ediliyor...")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=or_key.strip()
            )
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="google/gemma-4-26b-a4b-it:free",
                messages=[{"role": "user", "content": "Say 'Hello' and nothing else."}],
                temperature=0.1,
                max_tokens=10
            )
            
            latency = (time.time() - start_time) * 1000
            print(f"✅ OPENROUTER BAŞARILI! Cevap: '{response.choices[0].message.content.strip()}'")
            print(f"⏱️ OpenRouter Gecikmesi: {latency:.2f} ms")
        except Exception as e:
            print(f"❌ OPENROUTER BAĞLANTI HATASI: {e}")

    print("-" * 50)

    # ---------------------------------------------------------
    # 3. ÜÇÜNCÜL MOTOR: GEMINI (GÜVENLİK YEDEĞİ)
    # ---------------------------------------------------------
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ HATA: GEMINI_API_KEY .env dosyasında bulunamadı!")
    else:
        try:
            print("⏳ 3. KATMAN: Gemini 2.5 Flash test ediliyor...")
            client = genai.Client(api_key=gemini_key.strip()) 
            start_time = time.time()
            
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents="Say 'Hello' and nothing else."
            )
            
            latency = (time.time() - start_time) * 1000
            print(f"✅ GEMINI BAŞARILI! Cevap: '{response.text.strip()}'")
            print(f"⏱️ Gemini Gecikmesi: {latency:.2f} ms")
        except Exception as e:
            print(f"❌ GEMINI BAĞLANTI HATASI: {e}")
            
    print("=" * 50)

if __name__ == "__main__":
    test_apis()