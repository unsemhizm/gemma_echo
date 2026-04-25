import os
import sys

# Proje kök dizinini Python yoluna ekle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import time
try:
    from stt.transcriber import Transcriber     # stt/transcriber.py içindeki Transcriber sınıfı
    from llm.translator import Translator       # llm/translator.py içindeki Translator sınıfı
except ImportError as e:
    print(f"❌ Aktarma Hatası: {e}")
    print("İpucu: Sınıf isimlerinin (Transcriber, Translator) dosya içinde doğru yazıldığından emin ol.")
    sys.exit()


def main():
    print("\n" + "="*50)
    print("      GEMMA ECHO v7 — GÜN 2 TEST MODU      ")
    print("="*50 + "\n")

    try:
        # ─── Modülleri Başlat ─────────────────────────────────
        print("[SİSTEM] Modüller yükleniyor...")
        
        # 1. STT Engine (GPU, small)
        stt_engine = Transcriber()
        
        # 2. LLM Engine (Online mod — Groq → OpenRouter → Gemini)
        llm_engine = Translator()
        
        # 3. Ollama VRAM boşaltma (keep_alive=0) — rapordaki kritik kural
        llm_engine.release_ollama_vram()

        print("\n🚀 SİSTEM HAZIR! Ses dosyaları test ediliyor...\n")

        # ─── Test Ses Dosyaları ───────────────────────────────
        # Gün 2: Mikrofon henüz yok, audio/ klasöründen test ediyoruz
        audio_dir = os.path.join(BASE_DIR, "audio")
        audio_files = sorted([
            os.path.join(audio_dir, f) 
            for f in os.listdir(audio_dir) 
            if f.endswith(".wav")
        ])

        if not audio_files:
            print("❌ audio/ klasöründe .wav dosyası bulunamadı!")
            return

        # Context buffer — zamir çevirisi için önceki cümleleri tutar
        context_buffer = []

        for audio_path in audio_files:
            print(f"\n{'─'*50}")
            print(f"📁 Dosya: {os.path.basename(audio_path)}")
            
            # 1. STT: Ses → Türkçe Metin
            print("👂 Transkripsiyon yapılıyor...")
            stt_result = stt_engine.transcribe(audio_path)
            
            tr_text = stt_result["text"]
            no_speech = stt_result["no_speech_prob"]
            stt_latency = stt_result.get("latency_ms", 0)

            # Confidence Filtering: no_speech_prob > 0.6 → atla
            if no_speech > 0.6:
                print(f"⚠️ Gürültü tespit edildi (no_speech_prob={no_speech:.2f}), atlanıyor...")
                continue

            if not tr_text or len(tr_text.strip()) < 2:
                print("⚠️ Metin çok kısa, atlanıyor...")
                continue

            print(f"🇹🇷 (TR): {tr_text}")
            print(f"⏱️ STT: {stt_latency}ms | Güven: {1 - no_speech:.2f}")

            # 2. LLM: Türkçe → İngilizce
            print("🧠 Çevriliyor...")
            llm_result = llm_engine.translate(tr_text, context=context_buffer)
            
            en_text = llm_result["translation"]
            llm_latency = llm_result["latency_ms"]
            engine = llm_result["engine"]

            print(f"🇺🇸 (EN): {en_text}")
            print(f"⏱️ LLM: {llm_latency}ms | Motor: {engine}")

            # Context buffer'ı güncelle (son 3 cümle)
            context_buffer.append(tr_text)
            if len(context_buffer) > 3:
                context_buffer.pop(0)

            # Toplam latency
            total = stt_latency + llm_latency
            print(f"📊 Toplam: {total}ms (Hedef: < 2000ms)")

        print(f"\n{'='*50}")
        print("✅ Gün 2 testi tamamlandı.")
        print(f"{'='*50}\n")

    except KeyboardInterrupt:
        print("\n\n👋 Test durduruldu.")
    except Exception as e:
        print(f"\n[HATA] Beklenmedik bir sorun: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()