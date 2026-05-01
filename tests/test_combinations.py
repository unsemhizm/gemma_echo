import os
import sys
import time
import glob
import gc
import torch
import itertools

# Modul yollarini ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stt.transcriber import Transcriber
from llm.translator import Translator
from tts.synthesizer import Synthesizer

def main():
    print("="*60)
    print("      GEMMA ECHO — KAPSAMLI KOMBINASYON TESTI")
    print("="*60)

    audio_files = glob.glob("audio/*.wav")
    if not audio_files:
        print("[HATA] 'audio/' klasorunde test edilecek .wav dosyasi bulunamadi.")
        return

    # Ayni siralamayi garanti etmek icin dosyaları siralayalim
    audio_files.sort()

    print(f"[BILGI] Test edilecek ses dosyasi sayisi: {len(audio_files)}")

    # 1. Bilesenleri baslat
    print("\n[BILGI] Bilesenler baslatiliyor...")
    transcriber = Transcriber()
    translator = Translator()
    synthesizer = Synthesizer()

    # Test edilecek konfigurasyonlar
    stt_modes = ["cloud_deepgram", "cloud_groq", "local_gpu", "local_cpu"]
    llm_modes = ["online", "offline"]
    tts_modes = ["online", "gpu", "offline"]

    # 1. Kombinasyon Havuzunu Olustur
    valid_combinations = []
    for stt, llm, tts in itertools.product(stt_modes, llm_modes, tts_modes):
        # Cakisik (Tehlikeli) kombinasyonlari en bastan ele
        if llm == "offline" and tts == "gpu":
            continue
        valid_combinations.append({"stt": stt, "llm": llm, "tts": tts})

    print(f"[BILGI] Gecerli kombinasyon sayisi: {len(valid_combinations)}")

    print("\n" + "="*60)
    print("      TEST BASLIYOR")
    print("="*60)

    results = []

    # 2. Dis Dongu: Kombinasyonlar uzerinde don
    for idx, combo in enumerate(valid_combinations, 1):
        stt_mode = combo["stt"]
        llm_mode = combo["llm"]
        tts_mode = combo["tts"]
        
        print(f"\n[{idx}/{len(valid_combinations)}] KOMBINASYON KURULUYOR: STT={stt_mode} | LLM={llm_mode} | TTS={tts_mode}")
        
        # 3. Mod Kurulumu (Sadece 1 Kere)
        transcriber.set_mode(stt_mode)
        translator.set_mode(llm_mode)
        synthesizer.set_mode(tts_mode)
        
        # Eger TTS GPU modu secildiyse ve XTTS henuz RAM/VRAM'de degilse eager load
        if tts_mode == "gpu" and synthesizer.xtts_model is None:
            print("    [SISTEM] XTTS GPU'ya yukleniyor, bekleniyor...")
            synthesizer.preload_xtts_background(use_gpu=True)
            while synthesizer.xtts_model is None:
                time.sleep(1)
            print("    [SISTEM] XTTS hazir.")
            
        # Eger ElevenLabs modundaysak ama arka planda XTTS kalmissa ondan kurtul 
        # (transcriber set_mode vs bunu tam yonetmeyebilir bu test scripti icinde izole oldugu icin)
        if tts_mode == "online" and synthesizer._xtts_on_gpu:
            synthesizer.offload_xtts_from_gpu()
            
        print(">> Kurulum tamam, dosyalar isleniyor...")

        # 4. Ic Dongu: Secilen O TEK kombinasyon uzerinde tum ses dosyalarini isleyelim
        for audio_file in audio_files:
            file_name = os.path.basename(audio_file)
            print(f"\n--- Ses Dosyasi: {file_name} ---")
            
            # API Kotasi Korumasi (ElevenLabs icin)
            if tts_mode == "online" and audio_file != audio_files[0]:
                print(f"    [ATLANDI] API kotasini korumak icin ElevenLabs ({tts_mode}) sadece ilk dosyada calistirilir. Sonraki kombinasyona geciliyor...")
                # Sonraki dosyalara bakmanin anlami yok ElevenLabs ise, direkt dosyalar dongusunu bitirebiliriz.
                continue

            # --- A. STT ---
            print(f"[STT] {stt_mode} ile dinleniyor...")
            stt_start = time.time()
            stt_result = transcriber.transcribe(audio_file)
            stt_time = int((time.time() - stt_start) * 1000)
            text_tr = stt_result.get("text", "")
            print(f"[STT] Sure: {stt_time}ms | Metin: '{text_tr}'")

            if not text_tr:
                print(f"[UYARI] STT bos metin dondurdu. LLM ve TTS atlanıyor.")
                continue

            # --- B. LLM ---
            print(f"[LLM] {llm_mode} ile cevriliyor...")
            llm_start = time.time()
            llm_result = translator.translate(text_tr)
            llm_time = int((time.time() - llm_start) * 1000)
            text_en = llm_result.get("translation", "")
            engine = llm_result.get("engine", "unknown")
            print(f"[LLM] Sure: {llm_time}ms | Motor: {engine} | Ceviri: '{text_en}'")

            # --- C. TTS ---
            print(f"[TTS] {tts_mode} ile sentezleniyor...")
            tts_time = synthesizer.speak(text_en) or 0
            
            total_e2e = stt_time + llm_time + tts_time
            print(f"[TTS] Sure: {tts_time}ms")
            print(f"=> KOMBINE E2E SURE: {total_e2e}ms")
            
            # Kaydet
            results.append({
                "file": file_name,
                "stt": stt_mode,
                "llm": llm_mode,
                "tts": tts_mode,
                "stt_time": stt_time,
                "llm_time": llm_time,
                "tts_time": tts_time,
                "e2e": total_e2e
            })

        # 5. VRAM Temizligi: Bir kombinasyon tum dosyalari bitirdiginde sistemi bosalt
        print(f"\n[SISTEM] Kombinasyon ({stt_mode}/{llm_mode}/{tts_mode}) tamamlandi. VRAM temizleniyor...")
        gc.collect()
        torch.cuda.empty_cache()


    print("\n" + "="*60)
    print("      TUM TESTLER TAMAMLANDI - EN IYI SONUCLAR")
    print("="*60)
    
    # Sonuclari en hizlidan yavasa sirala
    results.sort(key=lambda x: x["e2e"])
    
    for i, res in enumerate(results[:30]): # Ilk 30 gosterilebilir
        print(f"{i+1:2d}. Dosya: {res['file']:15} | E2E: {res['e2e']:6}ms | STT: {res['stt']:15} ({res['stt_time']}ms) | LLM: {res['llm']:8} ({res['llm_time']}ms) | TTS: {res['tts']:8} ({res['tts_time']}ms)")

if __name__ == "__main__":
    main()
