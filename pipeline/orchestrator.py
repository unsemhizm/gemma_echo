import sys
import os
import time
from stt.transcriber import Transcriber
from llm.translator import Translator
from tts.synthesizer import Synthesizer

class Orchestrator:
    def __init__(self):
        print("\n" + "="*50)
        print("[SİSTEM] GEMMA ECHO ORKESTRA ŞEFİ BAŞLATILIYOR")
        print("="*50)
        
        self.transcriber = Transcriber()
        self.translator = Translator()
        self.synthesizer = Synthesizer()
        
        # Başlangıçta hepsi online modda
        self.translator.set_mode("online")
        self.synthesizer.set_mode("online")
        
        # Kritik VRAM Yönetimi: Ollama'yı boşalt
        print("[SİSTEM] Online mod aktif. Whisper için VRAM temizleniyor...")
        self.translator.release_ollama_vram()
        
        print("[SİSTEM] Orkestra Şefi hazır! Dinlemeye geçilebilir.")
        
    def process_audio(self, audio_path: str):
        """
        Uçtan uca ses çeviri hattı:
        STT -> LLM -> TTS
        """
        print(f"\n[ORCHESTRATOR] Ses işleniyor: {audio_path}")
        
        try:
            # 1. STT (Speech-to-Text)
            stt_start = time.time()
            stt_result = self.transcriber.transcribe(audio_path)
            
            # Gürültü kontrolü
            if stt_result.get("no_speech_prob", 0) > 0.6:
                print("[ORCHESTRATOR] Gürültü algılandı, çeviri iptal edildi.")
                return
                
            text_tr = stt_result.get("text", "")
            if not text_tr:
                print("[ORCHESTRATOR] Boş metin döndü, çeviri iptal edildi.")
                return
                
            print(f"[ORCHESTRATOR] STT Süresi: {int((time.time() - stt_start)*1000)}ms | Metin: '{text_tr}'")
            
            # 2. LLM (Çeviri)
            llm_result = self.translator.translate(text_tr)
            text_en = llm_result.get("translation", "")
            
            print(f"[ORCHESTRATOR] Çeviri: '{text_en}' | Motor: {llm_result.get('engine')}")
            
            # 3. TTS (Sentez)
            self.synthesizer.speak(text_en)
            
            print("[ORCHESTRATOR] İşlem tamamlandı.")
            
        except Exception as e:
            print(f"\n[ORCHESTRATOR] HATA YAKALANDI: {e}")
            self.fallback_mode(audio_path)
            
    def fallback_mode(self, audio_path: str):
        """
        Çevrimiçi servisler (Groq, ElevenLabs vb.) hata verirse XTTS ve Ollama'ya geçer.
        """
        print("\n" + "!"*50)
        print("[SİSTEM] BAĞLANTI HATASI! OFFLINE (HAYATTA KALMA) MODUNA GEÇİLİYOR...")
        print("!"*50)
        
        # Modları değiştir
        self.translator.set_mode("offline")
        self.synthesizer.set_mode("offline")
        
        # Süreci yeniden başlat (Offline)
        print(f"[ORCHESTRATOR] Offline işleniyor: {audio_path}")
        
        # Not: STT şu an hala GPU'da çalışıyor olabilir (transcriber.switch_to_cpu() Gün 4'te eklenecek)
        stt_result = self.transcriber.transcribe(audio_path)
        text_tr = stt_result.get("text", "")
        
        if text_tr:
            llm_result = self.translator.translate(text_tr)
            text_en = llm_result.get("translation", "")
            print(f"[ORCHESTRATOR] Offline Çeviri: '{text_en}'")
            
            self.synthesizer.speak(text_en)
            print("[ORCHESTRATOR] Offline İşlem tamamlandı.")
