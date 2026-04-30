import sys
import os
import time
from stt.transcriber import Transcriber
from llm.translator import Translator
from tts.synthesizer import Synthesizer

class Orchestrator:
    def __init__(self, transcriber: Transcriber, translator: Translator, synthesizer: Synthesizer):
        """
        Gemma Echo Orkestra Sefi.
        Bilesenleri disaridan alir (Dependency Injection) —
        main.py veya test dosyalari istenen konfigurasyonu gecebilir.
        """
        print("\n" + "="*50)
        print("[SISTEM] GEMMA ECHO ORKESTRA SEFI BASLATILIYOR")
        print("="*50)
        
        self.transcriber = transcriber
        self.translator = translator
        self.synthesizer = synthesizer
        
        # Baslangicta hepsi online modda
        self.translator.set_mode("online")
        self.synthesizer.set_mode("online")
        
        # Kritik VRAM Yonetimi: Ollama'yi bosalt
        print("[SISTEM] Online mod aktif. Whisper icin VRAM temizleniyor...")
        self.translator.release_ollama_vram()
        
        print("[SISTEM] Orkestra Sefi hazir! Dinlemeye gecilebilir.")
        
    def process(self, audio_path: str):
        """
        Uctan uca ses ceviri hatti:
        STT -> LLM -> TTS
        """
        print(f"\n[ORCHESTRATOR] Ses isleniyor: {audio_path}")
        total_start = time.time()
        
        try:
            # 1. STT (Speech-to-Text)
            stt_start = time.time()
            stt_result = self.transcriber.transcribe(audio_path)
            
            # Gurultu kontrolu
            if stt_result.get("no_speech_prob", 0) > 0.6:
                print("[ORCHESTRATOR] Gurultu algilandi, ceviri iptal edildi.")
                return
                
            text_tr = stt_result.get("text", "")
            if not text_tr:
                print("[ORCHESTRATOR] Bos metin dondu, ceviri iptal edildi.")
                return
            
            stt_ms = int((time.time() - stt_start) * 1000)
            print(f"[ORCHESTRATOR] STT Suresi: {stt_ms}ms | Metin: '{text_tr}'")
            
            # 2. LLM (Ceviri)
            llm_result = self.translator.translate(text_tr)
            text_en = llm_result.get("translation", "")
            llm_ms = llm_result.get("latency_ms", 0)
            
            print(f"[ORCHESTRATOR] Ceviri: '{text_en}' | Motor: {llm_result.get('engine')} | {llm_ms}ms")
            
            # 3. TTS (Sentez)
            tts_ms = self.synthesizer.speak(text_en) or 0
            
            total_ms = int((time.time() - total_start) * 1000)
            print(f"[ORCHESTRATOR] Islem tamamlandi. E2E: {total_ms}ms (STT:{stt_ms} + LLM:{llm_ms} + TTS:{tts_ms})")
            
        except Exception as e:
            print(f"\n[ORCHESTRATOR] HATA YAKALANDI: {e}")
            self._fallback(audio_path)
            
    def _fallback(self, audio_path: str):
        """
        Cevrimici servisler hata verirse XTTS ve Ollama'ya gecer.
        """
        print("\n" + "!"*50)
        print("[SISTEM] BAGLANTI HATASI! OFFLINE (HAYATTA KALMA) MODUNA GECILIYOR...")
        print("!"*50)
        
        # Modlari degistir
        self.transcriber.switch_to_cpu()  # VRAM'i Gemma icin bosalt
        self.translator.set_mode("offline")
        self.synthesizer.set_mode("offline")
        
        # Sureci yeniden baslat (Offline)
        print(f"[ORCHESTRATOR] Offline isleniyor: {audio_path}")
        
        stt_result = self.transcriber.transcribe(audio_path)
        text_tr = stt_result.get("text", "")
        
        if text_tr:
            llm_result = self.translator.translate(text_tr)
            text_en = llm_result.get("translation", "")
            print(f"[ORCHESTRATOR] Offline Ceviri: '{text_en}'")
            
            self.synthesizer.speak(text_en)
            print("[ORCHESTRATOR] Offline Islem tamamlandi.")
