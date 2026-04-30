import sys
import os
import argparse

# Modul yollarini ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stt.transcriber import Transcriber
from llm.translator import Translator
from tts.synthesizer import Synthesizer
from pipeline.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="Gemma Echo v7 — Hibrit Ceviri Sistemi")
    parser.add_argument("--offline", action="store_true", help="Sistemi dogrudan offline modda baslat")
    parser.add_argument("--input", type=str, default="audio/Kayıt (3).wav", help="Islenecek ses dosyasi")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("      GEMMA ECHO v7 — GUN 3: ORKESTRA SEFI")
    print("="*50)

    # 1. Bilesenleri Baslat
    transcriber = Transcriber()
    translator = Translator()
    synthesizer = Synthesizer()

    # 2. Orkestray Kur
    orchestrator = Orchestrator(transcriber, translator, synthesizer)

    # 3. Baslangic Modunu Ayarla
    if args.offline:
        print("[SISTEM] Manuel olarak OFFLINE mod secildi.")
        transcriber.switch_to_cpu()
        translator.set_mode("offline")
        synthesizer.set_mode("offline")

    # 4. Islemi Baslat
    print(f"\n[ISLEM] Hedef dosya: {args.input}")
    orchestrator.process(args.input)

    print("\n" + "="*50)
    print("      ISLEM TAMAMLANDI - GUN 4 ICIN HAZIRIZ")
    print("="*50)

if __name__ == "__main__":
    main()