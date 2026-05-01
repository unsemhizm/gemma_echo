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
    parser = argparse.ArgumentParser(description="Gemma Echo v8 — 4 Modlu Hibrit Ceviri Sistemi")
    parser.add_argument(
        "--mode", type=str, default="interactive",
        choices=["online", "online_xtts", "interactive", "offline", "hybrid_cloud_io", "hybrid_cloud_stt"],
        help="Calisma modu: online, online_xtts, interactive (varsayilan), offline, hybrid_cloud_io, hybrid_cloud_stt"
    )
    parser.add_argument(
        "--input", type=str, default="audio/Kayıt (3).wav",
        help="Islenecek ses dosyasi"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("      GEMMA ECHO v8 — QUAD-STATE ORKESTRA SEFI")
    print(f"      Mod: {args.mode.upper()}")
    print("="*60)

    # 1. Bilesenleri Baslat
    transcriber = Transcriber()
    translator = Translator()
    synthesizer = Synthesizer()

    # 2. Orkestrayi Kur (baslangic moduyla)
    orchestrator = Orchestrator(transcriber, translator, synthesizer, initial_mode=args.mode)

    # 3. Islemi Baslat
    print(f"\n[ISLEM] Hedef dosya: {args.input}")
    orchestrator.process(args.input)

    print("\n" + "="*60)
    print("      ISLEM TAMAMLANDI")
    print("="*60)


if __name__ == "__main__":
    main()