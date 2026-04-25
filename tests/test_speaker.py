import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts.synthesizer import Synthesizer


def test_tts():
    print("[TEST] Synthesizer Modulu Kontrol Ediliyor...\n")
    try:
        synthesizer = Synthesizer()
        test_text = (
            "Hello! I am your real-time AI translator. "
            "I am ready to work like a wolf council."
        )

        print(f"Metin: '{test_text}'")
        print("Ses uretiliyor ve caliniyor...")

        synthesizer.speak_online(test_text)
        print("[TEST] TTS TESTI BASARILI!")

    except Exception as e:
        print(f"[TEST HATASI] TTS TEST HATASI: {e}")


if __name__ == "__main__":
    test_tts()
