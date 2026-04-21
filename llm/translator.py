import time
import ollama
import re
import string

# ─────────────────────────────────────────────────────────────────────
# Sabit Kültürel Deyim Sözlüğü
# ─────────────────────────────────────────────────────────────────────
CULTURAL_MAP = {
    "hoş geldin": "Welcome.",
    "hoş bulduk": "Glad to be here.",
    "görüşürüz": "See you later.",
    "kendine iyi bak": "Take care of yourself.",
    "sağlıcakla kal": "Stay well.",
    "yolun açık olsun": "Safe travels.",
    "allah'a emanet ol": "May God protect you.",
    "hayırlı olsun": "Congratulations, best wishes.",
    "gözün aydın": "I am so happy for your good news.",
    "ellerine sağlık": "Well done, thank you.",
    "çok yaşa": "Bless you.",
    "sen de gör": "Thank you, you too.",
    "iyi ki doğdun": "Happy birthday!",
    "nice senelere": "Many happy returns.",
    "helal olsun": "Bravo, well done.",
    "sıhhatler olsun": "Enjoy your haircut/shower.", 
    "geçmiş olsun": "Get well soon.",
    "başınız sağ olsun": "I am sorry for your loss.",
    "allah rahmet eylesin": "May they rest in peace.",
    "canın sağ olsun": "Don't worry about it.",
    "üzme kendini": "Don't beat yourself up.",
    "kısmet değilmiş": "It wasn't meant to be.",
    "hayırlısı olsun": "Let's hope for the best.",
    "kolay gelsin": "May it be easy for you.",
    "afiyet olsun": "Enjoy your meal.",
    "bereket versin": "Thanks, may it bring abundance.",
    "ziyade olsun": "Thank you for the meal.",
    "iyi çalışmalar": "Have a good shift/work.",
    "eyvallah": "Thanks, alright.",
    "estağfurullah": "Not at all / Don't mention it.",
    "aman diyeyim": "Watch out / Be careful.",
    "hadi canım": "No way / You're kidding.",
    "yok artık": "Unbelievable.",
    "ne halt ettin sen": "What have you done!",
    "kurban olayım sana": "I would do anything for you.",
    "allah razı olsun": "May God bless you.",
    "allah korusun": "God forbid.",
    "allah rahatlık versin": "Good night, sleep tight." 
}

SYSTEM_PROMPT = (
    "You are a Turkish-to-English translation engine. "
    "Your output must follow these rules without exception:\n"
    "RULE 1: Output ONLY the English translation. Nothing else.\n"
    "RULE 2: No bullet points, no bold, no markdown, no asterisks.\n"
    "RULE 3: No explanations, no notes, no alternatives, no comments.\n"
    "RULE 4: One line only. No line breaks.\n"
    "RULE 5: If you add ANYTHING other than the translation, you have failed.\n"
    "Translate naturally. Preserve tone and register."
)

class Translator:
    def __init__(self, model_name="gemma2:2b"):
        self.model_name = model_name
        self.system_prompt = SYSTEM_PROMPT
        print(f"LLM Modeli Hazır: {self.model_name}")

    def _tr_lower(self, text):
        """Türkçe karakterleri güvenli bir şekilde küçültür."""
        return text.replace("İ", "i").replace("I", "ı").lower()

    def _check_cultural(self, text_tr: str):
        # Türkçe karakter düzeltmesi yapılmış temiz metin
        fixed_text = self._tr_lower(text_tr)
        clean_text = fixed_text.translate(str.maketrans('', '', string.punctuation)).strip()

        # 1. AŞAMA: TAM EŞLEŞME (Exact Match)
        for key, value in CULTURAL_MAP.items():
            clean_key = self._tr_lower(key).translate(str.maketrans('', '', string.punctuation)).strip()
            if clean_text == clean_key:
                return value, "exact"

        # 2. AŞAMA: KISMİ EŞLEŞME (Partial Match)
        for key, value in CULTURAL_MAP.items():
            if self._tr_lower(key) in fixed_text:
                return (key, value), "partial"

        return None, "none"

    def _clean(self, text: str) -> str:
        text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
        text = re.sub(r"^\s*[\*\-•]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(
            r"^(here is|here's|translation:|the translation is|in english)[:\s]+",
            "", text, flags=re.IGNORECASE
        )
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = " ".join(lines) if lines else text
        text = text.strip('"').strip("'").strip()
        return text

    def translate(self, text_tr: str, context: list = None) -> dict:
        if context is None:
            context = []
        if not text_tr.strip():
            return {"text_en": "", "duration_ms": 0.0, "source": "empty"}

        cultural_result, match_type = self._check_cultural(text_tr)

        if match_type == "exact":
            return {"text_en": cultural_result, "duration_ms": 0.0, "source": "cultural_map"}

        start_time = time.time()
        full_prompt = ""

        if context:
            full_prompt += "PREVIOUS CONTEXT:\n" + "\n".join(context) + "\n\n"

        if match_type == "partial":
            tr_idiom, en_idiom = cultural_result
            # Kuralı esnetiyoruz: Bu kısmı böyle çevir ama cümlenin kalanını da doğal çevir.
            full_prompt += (f"CRITICAL RULE: The text contains '{tr_idiom}'. "
                            f"Translate that specific part as '{en_idiom}', "
                            "and translate the rest of the sentence naturally.\n")

        full_prompt += f"TRANSLATE: {text_tr}"

        try:
            response = ollama.generate(
                model=self.model_name, system=self.system_prompt, prompt=full_prompt,
                options={"temperature": 0.1, "num_predict": 80, "num_ctx": 512}
            )
            raw = response["response"].strip()
            text_en = self._clean(raw)
            duration_ms = (time.time() - start_time) * 1000

            return {
                "text_en": text_en,
                "duration_ms": round(duration_ms, 1),
                "source": "llm_with_hint" if match_type == "partial" else "llm"
            }
        except Exception as e:
            return {"text_en": f"Error: {e}", "duration_ms": 0.0, "source": "error"}

if __name__ == "__main__":
    tr_model = Translator()
    print("Warm-up...")
    tr_model.translate("Merhaba")

    test_suite = [
        ("Başınız sağ olsun.", []),
        ("Allah razı olsun. Melih", []),
        ("Kurban olayım sana Yusuf'um.", []),
        ("Ellerine sağlık teyzeciğim.", []),
        ("İyi ki doğdun! Melisa", []), # ARTIK HINT OLARAK ÇALIŞACAK
        ("Ahmet projeyi bitirdi.", []),
        ("O şimdi çok mutlu.", ["Ahmet projeyi bitirdi."]),
    ]

    print("\n" + "=" * 60)
    print("Gemma Echo — Nihai Çeviri Testi")
    print("=" * 60)

    for text, ctx in test_suite:
        res = tr_model.translate(text, ctx)
        tag = f"[{res['source'].upper():<12}]"
        lat = f"{res['duration_ms']:6.1f}ms" if res['duration_ms'] > 0 else "  --  "
        print(f"  {tag} | {lat} | {text} -> {res['text_en']}")