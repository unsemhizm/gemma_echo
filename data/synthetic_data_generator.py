# -*- coding: utf-8 -*-
"""
synthetic_data_generator.py
-----------------------------────────
Gemini API kullanarak Türkçe → Japonca çeviri eğitim verisi üretir.

Çıktı  : ./data/ja_training_data.jsonl  (Unsloth / Alpaca formatı)
Hedef  : 5 mod × 100 örnek = 500 satır
Çalıştır: python data/synthetic_data_generator.py
"""

import os
import json
import time
import random
import sys
import io
from pathlib import Path

# Windows terminal UTF-8 zorlama (cp1254 hatasini onler)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from dotenv import load_dotenv

# ── proje kökünü sys.path'e ekle (translator gibi modüller import edilebilsin) ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from google import genai
from google.genai import types

# ════════════════════════════════════════════════════════════════
# AYARLAR
# ════════════════════════════════════════════════════════════════
GEMINI_MODEL      = "gemini-flash-lite-latest"  # Kotasi olmayan lite surumu
OUTPUT_FILE       = ROOT / "data" / "ja_training_data.jsonl"
EXAMPLES_PER_MODE = 100                          # 5 × 100 = 500 toplam
BATCH_SIZE        = 10                           # tek API çağrısında üretilecek örnek
RETRY_LIMIT       = 3
DELAY_BETWEEN_BATCHES = 2.0                      # saniye (rate limit koruması)

# ════════════════════════════════════════════════════════════════
# 5 MOD TANIMI
# Sistem promptu + Japonca üslup rehberi
# ════════════════════════════════════════════════════════════════
MODES: dict[str, dict] = {

    "emergency": {
        "label": "Acil Durum / Telsiz",
        "system": (
            "You are a Japanese Coast Guard (海上保安庁) and JMSDF radio communication expert. "
            "Generate emergency radio messages in authentic Japanese military/coast-guard style. "
            "Use: short imperative sentences, call signs (例: 第7ユニット), official codes, "
            "proper radio endings (どうぞ / 了解 / 以上). "
            "NEVER use casual or anime-style Japanese. Sound like a real JCG radio transcript."
        ),
        "instruction": (
            "Sen bir Japon Sahil Güvenlik telsiz operatörüsün. "
            "Aşağıdaki Türkçe acil durum mesajını otantik JCG telsiz jargonuyla Japoncaya çevir. "
            "Kısa, net ve komuta dayalı cümleler kur."
        ),
        "seeds": [
            "Birim 7, rotayı hemen değiştir.",
            "Bölgede şüpheli deniz aracı tespit edildi.",
            "Acil tıbbi tahliye gerekiyor, koordinatları bildirin.",
            "Fırtına uyarısı, tüm birimler limana dönün.",
            "Motor arızası, yardım isteği alındı.",
            "Kıyı muhafızları, hedef bölgeyi çevreyin.",
            "Görüş mesafesi sıfıra düşüyor, dikkatli olun.",
            "Yangın ihbarı alındı, kurtarma ekibi hazır olsun.",
            "İletişim kesildi, son bilinen konum paylaşıldı.",
            "Tüm birimler alarm durumuna geçiyor.",
        ],
    },

    "official": {
        "label": "Resmi / Diplomatik",
        "system": (
            "You are a senior Japanese diplomat and academic writer. "
            "Use the most formal register of Japanese (敬語 / keigo). "
            "Apply polite verb endings (-ます/-です), honorifics (御社, ご連絡), "
            "and formal connectors (なお、つきましては). "
            "Zero slang. Zero casual contractions."
        ),
        "instruction": (
            "Sen üst düzey bir Japon diplomat veya akademisyensin. "
            "Aşağıdaki Türkçe resmi metni tam anlamıyla resmi Japonca keigo üslubuyla çevir."
        ),
        "seeds": [
            "Toplantı yarın saat 10.00'a ertelenmiştir.",
            "Başvurunuz incelemeye alınmıştır, sonuç tarafınıza iletilecektir.",
            "Anlaşma koşulları her iki tarafça kabul edilmiştir.",
            "Projenin ilk aşaması başarıyla tamamlanmıştır.",
            "Yetkililer, durumu titizlikle takip etmektedir.",
            "Bildirge tüm üye devletler tarafından imzalanmıştır.",
            "Konu hakkında resmi açıklama yakında yapılacaktır.",
            "Komisyon, raporu değerlendirmek üzere toplanacaktır.",
            "Bütçe teklifi meclis gündemine alınmıştır.",
            "Ziyaret programı protokol kurallarına uygun olarak düzenlenmiştir.",
        ],
    },

    "streamer": {
        "label": "Yayıncı / İnternet Jargonu",
        "system": (
            "You are a popular Japanese gaming streamer on NicoNico/YouTube. "
            "Use authentic Japanese internet/gaming slang: "
            "草 (lol), ワロタ, めっちゃ, ガチで, やばい, 神ゲー, "
            "stream-chat expressions (いいね！コメントどんどん来い！), "
            "hype endings (〜じゃん！, 〜でしょ！). "
            "DO NOT translate English slang word-for-word. Find the Japanese internet equivalent."
        ),
        "instruction": (
            "Sen Japonya'nın en popüler oyun yayıncısısın. "
            "Aşağıdaki Türkçe yayıncı ifadesini Japon internet ve oyun kültürüne özgü "
            "güncel argo ile çevir. İngilizce argo kelimelerini birebir çevirme."
        ),
        "seeds": [
            "Bu oyun gerçekten çok efsane, herkese tavsiye ederim.",
            "Az önce inanılmaz bir hamle yaptım, izlediniz mi?",
            "Chat çılgına döndü, herkese teşekkürler!",
            "Bu boss gerçekten imkansız, beyin yaktı.",
            "Abone olun, her gün yeni içerik gelecek.",
            "Şu anki meta tamamen değişti, artık kimse bu karakteri kullanmıyor.",
            "Lag yüzünden öldüm, sunucular berbat.",
            "Klip atın, bu anı kaçırmayın!",
            "İlk denemede bitirdim, kolaylıktan ölüyorum.",
            "Sıradaki yayın çok daha büyük bir sürprizle geliyor.",
        ],
    },

    "casual": {
        "label": "Günlük / Samimi",
        "system": (
            "You are a young Japanese person talking to a close friend. "
            "Use everyday casual Japanese: plain form verbs (〜だ、〜じゃない), "
            "common contractions (〜てる、〜とく), friendly filler words (ねえ、さ、よ). "
            "Sound warm, natural and relaxed — like a real text message between friends."
        ),
        "instruction": (
            "Sen yakın bir Japon arkadaşınla konuşuyorsun. "
            "Aşağıdaki Türkçe günlük ifadeyi samimi, rahat ve doğal Japonca konuşma "
            "diline çevir. Resmi dil kullanma."
        ),
        "seeds": [
            "Bugün çok yoruldum, erken yatacağım.",
            "Ne yapıyorsun, bir şeyler yiyelim mi?",
            "O film gerçekten çok iyiydi, keşke daha önce izleseydim.",
            "Yarın sınav var ama hiç çalışmadım.",
            "Hava çok güzel, dışarıya çıkalım.",
            "Seni özledim, ne zaman görüşeceğiz?",
            "Kahve içmek ister misin?",
            "Bu hafta sonu plan yapıyor musun?",
            "Telefonu nerede bıraktım, bulamıyorum.",
            "Biraz önce çok komik bir şey oldu, anlatsam inanmazsın.",
        ],
    },

    "literary": {
        "label": "Edebi / Şiirsel",
        "system": (
            "You are a skilled Japanese literary translator and poet. "
            "Use vivid, expressive and elegant Japanese: "
            "preserve emotional tone, use metaphors, classical references if appropriate, "
            "rich vocabulary, and flowing sentence structure. "
            "The translation should feel like literature, not a conversation."
        ),
        "instruction": (
            "Sen yetenekli bir Japon edebi çevirmensin. "
            "Aşağıdaki Türkçe edebi/şiirsel ifadeyi duygusal tonu ve imgelerini "
            "koruyarak zarif ve akıcı Japoncaya çevir."
        ),
        "seeds": [
            "Güneş ufukta yavaşça battı, her şeyi altın rengine boyadı.",
            "Sessizlik bazen en güçlü çığlıktır.",
            "Geçmiş, aklımızın derinliklerinde asla kaybolmaz.",
            "Yağmur düşerken toprak o eski kokuya büründü.",
            "Gözlerin, söylenemeyen bütün sırları saklıyordu.",
            "Zamanın ağırlığı omuzlarımda her geçen günle biraz daha arttı.",
            "O gece yıldızlar hiç bu kadar yakın görünmemişti.",
            "Rüzgar ağaçların arasından geçerken eski türküler mırıldanır gibi oldu.",
            "Umut, karanlığın ortasında titreyen küçük bir alevdir.",
            "Elveda demek, bazen yeni bir başlangıcın ilk adımıdır.",
        ],
    },
}

# ════════════════════════════════════════════════════════════════
# BATCH ÜRETME PROMPTU
# ════════════════════════════════════════════════════════════════

BATCH_PROMPT_TEMPLATE = """\
You are a bilingual Turkish-Japanese training data generator.

MODE: {mode_label}
{system_context}

TASK:
Generate EXACTLY {n} unique Turkish→Japanese translation pairs in this MODE style.
Vary the topics and sentence lengths. Use the seed examples only as INSPIRATION — \
do NOT copy them. Create completely original sentences.

SEED EXAMPLES FOR INSPIRATION:
{seeds}

OUTPUT FORMAT — return ONLY a valid JSON array, no extra text:
[
  {{"tr": "Turkish sentence 1", "ja": "Japanese translation 1"}},
  {{"tr": "Turkish sentence 2", "ja": "Japanese translation 2"}},
  ...
]

CRITICAL RULES:
- Each "tr" must be natural Turkish (not a literal back-translation).
- Each "ja" must match the {mode_label} style exactly.
- Return ONLY the JSON array. No markdown, no explanation.
"""

# ════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ════════════════════════════════════════════════════════════════

def load_existing(path: Path) -> set[str]:
    """Önceki çalıştırmadan kaydedilmiş Türkçe cümleleri döner (tekrar üretme)."""
    seen = set()
    if not path.exists():
        return seen
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                seen.add(obj.get("input", ""))
            except json.JSONDecodeError:
                continue
    return seen


def count_by_mode(path: Path) -> dict[str, int]:
    """Mevcut JSONL dosyasındaki mod başına örnek sayısını sayar."""
    counts: dict[str, int] = {m: 0 for m in MODES}
    if not path.exists():
        return counts
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                m = obj.get("mode", "")
                if m in counts:
                    counts[m] += 1
            except json.JSONDecodeError:
                continue
    return counts


def generate_batch(client: genai.Client, mode_key: str, n: int) -> list[dict]:
    """Gemini API ile n adet örnek üretir. Başarısızlıkta [] döner."""
    mode = MODES[mode_key]
    seeds_text = "\n".join(f"  TR: {s}" for s in random.sample(mode["seeds"], min(5, len(mode["seeds"]))))

    prompt = BATCH_PROMPT_TEMPLATE.format(
        mode_label=mode["label"],
        system_context=mode["system"],
        n=n,
        seeds=seeds_text,
    )

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    temperature=0.85,
                    max_output_tokens=4096,
                ),
                contents=prompt,
            )
            raw = (resp.text or "").strip()

            # JSON bloğunu bul (```json ... ``` veya [ ... ])
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            pairs = json.loads(raw)
            if isinstance(pairs, list):
                return pairs
        except Exception as e:
            print(f"  ⚠  Deneme {attempt}/{RETRY_LIMIT} başarısız: {e}")
            time.sleep(2 ** attempt)

    return []


def pairs_to_jsonl(pairs: list[dict], mode_key: str, mode: dict) -> list[str]:
    """Ham çiftleri Unsloth / Alpaca JSONL formatına dönüştürür."""
    lines = []
    for p in pairs:
        tr = (p.get("tr") or "").strip()
        ja = (p.get("ja") or "").strip()
        if not tr or not ja:
            continue
        record = {
            "instruction": mode["instruction"],
            "input": tr,
            "output": ja,
            "mode": mode_key,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return lines


# ════════════════════════════════════════════════════════════════
# ANA DÖNGÜ
# ════════════════════════════════════════════════════════════════

def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("❌ GEMINI_API_KEY bulunamadı! .env dosyasını kontrol et.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing_sentences = load_existing(OUTPUT_FILE)
    mode_counts = count_by_mode(OUTPUT_FILE)

    total_existing = sum(mode_counts.values())
    print(f"\n{'='*60}")
    print(f"  Gemma Echo -- Turkce->Japonca Sentetik Veri Uretici")
    print(f"{'='*60}")
    print(f"  Hedef  : {len(MODES)} mod x {EXAMPLES_PER_MODE} = {len(MODES)*EXAMPLES_PER_MODE} ornek")
    print(f"  Mevcut : {total_existing} ornek zaten var")
    print(f"  Cikti  : {OUTPUT_FILE}")
    print(f"{'='*60}\n")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for mode_key, mode in MODES.items():
            already = mode_counts[mode_key]
            remaining = EXAMPLES_PER_MODE - already

            if remaining <= 0:
                print(f"  [OK] [{mode['label']}] zaten tamamlandi ({already}/{EXAMPLES_PER_MODE})")
                continue

            print(f"\n  [..] [{mode['label']}] -- {already}/{EXAMPLES_PER_MODE} mevcut, {remaining} uretiliyor...")

            generated = 0
            while generated < remaining:
                batch_n = min(BATCH_SIZE, remaining - generated)
                print(f"     Batch: {batch_n} ornek bekleniyor...", end=" ", flush=True)

                pairs = generate_batch(client, mode_key, batch_n)
                if not pairs:
                    print("ATLANDI (API hatasi)")
                    continue

                # Tekrar kontrol + yaz
                new_lines = []
                for line in pairs_to_jsonl(pairs, mode_key, mode):
                    obj = json.loads(line)
                    if obj["input"] in existing_sentences:
                        continue   # tekrar örnek → atla
                    existing_sentences.add(obj["input"])
                    new_lines.append(line)

                if new_lines:
                    out_f.write("\n".join(new_lines) + "\n")
                    out_f.flush()
                    generated += len(new_lines)
                    print(f"{len(new_lines)} eklendi  (bu modda toplam: {already + generated})")
                else:
                    print("0 yeni ornek (hepsi tekrar), yeniden deneniyor...")

                time.sleep(DELAY_BETWEEN_BATCHES)

            print(f"  [OK] [{mode['label']}] tamamlandi -- {already + generated}/{EXAMPLES_PER_MODE}")

    # ── Özet ──────────────────────────────────────────────────────
    final_counts = count_by_mode(OUTPUT_FILE)
    total = sum(final_counts.values())
    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI -- Toplam: {total} ornek")
    for m, cnt in final_counts.items():
        label = MODES[m]["label"]
        bar = "#" * (cnt // 5) + "." * ((EXAMPLES_PER_MODE - cnt) // 5)
        print(f"  {label:<30} {cnt:>3}/{EXAMPLES_PER_MODE}  {bar}")
    print(f"\n  Dosya: {OUTPUT_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
