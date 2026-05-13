"""
═══════════════════════════════════════════════════════════════════════════
GEMMA ECHO — Türkçe→Japonca Stil-Aware LoRA Fine-Tune (Kaggle Notebook)
═══════════════════════════════════════════════════════════════════════════
Hedef:  Gemma 2 modelini bizim 500 örnekli 5-modlu sentetik veri setiyle
        (emergency / official / streamer / casual / literary) LoRA ile
        Türkçe→Japonca çevirinin "stil-bilinçli" hâline getirmek.

⚠️  MİMARİ BİLİNÇ — Neden Gemma 2 (Gemma 4 değil)?
    ─────────────────────────────────────────────────────────────────────
    Mayıs 2026 itibarıyla iki teknolojik darboğaz vardır:

    1. GGUF EĞİTİLEMEZ:  HuggingFace PEFT/Unsloth pipeline'ı PyTorch tensörü
       (SafeTensors) bekler. Önceden kuantize edilmiş .gguf dosyası fine-tune
       sürecine doğrudan beslenemez.

    2. Gemma 4+ ↔ llama.cpp:  llama.cpp/convert_hf_to_gguf.py scripti, Gemma 4
       mimarisinin yeni tensor şekillerini ve özel layer'larını (yeni sliding
       window attention) henüz tam stabil desteklemiyor.

    Bu yüzden bu Kaggle pipeline'ı, jüriye METODOLOJİMİZİ uçtan uca
    kanıtlamak için (Proof-of-Concept) Gemma 2 üzerinde inşa edilmiştir.
    Pipeline tamamen mimari olarak hazır; ekosistem (Unsloth + llama.cpp)
    Gemma 4'ü stabilize edince sadece MODEL_NAME değişkeni güncellenerek
    migrate edilebilir. Detaylı açıklama: ./README.md (Mimari Bilinç Notu).
    ─────────────────────────────────────────────────────────────────────

Donanım:    Kaggle "GPU T4 x2"  (16 GB VRAM yeter; 4-bit + LoRA)
            Tahmini eğitim süresi:  ~12-18 dk (2 epoch, 500 örnek)

Önkoşul (Kaggle Notebook tarafında):
  1. Sağdaki "Add Input" → "Upload" → ja_training_data.jsonl yükle.
     Dataset adı: "gemma-echo-ja-training"  (slug aynı kalsın).
  2. Settings → Accelerator: GPU T4 x2  /  Internet: ON
  3. Bu .py dosyasını yeni bir notebook'a yapıştır (her '# %%' yeni hücre).

Çıktılar (Kaggle):
  /kaggle/working/gemma_echo_lora/         ← LoRA adapter (küçük, ~50MB)
  /kaggle/working/gemma_echo_merged_16bit/ ← merged model (HF)
  /kaggle/working/gemma_echo_q4.gguf       ← yerel GGUF (uygulamanın kullandığı)

Yerel kullanım:
  GGUF dosyasını indir → llm/translator.py içindeki LOCAL_MODEL_PATH'e koy.

═══════════════════════════════════════════════════════════════════════════
"""

# %% [markdown]
# ## 1. Bağımlılık Kurulumu
# Unsloth Kaggle T4 üzerinde 2x hızlı fine-tune sağlar; 4-bit kuantize ile
# 2B model 5 GB VRAM'e sığar.

# %%
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q --no-deps "trl<0.9.0" peft accelerate bitsandbytes
# !pip install -q sentencepiece protobuf datasets

# %% [markdown]
# ## 2. Konfigürasyon — Tek Yerden Tüm Hyperparams
# Modeli değiştirmek istiyorsan sadece `MODEL_NAME` değişkenini değiştir.

# %%
import os
import json
import torch
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────
# MODEL & VERİ
# ────────────────────────────────────────────────────────────────────────
# Alternatifler (Mayıs 2026 itibarıyla Unsloth Hub'da hazır olanlar):
#   "unsloth/gemma-2-2b-it-bnb-4bit"       (en stabil, T4 dostu, GGUF export sağlam) ← default
#   "unsloth/gemma-4-2b-it-bnb-4bit"       (daha yeni mimari, en küçük)
#   "unsloth/gemma-4-4b-it-bnb-4bit"       (daha güçlü; A100/L4 öneririm)
# NOT: Gemma 4 için Unsloth Hub modeli + stabil GGUF export zinciri henüz YOK.
#      Ekosistem hazır olunca aşağıdaki satırı "unsloth/gemma-4-...-bnb-4bit" ile
#      değiştirmek migrate için yeterli olacak (bkz. README "Mimari Bilinç Notu").
MODEL_NAME      = "unsloth/gemma-2-2b-it-bnb-4bit"
MAX_SEQ_LENGTH  = 1024              # cümleler kısa; 1024 fazlasıyla yeter
LOAD_IN_4BIT    = True              # 4-bit quant — VRAM tasarrufu

# Kaggle dataset path (yüklenen JSONL)
DATA_PATH = "/kaggle/input/gemma-echo-ja-training/ja_training_data.jsonl"
# Lokal test için fallback (Kaggle dışında çalıştırırsan):
if not Path(DATA_PATH).exists():
    DATA_PATH = str(Path(__file__).resolve().parent.parent / "ja_training_data.jsonl")

# ────────────────────────────────────────────────────────────────────────
# LoRA HYPERPARAMS
# ────────────────────────────────────────────────────────────────────────
LORA_R          = 32                # rank — 16'dan büyük, capacity yeter
LORA_ALPHA      = 64                # 2*r yaygın pratik
LORA_DROPOUT    = 0.05
TARGET_MODULES  = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ────────────────────────────────────────────────────────────────────────
# EĞİTİM HYPERPARAMS
# ────────────────────────────────────────────────────────────────────────
NUM_EPOCHS                  = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRAD_ACCUM_STEPS            = 4     # effective batch = 4 × 4 = 16
LEARNING_RATE               = 2e-4  # LoRA için yüksek-low arası standart
WARMUP_RATIO                = 0.03
WEIGHT_DECAY                = 0.01
LOGGING_STEPS               = 5
SAVE_STEPS                  = 50
SEED                        = 3407

# Çıktı yolları
OUTPUT_LORA   = "/kaggle/working/gemma_echo_lora"
OUTPUT_MERGED = "/kaggle/working/gemma_echo_merged_16bit"
OUTPUT_GGUF   = "/kaggle/working/gemma_echo_q4.gguf"

# bf16 vs fp16 — Ampere+ (sm_80) bf16 destekler; T4 (sm_75) fp16 zorunlu
COMPUTE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_FP16 = not COMPUTE_BF16
print(f"GPU       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK'}")
print(f"Precision : {'bf16' if COMPUTE_BF16 else 'fp16'}")
print(f"Veri      : {DATA_PATH}")

# %% [markdown]
# ## 3. Modeli Yükle (Unsloth, 4-bit)
# Unsloth, peft+bitsandbytes işlevini kendi optimize edilmiş yolundan yapar.

# %%
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    dtype           = None,             # otomatik (bf16/fp16)
    load_in_4bit    = LOAD_IN_4BIT,
)

# LoRA adapter'ı yapıştır
model = FastLanguageModel.get_peft_model(
    model,
    r                            = LORA_R,
    lora_alpha                   = LORA_ALPHA,
    lora_dropout                 = LORA_DROPOUT,
    target_modules               = TARGET_MODULES,
    bias                         = "none",
    use_gradient_checkpointing   = "unsloth",   # ekstra VRAM tasarrufu
    random_state                 = SEED,
    use_rslora                   = False,
    loftq_config                 = None,
)

print(f"\nTrainable params: "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown]
# ## 4. Dataset — JSONL → Gemma Chat Template
# Her örneği Gemma'nın `<start_of_turn>` formatına çeviriyoruz.
# `instruction` (sistem talimatı) + `input` (Türkçe) → `output` (Japonca).

# %%
from datasets import load_dataset

# JSONL'i HF Dataset olarak yükle
raw = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"Toplam örnek    : {len(raw)}")
print(f"Örnek anahtarlar: {list(raw.column_names)}")

# Mod dağılımı kontrol — bilgi amaçlı
from collections import Counter
mode_dist = Counter(raw["mode"])
print(f"Mod dağılımı     : {dict(mode_dist)}")

# Train/eval split — küçük dataset için %95/5
split = raw.train_test_split(test_size=0.05, seed=SEED)
train_ds = split["train"]
eval_ds  = split["test"]
print(f"Train / Eval     : {len(train_ds)} / {len(eval_ds)}")


def format_to_gemma_chat(example):
    """Alpaca formatını Gemma chat template'ine dönüştür.

    Mod bilgisini instruction'a entegre etmiyoruz — instruction zaten
    her modun stil sözleşmesini içeriyor (synthetic_data_generator.py'de
    tanımlandığı gibi).
    """
    user_msg = (
        f"{example['instruction']}\n\n"
        f"Türkçe: {example['input']}\n"
        f"Japonca:"
    )
    full_text = (
        f"<start_of_turn>user\n{user_msg}<end_of_turn>\n"
        f"<start_of_turn>model\n{example['output']}<end_of_turn>"
    )
    return {"text": full_text}


train_ds = train_ds.map(format_to_gemma_chat, remove_columns=raw.column_names)
eval_ds  = eval_ds.map(format_to_gemma_chat, remove_columns=raw.column_names)

# Sağlık kontrol — ilk örnek
print("\n─── Format Örneği ───")
print(train_ds[0]["text"][:600], "...")

# %% [markdown]
# ## 5. SFTTrainer Konfigürasyonu

# %%
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_ds,
    eval_dataset       = eval_ds,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    dataset_num_proc   = 2,
    packing            = False,   # küçük dataset; packing dağılımı bozar
    args = TrainingArguments(
        output_dir                  = "/kaggle/working/checkpoints",
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        warmup_ratio                = WARMUP_RATIO,
        num_train_epochs            = NUM_EPOCHS,
        learning_rate               = LEARNING_RATE,
        bf16                        = COMPUTE_BF16,
        fp16                        = COMPUTE_FP16,
        logging_steps               = LOGGING_STEPS,
        optim                       = "adamw_8bit",
        weight_decay                = WEIGHT_DECAY,
        lr_scheduler_type           = "cosine",
        seed                        = SEED,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 2,
        eval_strategy               = "steps",
        eval_steps                  = SAVE_STEPS,
        report_to                   = "none",        # Kaggle'da W&B kapalı
    ),
)

# %% [markdown]
# ## 6. Eğitim
# T4 üzerinde 500 örnek × 2 epoch ≈ 12-18 dakika.

# %%
# VRAM durumu (başlangıç)
def _vram(label):
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM-{label}] {used:.2f} GB / {total:.2f} GB")

_vram("PRE")
trainer_stats = trainer.train()
_vram("POST")

print("\n─── Eğitim Özeti ───")
print(f"Toplam adım     : {trainer_stats.global_step}")
print(f"Eğitim süresi   : {trainer_stats.metrics.get('train_runtime', 0):.1f}s")
print(f"Train loss      : {trainer_stats.metrics.get('train_loss', 0):.4f}")

# %% [markdown]
# ## 7. Inference Testi — Her 5 Modda Bir Örnek
# Eğitilen modelin her stilde gerçekten farklı çıktı üretip üretmediğini doğrula.

# %%
FastLanguageModel.for_inference(model)   # 2x hızlı çıkarım modu

TEST_CASES = [
    ("emergency",
     "Sen bir Japon Sahil Güvenlik telsiz operatörüsün. "
     "Aşağıdaki Türkçe acil durum mesajını otantik JCG telsiz jargonuyla Japoncaya çevir. "
     "Kısa, net ve komuta dayalı cümleler kur.",
     "Tüm birimler dikkat, sektör 4'te yardım sinyali alındı."),
    ("official",
     "Sen üst düzey bir Japon diplomat veya akademisyensin. "
     "Aşağıdaki Türkçe resmi metni tam anlamıyla resmi Japonca keigo üslubuyla çevir.",
     "Bütçe görüşmeleri önümüzdeki haftaya ertelenmiştir."),
    ("streamer",
     "Sen Japonya'nın en popüler oyun yayıncısısın. "
     "Aşağıdaki Türkçe yayıncı ifadesini Japon internet ve oyun kültürüne özgü "
     "güncel argo ile çevir. İngilizce argo kelimelerini birebir çevirme.",
     "Bu boss savaşı kesinlikle efsane geçti, klip kesinim!"),
    ("casual",
     "Sen yakın bir Japon arkadaşınla konuşuyorsun. "
     "Aşağıdaki Türkçe günlük ifadeyi samimi, rahat ve doğal Japonca konuşma "
     "diline çevir. Resmi dil kullanma.",
     "Bu akşam dışarı çıkmak ister misin, ramen yiyelim?"),
    ("literary",
     "Sen yetenekli bir Japon edebi çevirmensin. "
     "Aşağıdaki Türkçe edebi/şiirsel ifadeyi duygusal tonu ve imgelerini "
     "koruyarak zarif ve akıcı Japoncaya çevir.",
     "Sonbahar yaprakları, geçen zamanın sessiz haberçileri gibi düşüyordu."),
]

def generate(instruction: str, source_tr: str, max_new=128) -> str:
    user_msg = f"{instruction}\n\nTürkçe: {source_tr}\nJaponca:"
    prompt = (
        f"<start_of_turn>user\n{user_msg}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens   = max_new,
        do_sample        = True,
        temperature      = 0.4,
        top_p            = 0.9,
        repetition_penalty = 1.05,
        pad_token_id     = tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    # <end_of_turn> sonrasını kes
    if "<end_of_turn>" in text:
        text = text.split("<end_of_turn>")[0]
    return text.strip()


print("\n══════ STIL-AWARE ÇEVİRİ TESTİ ══════\n")
for mode, instr, tr in TEST_CASES:
    ja = generate(instr, tr)
    print(f"[{mode.upper():<10}] TR: {tr}")
    print(f"            JA: {ja}\n")

# %% [markdown]
# ## 8. Kayıt — LoRA + Merged + GGUF
# 3 farklı çıktı:
#   - **LoRA adapter** (küçük, base model + bunu yükle): paylaşmak için ideal
#   - **Merged 16-bit**: HF Hub'a push etmek istersen
#   - **GGUF Q4_K_M**: yerel uygulamamızın (`llama-cpp-python`) kullandığı format

# %%
# 1) LoRA adapter
model.save_pretrained(OUTPUT_LORA)
tokenizer.save_pretrained(OUTPUT_LORA)
print(f"[OK] LoRA adapter   → {OUTPUT_LORA}")

# 2) Merged 16-bit (HF format) — opsiyonel; disk maliyeti yüksek (~5GB)
SAVE_MERGED_16BIT = False
if SAVE_MERGED_16BIT:
    model.save_pretrained_merged(
        OUTPUT_MERGED, tokenizer,
        save_method="merged_16bit",
    )
    print(f"[OK] Merged 16-bit  → {OUTPUT_MERGED}")

# 3) GGUF (Q4_K_M) — uygulamanın yerel modeli
SAVE_GGUF = True
if SAVE_GGUF:
    # Unsloth llama.cpp'yi otomatik clone+build eder. ~3-4 dk sürebilir.
    model.save_pretrained_gguf(
        OUTPUT_GGUF.replace(".gguf", ""),  # klasör adı; içine .gguf yazar
        tokenizer,
        quantization_method = "q4_k_m",
    )
    print(f"[OK] GGUF Q4_K_M    → {OUTPUT_GGUF}*")

# %% [markdown]
# ## 9. (Opsiyonel) HuggingFace Hub'a Push
# Kaggle Secrets'a `HF_TOKEN` eklersen aşağıyı aç.

# %%
PUSH_TO_HUB = False
if PUSH_TO_HUB:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    HF_REPO  = "<your-username>/gemma-echo-tr-ja-lora"
    model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"[OK] Pushed → https://huggingface.co/{HF_REPO}")

# %% [markdown]
# ## 10. İndirme Talimatı
# Kaggle çıktıları otomatik olarak `/kaggle/working/` altında zip'lenip
# notebook çıktısı olarak indirilebilir hale gelir.
# `gemma_echo_q4.gguf` dosyasını yerel `models/` klasörüne koy ve
# `llm/translator.py` içinde GGUF path'ini güncelle.
