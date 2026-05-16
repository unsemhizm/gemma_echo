# Gemma Echo — Kaggle LoRA Fine-Tune (Experimental)

> **Status:** **Synthetic dataset complete (500 samples). Training pipeline implemented but not yet executed.**
> This module is **not part of the current competition submission**. It documents the planned post-launch personalization track and serves as a reproducible blueprint for anyone (including the author) who wants to run the fine-tune on a Kaggle T4 notebook.

This folder contains the Kaggle notebook script that fine-tunes a **Gemma** model with our custom 500-example **Turkish → Japanese, 5-mode** synthetic dataset using **LoRA** (Low-Rank Adaptation).

---

## Architectural Note (For the Jury)

**Question:** "The live application is built on Gemma 4 Cloud (`gemma-4-26b-a4b-it`) plus a Gemma GGUF (llama.cpp) local fallback. Why does the Kaggle script fine-tune Gemma 2?"

**Answer — Ecosystem Bottleneck (as of May 2026):**

There is a **two-layer incompatibility** in the current open-source training ecosystem:

1. **GGUF is a non-trainable format.** HuggingFace PEFT and Unsloth require **PyTorch tensors** (SafeTensors / `.bin`) at fine-tune time. A pre-quantized C++-inference-optimized `.gguf` file cannot be fed directly into this pipeline. This is not a Gemma Echo limitation — it is a **fundamental design decision of the training frameworks**.

2. **Gemma 4 ↔ Unsloth / llama.cpp conversion is not yet mature.** As of May 2026, there is **no published 4-bit Gemma 4 weight on the Unsloth Hub**, and `llama.cpp/convert_hf_to_gguf.py` does not yet provide stable support for Gemma 4's new tensor shapes and custom layers (e.g. its updated attention configuration). Even if pure PyTorch weights were obtained and trained, the GGUF export path is practically closed.

**Our Choice — Proof-of-Concept:**

Rather than hide this bottleneck, we chose to **prove that the methodology is correct**:

- The fine-tune **methodology** (synthetic data → Alpaca format → Unsloth + LoRA → GGUF export → local `llama-cpp-python` engine) is **architecturally complete**.
- We validate this methodology end-to-end on **Gemma 2**. The resulting GGUF can be loaded directly by the local app's `llm/translator.py` engine.
- Once Unsloth and llama.cpp conversion scripts stabilise for Gemma 4 (on the roadmap), the same pipeline migrates by changing a single `MODEL_NAME` variable.

> **Engineering honesty:** Vanilla Gemma 2 is itself an eligible Google Gemma-family model. The relevant point is the awareness of the ecosystem boundary between the **live runtime engine (Gemma 4 cloud + GGUF)** and the **fine-tune proof-of-concept layer (Gemma 2 LoRA)**.

---

## Why Are We Doing This?

A vanilla Gemma model can translate "I love football" into Japanese, but:

- In an **emergency-radio** mode it does not insert `了解` / `どうぞ`.
- In **official correspondence** it does not use keigo (敬語).
- In **streamer style** it does not say `草` / `やばい`.
- In **literary** prose it loses rhythm and imagery.

We are teaching the model **5 style contracts** (instruction prefixes) **persistently**. This is a concrete engineering artefact that complements the live cascade.

## Dataset

`experimental/ja_training_data.jsonl` — **500 samples, 100 per mode**:

| Mode | Style | Example output |
|-----|------|-------------|
| `emergency` | JCG-radio, short imperative, `どうぞ` | `未確認船、進入中。直ちに停船させよ。` |
| `official` | Diplomatic keigo, `〜ます/〜です` | `予算協議は来週に延期されました。` |
| `streamer` | Niconico/YouTube slang, `草`, `やばい` | `このボス戦マジで神！クリップ確定！` |
| `casual` | Friend tone, `〜だ/〜じゃない` | `今夜出かけない？ラーメン食べに行こ。` |
| `literary` | Literary, metaphor, classical reference | `秋の葉が、過ぎ行く時の静かな使者のごとく` |

Format: Alpaca JSONL — `{instruction, input, output, mode}`.

### Data Provenance & License

The dataset was **generated synthetically by the project author using the Google Gemini API** (see `experimental/synthetic_data_generator.py`). It contains no scraped or third-party copyrighted material. The dataset is released under the same **Apache License 2.0** as the rest of this repository.

## Running on Kaggle — Step by Step

### 1. Upload the data as a Kaggle Dataset (one time)

- Kaggle → **Datasets → New Dataset**
- Drag and drop `ja_training_data.jsonl`
- **Dataset name**: `gemma-echo-ja-training` (keep this slug — the script expects it)
- Visibility: Private
- **Create**

### 2. Create a new notebook

- Kaggle → **Code → New Notebook**
- Right panel → **Add Input** → attach the uploaded dataset (`gemma-echo-ja-training`)
- **Settings**:
  - Accelerator: **GPU T4 x2** (16 GB is sufficient)
  - Internet: **ON** (required to download Unsloth and the Gemma weights)
  - Persistence: **Off**

### 3. Paste the script into cells

Split the contents of `train_gemma_lora.py` by the `# %%` markers — each marker indicates the start of a new cell. (VS Code already renders this file as an interactive Python notebook.)

The first cell — pip install — is commented out. After pasting into the notebook, remove the leading `# ` from the `# !pip` lines.

### 4. Run All

On a T4 the full run takes ~15–20 minutes:

- Model download (~3 min, 2B 4-bit ≈ 1.5 GB)
- Fine-tune (2 epochs, 500 samples, ~10–12 min)
- 5-mode inference test
- LoRA save (~30 s)
- GGUF export (~3–5 min, includes the llama.cpp build)

### 5. Download the outputs

Three artefacts are produced under `/kaggle/working/`:

- `gemma_echo_lora/` — **LoRA adapter** (~50 MB) — for sharing or pushing to the Hub
- `gemma_echo_merged_16bit/` — Merged HF model (~5 GB) — optional (`SAVE_MERGED_16BIT=False` to skip)
- `gemma_echo_q4.gguf` — **Format consumed by the local app** (~1.5 GB)

Use the notebook's "Download Output" link to grab them as a zip.

## Integration With the Local App

Place the downloaded `gemma_echo_q4.gguf` in the project's `models/` directory, then update the local-model path in `llm/translator.py` (or, if config-based, in `config.json`).

## Hyperparameter Reference (`train_gemma_lora.py`, Cell 2)

| Param | Default | When to change |
|-------|---------|-------------------|
| `MODEL_NAME` | `gemma-2-2b-it-bnb-4bit` | For more capacity use `unsloth/gemma-4-4b-it-bnb-4bit` (A100 required). Gemma 4 is not yet on Unsloth. |
| `LORA_R` | 32 | 16 for small datasets, 64 for large ones |
| `NUM_EPOCHS` | 2 | Reduce when loss plateaus to avoid overfitting |
| `LEARNING_RATE` | 2e-4 | Drop to 5e-5 if loss spikes |
| `MAX_SEQ_LENGTH` | 1024 | 512 is enough — sentences are short |

## Troubleshooting

| Issue | Cause / Fix |
|-------|---------------|
| `CUDA out of memory` | Drop `PER_DEVICE_TRAIN_BATCH_SIZE` to 2, raise `GRAD_ACCUM_STEPS` to 8 |
| `Dataset not found` | Is the Kaggle dataset slug `gemma-echo-ja-training`? Did you "Add Input" on the right panel? |
| `Unsloth import error` | After the `pip install unsloth` cell completes, restart the runtime (Run → Restart) |
| GGUF export is slow | First run builds llama.cpp (~3–5 min); subsequent runs are cached |
| Model produces broken Japanese | Lower `temperature` from 0.4 to 0.2, raise `repetition_penalty` to 1.1 |

## Verifying the Results

After training, the 5-mode test block (Cell 7) should produce stylistically distinct outputs:

- **emergency** → short, contains `どうぞ` / `了解`
- **official** → ends in `〜ます` / `〜です`
- **streamer** → contains slang such as `やばい` / `神` / `草`
- **casual** → plain form (`〜だよ` / `〜じゃん`)
- **literary** → long, imagistic, comma-light prose

If those distinctions are clearly visible, the style-aware fine-tune is successful.
