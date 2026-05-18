# 🗣️ Gemma Echo 🌍

**Real-time Turkish-to-English voice translation with intelligent model orchestration and voice cloning.**

Gemma Echo is a desktop AI assistant that transcribes spoken Turkish, translates it to English using a self-healing model cascade, and speaks the result back in the original speaker's cloned voice — all on consumer hardware, entirely offline when needed.

---

## 💻 Platform Support

> ⚠️ **Tested platform: Windows 11 + NVIDIA GPU (CUDA 13).**
>
> This is the **only officially supported configuration** at this time. The project has been developed and verified end-to-end on this stack only.
>
> Linux and macOS are **not currently tested** and may require manual adaptation:
> - **Linux + NVIDIA:** likely works after installing `libportaudio2` / `libasound2-dev` and selecting the matching `torch` CUDA wheel; the WASAPI loopback recorder (system-audio capture) is Windows-only.
> - **macOS (Apple Silicon):** requires switching the inference device from `cuda` to `mps`, building `llama-cpp-python` with `CMAKE_ARGS="-DLLAMA_METAL=on"`, and installing Tcl/Tk (`brew install python-tk`); not tested by the author.
> - **macOS (Intel) / Linux without NVIDIA:** CPU-only mode is achievable but slow; `torch==2.11.0+cu130` in `requirements.txt` must be replaced with the appropriate non-CUDA wheel.
>
> Pull requests adding tested cross-platform support are welcome.

---

## ✨ Features & Workflows

Gemma Echo is a multi-modal translation suite. The cascade described below powers **five** distinct workflows, each accessible from the main GUI:

| Mode | Input | Output | Use Case |
|------|-------|--------|----------|
| 🎙️ **Live** | Microphone (push-to-talk or VAD) or system loopback (WASAPI) | Streaming text + cloned-voice audio | Real-time conversation, meetings, live calls |
| 🎬 **Media — Dubbing** | Video file (MP4, MKV, MOV, AVI, WebM) | Dubbed video with cloned speaker voice | Re-voicing Turkish videos in English |
| 📝 **Media — Subtitling** | Video file | Soft `.srt` track or hard-burned cinematic subtitles | YouTube uploads, accessibility, deliverables |
| 📄 **Book / Document** | PDF, DOCX, TXT | Translated `.txt` (with optional layout-preserving `.docx`) | Academic papers, books, long-form documents |
| 📁 **File / Text** | Audio/video file or pasted text | Translated transcript | Bulk transcription, ad-hoc text translation |

All five modes share the same self-healing translation cascade (Cultural Map → Gemma 4 Cloud → Gemini 2.5 Flash → Gemma 4 Q4 local) and switch between cloud and offline operation transparently.

---

## 🏗️ System Architecture

```
Microphone / Video File
        |
        v
[ faster-whisper STT ]   <-- VAD-gated, 16kHz mono, language=tr
        |
        v
[ Cultural Map ]          <-- 130 idioms across 7 languages (TR/AR/DE/ES/FR/IT/JA), zero-latency exact match
        |
        | (no match)
        v
[ Gemma 4 26B  via Gemini API ]   <-- Primary: quality-first
        |
        | dynamic timeout (40-120s) or API error
        v
[ Gemini 2.5 Flash via Gemini API ] <-- Speed fallback
        |
        | all cloud paths failed / offline mode
        v
[ Gemma 4 Q4 GGUF (Local Inference Engine) ]  <-- Zero-dependency local bedrock
        |
        v
[ XTTS-v2 Voice Cloning ]   <-- Clones original speaker from reference wav
        |
        v
Audio Output / Dubbed Video
```

---

## 🧠 Self-Healing Model Cascade

| Layer | Model | Provider | Trigger |
|-------|-------|----------|---------|
| 0 | Cultural Map (130 entries × 7 languages) | Local | Idiom detected in source language |
| 1 | Gemma 4 26B (`gemma-4-26b-a4b-it`) | Gemini API | Default online path |
| 2 | Gemini 2.5 Flash | Gemini API | Layer 1 timeout / error |
| 3 | Gemma 4 Q4 GGUF | Local Inference Engine | Offline mode / all cloud layers failed |

The cascade is **self-healing**: any layer can fail silently. The next layer activates automatically within milliseconds. In practice, the system almost always resolves at Layer 1 or 2; Layer 3 exists so the system never goes down — even with no internet at all.

### 🤝 Why Gemma 4 at Both Ends?

Gemma 4 appears at Layer 1 (cloud, 26B full precision) and Layer 3 (local, Q4 quantized) by design:

- **Layer 1 (cloud)**: Maximum translation quality for real-time conversation. The 26B parameter model handles complex Turkish grammar, pronoun resolution, and domain-specific vocabulary.
- **Layer 3 (local)**: The privacy-preserving, cost-free, network-independent fallback. When quotas expire, internet drops, or data must stay on-device, `gemma-4-q4.gguf` takes over with zero configuration change — no restart, no user intervention.

This creates a **quality-symmetric, fully Gemma-native** cascade: every translation layer in the system is a Google Gemma model.

---

## ⚡ VRAM Optimization & Performance

Consumer GPUs (8–12 GB VRAM) cannot hold all models simultaneously. Gemma Echo uses three strategies to make this work:

### 🐢 1. Lazy Loading
Models are not loaded at startup. The local Gemma 4 Q4 is loaded only when first needed (offline mode or video dubbing). XTTS-v2 loads only when TTS mode is switched to offline/GPU.

```python
# translator.py — load_local_model() called on first offline request
# (Llama is the entry-point class of llama-cpp-python, a generic C++ inference
#  engine for GGUF models. Here it is used to run a Google Gemma 4 weight file.)
self.local_llm = Llama(model_path="./models/gemma-4-q4.gguf", n_gpu_layers=-1)
```

### 🥷 2. Background Preloading (Ambush Mode)
When the user is in online mode, XTTS-v2 silently preloads into system RAM on a daemon thread. If the user switches to offline mode, the model is already warm — no perceived latency.

```python
# synthesizer.py — starts while online mode is active
synthesizer.preload_xtts_background(use_gpu=False)
```

### 🔄 3. Hot-Swap with Cache Eviction
Switching between GPU and CPU modes triggers controlled VRAM eviction before loading the new configuration, preventing CUDA OOM errors.

```python
# synthesizer.py — offload_xtts()
del self.xtts_model
gc.collect()
torch.cuda.empty_cache()   # VRAM fully released before next load
```

All three strategies compose: the system can run on a single RTX 3060 Ti (8 GB VRAM) handling real-time translation in online mode and full video dubbing in offline mode, simply by hot-swapping which model occupies VRAM at any given moment.

---

## ⚙️ Backend Configuration

The Settings page exposes an STT × LLM × TTS matrix. Each axis can be picked independently, producing 36+ valid combinations. The presets below are the most common; **`Custom`** lets you mix any STT engine with any translation backend and any TTS sink.

| Preset | STT | Translation | TTS | Internet |
|--------|-----|-------------|-----|----------|
| 🟢 **Online (default)** | faster-whisper local-GPU | Gemma 4 26B → Gemini 2.5 Flash | ElevenLabs Turbo | Required |
| ☁️ **Cloud STT accelerator** | Groq Whisper-large-v3 *or* Deepgram Nova | Gemma 4 26B → Gemini 2.5 Flash | ElevenLabs Turbo | Required |
| 🛡️ **Offline (CPU)** | faster-whisper CPU | Gemma 4 Q4 GGUF (CPU) | XTTS-v2 CPU | Not needed |
| 🚀 **Offline (GPU)** | faster-whisper local-GPU | Gemma 4 Q4 GGUF (GPU) | XTTS-v2 GPU | Not needed |
| ⚖️ **Hybrid (recommended)** | faster-whisper local-GPU | Gemma 4 26B → Gemini 2.5 Flash → Gemma 4 Q4 (auto-fallback) | XTTS-v2 GPU | Optional |
| 🎥 **Video Dubbing** | faster-whisper medium (timestamped) + Demucs vocal split | Gemma 4 26B → Gemini 2.5 Flash → Gemma 4 Q4 | XTTS-v2 (voice clone) | Optional |
| 🧩 **Custom** | any of the above | any of the above | any of the above | depends |

---

## 🎬 Video Dubbing Pipeline

Gemma Echo can dub a Turkish video into English, preserving the original speaker's voice:

```
1. ffmpeg          Extract 16kHz mono WAV from video
2. faster-whisper  Timestamped transcription (start/end/text per segment)
3. Gemma 4 Q4      Translate each segment locally (no API cost, no rate limits)
4. XTTS-v2         Extract speaker reference from longest clean segment (≤8s)
5. XTTS-v2         get_conditioning_latents() computed once, reused per segment
6. XTTS-v2         inference() per segment → English speech in cloned voice
7. ffmpeg atempo   Time-stretch English audio to fit original segment duration
8. ffmpeg          Mux new audio track into original video (stream copy, no re-encode)
```

Output: `<source_video>_dubbed.mp4`

The dubbing pipeline uses **only local models** (Steps 3–6), making it suitable for sensitive content and long videos without API cost concerns.

### 📝 Subtitling (alternative to dubbing)

Same Steps 1–3 (extract → transcribe → translate), then a different finish:

```
4. SRT writer       Sentence-aware splitting, max 42 chars × 2 lines per cue
5a. soft mux         ffmpeg copies the .srt as a selectable subtitle track (no re-encode)
5b. hard burn-in     ffmpeg subtitles filter renders cinematic-style captions
                    onto the video frames (white Arial bold, black outline,
                    soft shadow — Netflix-style; no opaque background box)
```

Output: `<source_video>_subtitled.mp4` (soft) or `<source_video>_burned.mp4` (hard).

---

## 📄 Document Translation Pipeline

Long-form translation (PDF / DOCX / TXT) is engineered separately from the live conversational path. The naïve approach — feed the whole document to one LLM call — fails on terminology consistency, exceeds context windows, and produces drift across chapters. Gemma Echo uses an 8-stage pipeline:

```
1. pdfplumber/python-docx   Extract text while preserving paragraph boundaries
2. Sliding Window           Group paragraphs into ~400-word chunks with 15% overlap
3. Term Extraction          First-pass scan extracts proper nouns, citations,
                            domain terms → builds a per-document glossary
4. Cultural Map             Pre-translate idioms and fixed expressions (zero LLM cost)
5. Translation Cascade      Each chunk: Gemma 4 Cloud → Gemini 2.5 Flash → Gemma 4 Q4
                            (per-paragraph fallback if a chunk fails the boundary check)
6. Rolling Summary          After every 5 chunks, regenerate a 2-sentence summary
                            of the document so far → fed back as context to subsequent
                            chunks (long-document coherence, pronoun resolution)
7. Reassembly               Concatenate translated chunks; deduplicate the overlap
8. Output writer            Plain `.txt` (always) or layout-preserving `.docx`
                            (optional, retains paragraph structure)
```

Output: `<source>_<lang>.txt` and/or `<source>_<lang>.docx`.

The glossary (Stage 3) and rolling summary (Stage 6) are the difference between machine-translation slop and a publishable draft. A 50-page paper translated with this pipeline maintains consistent terminology end-to-end without any human pre-processing.

---

## 🚀 Setup & Installation

### 📋 Requirements

- Python 3.11 (3.11.x recommended — Coqui XTTS is verified on this line)
- NVIDIA GPU driver supporting CUDA 13.0 (driver 580+ on Windows; PyTorch ships its own CUDA runtime)
- ffmpeg ≥ 6.0 in PATH (`ffmpeg -version` should resolve)
- ~12 GB free disk space (≈4 GB Gemma 4 GGUF, ≈2 GB XTTS-v2, the rest for venv)

### 📦 Install

```bash
git clone https://github.com/unsemhizm/gemma_echo.git
cd gemma_echo
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 🧠 Download Gemma 4 Local Model

Gemma Echo expects a **GGUF-quantized Gemma 4** weight file at `models/gemma-4-q4.gguf`. This file is **not bundled with the repository** (it is ~4 GB and outside Git LFS limits) and must be downloaded manually.

#### 📥 Recommended — Direct Browser Download

This is the safest path on Windows because it does **not** touch your Python environment (the new Hugging Face CLI requires `huggingface_hub>=1.0`, which is incompatible with the `transformers==4.38.2` / `tokenizers==0.15.2` versions this project pins for translation stability — upgrading the hub library will break local inference).

1. Open [Gemma 4 26B GGUF on Hugging Face](https://huggingface.co/google/gemma-4-26b-a4b-it-qat-q4_0-gguf).
2. Sign in with a free Hugging Face account and accept the **Gemma Terms of Use** once.
3. Download the `gemma-4-26b-a4b-it-q4_0.gguf` file (~4 GB).
4. Place it in the project's `models/` folder and rename it to `gemma-4-q4.gguf`.

#### 💻 Optional — `hf` CLI (advanced users only)

The legacy `huggingface-cli download …` command is **deprecated** as of `huggingface_hub` 1.x and is replaced by `hf`. If you already have `hf` on your `PATH` and a valid token (`hf auth login`), you can do:

```bash
hf download google/gemma-4-26b-a4b-it-qat-q4_0-gguf gemma-4-26b-a4b-it-q4_0.gguf --local-dir ./models
# Then rename to the path the project expects:
ren .\models\gemma-4-26b-a4b-it-q4_0.gguf gemma-4-q4.gguf     # PowerShell / Windows
# mv  ./models/gemma-4-26b-a4b-it-q4_0.gguf ./models/gemma-4-q4.gguf   # macOS / Linux
```

> **Do not** run `pip install -U "huggingface_hub[cli]"` inside this project's `venv`. It will silently upgrade `huggingface_hub` past 1.0 and break `transformers 4.38.2` + `tokenizers 0.15.2`. If you accidentally did so, restore the pinned version with:
>
> ```bash
> pip install huggingface_hub==0.36.2
> ```

#### 📂 Expected Final Layout

```
models/
  gemma-4-q4.gguf      # ~4 GB, Q4_K_M quantization (Google Gemma 4 26B)
```

Any other Q4 GGUF build of Gemma 4 works as long as the final filename is `gemma-4-q4.gguf` (this is the path hard-coded in `llm/translator.py`).

**XTTS-v2** is downloaded automatically by the Coqui `TTS` library on first offline-TTS use (~2 GB into `~/.local/share/tts/` or the Windows equivalent). No manual step required.

> **Note for jury / first-time users:** if you skip this step the project will still launch and online translation (Gemini API) will work, but **Offline mode, Hybrid auto-fallback, and Video Dubbing will fail** with a `models/gemma-4-q4.gguf not found` error in the logs.

### 🔑 Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key            # Translation (Gemma 4 + Gemini Flash)
ELEVENLABS_API_KEY=your_elevenlabs_api_key    # High-quality online TTS (optional)
GROQ_API_KEY=your_groq_api_key                # Cloud Whisper-large-v3 STT acceleration (optional)
DEEPGRAM_API_KEY=your_deepgram_api_key        # Cloud STT fallback (optional)
```

The app runs fully offline without any of these keys (using only local Gemma 4 Q4 + faster-whisper + XTTS-v2). The Groq and Deepgram keys only accelerate cloud speech-to-text on low-VRAM machines; they are **not used for translation**.

---

## 🎮 Usage

```bash
# Launch GUI
python -m gui.app

# Or directly
python gui/app.py
```

---

## 🛠️ Technology Stack

| Component | Library |
|-----------|---------|
| GUI | CustomTkinter |
| STT (local) | faster-whisper (CTranslate2 backend) |
| STT (cloud accelerator, optional) | Groq Whisper-large-v3, Deepgram Nova |
| VAD (voice activity detection) | webrtcvad |
| Translation (cloud) | Google Gemini API — Gemma 4 26B → Gemini 2.5 Flash |
| Translation (local) | Gemma 4 Q4 GGUF via `llama-cpp-python` |
| TTS (online) | ElevenLabs |
| TTS (offline / voice cloning) | Coqui XTTS-v2 † |
| Vocal/instrumental separation (dubbing) | Demucs htdemucs (Meta, MIT) |
| Document parsing (book translation) | pdfplumber, python-docx |
| Audio I/O | sounddevice, soundfile, soundcard (WASAPI loopback) |
| Video processing | ffmpeg (CLI subprocess) |

> **† TTS Engine Licensing Disclaimer.** The core orchestration framework of Gemma Echo is licensed under Apache 2.0. However, the **default** offline TTS engine (Coqui XTTS-v2) uses model weights licensed under the **Coqui Public Model License (Non-Commercial)**. Gemma Echo provides the architecture to integrate any TTS engine. For commercial deployment, users must replace the XTTS-v2 model weights with a commercially permissive alternative (e.g., VITS, Piper) or obtain a commercial license from Coqui GmbH. The Apache 2.0 license of Gemma Echo itself is unaffected.

---

## 🗺️ Roadmap & Experimental Track

The [`experimental/`](experimental/) directory contains research-and-development work that is **not part of the current competition submission** but documents the project's planned post-launch personalization track:

- **Mode-aware Japanese fine-tuning** — A 500-sample synthetic dataset (5 stylistic modes: emergency, official, streamer, casual, literary) generated with the Gemini API, plus a complete Unsloth + LoRA training script ready to run on a Kaggle T4 notebook. Training has not yet been executed; the dataset and pipeline are provided as a reproducible blueprint. See [`experimental/kaggle_finetune/README.md`](experimental/kaggle_finetune/README.md).

These materials demonstrate the engineering direction for adding Japanese support and style-aware translation in future versions, and are released under the same Apache 2.0 license as the rest of the project.

---

## 👨‍💻 Developer & License

Developed by **Yusuf Semih Öksüzoğlu** for the **Google Gemma AI Hackathon 2026**.

📝 **License:** Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2026 Yusuf Semih Öksüzoğlu
