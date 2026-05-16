# Gemma Echo

**Real-time Turkish-to-English voice translation with intelligent model orchestration and voice cloning.**

Gemma Echo is a desktop AI assistant that transcribes spoken Turkish, translates it to English using a self-healing model cascade, and speaks the result back in the original speaker's cloned voice — all on consumer hardware, entirely offline when needed.

---

## Architecture

```
Microphone / Video File
        |
        v
[ faster-whisper STT ]   <-- VAD-gated, 16kHz mono, language=tr
        |
        v
[ Cultural Map ]          <-- 50+ Turkish idioms, zero-latency exact match
        |
        | (no match)
        v
[ Gemma 4 26B  via Gemini API ]   <-- Primary: quality-first
        |
        | timeout (8s) or API error
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

## Model Cascade

| Layer | Model | Provider | Trigger |
|-------|-------|----------|---------|
| 0 | Cultural Map (dictionary) | Local | Turkish idiom detected |
| 1 | Gemma 4 26B (`gemma-4-26b-a4b-it`) | Gemini API | Default online path |
| 2 | Gemini 2.5 Flash | Gemini API | Layer 1 timeout / error |
| 3 | Gemma 4 Q4 GGUF | Local Inference Engine | Offline mode / all cloud layers failed |

The cascade is **self-healing**: any layer can fail silently. The next layer activates automatically within milliseconds. In practice, the system almost always resolves at Layer 1 or 2; Layer 3 exists so the system never goes down — even with no internet at all.

### Why Gemma 4 at Both Ends?

Gemma 4 appears at Layer 1 (cloud, 26B full precision) and Layer 3 (local, Q4 quantized) by design:

- **Layer 1 (cloud)**: Maximum translation quality for real-time conversation. The 26B parameter model handles complex Turkish grammar, pronoun resolution, and domain-specific vocabulary.
- **Layer 3 (local)**: The privacy-preserving, cost-free, network-independent fallback. When quotas expire, internet drops, or data must stay on-device, `gemma-4-q4.gguf` takes over with zero configuration change — no restart, no user intervention.

This creates a **quality-symmetric, fully Gemma-native** cascade: every translation layer in the system is a Google Gemma model.

---

## VRAM Optimization

Consumer GPUs (8–12 GB VRAM) cannot hold all models simultaneously. Gemma Echo uses three strategies to make this work:

### 1. Lazy Loading
Models are not loaded at startup. The local Gemma 4 Q4 is loaded only when first needed (offline mode or video dubbing). XTTS-v2 loads only when TTS mode is switched to offline/GPU.

```python
# translator.py — load_local_model() called on first offline request
# (Llama is the entry-point class of llama-cpp-python, a generic C++ inference
#  engine for GGUF models. Here it is used to run a Google Gemma 4 weight file.)
self.local_llm = Llama(model_path="./models/gemma-4-q4.gguf", n_gpu_layers=-1)
```

### 2. Background Preloading (Ambush Mode)
When the user is in online mode, XTTS-v2 silently preloads into system RAM on a daemon thread. If the user switches to offline mode, the model is already warm — no perceived latency.

```python
# synthesizer.py — starts while online mode is active
synthesizer.preload_xtts_background(use_gpu=False)
```

### 3. Hot-Swap with Cache Eviction
Switching between GPU and CPU modes triggers controlled VRAM eviction before loading the new configuration, preventing CUDA OOM errors.

```python
# synthesizer.py — offload_xtts()
del self.xtts_model
gc.collect()
torch.cuda.empty_cache()   # VRAM fully released before next load
```

All three strategies compose: the system can run on a single RTX 3060 Ti (8 GB VRAM) handling real-time translation in online mode and full video dubbing in offline mode, simply by hot-swapping which model occupies VRAM at any given moment.

---

## Modes

| Mode | STT | Translation | TTS | Internet |
|------|-----|-------------|-----|----------|
| Online | faster-whisper | Gemma 4 26B → Gemini 2.5 Flash | ElevenLabs Turbo | Required |
| Offline | faster-whisper | Gemma 4 Q4 (local) | XTTS-v2 CPU | Not needed |
| GPU | faster-whisper | Gemma 4 Q4 (local, GPU) | XTTS-v2 GPU | Not needed |
| Video Dubbing | faster-whisper (timestamped) | Gemma 4 Q4 (local) | XTTS-v2 (voice clone) | Not needed |

---

## Video Dubbing Pipeline

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

---

## Setup

### Requirements

- Python 3.11+
- CUDA 11.8+ (optional, for GPU acceleration)
- ffmpeg in PATH

### Install

```bash
git clone https://github.com/yourusername/gemma_echo.git
cd gemma_echo
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Models

Place the quantized Gemma 4 model in the `models/` directory:

```
models/
  gemma-4-q4.gguf      # ~4 GB, Q4_K_M quantization
```

XTTS-v2 downloads automatically via the Coqui TTS library on first use.

### Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key            # Translation (Gemma 4 + Gemini Flash)
ELEVENLABS_API_KEY=your_elevenlabs_api_key    # High-quality online TTS (optional)
GROQ_API_KEY=your_groq_api_key                # Cloud Whisper-large-v3 STT acceleration (optional)
DEEPGRAM_API_KEY=your_deepgram_api_key        # Cloud STT fallback (optional)
```

The app runs fully offline without any of these keys (using only local Gemma 4 Q4 + faster-whisper + XTTS-v2). The Groq and Deepgram keys only accelerate cloud speech-to-text on low-VRAM machines; they are **not used for translation**.

---

## Usage

```bash
# Launch GUI
python -m gui.app

# Or directly
python gui/app.py
```

---

## Stack

| Component | Library |
|-----------|---------|
| GUI | CustomTkinter |
| STT (local) | faster-whisper |
| STT (cloud accelerator, optional) | Groq Whisper-large-v3, Deepgram Nova |
| Translation (cloud) | Google Gemini API — Gemma 4 26B → Gemini 2.5 Flash |
| Translation (local) | Gemma 4 Q4 GGUF on a local C++ inference engine |
| TTS (online) | ElevenLabs |
| TTS (offline / voice cloning) | Coqui XTTS-v2 † |
| Audio I/O | sounddevice, soundfile |
| Video processing | ffmpeg |

> **† TTS Engine Licensing Disclaimer.** The core orchestration framework of Gemma Echo is licensed under Apache 2.0. However, the **default** offline TTS engine (Coqui XTTS-v2) uses model weights licensed under the **Coqui Public Model License (Non-Commercial)**. Gemma Echo provides the architecture to integrate any TTS engine. For commercial deployment, users must replace the XTTS-v2 model weights with a commercially permissive alternative (e.g., VITS, Piper) or obtain a commercial license from Coqui GmbH. The Apache 2.0 license of Gemma Echo itself is unaffected.

---

## Roadmap / Experimental Track

The [`experimental/`](experimental/) directory contains research-and-development work that is **not part of the current competition submission** but documents the project's planned post-launch personalization track:

- **Mode-aware Japanese fine-tuning** — A 500-sample synthetic dataset (5 stylistic modes: emergency, official, streamer, casual, literary) generated with the Gemini API, plus a complete Unsloth + LoRA training script ready to run on a Kaggle T4 notebook. Training has not yet been executed; the dataset and pipeline are provided as a reproducible blueprint. See [`experimental/kaggle_finetune/README.md`](experimental/kaggle_finetune/README.md).

These materials demonstrate the engineering direction for adding Japanese support and style-aware translation in future versions, and are released under the same Apache 2.0 license as the rest of the project.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2026 Yusuf Semih Öksüzoğlu
