# VibeVoice Server Research Notes

## Overview

This document contains research findings for building a VibeVoice server that acts as a **drop-in replacement** for the F5-TTS server.

---

## F5-TTS Server API (Target Interface)

The server we need to replicate has these endpoints:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/base_tts/` | TTS with default English voice |
| `POST` | `/change_voice/` | Voice conversion on existing audio |
| `POST` | `/upload_audio/` | Upload reference audio for cloning |
| `GET` | `/synthesize_speech/` | TTS with custom voice |

### Endpoint Details

#### 1. `GET /base_tts/`
- **Parameters:** `text` (str, required), `speed` (float, optional, default=1.0)
- **Response:** WAV audio stream

#### 2. `POST /change_voice/`
- **Parameters:** `reference_speaker` (Form), `file` (UploadFile)
- **Response:** WAV audio stream

#### 3. `POST /upload_audio/`
- **Parameters:** `audio_file_label` (Form), `file` (UploadFile)
- **Constraints:** wav/mp3/flac/ogg, max 5MB
- **Response:** JSON success/error

#### 4. `GET /synthesize_speech/`
- **Parameters:** `text` (str), `voice` (str), `speed` (float, optional, default=1.0)
- **Response:** WAV audio stream with headers `X-Elapsed-Time`, `X-Device-Used`

---

## VibeVoice Model Information

### Model Variants

| Model | Size | VRAM | Use Case |
|-------|------|------|----------|
| **VibeVoice-1.5B** | ~5.4GB | ~6GB | Quick prototyping, single voices |
| **VibeVoice-Large (7B)** | ~18.7GB | ~20GB | Highest quality, multi-speaker |
| **VibeVoice-Large-Q8** | ~11.6GB | ~12GB | Production quality, reduced VRAM |
| **VibeVoice-Large-Q4** | ~6.6GB | ~8GB | Maximum VRAM savings |

### Key Technical Details

- **Framework:** Microsoft's VibeVoice - novel TTS framework for long-form, multi-speaker audio
- **Architecture:** LLM (Qwen2.5) + Diffusion head + Acoustic/Semantic tokenizers
- **Sample Rate:** 24kHz output
- **Frame Rate:** 7.5 Hz (ultra-low, efficient for long sequences)
- **Max Duration:** Up to 90 minutes of speech
- **Max Speakers:** Up to 4 distinct speakers
- **License:** MIT

### HuggingFace Model Links

- `microsoft/VibeVoice-1.5B`
- `aoi-ot/VibeVoice-Large` (community mirror of 7B)
- `FabioSarracino/VibeVoice-Large-Q8`
- `DevParker/VibeVoice7b-low-vram` (Q4)

### Required Tokenizer

- **Qwen2.5-1.5B tokenizer** from `Qwen/Qwen2.5-1.5B`
- Required files: `tokenizer_config.json`, `vocab.json`, `merges.txt`, `tokenizer.json`

---

## VibeVoice Inference Parameters

From the ComfyUI wrapper, these are the key generation parameters:

### Core Parameters
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `text` | str | - | - | Input text to synthesize |
| `diffusion_steps` | int | 20 | 5-100 | Denoising steps (quality vs speed) |
| `seed` | int | 42 | - | Random seed for reproducibility |
| `cfg_scale` | float | 1.3 | 1.0-2.0 | Classifier-free guidance |
| `use_sampling` | bool | False | - | Deterministic vs sampling mode |

### Optional/Advanced Parameters
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `temperature` | float | 0.95 | 0.1-2.0 | Sampling temperature |
| `top_p` | float | 0.95 | 0.1-1.0 | Nucleus sampling |
| `max_words_per_chunk` | int | 250 | 100-500 | Text chunking for long texts |
| `voice_speed_factor` | float | 1.0 | 0.8-1.2 | Speech rate adjustment |
| `attention_type` | str | "auto" | auto/eager/sdpa/flash_attention_2/sage | Attention mechanism |
| `quantize_llm` | str | "full precision" | full/4bit/8bit | Dynamic quantization |

### Voice Cloning Requirements
- Clear audio with minimal background noise
- Minimum 3-10 seconds, recommended 30+ seconds for better quality
- Automatically resampled to 24kHz

---

## Implementation Approach

### Key Differences from F5-TTS

1. **No built-in transcription:** F5-TTS uses `model.transcribe()` for reference audio. VibeVoice uses voice cloning via audio prefill, not text-based reference.

2. **Voice cloning mechanism:** VibeVoice clones voice from audio samples directly without needing transcription of the reference audio.

3. **Speed control:** F5-TTS uses a `speed` parameter directly. VibeVoice uses `voice_speed_factor` which modifies the reference voice (0.8-1.2 range vs arbitrary speed multiplier).

4. **Multi-speaker support:** VibeVoice natively supports up to 4 speakers with speaker labels in text format like `Speaker 1: text`.

### Mapping F5-TTS API to VibeVoice

| F5-TTS Feature | VibeVoice Equivalent |
|----------------|---------------------|
| `speed` parameter | `voice_speed_factor` (needs mapping: 1.0 → 1.0, clamp to 0.8-1.2) |
| Reference audio + transcription | Voice cloning via audio prefill (no transcription needed) |
| `model.infer()` | VibeVoice inference pipeline |
| `nfe_step=32` | `diffusion_steps=20` (default) |
| `cfg_strength=2.0` | `cfg_scale=1.3` (default) |

### Required Components

1. **Model loading:** Load VibeVoice model from HuggingFace or local path
2. **Tokenizer:** Load Qwen2.5-1.5B tokenizer
3. **Voice storage:** `resources/` directory for uploaded voice samples
4. **Audio processing:** Convert uploaded audio to 24kHz WAV
5. **Inference pipeline:** Text → VibeVoice → WAV output

### Community Fork Reference

The best reference implementation is: `https://github.com/vibevoice-community/VibeVoice`

Key files to study:
- `demo/inference_from_file.py` - File-based inference
- `demo/gradio_demo.py` - Interactive demo
- `vibevoice/` - Core model code

### Available Voice Presets (from community fork)

English voices:
- `en-Alice_woman`
- `en-Carter_man`
- `en-Frank_man`
- `en-Mary_woman_bgm`
- `en-Maya_woman`
- `in-Samuel_man` (Indian English)

Chinese voices:
- `zh-Anchen_man_bgm`
- `zh-Bowen_man`
- `zh-Xinran_woman`

---

## Implementation Plan

### Phase 1: Basic Setup
1. Clone/install vibevoice-community/VibeVoice
2. Download VibeVoice-Large model (or Q8 for lower VRAM)
3. Download Qwen2.5-1.5B tokenizer
4. Set up FastAPI server structure

### Phase 2: Core Endpoints
1. Implement `/upload_audio/` - Store voice samples in `resources/`
2. Implement `/synthesize_speech/` - Main TTS with voice cloning
3. Implement `/base_tts/` - TTS with default voice (e.g., en-Alice_woman)
4. Implement `/change_voice/` - Voice conversion (may need ASR for transcription)

### Phase 3: Compatibility
1. Match response formats (WAV streaming, headers)
2. Handle speed parameter mapping
3. Add CORS middleware
4. Test with existing F5-TTS clients

---

## Open Questions

1. **Voice conversion (`/change_voice/`):** F5-TTS transcribes input audio and regenerates with new voice. VibeVoice doesn't have built-in ASR. Options:
   - Use external ASR (Whisper) to transcribe input audio
   - Use VibeVoice-ASR (newly released, handles 60-min audio)

2. **Speed mapping:** F5-TTS allows arbitrary speed values. VibeVoice clamps to 0.8-1.2. Need to decide how to handle out-of-range values.

3. **Default voice:** Which preset voice to use for `/base_tts/`? Suggest `en-Alice_woman` or `en-Frank_man`.

---

## Dependencies

Based on ComfyUI wrapper and community fork:

```
torch>=2.0
transformers>=4.51.3
torchaudio
soundfile
pydub
fastapi
uvicorn
python-multipart
python-magic
huggingface_hub
```

For GPU acceleration:
- CUDA 11.8+
- flash-attn (optional, for flash_attention_2)

---

## Model Installation Script

For reference, here's the installation script used for ComfyUI setup:

```bash
echo "Installing VibeVoice"
mkdir -p /ComfyUI/models/vibevoice
cd /ComfyUI/models/vibevoice
git clone https://huggingface.co/aoi-ot/VibeVoice-Large VibeVoice-Large

echo "Installing Qwen tokenizer"
mkdir -p /ComfyUI/models/vibevoice/tokenizer
cd /ComfyUI/models/vibevoice/tokenizer
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B .
```

**Model paths used:**
- Model: `/ComfyUI/models/vibevoice/VibeVoice-Large/`
- Tokenizer: `/ComfyUI/models/vibevoice/tokenizer/`

For the server, we can adapt these paths or use environment variables to configure model locations.

---

## References

- [VibeVoice-ComfyUI](https://github.com/Enemyx-net/VibeVoice-ComfyUI) - ComfyUI integration with embedded code
- [vibevoice-community/VibeVoice](https://github.com/vibevoice-community/VibeVoice) - Community fork with full code
- [aoi-ot/VibeVoice-Large](https://huggingface.co/aoi-ot/VibeVoice-Large) - Model weights
- [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) - Original (limited) Microsoft repo
- [KDnuggets Guide](https://www.kdnuggets.com/beginners-guide-to-vibevoice) - Beginner tutorial
