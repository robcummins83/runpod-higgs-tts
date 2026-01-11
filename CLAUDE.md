# RunPod Higgs TTS - Claude Instructions

## Overview

RunPod serverless endpoint for **Higgs Audio V2** text-to-speech with voice cloning. This is a 3B parameter TTS model from Boson AI that uses in-context learning for voice cloning.

**Architecture**: Single-chunk processing. The handler processes ONE text chunk at a time. Client handles chunking (~500 chars) and concatenation.

## Files

| File | Purpose |
|------|---------|
| `handler.py` | RunPod serverless handler - processes single chunks |
| `Dockerfile` | Container build with CUDA, PyTorch, Higgs Audio |
| `.github/workflows/build-push.yml` | Auto-build on push to main |

## API Interface

### Input Parameters

```json
{
  "input": {
    "prompt": "Text to convert to speech (single chunk)",
    "audio_url": "https://example.com/voice_sample.wav",
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 50,
    "max_new_tokens": 1024,
    "seed": 42
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text chunk to convert (keep under 500 chars) |
| `audio_url` | string | optional | URL to voice sample WAV for cloning |
| `temperature` | float | 0.3 | Sampling temperature (lower = more consistent) |
| `top_p` | float | 0.95 | Top-p nucleus sampling |
| `top_k` | int | 50 | Top-k sampling |
| `max_new_tokens` | int | 1024 | Max audio tokens (~40s at 25 fps) |
| `seed` | int | optional | Random seed for reproducibility |

### Output

```json
{
  "audio_base64": "base64_encoded_wav",
  "duration": 12.5,
  "sample_rate": 24000
}
```

## Voice Cloning

Voice cloning uses in-context learning with a hardcoded transcript:

```
"Hi there, my name is Rob and I work in financial modeling..."
```

The voice sample URL must match this transcript. The transcript is defined in `handler.py` lines 139-145.

**Message structure for voice cloning:**
1. System message with scene description
2. User message with voice sample transcript
3. Assistant message with AudioContent pointing to voice sample
4. User message with text to generate

## Client-Side Chunking

The handler processes ONE chunk at a time to avoid RunPod response size limits (10-20 MB). The client must:

1. Split text into ~500 char chunks at sentence boundaries
2. Make separate API calls for each chunk
3. Concatenate audio files with ffmpeg

See `C:\Users\RobertCummins\Desktop\higgs_tts_test\test_higgs.py` for reference implementation.

## Deployment

### Auto-Build

Push to `main` triggers GitHub Actions to build and push:
```
ghcr.io/robcummins83/runpod-higgs-tts:latest
```

### RunPod Endpoint Setup

1. Go to RunPod Serverless console
2. Create endpoint with image: `ghcr.io/robcummins83/runpod-higgs-tts:latest`
3. **GPU**: RTX A6000 or A100 (24GB+ VRAM required)
4. **Idle Timeout**: 60+ seconds (model load takes time)
5. Add endpoint ID to `.env` as `RUNPOD_HIGGS_ENDPOINT_ID`

## Dependencies

Key packages in Dockerfile:
- PyTorch (NVIDIA container base)
- `boson-ai/higgs-audio` (cloned from GitHub)
- `torchcodec` (required by torchaudio 2.9+ for audio save)
- `runpod` SDK
- `librosa` (audio validation)
- `scipy` (WAV file writing)

## Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| TorchCodec required error | Added `torchcodec` to Dockerfile |
| Voice sample shape error | Use file path with AudioContent, not base64 |
| Audio load errors | Use librosa instead of torchaudio for validation |
| Audio save errors | Use scipy.io.wavfile instead of torchaudio.save |
| Response size exceeded | Client-side chunking (not server-side) |

## Environment Variables

Required in video_agent `.env`:
```
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_HIGGS_ENDPOINT_ID=your_endpoint_id
VOICE_SAMPLE_URL=https://url_to_voice_sample.wav
```

## Testing

Test script location: `C:\Users\RobertCummins\Desktop\higgs_tts_test\test_higgs.py`

```bash
cd C:\Users\RobertCummins\Desktop\higgs_tts_test
python test_higgs.py
```

## Related Projects

- **video_agent**: Main video pipeline that uses this endpoint
- **runpod_chatterbox**: Alternative TTS endpoint (has accent drift issues)
