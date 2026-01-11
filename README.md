# RunPod Higgs TTS

Serverless RunPod endpoint for Higgs Audio V2 text-to-speech with voice cloning.

## Deployment

### 1. Build Docker Image (Automatic)

The Docker image is automatically built and pushed to GitHub Container Registry on every push to `main`.

Image: `ghcr.io/robcummins83/runpod-higgs-tts:latest`

To trigger a manual build, go to Actions → Build and Push Docker Image → Run workflow.

### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Create new endpoint
3. Use Docker image: `ghcr.io/robcummins83/runpod-higgs-tts:latest`
4. **GPU**: RTX A6000 or A100 (24GB+ VRAM required)
5. **Idle Timeout**: 60 seconds (model takes time to load)
6. **Max Workers**: 1 (for testing)

### 3. Add Endpoint ID to Environment

Add to your `.env`:
```
RUNPOD_HIGGS_ENDPOINT_ID=your_endpoint_id_here
```

## API Usage

### Request

```json
{
  "input": {
    "prompt": "Text to convert to speech",
    "audio_url": "https://example.com/voice_sample.wav",
    "temperature": 0.3,
    "seed": 42
  }
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| prompt | string | Yes | - | Text to convert to speech |
| audio_url | string | No | - | URL to voice sample for cloning |
| temperature | float | No | 0.3 | Sampling temperature (lower = more consistent) |
| top_p | float | No | 0.95 | Top-p sampling |
| top_k | int | No | 50 | Top-k sampling |
| seed | int | No | - | Random seed for reproducibility |

### Response

```json
{
  "audio_base64": "base64_encoded_wav_audio",
  "duration": 45.2,
  "sample_rate": 24000,
  "chunks": 5
}
```

## Long-Form Handling

Text is automatically chunked at sentence boundaries (~80 words per chunk). Conversation history is maintained across chunks to preserve voice consistency.

## GPU Requirements

- **Minimum**: 24GB VRAM (RTX 4090, A5000, A6000)
- **Recommended**: A100 40GB for faster processing

Performance:
- A100 (40GB): ~60 seconds audio per second of processing
- RTX 4090 (24GB): ~24 seconds audio per second of processing
