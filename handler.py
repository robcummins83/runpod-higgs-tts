"""
RunPod Serverless Handler for Higgs Audio V2
Generates TTS audio with voice cloning support
"""

print("[DEBUG] Handler starting...", flush=True)

import sys
print(f"[DEBUG] Python: {sys.version}", flush=True)

try:
    import os
    import base64
    import tempfile
    import requests
    import numpy as np
    print("[DEBUG] Basic imports OK", flush=True)

    import torch
    print(f"[DEBUG] PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)

    import runpod
    from runpod.serverless.utils import rp_upload
    print("[DEBUG] RunPod SDK OK", flush=True)

    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
    print("[DEBUG] Higgs Audio imports OK", flush=True)

except Exception as e:
    print(f"[ERROR] Import failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Configuration - defaults only, all can be overridden via job input
CONFIG = {
    "model_path": "bosonai/higgs-audio-v2-generation-3B-base",
    "tokenizer_path": "bosonai/higgs-audio-v2-tokenizer",
    "default_temperature": 0.3,
    "default_top_p": 0.95,
    "default_top_k": 50,
    "default_max_new_tokens": 1024,  # ~40s max per chunk at 25fps
    "chunk_max_words": 80,
}

# Global model instance (loaded once, reused across requests)
_serve_engine = None
_device = None


def get_serve_engine():
    """Load model once and cache it."""
    global _serve_engine, _device

    if _serve_engine is None:
        print("[INIT] Loading Higgs Audio V2 model...")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _serve_engine = HiggsAudioServeEngine(
            CONFIG["model_path"],
            CONFIG["tokenizer_path"],
            device=_device
        )
        print(f"[INIT] Model loaded on {_device}")

    return _serve_engine


def download_audio(url: str, suffix: str = ".wav") -> str:
    """Download audio file from URL to temp file."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    audio_bytes = response.content

    if len(audio_bytes) < 1000:
        raise ValueError(f"Audio file too small: {len(audio_bytes)} bytes")

    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file.write(audio_bytes)
    temp_file.close()

    # Validate audio can be loaded
    try:
        import librosa
        audio_data, sr = librosa.load(temp_file.name, sr=None)
        duration = len(audio_data) / sr
        print(f"[AUDIO] Validated: {duration:.1f}s, {sr}Hz, mono")
    except Exception as e:
        os.unlink(temp_file.name)
        raise ValueError(f"Invalid audio file: {e}")

    return temp_file.name


def chunk_text(text: str, max_words: int = 80) -> list:
    """Split text into chunks at sentence boundaries."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_word_count + sentence_words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def generate_audio(
    text: str,
    voice_sample_path: str = None,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    max_new_tokens: int = None,
    seed: int = None,
) -> tuple:
    """Generate audio for text, optionally with voice cloning."""
    serve_engine = get_serve_engine()

    # Use defaults if not specified
    temperature = temperature if temperature is not None else CONFIG["default_temperature"]
    top_p = top_p if top_p is not None else CONFIG["default_top_p"]
    top_k = top_k if top_k is not None else CONFIG["default_top_k"]
    max_new_tokens = max_new_tokens if max_new_tokens is not None else CONFIG["default_max_new_tokens"]

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Transcript of the voice sample (required for voice cloning)
    VOICE_SAMPLE_TRANSCRIPT = (
        "Hi there, my name is Rob and I work in financial modeling. "
        "Over the years I've helped businesses make better decisions by turning complex numbers into clear, actionable insights. "
        "Whether you are looking into forecasts, valuations or scenario planning, the key is always to keep things simple and practical. "
        "Today, I would like to share a few thoughts on how technology is changing the way we approach these challenges. "
        "Let's dive in."
    )

    system_content = (
        "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "Audio is recorded from a quiet room with professional microphone. "
        "The speaker maintains a consistent, engaging tone throughout.\n"
        "<|scene_desc_end|>"
    )

    chunks = chunk_text(text, CONFIG["chunk_max_words"])
    print(f"[GEN] Processing {len(chunks)} chunk(s), max_new_tokens={max_new_tokens}...")

    all_audio = []
    sample_rate = None

    conversation_history = [
        Message(role="system", content=system_content)
    ]

    if voice_sample_path:
        print(f"[GEN] Adding voice sample for cloning...")
        conversation_history.append(
            Message(role="user", content=VOICE_SAMPLE_TRANSCRIPT)
        )
        conversation_history.append(
            Message(role="assistant", content=AudioContent(audio_url=voice_sample_path))
        )

    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        print(f"[GEN] Chunk {i+1}/{len(chunks)} ({word_count} words)...")

        # Fresh context each chunk - just system + voice sample + current text
        messages = conversation_history + [
            Message(role="user", content=chunk)
        ]

        output = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )

        if output.audio is not None and len(output.audio) > 0:
            audio_duration = len(output.audio) / output.sampling_rate
            all_audio.append(torch.from_numpy(output.audio))
            sample_rate = output.sampling_rate
            print(f"[GEN] Chunk {i+1} complete: {audio_duration:.1f}s")
        else:
            print(f"[WARN] Chunk {i+1} produced no audio")

    if not all_audio:
        raise RuntimeError("No audio generated")

    combined_audio = torch.cat(all_audio, dim=0)
    return combined_audio, sample_rate


def handler(job):
    """
    RunPod serverless handler.

    Input:
        - prompt: Text to convert to speech (required)
        - audio_url: URL to voice sample for cloning (optional)
        - temperature: Sampling temperature (default 0.3)
        - top_p: Top-p sampling (default 0.95)
        - top_k: Top-k sampling (default 50)
        - max_new_tokens: Max tokens per chunk (default 1024, ~40s audio)
        - seed: Random seed for reproducibility (optional)

    Output:
        - audio_url: URL to download generated WAV audio
        - duration: Audio duration in seconds
        - sample_rate: Audio sample rate
        - chunks: Number of chunks processed
    """
    try:
        job_input = job["input"]
        job_id = job.get("id", "unknown")

        text = job_input.get("prompt")
        if not text:
            return {"error": "No prompt provided"}

        audio_url = job_input.get("audio_url")
        temperature = job_input.get("temperature")
        top_p = job_input.get("top_p")
        top_k = job_input.get("top_k")
        max_new_tokens = job_input.get("max_new_tokens")
        seed = job_input.get("seed")

        print(f"[JOB] Received: {len(text)} chars, voice_clone={bool(audio_url)}, max_tokens={max_new_tokens or 'default'}")

        # Download voice sample if provided
        voice_sample_path = None
        if audio_url:
            print(f"[JOB] Downloading voice sample...")
            voice_sample_path = download_audio(audio_url)
            print(f"[JOB] Voice sample ready: {voice_sample_path}")

        # Generate audio
        audio, sample_rate = generate_audio(
            text=text,
            voice_sample_path=voice_sample_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

        # Calculate duration
        duration = len(audio) / sample_rate
        print(f"[JOB] Generated {duration:.1f}s of audio")

        # Save to temp file using scipy
        import scipy.io.wavfile as wavfile
        temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_np = audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wavfile.write(temp_output.name, sample_rate, audio_int16)

        file_size_mb = os.path.getsize(temp_output.name) / (1024 * 1024)
        print(f"[JOB] File size: {file_size_mb:.1f} MB")

        # Upload to RunPod storage (base64 too large for API response)
        print(f"[JOB] Uploading to storage...")
        uploaded_url = rp_upload.upload_file_to_bucket(
            file_name=f"higgs_tts_{job_id}.wav",
            file_location=temp_output.name
        )
        print(f"[JOB] Uploaded: {uploaded_url}")

        # Cleanup
        os.unlink(temp_output.name)
        if voice_sample_path:
            os.unlink(voice_sample_path)

        print(f"[JOB] Complete: {duration:.1f}s audio")

        return {
            "audio_url": uploaded_url,
            "duration": duration,
            "sample_rate": sample_rate,
            "chunks": len(chunk_text(text, CONFIG["chunk_max_words"])),
        }

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    print("[STARTUP] Higgs Audio V2 RunPod Handler")
    print(f"[STARTUP] Model: {CONFIG['model_path']}")
    get_serve_engine()
    runpod.serverless.start({"handler": handler})
