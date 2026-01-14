"""
RunPod Serverless Handler for Higgs Audio V2
Generates TTS audio with voice cloning support
Processes ONE chunk at a time - client handles chunking and concatenation
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
    print("[DEBUG] RunPod SDK OK", flush=True)

    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
    print("[DEBUG] Higgs Audio imports OK", flush=True)

except Exception as e:
    print(f"[ERROR] Import failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Configuration
CONFIG = {
    "model_path": "bosonai/higgs-audio-v2-generation-3B-base",
    "tokenizer_path": "bosonai/higgs-audio-v2-tokenizer",
}

# Global model instance
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

    # Validate audio
    try:
        import librosa
        audio_data, sr = librosa.load(temp_file.name, sr=None)
        duration = len(audio_data) / sr
        print(f"[AUDIO] Validated: {duration:.1f}s, {sr}Hz")
    except Exception as e:
        os.unlink(temp_file.name)
        raise ValueError(f"Invalid audio file: {e}")

    return temp_file.name


def handler(job):
    """
    RunPod serverless handler - processes ONE text chunk with audio continuity.
    Client handles chunking and concatenation.

    Input:
        - prompt: Text to convert to speech (required)
        - temperature: Sampling temperature (required)
        - top_p: Top-p sampling (required)
        - top_k: Top-k sampling (required)
        - max_new_tokens: Max tokens to generate (required)
        - system_prompt: Scene description for TTS (required)
        - audio_url: URL to voice sample for cloning (optional)
        - previous_audio_base64: Base64-encoded audio from previous chunk (optional, for continuity)
        - previous_text: Text from the previous chunk (optional, for continuity)
        - seed: Random seed for reproducibility (optional)

    Output:
        - audio_base64: Base64-encoded WAV audio for this chunk
        - duration: Audio duration in seconds
        - sample_rate: Audio sample rate
    """
    try:
        job_input = job["input"]

        text = job_input.get("prompt")
        if not text:
            return {"error": "No prompt provided"}

        audio_url = job_input.get("audio_url")
        previous_audio_base64 = job_input.get("previous_audio_base64")
        previous_text = job_input.get("previous_text")
        temperature = job_input.get("temperature")
        top_p = job_input.get("top_p")
        top_k = job_input.get("top_k")
        max_new_tokens = job_input.get("max_new_tokens")
        seed = job_input.get("seed")
        system_prompt = job_input.get("system_prompt")

        # Validate required parameters - no fallback defaults
        missing = []
        if temperature is None: missing.append("temperature")
        if top_p is None: missing.append("top_p")
        if top_k is None: missing.append("top_k")
        if max_new_tokens is None: missing.append("max_new_tokens")
        if system_prompt is None: missing.append("system_prompt")
        if missing:
            return {"error": f"Missing required parameters: {', '.join(missing)}"}

        has_continuity = bool(previous_audio_base64 and previous_text)
        print(f"[JOB] Text: {len(text)} chars, voice_clone={bool(audio_url)}, continuity={has_continuity}")

        serve_engine = get_serve_engine()

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Voice sample transcript (required for voice cloning)
        VOICE_SAMPLE_TRANSCRIPT = (
            "Hi there, my name is Rob and I work in financial modeling. "
            "Over the years I've helped businesses make better decisions by turning complex numbers into clear, actionable insights. "
            "Whether you are looking into forecasts, valuations or scenario planning, the key is always to keep things simple and practical. "
            "Today, I would like to share a few thoughts on how technology is changing the way we approach these challenges. "
            "Let's dive in."
        )

        # Build messages
        messages = [Message(role="system", content=system_prompt)]

        # Add voice cloning context if provided
        voice_sample_path = None
        if audio_url:
            print(f"[JOB] Downloading voice sample...")
            voice_sample_path = download_audio(audio_url)
            messages.append(Message(role="user", content=VOICE_SAMPLE_TRANSCRIPT))
            messages.append(Message(role="assistant", content=AudioContent(audio_url=voice_sample_path)))

        # Add previous chunk context for continuity (if provided)
        previous_audio_path = None
        if previous_audio_base64 and previous_text:
            print(f"[JOB] Adding continuity context ({len(previous_text)} chars)...")
            # Decode previous audio and save to temp file
            previous_audio_bytes = base64.b64decode(previous_audio_base64)
            previous_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            previous_audio_file.write(previous_audio_bytes)
            previous_audio_file.close()
            previous_audio_path = previous_audio_file.name

            # Add as user text + assistant audio (shows model what was just spoken)
            messages.append(Message(role="user", content=previous_text))
            messages.append(Message(role="assistant", content=AudioContent(audio_url=previous_audio_path)))

        # Add the text to generate
        messages.append(Message(role="user", content=text))

        # Generate
        print(f"[GEN] Generating audio...")
        output = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )

        if output.audio is None or len(output.audio) == 0:
            return {"error": "No audio generated"}

        audio = torch.from_numpy(output.audio)
        sample_rate = output.sampling_rate
        duration = len(audio) / sample_rate
        print(f"[GEN] Generated {duration:.1f}s audio")

        # Save to temp file
        import scipy.io.wavfile as wavfile
        temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_np = audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wavfile.write(temp_output.name, sample_rate, audio_int16)

        # Encode as base64
        with open(temp_output.name, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        file_size_kb = os.path.getsize(temp_output.name) / 1024
        print(f"[JOB] File size: {file_size_kb:.1f} KB")

        # Cleanup
        os.unlink(temp_output.name)
        if voice_sample_path:
            os.unlink(voice_sample_path)
        if previous_audio_path:
            os.unlink(previous_audio_path)

        print(f"[JOB] Complete")

        return {
            "audio_base64": audio_base64,
            "duration": duration,
            "sample_rate": sample_rate,
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
