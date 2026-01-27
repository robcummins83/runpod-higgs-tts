# RunPod Serverless Handler for Higgs Audio V2
# Based on NVIDIA PyTorch container for CUDA compatibility

# Use PyTorch container with CUDA 12.4 for RunPod compatibility
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone and install Higgs Audio
RUN git clone https://github.com/boson-ai/higgs-audio.git /app/higgs-audio \
    && cd /app/higgs-audio \
    && pip install -r requirements.txt \
    && pip install -e .

# Install RunPod SDK and torchcodec (required by torchaudio 2.9+ for audio save)
RUN pip install runpod requests torchcodec

# Remove flash_attn to avoid CUDA compatibility issues
# The model will fall back to standard attention
RUN pip uninstall -y flash_attn || true

# Set environment variables BEFORE downloading models
# so they're cached to the same location used at runtime
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Pre-download models to bake into image (faster cold starts)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('bosonai/higgs-audio-v2-generation-3B-base'); \
    snapshot_download('bosonai/higgs-audio-v2-tokenizer')"

# Copy handler
COPY handler.py /app/handler.py

# Run the handler
CMD ["python", "-u", "/app/handler.py"]
