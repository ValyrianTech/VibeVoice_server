# =============================================================================
# Stage 1: Builder - Install dependencies and build wheels
# =============================================================================
ARG DOCKER_FROM=nvidia/cuda:12.8.0-runtime-ubuntu22.04
FROM ${DOCKER_FROM} AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install PyTorch (CPU for building, runtime will use CUDA)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Clone and install VibeVoice
WORKDIR /build
RUN git clone https://github.com/vibevoice-community/VibeVoice.git \
    && cd VibeVoice \
    && pip install --no-cache-dir -e .

# Install all Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    python-magic \
    pydub \
    openai-whisper \
    soundfile \
    transformers \
    huggingface_hub

# Install flash-attention (optional, may fail)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# =============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# =============================================================================
FROM ${DOCKER_FROM} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

LABEL github_repo="https://github.com/vibevoice-community/VibeVoice"

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsndfile1 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy VibeVoice installation
COPY --from=builder /build/VibeVoice /workspace/VibeVoice

ENV SHELL=/bin/bash

# Create necessary directories
RUN mkdir -p /workspace/VibeVoice/server/outputs \
    /workspace/VibeVoice/server/resources \
    /workspace/models/vibevoice

# Copy the server files
COPY server.py /workspace/VibeVoice/server/
COPY start.sh /workspace/VibeVoice/server/

# Fix line endings (in case of Windows CRLF) and make executable
RUN sed -i 's/\r$//' /workspace/VibeVoice/server/start.sh \
    && chmod +x /workspace/VibeVoice/server/start.sh

# Set environment variables for model paths
ENV VIBEVOICE_MODEL_PATH=/workspace/models/vibevoice/VibeVoice-Large
ENV VIBEVOICE_TOKENIZER_PATH=/workspace/models/vibevoice/tokenizer

# Set the working directory to the server directory
WORKDIR /workspace/VibeVoice/server

# Expose port
EXPOSE 7860

# Set the entrypoint to our start script
ENTRYPOINT ["/bin/bash", "/workspace/VibeVoice/server/start.sh"]
