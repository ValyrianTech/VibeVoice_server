#!/bin/bash

# Script to install VibeVoice models
# Run this before starting the server or mount the models directory
#
# On RunPod: Models are stored in /workspace which is the network volume,
# so they persist across container restarts.

# Base directory for all models (not the model path itself)
BASE_DIR="/workspace/models/vibevoice"

echo "=========================================="
echo "Installing VibeVoice models to $BASE_DIR"
echo "(This location persists on RunPod network volumes)"
echo "=========================================="

mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

echo "Installing VibeVoice-Large model..."
if [ ! -d "VibeVoice-Large" ]; then
    git clone https://huggingface.co/aoi-ot/VibeVoice-Large VibeVoice-Large
else
    echo "VibeVoice-Large already exists, skipping..."
fi

echo "Installing Qwen tokenizer..."
if [ ! -d "tokenizer" ] || [ ! -f "tokenizer/tokenizer.json" ]; then
    rm -rf tokenizer
    git clone https://huggingface.co/Qwen/Qwen2.5-1.5B tokenizer
else
    echo "Tokenizer already exists, skipping..."
fi

echo ""
echo "=========================================="
echo "Model installation complete!"
echo "=========================================="
echo "Model path: $BASE_DIR/VibeVoice-Large"
echo "Tokenizer path: $BASE_DIR/tokenizer"
