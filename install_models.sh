#!/bin/bash

# Script to install VibeVoice models
# Run this before starting the server or mount the models directory

MODEL_DIR="${VIBEVOICE_MODEL_PATH:-/workspace/models/vibevoice}"

echo "Installing VibeVoice models to $MODEL_DIR"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

echo "Installing VibeVoice-Large model..."
if [ ! -d "VibeVoice-Large" ]; then
    git clone https://huggingface.co/aoi-ot/VibeVoice-Large VibeVoice-Large
else
    echo "VibeVoice-Large already exists, skipping..."
fi

echo "Installing Qwen tokenizer..."
mkdir -p tokenizer
cd tokenizer
if [ ! -f "tokenizer.json" ]; then
    git clone https://huggingface.co/Qwen/Qwen2.5-1.5B .
else
    echo "Tokenizer already exists, skipping..."
fi

echo "Model installation complete!"
echo ""
echo "Model path: $MODEL_DIR/VibeVoice-Large"
echo "Tokenizer path: $MODEL_DIR/tokenizer"
