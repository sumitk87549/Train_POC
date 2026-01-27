#!/bin/bash

# AMD GPU Setup Script for natural_listen/tts.py
# Installs PyTorch with ROCm support and other dependencies globally.

set -e  # Exit on error

echo "==================================================="
echo "🚀 Setting up AMD GPU Support for Radeon 610M"
echo "==================================================="

# 1. Install PyTorch with ROCm support
# Trying ROCm 6.1 which is often better supported for new envs
echo "📦 Installing PyTorch with ROCm support (Global)..."
pip uninstall -y torch torchvision torchaudio || true

# Try installing from the ROCm 6.1 index
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1 || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0 || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# 2. Install other dependencies from requirements.txt
echo "📦 Installing other dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found! Skipping..."
fi

# 3. Verification
echo "🔍 Verifying installation..."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python3 -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "==================================================="
echo "✅ Setup Complete!"
echo "Run ./run_tts_amd.sh to start the TTS system."
echo "==================================================="
