# Wrapper script to run tts.py with AMD GPU configuration

# Override GFX version for Radeon 610M (RDNA 2 APU) compatibility with ROCm
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Critical fixes for some RDNA 2 APUs to prevent SegFaults
export MIOPEN_DEBUG_COMGR_HIP_COMPILER_ENFORCE_DEVICE_LOWERING=1
export MIOPEN_DISABLE_CACHE=1 

echo "🚀 Starting TTS with AMD GPU support..."
echo "env: HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"

# Forward all arguments to the python script
python3 natural_listen/tts.py "$@"
