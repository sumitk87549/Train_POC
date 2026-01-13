#!/usr/bin/env python3
"""
Simple test script to verify TTS functionality after installation.
"""

def test_imports():
    """Test if all required modules can be imported."""
    try:
        import torch
        print("✅ torch imported successfully")

        import transformers
        print("✅ transformers imported successfully")

        import numpy
        print("✅ numpy imported successfully")

        import scipy
        print("✅ scipy imported successfully")

        import soundfile
        print("✅ soundfile imported successfully")

        import pydub
        print("✅ pydub imported successfully")

        # Test TTS import (optional)
        try:
            from TTS.api import TTS
            print("✅ Coqui TTS available")
        except ImportError:
            print("ℹ️  Coqui TTS not available (expected)")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_tts():
    """Test basic TTS functionality."""
    try:
        from transformers import pipeline

        # Test text-to-speech pipeline
        print("🎙️  Testing basic TTS pipeline...")
        synthesizer = pipeline("text-to-speech", model="facebook/mms-tts-eng")

        # Generate a short test
        result = synthesizer("Hello, this is a test of the text to speech system.")
        print(f"✅ TTS test successful! Generated audio with shape: {result['audio'].shape}")

        return True

    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TTS Installation Test")
    print("=" * 50)

    if test_imports():
        print("\n✅ All imports successful!")
        if test_basic_tts():
            print("✅ TTS functionality working!")
        else:
            print("⚠️  Imports OK but TTS test failed")
    else:
        print("\n❌ Some imports failed")

    print("\n💡 You can now use: python3 listen.py -f your_text_file.txt -t BASIC")
