"""
Audio Engine for ReadLyte MVP
Handles text-to-speech generation using HuggingFace or Coqui TTS
"""

import os
import re
import tempfile
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Check dependencies
DEPS_OK = True

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    DEPS_OK = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    DEPS_OK = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# TTS Models
try:
    from transformers import VitsModel, VitsTokenizer
    VITS_AVAILABLE = True
except ImportError:
    VITS_AVAILABLE = False

try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False


# Model configurations
TTS_MODELS = {
    "BASIC": {
        "huggingface": [
            {
                "name": "facebook/mms-tts-hin",
                "lang": "hindi",
                "type": "vits",
                "description": "Fast Hindi TTS from Meta"
            },
            {
                "name": "facebook/mms-tts-eng",
                "lang": "english",
                "type": "vits",
                "description": "Fast English TTS from Meta"
            }
        ]
    },
    "INTERMEDIATE": {
        "huggingface": [
            {
                "name": "suno/bark-small",
                "lang": "multilingual",
                "type": "bark",
                "description": "Bark small - decent quality"
            }
        ]
    },
    "ADVANCED": {
        "huggingface": [
            {
                "name": "suno/bark",
                "lang": "multilingual",
                "type": "bark",
                "description": "Bark full - excellent quality"
            }
        ]
    }
}


class TTSEngine:
    """Text-to-Speech engine wrapper."""
    
    def __init__(self, model_name: str = None, tier: str = "BASIC", provider: str = "huggingface"):
        self.model_name = model_name
        self.tier = tier
        self.provider = provider
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        # Auto-select model if not specified
        if not self.model_name:
            models = TTS_MODELS.get(tier, TTS_MODELS["BASIC"]).get(provider, [])
            if models:
                self.model_name = models[0]["name"]
            else:
                self.model_name = "facebook/mms-tts-hin"
        
        # Determine model type
        if 'mms-tts' in self.model_name:
            self.model_type = "vits"
        elif 'bark' in self.model_name:
            self.model_type = "bark"
        else:
            self.model_type = "vits"
    
    def load_model(self):
        """Load the TTS model."""
        if self.model_type == "vits":
            return self._load_vits()
        elif self.model_type == "bark":
            return self._load_bark()
        return False
    
    def _load_vits(self):
        """Load VITS model (facebook/mms-tts-*)."""
        if not VITS_AVAILABLE:
            raise ImportError("transformers not installed. Run: pip install transformers")
        
        self.tokenizer = VitsTokenizer.from_pretrained(self.model_name)
        self.model = VitsModel.from_pretrained(self.model_name).to(self.device)
        return True
    
    def _load_bark(self):
        """Load Bark model."""
        from transformers import AutoProcessor, BarkModel
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = BarkModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        return True
    
    def generate_audio(self, text: str, output_path: str = None) -> bytes:
        """
        Generate audio from text.
        
        Args:
            text: Text to convert to speech
            output_path: Optional path to save audio file
        
        Returns:
            Audio data as bytes (WAV format)
        """
        if self.model is None:
            self.load_model()
        
        if self.model_type == "vits":
            return self._generate_vits(text, output_path)
        elif self.model_type == "bark":
            return self._generate_bark(text, output_path)
    
    def _generate_vits(self, text: str, output_path: str = None) -> bytes:
        """Generate audio using VITS model."""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model(**inputs)
        
        # Extract waveform
        waveform = output.waveform[0].cpu().numpy()
        sample_rate = self.model.config.sampling_rate
        
        # Save to file or return bytes
        if output_path:
            sf.write(output_path, waveform, sample_rate)
            with open(output_path, 'rb') as f:
                return f.read()
        else:
            # Write to temp file and read back
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, waveform, sample_rate)
                tmp.flush()
                with open(tmp.name, 'rb') as f:
                    audio_data = f.read()
                os.unlink(tmp.name)
                return audio_data
    
    def _generate_bark(self, text: str, output_path: str = None) -> bytes:
        """Generate audio using Bark model."""
        # Process text
        inputs = self.processor(text, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            audio_array = self.model.generate(**inputs)
        
        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = 24000  # Bark outputs at 24kHz
        
        # Save or return
        if output_path:
            sf.write(output_path, audio_array, sample_rate)
            with open(output_path, 'rb') as f:
                return f.read()
        else:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio_array, sample_rate)
                tmp.flush()
                with open(tmp.name, 'rb') as f:
                    audio_data = f.read()
                os.unlink(tmp.name)
                return audio_data


def chunk_text_for_tts(text: str, max_chars: int = 500) -> list:
    """Split text into chunks suitable for TTS processing."""
    # Split by sentences
    sentences = re.split(r'([.!?।])', text)
    
    # Reconstruct sentences with punctuation
    reconstructed = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            reconstructed.append(sentences[i] + sentences[i + 1])
        else:
            reconstructed.append(sentences[i])
    
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        reconstructed.append(sentences[-1])
    
    # Build chunks
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sent in reconstructed:
        sent = sent.strip()
        if not sent:
            continue
        
        if current_length + len(sent) > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sent]
            current_length = len(sent)
        else:
            current_chunk.append(sent)
            current_length += len(sent)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def merge_audio_files(audio_files: list, output_path: str, pause_ms: int = 500) -> str:
    """Merge multiple audio files into one."""
    if not PYDUB_AVAILABLE:
        raise ImportError("pydub not installed. Run: pip install pydub")
    
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=pause_ms)
    
    for i, file_path in enumerate(audio_files):
        audio = AudioSegment.from_file(file_path)
        if i < len(audio_files) - 1:
            combined += audio + silence
        else:
            combined += audio
    
    combined.export(output_path, format="mp3", bitrate="192k")
    return output_path


def generate_audio(text: str, model: str = None, tier: str = "BASIC",
                   provider: str = "huggingface", output_path: str = None) -> bytes:
    """
    Main audio generation function.
    
    Args:
        text: Text to convert to speech
        model: Model name (optional, auto-selected based on tier)
        tier: Quality tier - BASIC, INTERMEDIATE, ADVANCED
        provider: "huggingface" or "coqui"
        output_path: Optional path to save audio file
    
    Returns:
        Audio data as bytes
    """
    if not text or not text.strip():
        return b""
    
    engine = TTSEngine(model, tier, provider)
    
    # For long text, chunk and merge
    if len(text) > 500:
        chunks = chunk_text_for_tts(text, 500)
        
        # Generate audio for each chunk
        temp_files = []
        temp_dir = tempfile.mkdtemp()
        
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
            engine.generate_audio(chunk, chunk_path)
            temp_files.append(chunk_path)
        
        # Merge
        if output_path:
            final_path = output_path
        else:
            final_path = os.path.join(temp_dir, "final.mp3")
        
        merge_audio_files(temp_files, final_path)
        
        with open(final_path, 'rb') as f:
            audio_data = f.read()
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return audio_data
    else:
        return engine.generate_audio(text, output_path)


def get_available_models(tier: str = "BASIC", provider: str = "huggingface") -> list:
    """Get list of available TTS models for a tier."""
    models = TTS_MODELS.get(tier, TTS_MODELS["BASIC"]).get(provider, [])
    return [m["name"] for m in models]


def get_model_info() -> dict:
    """Get all model information for UI display."""
    return TTS_MODELS


def check_dependencies() -> dict:
    """Check which TTS dependencies are available."""
    return {
        "torch": TORCH_AVAILABLE,
        "soundfile": SOUNDFILE_AVAILABLE,
        "pydub": PYDUB_AVAILABLE,
        "vits": VITS_AVAILABLE,
        "coqui": COQUI_AVAILABLE,
        "all_ok": DEPS_OK
    }


if __name__ == "__main__":
    # Test audio generation
    deps = check_dependencies()
    print("🔍 Dependency check:")
    for dep, status in deps.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {dep}")
    
    if deps["all_ok"]:
        print("\n🧪 Testing audio generation...")
        test_text = "Hello, this is a test of the text to speech system."
        
        try:
            audio_data = generate_audio(test_text, tier="BASIC")
            print(f"✅ Generated {len(audio_data)} bytes of audio")
        except Exception as e:
            print(f"❌ Error: {e}")
