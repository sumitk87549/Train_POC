#!/usr/bin/env python3
"""
Context-Aware Text-to-Speech Audio Generator
Supports BASIC, INTERMEDIATE, and ADVANCED quality tiers
Optimized for Hindi and English audiobook narration
"""

import os
import sys
import argparse
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
import subprocess

warnings.filterwarnings("ignore")

# Check and install dependencies
def check_dependencies():
    """Check and suggest installation of required packages."""
    required = {
        'torch': 'torch',
        'transformers': 'transformers',
        'TTS': 'TTS',  # Coqui TTS
        'pydub': 'pydub',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'soundfile': 'soundfile',
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        print(f"💡 Install with: pip install {' '.join(missing)}")
        return False
    return True

# Import after dependency check
try:
    import torch
    import numpy as np
    import soundfile as sf
    from transformers import AutoProcessor, AutoModel, pipeline
    from pydub import AudioSegment
    DEPS_OK = True
except ImportError:
    DEPS_OK = False
    print("⚠️  Some dependencies not installed. Run with dependencies check first.")

# Try to import TTS
try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except:
    COQUI_AVAILABLE = False

# Try to import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except:
    OLLAMA_AVAILABLE = False

# ==== HARDWARE DETECTION ====
class HardwareManager:
    """Detect and configure hardware for optimal performance."""

    def __init__(self):
        self.device = "cpu"
        self.gpu_type = None
        self.gpu_available = False
        self.detect_hardware()

    def detect_hardware(self):
        """Detect available hardware and configure accordingly."""
        if not DEPS_OK:
            return

        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            self.device = "cuda"
            self.gpu_type = "nvidia"
            self.gpu_available = True
            print(f"✅ NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Check for ROCm (AMD)
        elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
            self.device = "cuda"  # ROCm uses cuda device name in PyTorch
            self.gpu_type = "amd"
            self.gpu_available = True
            print(f"✅ AMD GPU detected (ROCm)")
            print(f"   ROCm Version: {torch.version.hip}")

        # CPU only
        else:
            self.device = "cpu"
            self.gpu_available = False
            print(f"ℹ️  Using CPU (no GPU detected)")
            print(f"   CPU Threads: {torch.get_num_threads()}")

    def setup_gpu_environment(self):
        """Setup GPU environment variables if needed."""
        if self.gpu_type == "amd":
            # ROCm environment variables
            os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # For Ryzen 5 7520U
            os.environ['ROCM_PATH'] = '/opt/rocm'
            print("🔧 ROCm environment configured")

    def get_optimal_device(self, tier):
        """Get optimal device based on tier and available hardware."""
        if tier == "ADVANCED":
            if not self.gpu_available:
                raise RuntimeError("ADVANCED tier requires GPU but none available!")
            return self.device
        elif tier == "INTERMEDIATE":
            return self.device  # Use GPU if available, else CPU
        else:  # BASIC
            return "cpu"

# ==== MODEL CONFIGURATIONS ====
TTS_MODELS = {
    "BASIC": {
        "huggingface": [
            {
                "name": "facebook/mms-tts-hin",
                "lang": "hindi",
                "type": "transformers",
                "description": "Fast, lightweight Hindi TTS from Meta",
                "ram": "2-4GB",
                "quality": "Good for basic narration"
            },
            {
                "name": "facebook/mms-tts-eng",
                "lang": "english",
                "type": "transformers",
                "description": "Fast English TTS",
                "ram": "2-4GB",
                "quality": "Good for basic narration"
            },
            {
                "name": "suno/bark-small",
                "lang": "multilingual",
                "type": "bark",
                "description": "Bark small - decent quality, context-aware",
                "ram": "4-6GB",
                "quality": "Good emotional range"
            }
        ],
        "coqui": [
            {
                "name": "tts_models/hi/custom/female",
                "lang": "hindi",
                "description": "Coqui Hindi female voice",
                "ram": "2-4GB"
            }
        ]
    },

    "INTERMEDIATE": {
        "huggingface": [
            {
                "name": "suno/bark",
                "lang": "multilingual",
                "type": "bark",
                "description": "Bark full - excellent quality, emotional, context-aware",
                "ram": "8-12GB",
                "quality": "Excellent for audiobooks, natural prosody"
            },
            {
                "name": "microsoft/speecht5_tts",
                "lang": "english",
                "type": "transformers",
                "description": "Microsoft SpeechT5 - high quality English",
                "ram": "4-8GB",
                "quality": "Very natural English speech"
            }
        ],
        "coqui": [
            {
                "name": "tts_models/multilingual/multi-dataset/xtts_v2",
                "lang": "multilingual",
                "description": "XTTS v2 - state-of-art multilingual, emotion-aware",
                "ram": "6-10GB",
                "quality": "Excellent, can clone voices"
            }
        ]
    },

    "ADVANCED": {
        "huggingface": [
            {
                "name": "suno/bark",
                "lang": "multilingual",
                "type": "bark",
                "description": "Bark with advanced processing - maximum quality",
                "ram": "12-16GB",
                "quality": "Commercial grade, full context awareness",
                "gpu_required": True
            }
        ],
        "coqui": [
            {
                "name": "tts_models/multilingual/multi-dataset/xtts_v2",
                "lang": "multilingual",
                "description": "XTTS v2 with advanced context processing",
                "ram": "12-20GB",
                "quality": "State-of-art, commercial grade",
                "gpu_required": True
            }
        ]
    }
}

# ==== CONTEXT-AWARE TEXT PROCESSING ====
class ContextProcessor:
    """Process text to add context and emotion markers for better narration."""

    def __init__(self, tier, llm_provider=None, llm_model=None):
        self.tier = tier
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def enhance_text(self, text):
        """Enhance text with context markers based on tier."""
        if self.tier == "BASIC":
            return self._basic_processing(text)
        elif self.tier == "INTERMEDIATE":
            return self._intermediate_processing(text)
        else:  # ADVANCED
            return self._advanced_processing(text)

    def _basic_processing(self, text):
        """Basic text cleanup."""
        # Add pauses at punctuation
        text = text.replace('. ', '. [pause] ')
        text = text.replace('। ', '। [pause] ')
        text = text.replace('? ', '? [pause] ')
        text = text.replace('! ', '! [pause] ')
        return text

    def _intermediate_processing(self, text):
        """Add emotion and context markers."""
        # Split into sentences
        sentences = text.replace('।', '.').split('.')
        enhanced = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Add emotion markers based on content
            if '!' in sent or 'excited' in sent.lower():
                sent = f"[excited] {sent}"
            elif '?' in sent:
                sent = f"[questioning] {sent}"
            elif any(word in sent.lower() for word in ['sad', 'unfortunately', 'दुर्भाग्य']):
                sent = f"[sad] {sent}"
            elif any(word in sent.lower() for word in ['laugh', 'funny', 'हंस']):
                sent = f"[laughing] {sent}"

            enhanced.append(sent)

        return ' [pause] '.join(enhanced)

    def _advanced_processing(self, text):
        """Use LLM to add sophisticated context and emotion markers."""
        if not self.llm_provider or not self.llm_model:
            return self._intermediate_processing(text)

        prompt = f"""Analyze this text and add appropriate emotion and context markers for audiobook narration.

Available markers: [excited], [sad], [questioning], [thoughtful], [urgent], [whisper], [shouting], [laughing], [serious], [pause]

Text:
{text}

Return the text with emotion markers inserted naturally. Keep the original text but add markers in square brackets."""

        try:
            if self.llm_provider == "ollama" and OLLAMA_AVAILABLE:
                response = ollama.chat(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response["message"]["content"]
            else:
                # Fallback to intermediate processing
                return self._intermediate_processing(text)
        except:
            return self._intermediate_processing(text)

    def chunk_for_tts(self, text, max_chars=500):
        """Split text into manageable chunks for TTS processing."""
        # Split by sentences first
        sentences = text.replace('।', '.').split('.')
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            sent_length = len(sent)

            if current_length + sent_length > max_chars and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sent]
                current_length = sent_length
            else:
                current_chunk.append(sent)
                current_length += sent_length

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks

# ==== TTS ENGINES ====
class TTSEngine:
    """Unified interface for different TTS engines."""

    def __init__(self, model_name, model_type, device="cpu", tier="BASIC"):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.tier = tier
        self.model = None
        self.processor = None

    def load_model(self):
        """Load the TTS model."""
        print(f"📥 Loading model: {self.model_name}")
        print(f"   Device: {self.device}")

        try:
            if self.model_type == "bark":
                return self._load_bark()
            elif self.model_type == "transformers":
                return self._load_transformers()
            elif self.model_type == "coqui":
                return self._load_coqui()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            raise

    def _load_bark(self):
        """Load Bark model."""
        from transformers import AutoProcessor, BarkModel

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = BarkModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # Enable optimization for GPU
        if self.device == "cuda":
            self.model = self.model.to_bettertransformer()

        print("✅ Bark model loaded")
        return True

    def _load_transformers(self):
        """Load standard Transformers TTS model."""
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        print("✅ Transformers model loaded")
        return True

    def _load_coqui(self):
        """Load Coqui TTS model."""
        if not COQUI_AVAILABLE:
            raise ImportError("Coqui TTS not installed. Install with: pip install TTS")

        self.model = CoquiTTS(model_name=self.model_name, gpu=(self.device=="cuda"))
        print("✅ Coqui TTS model loaded")
        return True

    def generate_audio(self, text, output_path):
        """Generate audio from text."""
        if self.model_type == "bark":
            return self._generate_bark(text, output_path)
        elif self.model_type == "transformers":
            return self._generate_transformers(text, output_path)
        elif self.model_type == "coqui":
            return self._generate_coqui(text, output_path)

    def _generate_bark(self, text, output_path):
        """Generate audio using Bark."""
        print(f"🎙️  Generating audio with Bark...")

        # Process text through Bark
        inputs = self.processor(text, return_tensors="pt").to(self.device)

        # Set generation parameters based on tier
        if self.tier == "ADVANCED":
            do_sample = True
            temperature = 0.9
            semantic_temperature = 0.8
        elif self.tier == "INTERMEDIATE":
            do_sample = True
            temperature = 0.7
            semantic_temperature = 0.7
        else:
            do_sample = False
            temperature = 0.6
            semantic_temperature = 0.6

        with torch.no_grad():
            audio_array = self.model.generate(
                **inputs,
                do_sample=do_sample,
                semantic_temperature=semantic_temperature
            )

        # Convert to numpy and save
        audio_array = audio_array.cpu().numpy().squeeze()

        # Bark outputs at 24kHz
        sample_rate = 24000
        sf.write(output_path, audio_array, sample_rate)

        print(f"✅ Audio generated: {output_path}")
        return output_path

    def _generate_transformers(self, text, output_path):
        """Generate audio using Transformers model."""
        print(f"🎙️  Generating audio with Transformers...")

        inputs = self.processor(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            speech = self.model.generate(**inputs)

        # Extract audio
        audio_array = speech.cpu().numpy().squeeze()
        sample_rate = 16000  # Most TTS models use 16kHz

        sf.write(output_path, audio_array, sample_rate)
        print(f"✅ Audio generated: {output_path}")
        return output_path

    def _generate_coqui(self, text, output_path):
        """Generate audio using Coqui TTS."""
        print(f"🎙️  Generating audio with Coqui TTS...")

        # Coqui handles file output directly
        self.model.tts_to_file(
            text=text,
            file_path=output_path,
            emotion="neutral" if self.tier == "BASIC" else "expressive"
        )

        print(f"✅ Audio generated: {output_path}")
        return output_path

# ==== AUDIO POST-PROCESSING ====
class AudioProcessor:
    """Post-process generated audio for better quality."""

    def __init__(self, tier):
        self.tier = tier

    def process(self, audio_path):
        """Apply post-processing based on tier."""
        if self.tier == "BASIC":
            return audio_path  # No processing

        try:
            audio = AudioSegment.from_file(audio_path)

            if self.tier == "INTERMEDIATE":
                # Normalize audio
                audio = self._normalize_audio(audio)

            elif self.tier == "ADVANCED":
                # Full processing pipeline
                audio = self._normalize_audio(audio)
                audio = self._enhance_quality(audio)
                audio = self._add_subtle_effects(audio)

            # Export
            audio.export(audio_path, format="mp3", bitrate="192k")
            print(f"✅ Audio post-processed")

        except Exception as e:
            print(f"⚠️  Post-processing failed: {e}")

        return audio_path

    def _normalize_audio(self, audio):
        """Normalize audio levels."""
        # Target -20 dBFS
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - audio.dBFS
        return audio.apply_gain(change_in_dBFS)

    def _enhance_quality(self, audio):
        """Enhance audio quality."""
        # Apply compression
        audio = audio.compress_dynamic_range(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0
        )
        return audio

    def _add_subtle_effects(self, audio):
        """Add subtle effects for professional sound."""
        # Add slight reverb effect (if available)
        # This is a placeholder - would need additional libraries for real reverb
        return audio

# ==== MAIN AUDIO GENERATOR ====
class AudioGenerator:
    """Main audio generation orchestrator."""

    def __init__(self, args):
        self.args = args
        self.hardware = HardwareManager()
        self.tier = args.tier
        self.output_dir = Path(args.output) / "audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup context processor
        self.context_processor = ContextProcessor(
            self.tier,
            llm_provider=args.llm_provider if hasattr(args, 'llm_provider') else None,
            llm_model=args.llm_model if hasattr(args, 'llm_model') else None
        )

        # Setup audio processor
        self.audio_processor = AudioProcessor(self.tier)

    def read_input(self):
        """Read input text file."""
        print(f"📖 Reading input: {self.args.file}")
        with open(self.args.file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        print(f"📊 Input: {len(text)} characters, ~{len(text.split())} words")
        return text

    def select_model(self):
        """Select appropriate model based on tier and provider."""
        provider = self.args.provider
        models = TTS_MODELS[self.tier][provider]

        if self.args.model:
            # Use specified model
            model_name = self.args.model
            # Find model type
            model_info = next((m for m in models if m["name"] == model_name), None)
            if not model_info:
                # Assume transformers type if not found
                model_type = "transformers"
            else:
                model_type = model_info.get("type", "transformers")
        else:
            # Use recommended model
            model_info = models[0]
            model_name = model_info["name"]
            model_type = model_info.get("type", "coqui" if provider == "coqui" else "transformers")

        print(f"\n🤖 Selected Model: {model_name}")
        print(f"   Type: {model_type}")
        print(f"   Tier: {self.tier}")

        return model_name, model_type

    def generate(self):
        """Main generation process."""
        print("=" * 70)
        print("🎙️  CONTEXT-AWARE AUDIOBOOK GENERATOR")
        print("=" * 70)
        print(f"⚡ Tier: {self.tier}")
        print(f"🔧 Provider: {self.args.provider}")
        print(f"💾 Output: {self.output_dir}")
        print("=" * 70)

        # Read input
        text = self.read_input()

        # Enhance text with context
        print(f"\n🧠 Processing text with context awareness ({self.tier} mode)...")
        enhanced_text = self.context_processor.enhance_text(text)

        # Chunk text
        print(f"📦 Chunking text for optimal processing...")
        max_chunk_size = 300 if self.tier == "BASIC" else (500 if self.tier == "INTERMEDIATE" else 1000)
        chunks = self.context_processor.chunk_for_tts(enhanced_text, max_chunk_size)
        print(f"✅ Created {len(chunks)} chunks")

        # Select model
        model_name, model_type = self.select_model()

        # Get optimal device
        device = self.hardware.get_optimal_device(self.tier)

        # Initialize TTS engine
        print(f"\n🚀 Initializing TTS engine...")
        engine = TTSEngine(model_name, model_type, device, self.tier)
        engine.load_model()

        # Generate audio for each chunk
        audio_files = []
        start_time = time.time()

        print(f"\n{'='*70}")
        print(f"🎵 GENERATING AUDIO")
        print(f"{'='*70}")

        for i, chunk in enumerate(chunks, 1):
            print(f"\n📝 Chunk {i}/{len(chunks)}")
            print(f"   Text length: {len(chunk)} chars")

            chunk_start = time.time()

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = model_name.split('/')[-1].replace(':', '_')
            chunk_file = self.output_dir / f"{model_short}_{timestamp}_chunk{i:03d}.wav"

            # Generate audio
            try:
                engine.generate_audio(chunk, str(chunk_file))
                audio_files.append(str(chunk_file))

                chunk_time = time.time() - chunk_start
                print(f"   ⏱️  Generated in {chunk_time:.1f}s")

            except Exception as e:
                print(f"   ❌ Error generating chunk {i}: {e}")
                continue

        # Merge audio files
        if len(audio_files) > 1:
            print(f"\n🔗 Merging {len(audio_files)} audio chunks...")
            final_audio = self._merge_audio_files(audio_files)
        elif len(audio_files) == 1:
            final_audio = audio_files[0]
        else:
            print("❌ No audio files generated!")
            return None

        # Post-process
        print(f"\n🎛️  Post-processing audio...")
        final_audio = self.audio_processor.process(final_audio)

        # Rename to final format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model_name.split('/')[-1].replace(':', '_')
        final_path = self.output_dir / f"{model_short}_{timestamp}.mp3"

        if Path(final_audio).suffix != '.mp3':
            audio = AudioSegment.from_file(final_audio)
            audio.export(str(final_path), format="mp3", bitrate="192k")
            # Clean up temp files
            for f in audio_files:
                try:
                    Path(f).unlink()
                except:
                    pass
        else:
            Path(final_audio).rename(final_path)

        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"🎉 AUDIO GENERATION COMPLETE!")
        print(f"{'='*70}")
        print(f"⏱️  Total time: {total_time/60:.2f} minutes")
        print(f"📦 Chunks processed: {len(chunks)}")
        print(f"💾 Output file: {final_path}")
        print(f"📊 File size: {final_path.stat().st_size / 1e6:.2f} MB")
        print(f"{'='*70}")

        return str(final_path)

    def _merge_audio_files(self, files):
        """Merge multiple audio files into one."""
        combined = AudioSegment.empty()

        for i, file in enumerate(files, 1):
            print(f"   Adding chunk {i}/{len(files)}...")
            audio = AudioSegment.from_file(file)

            # Add slight pause between chunks (except BASIC tier)
            if self.tier != "BASIC" and i < len(files):
                silence = AudioSegment.silent(duration=500)  # 500ms pause
                combined += audio + silence
            else:
                combined += audio

        # Save merged file
        merged_path = self.output_dir / "merged_temp.wav"
        combined.export(str(merged_path), format="wav")

        return str(merged_path)

# ==== MAIN FUNCTION ====
def main():
    parser = argparse.ArgumentParser(
        description='Context-Aware Text-to-Speech Audio Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # BASIC tier (CPU, 8GB RAM)
  python listen.py -f book.txt -p huggingface -t BASIC
  python listen.py -f book.txt -p coqui -m tts_models/hi/custom/female -t BASIC
  
  # INTERMEDIATE tier (CPU/GPU, better quality)
  python listen.py -f book.txt -p huggingface -m suno/bark -t INTERMEDIATE
  python listen.py -f book.txt -p coqui -m tts_models/multilingual/multi-dataset/xtts_v2 -t INTERMEDIATE
  
  # ADVANCED tier (GPU required, commercial quality)
  python listen.py -f book.txt -p coqui -t ADVANCED --llm-provider ollama --llm-model qwen2.5:7b
  
  # List available models
  python listen.py --list-models
  
  # Check dependencies
  python listen.py --check-deps
        """
    )

    parser.add_argument('-f', '--file', help='Input text file')
    parser.add_argument('-o', '--output', default='.', help='Output directory (default: current dir)')
    parser.add_argument('-p', '--provider', choices=['huggingface', 'coqui'],
                        help='TTS provider')
    parser.add_argument('-m', '--model', help='Specific model name (optional)')
    parser.add_argument('-t', '--tier', choices=['BASIC', 'INTERMEDIATE', 'ADVANCED'],
                        help='Quality tier')
    parser.add_argument('--llm-provider', choices=['ollama', 'huggingface'],
                        help='LLM provider for context enhancement (ADVANCED tier)')
    parser.add_argument('--llm-model', help='LLM model for context enhancement')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies')

    args = parser.parse_args()

    # Check dependencies
    if args.check_deps or not DEPS_OK:
        check_dependencies()
        return

    # List models
    if args.list_models:
        print("🎙️  Available TTS Models:\n")
        for tier, providers in TTS_MODELS.items():
            print(f"{'='*70}")
            print(f"⚡ {tier} TIER")
            print(f"{'='*70}")
            for provider, models in providers.items():
                print(f"\n📦 {provider.upper()} Provider:")
                for model in models:
                    print(f"   • {model['name']}")
                    print(f"     - Language: {model['lang']}")
                    print(f"     - RAM: {model['ram']}")
                    print(f"     - {model['description']}")
                    if 'quality' in model:
                        print(f"     - Quality: {model['quality']}")
                    print()
        return

    # Validate required arguments
    if not args.file or not args.provider or not args.tier:
        parser.print_help()
        print("\n❌ Error: --file, --provider, and --tier are required!")
        sys.exit(1)

    # Check input file
    if not Path(args.file).exists():
        print(f"❌ Error: Input file not found: {args.file}")
        sys.exit(1)

    # Initialize and generate
    try:
        generator = AudioGenerator(args)
        output_file = generator.generate()

        if output_file:
            print(f"\n✅ Success! Audio saved to: {output_file}")
        else:
            print(f"\n❌ Audio generation failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()