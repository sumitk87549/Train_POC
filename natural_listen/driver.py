#!/usr/bin/env python3
"""
Complete Audiobook Production Pipeline
One-command solution: Book Text → Human Narration → Professional Audio
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


class AudiobookPipeline:
    """Complete audiobook production pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.temp_dir = Path(".audiobook_temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.narrator_config = {
            'provider': args.narrator_provider,
            'model': args.narrator_model,
            'quality': args.quality
        }
        
        self.tts_config = {
            'model': args.tts_model,
            'type': args.tts_type,
            'device': args.device
        }
    
    def run(self):
        """Run the complete pipeline."""
        print("=" * 80)
        print("🎬 PROFESSIONAL AUDIOBOOK PRODUCTION PIPELINE")
        print("=" * 80)
        print(f"📖 Input: {self.args.input}")
        print(f"🎭 Narrator: {self.narrator_config['provider']}/{self.narrator_config['model']}")
        print(f"🎙️ TTS: {self.tts_config['model']}")
        print(f"💾 Output: {self.args.output}")
        print("=" * 80)
        
        try:
            # Step 1: Generate human-like transcription
            print(f"\n{'='*80}")
            print(f"STEP 1: CREATING HUMAN-LIKE NARRATION SCRIPT")
            print(f"{'='*80}")
            transcription_file = self._run_narrator()
            
            if not transcription_file or not Path(transcription_file).exists():
                raise RuntimeError("Transcription generation failed")
            
            print(f"\n✅ Transcription ready: {transcription_file}")
            
            # Step 2: Generate audio from transcription
            print(f"\n{'='*80}")
            print(f"STEP 2: GENERATING PROFESSIONAL AUDIO")
            print(f"{'='*80}")
            audio_file = self._run_tts(transcription_file)
            
            if not audio_file or not Path(audio_file).exists():
                raise RuntimeError("Audio generation failed")
            
            print(f"\n✅ Audio ready: {audio_file}")
            
            # Step 3: Copy to output location
            output_path = Path(self.args.output)
            if output_path.is_dir():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_audio = output_path / f"audiobook_{timestamp}.mp3"
            else:
                final_audio = output_path
            
            import shutil
            shutil.copy2(audio_file, final_audio)
            
            # Cleanup if requested
            if self.args.cleanup:
                print(f"\n🧹 Cleaning up temporary files...")
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            else:
                print(f"\n💾 Transcription saved: {transcription_file}")
                print(f"   (Use --cleanup to remove temp files)")
            
            # Final summary
            file_size = final_audio.stat().st_size / 1e6
            
            print(f"\n{'='*80}")
            print(f"🎉 AUDIOBOOK PRODUCTION COMPLETE!")
            print(f"{'='*80}")
            print(f"🎵 Audio file: {final_audio}")
            print(f"📊 File size: {file_size:.2f} MB")
            print(f"📝 Transcription: {transcription_file}")
            print(f"{'='*80}")
            print(f"\n✨ Your professional audiobook is ready!")
            
            return str(final_audio)
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise
    
    def _run_narrator(self):
        """Run the narrator script."""
        cmd = [
            sys.executable, "narrator.py",
            "-f", str(self.args.input),
            "-p", self.narrator_config['provider'],
            "-o", str(self.temp_dir),
            "--quality", self.narrator_config['quality']
        ]
        
        if self.narrator_config['model']:
            cmd.extend(["-m", self.narrator_config['model']])
        
        if self.args.device == "cuda":
            cmd.extend(["--device", "cuda"])
        
        print(f"\n▶️ Running: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            raise RuntimeError(f"Narrator failed with code {result.returncode}")
        
        # Find the generated transcription file
        transcriptions = list(self.temp_dir.glob("transcriptions/transcription_*.txt"))
        if not transcriptions:
            raise RuntimeError("No transcription file found")
        
        # Return the most recent one
        return str(sorted(transcriptions)[-1])
    
    def _run_tts(self, transcription_file):
        """Run the TTS script."""
        cmd = [
            sys.executable, "enhanced_tts.py",
            "-f", str(transcription_file),
            "-m", self.tts_config['model'],
            "-t", self.tts_config['type'],
            "-o", str(self.temp_dir),
            "--device", self.args.device
        ]
        
        print(f"\n▶️ Running: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            raise RuntimeError(f"TTS failed with code {result.returncode}")
        
        # Find the generated audio file
        audio_files = list(self.temp_dir.glob("audio/narration_*.mp3"))
        if not audio_files:
            raise RuntimeError("No audio file found")
        
        # Return the most recent one
        return str(sorted(audio_files)[-1])


def main():
    parser = argparse.ArgumentParser(
        description='Complete Audiobook Production Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script combines narrator.py and enhanced_tts.py into one seamless workflow.

QUICK START EXAMPLES:

1. Best Quality (GPU recommended):
   python audiobook.py -i book.txt -o audiobooks/ \\
       --narrator-provider ollama --narrator-model qwen2.5:7b \\
       --tts-model suno/bark --device cuda

2. Good Quality (CPU friendly):
   python audiobook.py -i book.txt -o audiobooks/ \\
       --narrator-provider ollama --narrator-model llama3.2:3b \\
       --tts-model facebook/mms-tts-hin

3. Fast Processing:
   python audiobook.py -i book.txt -o audiobooks/ \\
       --narrator-provider ollama --narrator-model llama3.2:1b \\
       --tts-model facebook/mms-tts-hin --quality medium

4. Maximum Quality (GPU required):
   python audiobook.py -i book.txt -o audiobooks/ \\
       --narrator-provider ollama --narrator-model qwen2.5:14b \\
       --tts-model tts_models/multilingual/multi-dataset/xtts_v2 \\
       --tts-type coqui --device cuda --quality high

RECOMMENDED CONFIGURATIONS:

For Hindi Books:
  Narrator: ollama/qwen2.5:7b (best for Hindi understanding)
  TTS: facebook/mms-tts-hin (fast, good quality)
  
For English Books:
  Narrator: ollama/qwen2.5:7b or mistral:7b
  TTS: suno/bark (best emotional expression)
  
For Mixed Hindi-English:
  Narrator: ollama/qwen2.5:7b
  TTS: suno/bark or tts_models/multilingual/multi-dataset/xtts_v2

QUALITY LEVELS:
  - medium: Faster processing, good for drafts
  - high: Best quality, recommended for final production

REQUIREMENTS:
  1. Ollama installed and running (for narrator)
  2. Python packages: transformers, torch, soundfile, pydub
  3. Optional: TTS package for Coqui models
  4. ffmpeg installed (for audio processing)

SETUP:
  # Install Ollama
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen2.5:7b
  
  # Install Python packages
  pip install transformers torch soundfile pydub ollama
  
  # Optional for Coqui
  pip install TTS
        """
    )
    
    # Input/Output
    parser.add_argument('-i', '--input', required=True,
                        help='Input text file (book/summary)')
    parser.add_argument('-o', '--output', default='audiobooks',
                        help='Output directory or file path')
    
    # Narrator settings
    parser.add_argument('--narrator-provider', choices=['ollama', 'huggingface'],
                        default='ollama',
                        help='LLM provider for narration (default: ollama)')
    parser.add_argument('--narrator-model',
                        help='Narrator model (default: qwen2.5:7b for ollama)')
    parser.add_argument('--quality', choices=['medium', 'high'], default='high',
                        help='Narration quality level')
    
    # TTS settings
    parser.add_argument('--tts-model', default='suno/bark',
                        help='TTS model name')
    parser.add_argument('--tts-type', choices=['bark', 'vits', 'coqui'],
                        default='bark',
                        help='TTS model type (auto-detected from model name if not specified)')
    
    # Hardware
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='Device to use for processing')
    
    # Options
    parser.add_argument('--cleanup', action='store_true',
                        help='Remove temporary files after completion')
    parser.add_argument('--keep-transcription', action='store_true',
                        help='Keep transcription file (deprecated, always kept unless --cleanup)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Auto-detect TTS type from model name if not specified
    if '--tts-type' not in sys.argv:
        model_lower = args.tts_model.lower()
        if 'bark' in model_lower:
            args.tts_type = 'bark'
        elif 'mms-tts' in model_lower or 'vits' in model_lower:
            args.tts_type = 'vits'
        elif 'xtts' in model_lower or 'coqui' in model_lower:
            args.tts_type = 'coqui'
    
    # Check dependencies
    missing = []
    
    if args.narrator_provider == 'ollama':
        try:
            import ollama
            # Test connection
            ollama.list()
        except ImportError:
            missing.append("ollama (pip install ollama)")
        except Exception as e:
            print(f"⚠️ Warning: Cannot connect to Ollama: {e}")
            print("   Make sure Ollama is running: ollama serve")
    
    try:
        import torch
        import transformers
    except ImportError:
        missing.append("torch transformers (pip install torch transformers)")
    
    try:
        import soundfile
        import pydub
    except ImportError:
        missing.append("soundfile pydub (pip install soundfile pydub)")
    
    if args.tts_type == 'coqui':
        try:
            from TTS.api import TTS
        except ImportError:
            missing.append("TTS (pip install TTS)")
    
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        sys.exit(1)
    
    # Run pipeline
    try:
        pipeline = AudiobookPipeline(args)
        output_file = pipeline.run()
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()