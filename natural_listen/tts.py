#!/usr/bin/env python3
"""
Enhanced Multilingual TTS with Human-like Narration Support (Hindi + English)
Processes transcriptions with emotional markers and pronunciation guides
Language-agnostic markers ensure compatibility across languages
"""

import os
import sys
import argparse
import json
import time
import warnings
import re
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import torch
    import numpy as np
    import soundfile as sf
    from transformers import (AutoProcessor, AutoModel, VitsModel, VitsTokenizer, 
                            BarkModel, BarkProcessor, SpeechT5Processor, SpeechT5ForTextToSpeech,
                            SpeechT5HifiGan)
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    DEPS_OK = True
except ImportError:
    DEPS_OK = False

try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except:
    COQUI_AVAILABLE = False


class TranscriptionParser:
    """Parse human-like transcription with language-agnostic markers."""
    
    def __init__(self):
        # Language-agnostic markers (English only for TTS compatibility)
        self.tone_markers = re.compile(r'\[TONE:\s*(\w+)\]')
        self.pause_markers = re.compile(r'\[PAUSE-(SHORT|MEDIUM|LONG)\]')
        self.pronounce_markers = re.compile(r'(\S+)\s*\[PRONOUNCE:\s*([^\]]+)\]')
        self.emphasis_markers = re.compile(r'\[EMPHASIS:\s*([^\]]+)\]')
    
    def parse(self, text):
        """Parse transcription and extract emotional/contextual information."""
        segments = []
        current_pos = 0
        current_tone = "neutral"
        
        # Find all markers
        markers = []
        
        # Tone markers
        for match in self.tone_markers.finditer(text):
            markers.append(('tone', match.start(), match.end(), match.group(1)))
        
        # Pause markers
        for match in self.pause_markers.finditer(text):
            markers.append(('pause', match.start(), match.end(), match.group(1)))
        
        # Sort markers by position
        markers.sort(key=lambda x: x[1])
        
        # Build segments
        for marker_type, start, end, value in markers:
            # Get text before marker
            if start > current_pos:
                segment_text = text[current_pos:start].strip()
                if segment_text:
                    segment_text = self._clean_markers(segment_text)
                    segments.append({
                        'text': segment_text,
                        'tone': current_tone,
                        'type': 'speech'
                    })
            
            # Process marker
            if marker_type == 'tone':
                current_tone = value
            elif marker_type == 'pause':
                pause_duration = {
                    'SHORT': 0.3,
                    'MEDIUM': 0.6,
                    'LONG': 1.0
                }.get(value, 0.5)
                
                segments.append({
                    'text': '',
                    'duration': pause_duration,
                    'type': 'pause'
                })
            
            current_pos = end
        
        # Get remaining text
        if current_pos < len(text):
            segment_text = text[current_pos:].strip()
            if segment_text:
                segment_text = self._clean_markers(segment_text)
                segments.append({
                    'text': segment_text,
                    'tone': current_tone,
                    'type': 'speech'
                })
        
        return segments
    
    def _clean_markers(self, text):
        """Remove all markers from text."""
        text = self.tone_markers.sub('', text)
        text = self.pause_markers.sub('', text)
        text = self.pronounce_markers.sub(r'\1', text)
        text = self.emphasis_markers.sub(r'\1', text)
        return text.strip()
    
    def extract_pronunciation_guides(self, text):
        """Extract pronunciation guides for difficult words."""
        guides = {}
        for match in self.pronounce_markers.finditer(text):
            word = match.group(1)
            pronunciation = match.group(2)
            guides[word] = pronunciation
        return guides


class MultilingualTTSEngine:
    """TTS engine with multilingual and emotional awareness."""
    
    def __init__(self, model_name, model_type, device="cpu", language="auto"):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.language = language
        self.model = None
        self.processor = None
        self.vocoder = None
        
        # Bark voice presets for different emotions
        self.bark_voice_presets = {
            'neutral': 'v2/en_speaker_6',
            'happy': 'v2/en_speaker_9',
            'sad': 'v2/en_speaker_3',
            'excited': 'v2/en_speaker_9',
            'serious': 'v2/en_speaker_1',
            'thoughtful': 'v2/en_speaker_6',
            'angry': 'v2/en_speaker_1',
            'calm': 'v2/en_speaker_6',
            'worried': 'v2/en_speaker_3',
            'determined': 'v2/en_speaker_1',
            'curious': 'v2/en_speaker_9',
        }
        
        self.load_model()
    
    def load_model(self):
        """Load TTS model."""
        print(f"📥 Loading {self.model_type} model: {self.model_name}")
        
        if self.model_type == "bark":
            self.processor = BarkProcessor.from_pretrained(self.model_name)
            self.model = BarkModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            if self.device == "cuda":
                try:
                    self.model = self.model.to_bettertransformer()
                    print("   ✅ Optimized for GPU")
                except:
                    pass
        
        elif self.model_type == "vits":
            # VITS supports multiple languages including Hindi
            print(f"   Language: {self.language}")
            self.tokenizer = VitsTokenizer.from_pretrained(self.model_name)
            self.model = VitsModel.from_pretrained(self.model_name).to(self.device)
        
        elif self.model_type == "speecht5":
            # Microsoft's SpeechT5 - good for English
            self.processor = SpeechT5Processor.from_pretrained(self.model_name)
            self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_name).to(self.device)
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        elif self.model_type == "coqui":
            if not COQUI_AVAILABLE:
                raise ImportError("Coqui TTS not installed")
            # Coqui TTS supports many languages including Hindi
            self.model = CoquiTTS(model_name=self.model_name, gpu=(self.device=="cuda"))
        
        print("✅ Model loaded")
    
    def detect_language(self, text):
        """Detect text language."""
        # Count Hindi (Devanagari) characters
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        # Count English characters
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = hindi_chars + english_chars
        if total_chars == 0:
            return "en"
        
        hindi_ratio = hindi_chars / total_chars
        return "hi" if hindi_ratio > 0.3 else "en"
    
    def generate_with_emotion(self, text, tone="neutral", sample_rate=24000):
        """Generate audio with emotional context."""
        # Detect language if auto
        if self.language == "auto":
            detected_lang = self.detect_language(text)
        else:
            detected_lang = self.language
        
        if self.model_type == "bark":
            return self._generate_bark_emotional(text, tone, sample_rate)
        elif self.model_type == "vits":
            return self._generate_vits(text, sample_rate, detected_lang)
        elif self.model_type == "speecht5":
            return self._generate_speecht5(text, sample_rate)
        elif self.model_type == "coqui":
            return self._generate_coqui_emotional(text, tone, detected_lang)
    
    def _generate_bark_emotional(self, text, tone, sample_rate):
        """Generate with Bark using emotional voice presets."""
        voice_preset = self.bark_voice_presets.get(tone, 'v2/en_speaker_6')
        
        # Add emotion cues to text for better expression
        emotion_cues = {
            'happy': '♪',
            'excited': '!',
            'sad': '...',
            'serious': '.',
            'thoughtful': '...',
            'angry': '!',
            'worried': '...',
        }
        
        if tone in emotion_cues and not text.endswith(emotion_cues[tone]):
            text = text + ' ' + emotion_cues[tone]
        
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            audio_array = self.model.generate(
                **inputs,
                do_sample=True,
                semantic_temperature=0.8,
                coarse_temperature=0.7,
                fine_temperature=0.6
            )
        
        audio_array = audio_array.cpu().numpy().squeeze()
        return audio_array, 24000
    
    def _generate_vits(self, text, sample_rate, lang="en"):
        """Generate with VITS (supports Hindi and English)."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model(**inputs)
        
        waveform = output.waveform[0].cpu().numpy()
        return waveform, self.model.config.sampling_rate
    
    def _generate_speecht5(self, text, sample_rate):
        """Generate with SpeechT5."""
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
        # Load speaker embeddings (default speaker)
        speaker_embeddings = torch.zeros((1, 512)).to(self.device)
        
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"], 
                speaker_embeddings, 
                vocoder=self.vocoder
            )
        
        waveform = speech.cpu().numpy()
        return waveform, 16000
    
    def _generate_coqui_emotional(self, text, tone, lang="en"):
        """Generate with Coqui TTS with emotion and language support."""
        # Coqui XTTS supports emotion and multiple languages
        emotion_map = {
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'neutral': 'neutral',
            'excited': 'happy',
            'serious': 'neutral',
            'thoughtful': 'neutral',
            'calm': 'neutral',
            'worried': 'sad',
            'determined': 'angry',
            'curious': 'happy',
        }
        
        emotion = emotion_map.get(tone, 'neutral')
        
        # Language mapping
        lang_map = {
            'en': 'en',
            'hi': 'hi',
            'auto': 'en'
        }
        
        tts_lang = lang_map.get(lang, 'en')
        
        # Generate to temporary file
        temp_file = "/tmp/temp_tts.wav"
        
        # Check if model supports emotion and language parameters
        try:
            self.model.tts_to_file(
                text=text,
                file_path=temp_file,
                emotion=emotion,
                language=tts_lang
            )
        except TypeError:
            # Fallback if emotion/language not supported
            self.model.tts_to_file(
                text=text,
                file_path=temp_file
            )
        
        # Load and return
        audio, sr = sf.read(temp_file)
        return audio, sr


class HumanLikeTTSGenerator:
    """Generate human-like TTS from transcription (Hindi + English)."""
    
    def __init__(self, model_name, model_type, device="cpu", output_dir=".", language="auto"):
        self.engine = MultilingualTTSEngine(model_name, model_type, device, language)
        self.parser = TranscriptionParser()
        self.output_dir = Path(output_dir) / "audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_from_transcription(self, transcription_file):
        """Generate audio from transcription file."""
        print("=" * 70)
        print("🎙️ MULTILINGUAL HUMAN-LIKE TTS GENERATOR (Hindi + English)")
        print("=" * 70)
        
        # Load transcription
        print(f"\n📖 Loading transcription: {transcription_file}")
        
        if transcription_file.endswith('.json'):
            with open(transcription_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = self._extract_text_from_json(data)
            primary_lang = data.get('metadata', {}).get('primary_language', 'english')
        else:
            with open(transcription_file, 'r', encoding='utf-8') as f:
                text = f.read()
            # Auto-detect from text
            primary_lang = "hindi" if self.engine.detect_language(text) == "hi" else "english"
        
        print(f"📊 Text length: {len(text)} characters")
        print(f"🌐 Primary language: {primary_lang.upper()}")
        
        # Parse transcription
        print(f"\n🎭 Parsing emotional markers...")
        segments = self.parser.parse(text)
        print(f"✅ Found {len(segments)} segments")
        
        # Count segment types
        speech_segments = [s for s in segments if s['type'] == 'speech']
        pause_segments = [s for s in segments if s['type'] == 'pause']
        print(f"   Speech segments: {len(speech_segments)}")
        print(f"   Pause segments: {len(pause_segments)}")
        
        # Generate audio for each segment
        print(f"\n🎵 Generating audio segments...")
        audio_segments = []
        
        start_time = time.time()
        
        for i, segment in enumerate(segments, 1):
            if segment['type'] == 'pause':
                # Create silence
                duration_ms = int(segment['duration'] * 1000)
                silence = AudioSegment.silent(duration=duration_ms)
                audio_segments.append(silence)
                print(f"   [{i}/{len(segments)}] 🔇 Pause ({segment['duration']}s)")
            
            else:  # speech
                text = segment['text']
                tone = segment.get('tone', 'neutral')
                
                if not text.strip():
                    continue
                
                # Detect segment language
                seg_lang = self.engine.detect_language(text)
                lang_label = "HI" if seg_lang == "hi" else "EN"
                
                # Truncate display text
                display_text = text[:50] + "..." if len(text) > 50 else text
                print(f"   [{i}/{len(segments)}] 🎙️ [{lang_label}] ({tone}): {display_text}")
                
                try:
                    # Generate audio
                    seg_start = time.time()
                    audio_array, sample_rate = self.engine.generate_with_emotion(text, tone)
                    seg_time = time.time() - seg_start
                    
                    # Convert to AudioSegment
                    audio_array = (audio_array * 32767).astype(np.int16)
                    audio_seg = AudioSegment(
                        audio_array.tobytes(),
                        frame_rate=sample_rate,
                        sample_width=2,
                        channels=1
                    )
                    
                    audio_segments.append(audio_seg)
                    print(f"       ✅ Generated in {seg_time:.1f}s")
                
                except Exception as e:
                    print(f"       ⚠️ Failed: {e}")
                    # Add short silence as fallback
                    audio_segments.append(AudioSegment.silent(duration=500))
                    continue
        
        # Combine all segments
        print(f"\n🔗 Combining {len(audio_segments)} audio segments...")
        final_audio = AudioSegment.empty()
        for seg in audio_segments:
            final_audio += seg
        
        # Post-process
        print(f"🎛️ Post-processing audio...")
        final_audio = self._post_process(final_audio)
        
        # Export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"narration_{primary_lang}_{timestamp}.mp3"
        
        print(f"💾 Exporting to: {output_file}")
        final_audio.export(
            str(output_file),
            format="mp3",
            bitrate="192k",
            parameters=["-ar", "44100"]
        )
        
        # Summary
        total_time = time.time() - start_time
        duration_sec = len(final_audio) / 1000
        
        print(f"\n{'=' * 70}")
        print(f"🎉 AUDIO GENERATION COMPLETE!")
        print(f"{'=' * 70}")
        print(f"🌐 Language: {primary_lang.upper()}")
        print(f"⏱️ Generation time: {total_time/60:.2f} minutes")
        print(f"🎵 Audio duration: {duration_sec/60:.2f} minutes")
        print(f"⚡ Speed: {duration_sec/total_time:.2f}x realtime")
        print(f"📊 File size: {output_file.stat().st_size / 1e6:.2f} MB")
        print(f"💾 Output: {output_file}")
        print(f"{'=' * 70}")
        
        return str(output_file)
    
    def _extract_text_from_json(self, data):
        """Extract narration text from JSON transcription."""
        text_parts = []
        
        for chapter in data.get('chapters', []):
            # Add chapter title
            if chapter.get('title'):
                text_parts.append(f"[PAUSE-SHORT] {chapter['title']} [PAUSE-MEDIUM]")
            
            # Add chunks
            for chunk in chapter.get('chunks', []):
                if chunk.get('narration'):
                    text_parts.append(chunk['narration'])
                    text_parts.append('[PAUSE-SHORT]')
        
        return '\n\n'.join(text_parts)
    
    def _post_process(self, audio):
        """Post-process audio for quality."""
        # Normalize
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)
        
        # Compress dynamic range for consistent volume
        audio = compress_dynamic_range(
            audio,
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0
        )
        
        # Normalize again
        audio = normalize(audio)
        
        return audio


def main():
    parser = argparse.ArgumentParser(
        description='Multilingual Human-like TTS Generator (Hindi + English)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect language
  python tts_multilingual.py -f transcription.txt -m suno/bark
  
  # Force Hindi
  python tts_multilingual.py -f .txt -m facebook/mms-tts-hin -t vits --language hi
  
  # Force English
  python tts_multilingual.py -f transcription_english.txt -m suno/bark --language en
  
  # With GPU acceleration
  python tts_multilingual.py -f transcription.txt -m suno/bark --device cuda
  
  # Using Coqui XTTS (best quality, multilingual)
  python tts_multilingual.py -f transcription.txt -m tts_models/multilingual/multi-dataset/xtts_v2 -t coqui

Recommended Models:
  Hindi:
    - facebook/mms-tts-hin (Fast, good quality)
    - tts_models/multilingual/multi-dataset/xtts_v2 (Best quality, Coqui)
  
  English:
    - suno/bark (Best emotional expression)
    - microsoft/speecht5_tts (Fast, good quality)
    - tts_models/multilingual/multi-dataset/xtts_v2 (Best quality, Coqui)
  
  Multilingual:
    - suno/bark (Good for both, emotional)
    - tts_models/multilingual/multi-dataset/xtts_v2 (Best quality)
        """
    )
    
    parser.add_argument('-f', '--file', required=True, help='Transcription file (TXT or JSON)')
    parser.add_argument('-m', '--model', required=True, help='TTS model name')
    parser.add_argument('-t', '--type', choices=['bark', 'vits', 'speecht5', 'coqui'],
                        default='bark', help='Model type')
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='Device to use')
    parser.add_argument('--language', choices=['auto', 'en', 'hi'], default='auto',
                        help='Language (auto-detect, English, Hindi)')
    
    args = parser.parse_args()
    
    # Check file exists
    if not Path(args.file).exists():
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)
    
    # Check dependencies
    if not DEPS_OK:
        print("❌ Error: Missing dependencies")
        print("   Install: pip install torch transformers soundfile pydub")
        sys.exit(1)
    
    if args.type == "coqui" and not COQUI_AVAILABLE:
        print("❌ Error: Coqui TTS not installed")
        print("   Install: pip install TTS")
        sys.exit(1)
    
    try:
        generator = HumanLikeTTSGenerator(
            model_name=args.model,
            model_type=args.type,
            device=args.device,
            output_dir=args.output,
            language=args.language
        )
        
        output_file = generator.generate_from_transcription(args.file)
        
        print(f"\n✅ Success! Audio saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Generation interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()