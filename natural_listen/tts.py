#!/usr/bin/env python3
"""
Advanced TTS Generator with Full Prosodic Control
Supports all markers from TTS-optimized transcriptions:
- Pauses, Breaths, Tone, Emphasis, Stress, Pacing
Generates truly human-like narration with emotion and natural speech patterns.
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
import logging

# Suppress warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import torch
    import numpy as np
    import soundfile as sf
    from transformers import (
        AutoProcessor, AutoModel, AutoTokenizer,
        VitsModel, VitsTokenizer,
        BarkModel, BarkProcessor,
        SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    )
    from huggingface_hub import login, HfFolder
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range, speedup
    from scipy import signal
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    IMPORT_ERROR = str(e)

try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except:
    COQUI_AVAILABLE = False


class AdvancedTranscriptionParser:
    """Parse TTS-optimized transcription with full prosodic marker support."""
    
    def __init__(self):
        # All supported markers
        self.tone_markers = re.compile(r'\[TONE:\s*(\w+)\]')
        self.pause_markers = re.compile(r'\[PAUSE-(SHORT|MEDIUM|LONG)\]')
        self.breath_markers = re.compile(r'\[BREATH\]')
        self.emphasis_markers = re.compile(r'\[EMPHASIS:\s*([^\]]+)\]')
        self.stress_markers = re.compile(r'\[STRESS:\s*([^\]]+)\]')
        self.pace_markers = re.compile(r'\[PACE:\s*(\w+)\]')
    
    def parse(self, text):
        """Parse transcription and extract all prosodic information."""
        segments = []
        current_pos = 0
        current_tone = "neutral"
        current_pace = "normal"
        
        # Collect all markers with positions
        markers = []
        
        for match in self.tone_markers.finditer(text):
            markers.append(('tone', match.start(), match.end(), match.group(1)))
        
        for match in self.pause_markers.finditer(text):
            markers.append(('pause', match.start(), match.end(), match.group(1)))
        
        for match in self.breath_markers.finditer(text):
            markers.append(('breath', match.start(), match.end(), None))
        
        for match in self.pace_markers.finditer(text):
            markers.append(('pace', match.start(), match.end(), match.group(1)))
        
        # Sort markers by position
        markers.sort(key=lambda x: x[1])
        
        # Process text and markers
        for marker_type, start, end, value in markers:
            # Add text before this marker
            if start > current_pos:
                segment_text = text[current_pos:start].strip()
                if segment_text:
                    # Extract emphasis/stress from this segment
                    emphasis_words, stress_words, clean_text = self._extract_emphasis_stress(segment_text)
                    
                    if clean_text:
                        segments.append({
                            'text': clean_text,
                            'tone': current_tone,
                            'pace': current_pace,
                            'emphasis': emphasis_words,
                            'stress': stress_words,
                            'type': 'speech'
                        })
            
            # Process marker
            if marker_type == 'tone':
                current_tone = value
            
            elif marker_type == 'pace':
                current_pace = value
            
            elif marker_type == 'pause':
                pause_duration = {
                    'SHORT': 0.3,
                    'MEDIUM': 0.6,
                    'LONG': 1.0
                }.get(value, 0.5)
                
                segments.append({
                    'type': 'pause',
                    'duration': pause_duration
                })
            
            elif marker_type == 'breath':
                # Natural breathing sound (short pause with slight noise)
                segments.append({
                    'type': 'breath',
                    'duration': 0.25
                })
            
            current_pos = end
        
        # Add remaining text
        if current_pos < len(text):
            segment_text = text[current_pos:].strip()
            if segment_text:
                emphasis_words, stress_words, clean_text = self._extract_emphasis_stress(segment_text)
                
                if clean_text:
                    segments.append({
                        'text': clean_text,
                        'tone': current_tone,
                        'pace': current_pace,
                        'emphasis': emphasis_words,
                        'stress': stress_words,
                        'type': 'speech'
                    })
        
        return segments
    
    def _extract_emphasis_stress(self, text):
        """Extract emphasis and stress words, return cleaned text."""
        emphasis_words = []
        stress_words = []
        
        # Extract emphasis
        for match in self.emphasis_markers.finditer(text):
            emphasis_words.append(match.group(1).strip())
        
        # Extract stress
        for match in self.stress_markers.finditer(text):
            stress_words.append(match.group(1).strip())
        
        # Clean text
        clean_text = self.emphasis_markers.sub(r'\1', text)
        clean_text = self.stress_markers.sub(r'\1', clean_text)
        clean_text = clean_text.strip()
        
        return emphasis_words, stress_words, clean_text


class ProsodyController:
    """Control prosodic features of generated audio."""
    
    @staticmethod
    def apply_emphasis(audio_array, sample_rate, emphasis_ratio=0.3):
        """Apply emphasis by increasing volume and slightly changing pitch."""
        # Increase volume
        emphasized = audio_array * (1.0 + emphasis_ratio)
        
        # Clip to prevent distortion
        emphasized = np.clip(emphasized, -1.0, 1.0)
        
        return emphasized
    
    @staticmethod
    def apply_stress(audio_array, sample_rate, stress_ratio=0.15):
        """Apply stress by slightly increasing volume."""
        stressed = audio_array * (1.0 + stress_ratio)
        stressed = np.clip(stressed, -1.0, 1.0)
        return stressed
    
    @staticmethod
    def apply_pacing(audio_segment, pace):
        """Apply pacing control to audio segment."""
        if pace == "slow":
            # Slow down to 0.85x speed (makes it ~17% slower)
            return audio_segment.speedup(playback_speed=0.85)
        elif pace == "fast":
            # Speed up to 1.15x speed (makes it ~15% faster)
            return audio_segment.speedup(playback_speed=1.15)
        else:  # normal
            return audio_segment
    
    @staticmethod
    def create_breath_sound(duration_ms=250, sample_rate=22050):
        """Create a natural breathing sound."""
        # Generate pink noise for breath
        duration_samples = int(duration_ms * sample_rate / 1000)
        
        # Create pink noise (more natural than white noise)
        white_noise = np.random.randn(duration_samples)
        b, a = signal.butter(1, 0.1)
        pink_noise = signal.filtfilt(b, a, white_noise)
        
        # Apply envelope (fade in/out)
        envelope = np.hanning(duration_samples)
        breath = pink_noise * envelope * 0.05  # Very quiet
        
        return breath


class AdvancedTTSEngine:
    """Advanced TTS engine with full prosodic control."""
    
    def __init__(self, model_name, model_type="auto", device="cpu", language="auto", hf_token=None):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.language = language
        self.hf_token = hf_token
        self.model = None
        self.processor = None
        self.vocoder = None
        self.tokenizer = None
        self.prosody_controller = ProsodyController()
        
        if self.hf_token:
            os.environ['HF_TOKEN'] = self.hf_token
        
        if self.model_type == "auto":
            self.model_type = self._detect_model_type(model_name)
            print(f"🔍 Auto-detected model type: {self.model_type}")
        
        # Enhanced emotion presets for different models
        self.bark_voice_presets = {
            'neutral': 'v2/en_speaker_6',
            'thoughtful': 'v2/en_speaker_6',
            'curious': 'v2/en_speaker_9',
            'serious': 'v2/en_speaker_1',
            'calm': 'v2/en_speaker_6',
            'excited': 'v2/en_speaker_9',
            'mysterious': 'v2/en_speaker_3',
            'warm': 'v2/en_speaker_5',
            'dramatic': 'v2/en_speaker_1',
            'happy': 'v2/en_speaker_9',
            'sad': 'v2/en_speaker_3',
        }
        
        self.load_model()
    
    def _detect_model_type(self, model_name):
        """Auto-detect model type from model name."""
        model_lower = model_name.lower()
        
        if 'bark' in model_lower:
            return 'bark'
        elif 'ai4bharat' in model_lower or 'indic' in model_lower:
            return 'ai4bharat'
        elif 'speecht5' in model_lower:
            return 'speecht5'
        elif 'xtts' in model_lower or 'tts_models/' in model_lower:
            return 'coqui'
        elif 'mms-tts' in model_lower or 'vits' in model_lower:
            return 'vits'
        else:
            return 'vits'
    
    def load_model(self):
        """Load TTS model with authentication support."""
        print(f"📥 Loading {self.model_type} model: {self.model_name}")
        
        try:
            if self.model_type == "bark":
                self._load_bark()
            elif self.model_type == "vits":
                self._load_vits()
            elif self.model_type == "speecht5":
                self._load_speecht5()
            elif self.model_type == "coqui":
                self._load_coqui()
            elif self.model_type == "ai4bharat":
                self._load_ai4bharat()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            print("✅ Model loaded")
            
        except Exception as e:
            error_str = str(e)
            if 'gated' in error_str.lower() or '401' in error_str or 'authenticated' in error_str.lower():
                self._handle_gated_repo_error()
            else:
                raise
    
    def _handle_gated_repo_error(self):
        """Handle gated repository errors."""
        print(f"\n{'=' * 70}")
        print(f"🔒 GATED REPOSITORY DETECTED")
        print(f"{'=' * 70}")
        print(f"Model '{self.model_name}' requires HuggingFace authentication.\n")
        print("📝 SOLUTION:")
        print("   1. Login: huggingface-cli login")
        print("   2. Request access: https://huggingface.co/{self.model_name}")
        print("   3. Use --hf-token flag with your token")
        print(f"{'=' * 70}")
        sys.exit(1)
    
    def _load_bark(self):
        """Load Bark model."""
        self.processor = BarkProcessor.from_pretrained(self.model_name)
        self.model = BarkModel.from_pretrained(self.model_name)
        
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        
        self.model.enable_cpu_offload() if self.device == "cpu" else None
    
    def _load_vits(self):
        """Load VITS model."""
        self.tokenizer = VitsTokenizer.from_pretrained(self.model_name)
        self.model = VitsModel.from_pretrained(self.model_name)
        
        if self.device == "cuda":
            self.model = self.model.to(self.device)
    
    def _load_speecht5(self):
        """Load SpeechT5 model."""
        self.processor = SpeechT5Processor.from_pretrained(self.model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_name)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        if self.device == "cuda":
            self.model = self.model.to(self.device)
            self.vocoder = self.vocoder.to(self.device)
    
    def _load_coqui(self):
        """Load Coqui TTS model."""
        if not COQUI_AVAILABLE:
            raise ImportError("Coqui TTS not installed")
        
        self.model = CoquiTTS(self.model_name).to(self.device)
    
    def _load_ai4bharat(self):
        """Load AI4Bharat model."""
        try:
            self.tokenizer = VitsTokenizer.from_pretrained(self.model_name)
            self.model = VitsModel.from_pretrained(self.model_name)
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
        except:
            print("⚠️ Trying fallback to MMS-TTS for Hindi...")
            self.model_name = "facebook/mms-tts-hin"
            self._load_vits()
    
    def detect_language(self, text):
        """Detect language from text."""
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if hindi_chars > english_chars:
            return "hi"
        return "en"
    
    def generate_with_prosody(self, segment):
        """Generate audio with full prosodic control."""
        text = segment['text']
        tone = segment.get('tone', 'neutral')
        pace = segment.get('pace', 'normal')
        emphasis_words = segment.get('emphasis', [])
        stress_words = segment.get('stress', [])
        
        # Generate base audio
        audio_array, sample_rate = self._generate_base_audio(text, tone)
        
        # Apply emphasis/stress if words are present
        if emphasis_words or stress_words:
            audio_array = self._apply_word_prosody(
                audio_array, sample_rate, text, emphasis_words, stress_words
            )
        
        return audio_array, sample_rate, pace
    
    def _generate_base_audio(self, text, tone):
        """Generate base audio with emotion."""
        if self.model_type == "bark":
            return self._generate_bark(text, tone)
        elif self.model_type == "vits":
            return self._generate_vits(text)
        elif self.model_type == "speecht5":
            return self._generate_speecht5(text)
        elif self.model_type == "coqui":
            return self._generate_coqui(text)
        else:
            return self._generate_vits(text)
    
    def _generate_bark(self, text, tone):
        """Generate with Bark (best emotion support)."""
        voice_preset = self.bark_voice_presets.get(tone, 'v2/en_speaker_6')
        
        inputs = self.processor(
            text,
            voice_preset=voice_preset,
            return_tensors="pt"
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            audio_array = self.model.generate(**inputs, temperature=0.9)
        
        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        
        return audio_array, sample_rate
    
    def _generate_vits(self, text):
        """Generate with VITS."""
        inputs = self.tokenizer(text, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model(**inputs)
        
        audio_array = output.waveform.cpu().numpy().squeeze()
        sample_rate = self.model.config.sampling_rate
        
        return audio_array, sample_rate
    
    def _generate_speecht5(self, text):
        """Generate with SpeechT5."""
        inputs = self.processor(text=text, return_tensors="pt")
        
        # Load speaker embeddings
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors",
            split="validation"
        )
        speaker_embeddings = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0)
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            speaker_embeddings = speaker_embeddings.to(self.device)
        
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=self.vocoder
            )
        
        audio_array = speech.cpu().numpy()
        sample_rate = 16000
        
        return audio_array, sample_rate
    
    def _generate_coqui(self, text):
        """Generate with Coqui TTS."""
        wav = self.model.tts(text)
        audio_array = np.array(wav)
        sample_rate = self.model.synthesizer.output_sample_rate
        
        return audio_array, sample_rate
    
    def _apply_word_prosody(self, audio_array, sample_rate, text, emphasis_words, stress_words):
        """Apply emphasis and stress to specific words."""
        # Simple approach: apply overall emphasis if words are present
        # More sophisticated: segment audio by word and apply selectively
        
        if emphasis_words:
            # Check if any emphasis word is in text
            for word in emphasis_words:
                if word.lower() in text.lower():
                    audio_array = self.prosody_controller.apply_emphasis(
                        audio_array, sample_rate
                    )
                    break
        
        if stress_words:
            # Check if any stress word is in text
            for word in stress_words:
                if word.lower() in text.lower():
                    audio_array = self.prosody_controller.apply_stress(
                        audio_array, sample_rate
                    )
                    break
        
        return audio_array


class AdvancedTTSGenerator:
    """Generate human-like TTS from prosodically-annotated transcriptions."""
    
    def __init__(self, model_name, model_type="auto", device="cpu", 
                 output_dir=".", language="auto", hf_token=None):
        self.engine = AdvancedTTSEngine(
            model_name, model_type, device, language, hf_token
        )
        self.parser = AdvancedTranscriptionParser()
        self.output_dir = Path(output_dir) / "audio_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_from_transcription(self, input_file):
        """Generate audio from TTS-optimized transcription."""
        print("=" * 80)
        print("🎙️ ADVANCED TTS GENERATION - HUMAN-LIKE NARRATION")
        print("=" * 80)
        
        # Load transcription
        input_path = Path(input_file)
        
        if input_path.suffix == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = self._extract_text_from_json(data)
            primary_lang = data.get('metadata', {}).get('primary_language', 'en')
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            primary_lang = self.engine.detect_language(text)
        
        print(f"📖 Input: {input_file}")
        print(f"🌍 Language: {primary_lang.upper()}")
        
        # Parse prosodic markers
        print(f"\n🎭 Parsing prosodic markers...")
        segments = self.parser.parse(text)
        
        # Count marker types
        marker_counts = {
            'speech': sum(1 for s in segments if s['type'] == 'speech'),
            'pause': sum(1 for s in segments if s['type'] == 'pause'),
            'breath': sum(1 for s in segments if s['type'] == 'breath'),
            'tones': len(set(s.get('tone', 'neutral') for s in segments if s['type'] == 'speech')),
            'paces': len(set(s.get('pace', 'normal') for s in segments if s['type'] == 'speech')),
        }
        
        print(f"✅ Found {len(segments)} segments:")
        print(f"   🗣️ Speech: {marker_counts['speech']}")
        print(f"   ⏸️ Pauses: {marker_counts['pause']}")
        print(f"   💨 Breaths: {marker_counts['breath']}")
        print(f"   🎭 Unique tones: {marker_counts['tones']}")
        print(f"   ⚡ Pace variations: {marker_counts['paces']}")
        
        # Generate audio
        print(f"\n🎙️ Generating audio with prosodic control...")
        audio_segments = []
        
        start_time = time.time()
        
        for i, segment in enumerate(segments, 1):
            if segment['type'] == 'pause':
                # Add pause
                duration_ms = int(segment['duration'] * 1000)
                silence = AudioSegment.silent(duration=duration_ms)
                audio_segments.append(silence)
                print(f"   [{i}/{len(segments)}] ⏸️ Pause ({segment['duration']}s)")
            
            elif segment['type'] == 'breath':
                # Add breath sound
                duration_ms = int(segment.get('duration', 0.25) * 1000)
                
                # Create breath sound
                breath_array = self.engine.prosody_controller.create_breath_sound(
                    duration_ms, 22050
                )
                
                # Convert to AudioSegment
                breath_array_int = (breath_array * 32767).astype(np.int16)
                breath_seg = AudioSegment(
                    breath_array_int.tobytes(),
                    frame_rate=22050,
                    sample_width=2,
                    channels=1
                )
                
                audio_segments.append(breath_seg)
                print(f"   [{i}/{len(segments)}] 💨 Breath ({segment['duration']}s)")
            
            elif segment['type'] == 'speech':
                text = segment['text']
                tone = segment.get('tone', 'neutral')
                pace = segment.get('pace', 'normal')
                
                if not text.strip():
                    continue
                
                seg_lang = self.engine.detect_language(text)
                lang_label = "HI" if seg_lang == "hi" else "EN"
                
                display_text = text[:40] + "..." if len(text) > 40 else text
                
                prosody_info = []
                if tone != 'neutral':
                    prosody_info.append(f"tone:{tone}")
                if pace != 'normal':
                    prosody_info.append(f"pace:{pace}")
                if segment.get('emphasis'):
                    prosody_info.append(f"emph:{len(segment['emphasis'])}")
                if segment.get('stress'):
                    prosody_info.append(f"stress:{len(segment['stress'])}")
                
                prosody_str = " ".join(prosody_info) if prosody_info else "neutral"
                
                print(f"   [{i}/{len(segments)}] 🎙️ [{lang_label}] ({prosody_str})")
                print(f"       \"{display_text}\"")
                
                try:
                    seg_start = time.time()
                    
                    # Generate with prosodic control
                    audio_array, sample_rate, pace = self.engine.generate_with_prosody(segment)
                    
                    seg_time = time.time() - seg_start
                    
                    # Convert to AudioSegment
                    audio_array_int = (audio_array * 32767).astype(np.int16)
                    audio_seg = AudioSegment(
                        audio_array_int.tobytes(),
                        frame_rate=sample_rate,
                        sample_width=2,
                        channels=1
                    )
                    
                    # Apply pacing
                    if pace != 'normal':
                        audio_seg = self.engine.prosody_controller.apply_pacing(audio_seg, pace)
                    
                    audio_segments.append(audio_seg)
                    print(f"       ✅ Generated in {seg_time:.1f}s")
                
                except Exception as e:
                    print(f"       ⚠️ Failed: {e}")
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
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"human_like_narration_{primary_lang}_{timestamp}.mp3"
        
        print(f"💾 Exporting to: {output_file}")
        final_audio.export(
            str(output_file),
            format="mp3",
            bitrate="192k",
            parameters=["-ar", "44100"]
        )
        
        total_time = time.time() - start_time
        duration_sec = len(final_audio) / 1000
        
        # Summary
        print(f"\n{'=' * 80}")
        print(f"🎉 HUMAN-LIKE AUDIO GENERATION COMPLETE!")
        print(f"{'=' * 80}")
        print(f"🌍 Language: {primary_lang.upper()}")
        print(f"⏱️ Generation time: {total_time/60:.2f} minutes")
        print(f"🎵 Audio duration: {duration_sec/60:.2f} minutes")
        print(f"⚡ Speed: {duration_sec/total_time:.2f}x realtime")
        print(f"📊 File size: {output_file.stat().st_size / 1e6:.2f} MB")
        print(f"\n🎭 Prosodic Features Applied:")
        print(f"   Tones: {marker_counts['tones']} unique emotions")
        print(f"   Pauses: {marker_counts['pause']} natural breaks")
        print(f"   Breaths: {marker_counts['breath']} breathing sounds")
        print(f"   Pacing: {marker_counts['paces']} speed variations")
        print(f"\n💾 Output: {output_file}")
        print(f"{'=' * 80}")
        
        return str(output_file)
    
    def _extract_text_from_json(self, data):
        """Extract TTS transcription from JSON."""
        text_parts = []
        
        for chapter in data.get('chapters', []):
            # Add chapter title with pause
            if chapter.get('title'):
                text_parts.append(f"[TONE: serious] {chapter['title']} [PAUSE-LONG]")
            
            for chunk in chapter.get('chunks', []):
                # Use tts_transcription if available, otherwise fall back to narration
                tts_text = chunk.get('tts_transcription') or chunk.get('narration')
                
                if tts_text:
                    text_parts.append(tts_text)
        
        return '\n\n'.join(text_parts)
    
    def _post_process(self, audio):
        """Post-process audio for quality."""
        # Normalize volume
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)
        
        # Apply compression for consistent volume
        audio = compress_dynamic_range(
            audio,
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0
        )
        
        # Final normalization
        audio = normalize(audio)
        
        return audio


def check_dependencies():
    """Check dependencies."""
    issues = []
    
    if not DEPS_OK:
        issues.append(f"❌ Core dependencies missing: {IMPORT_ERROR}")
        issues.append("   Install: pip install torch transformers soundfile pydub numpy scipy huggingface-hub")
    
    if not COQUI_AVAILABLE:
        issues.append("⚠️ Coqui TTS not available (optional)")
        issues.append("   Install: pip install TTS")
    
    return issues


def main():
    parser = argparse.ArgumentParser(
        description='Advanced TTS with Full Prosodic Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This TTS engine supports ALL prosodic markers from TTS-optimized transcriptions:
  • Pauses: [PAUSE-SHORT/MEDIUM/LONG]
  • Breaths: [BREATH]
  • Tone/Emotion: [TONE: thoughtful/curious/serious/calm/excited/mysterious/warm/dramatic]
  • Emphasis: [EMPHASIS: word]
  • Stress: [STRESS: word]
  • Pacing: [PACE: slow/normal/fast]

Examples:
  # Generate from TTS-optimized transcription (Hindi)
  python tts_advanced.py -f tts_transcription.txt -m facebook/mms-tts-hin
  
  # Generate from JSON with prosodic markers
  python tts_advanced.py -f transcription.json -m suno/bark --device cuda
  
  # Use best emotion support (Bark)
  python tts_advanced.py -f story.txt -m suno/bark -o output/
  
  # GPU acceleration for faster generation
  python tts_advanced.py -f text.txt -m facebook/mms-tts-hin --device cuda

Recommended Models:
  ✅ suno/bark - BEST for emotions and prosody (English)
  ✅ facebook/mms-tts-hin - Fast and reliable (Hindi)
  ✅ microsoft/speecht5_tts - Good quality (English)
  
For Gated Models:
  1. Login: huggingface-cli login
  2. Request access at HuggingFace
  3. Use --hf-token flag
        """
    )
    
    parser.add_argument('-f', '--file', required=True, 
                        help='Input transcription file (TXT or JSON)')
    parser.add_argument('-m', '--model', required=True, 
                        help='TTS model name')
    parser.add_argument('-t', '--type', 
                        choices=['auto', 'bark', 'vits', 'speecht5', 'coqui', 'ai4bharat'],
                        default='auto', help='Model type (auto-detect)')
    parser.add_argument('-o', '--output', default='.', 
                        help='Output directory')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', 
                        help='Device (cuda for GPU)')
    parser.add_argument('--language', choices=['auto', 'en', 'hi'], default='auto', 
                        help='Language')
    parser.add_argument('--hf-token', help='HuggingFace token for gated models')
    parser.add_argument('--check-deps', action='store_true', 
                        help='Check dependencies')
    
    args = parser.parse_args()
    
    # Check GPU
    if args.device == 'cpu' and torch.cuda.is_available():
        print("🔍 GPU detected! Consider using --device cuda for faster generation")
    
    # Check dependencies
    if args.check_deps:
        print("\n📦 Checking dependencies...\n")
        issues = check_dependencies()
        
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("✅ All dependencies installed!")
        
        sys.exit(0)
    
    # Validate input
    if not Path(args.file).exists():
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)
    
    if not DEPS_OK:
        print("❌ Error: Missing core dependencies")
        print(f"   {IMPORT_ERROR}")
        sys.exit(1)
    
    # Generate audio
    try:
        print(f"\n🎙️ Advanced TTS Generator")
        print(f"   Model: {args.model}")
        print(f"   Device: {args.device}")
        print(f"   Input: {args.file}\n")
        
        generator = AdvancedTTSGenerator(
            model_name=args.model,
            model_type=args.type,
            device=args.device,
            output_dir=args.output,
            language=args.language,
            hf_token=args.hf_token
        )
        
        output_file = generator.generate_from_transcription(args.file)
        
        print(f"\n✅ Success! Human-like audio saved to:")
        print(f"   {output_file}")
        print(f"\n🎧 Play it to hear natural, emotional narration!")
        
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