#!/usr/bin/env python3
"""
Enhanced Multilingual TTS with Human-like Narration Support (Hindi + English)
Supports: Bark, VITS, SpeechT5, Coqui XTTS, AI4Bharat models
NO WARNINGS - HuggingFace Auth Supported
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

# Completely suppress all transformer warnings
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
    from pydub.effects import normalize, compress_dynamic_range
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    IMPORT_ERROR = str(e)

try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except:
    COQUI_AVAILABLE = False


# Alternative AI4Bharat models (non-gated)
ALTERNATIVE_HINDI_MODELS = {
    'ai4bharat/indic-parler-tts': [
        'facebook/mms-tts-hin',  # Fast, reliable
        'ai4bharat/indic-tts-coqui-inference-hindi',  # If available
    ],
}


class TranscriptionParser:
    """Parse human-like transcription with language-agnostic markers."""
    
    def __init__(self):
        self.tone_markers = re.compile(r'\[TONE:\s*(\w+)\]')
        self.pause_markers = re.compile(r'\[PAUSE-(SHORT|MEDIUM|LONG)\]')
        self.pronounce_markers = re.compile(r'(\S+)\s*\[PRONOUNCE:\s*([^\]]+)\]')
        self.emphasis_markers = re.compile(r'\[EMPHASIS:\s*([^\]]+)\]')
    
    def parse(self, text):
        """Parse transcription and extract emotional/contextual information."""
        segments = []
        current_pos = 0
        current_tone = "neutral"
        
        markers = []
        
        for match in self.tone_markers.finditer(text):
            markers.append(('tone', match.start(), match.end(), match.group(1)))
        
        for match in self.pause_markers.finditer(text):
            markers.append(('pause', match.start(), match.end(), match.group(1)))
        
        markers.sort(key=lambda x: x[1])
        
        for marker_type, start, end, value in markers:
            if start > current_pos:
                segment_text = text[current_pos:start].strip()
                if segment_text:
                    segment_text = self._clean_markers(segment_text)
                    segments.append({
                        'text': segment_text,
                        'tone': current_tone,
                        'type': 'speech'
                    })
            
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


class MultilingualTTSEngine:
    """TTS engine with multilingual support and HuggingFace auth."""
    
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
        
        # Set HuggingFace token if provided
        if self.hf_token:
            os.environ['HF_TOKEN'] = self.hf_token
        
        if self.model_type == "auto":
            self.model_type = self._detect_model_type(model_name)
            print(f"🔍 Auto-detected model type: {self.model_type}")
        
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
            print(f"⚠️ Could not auto-detect model type, defaulting to 'vits'")
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
        """Handle gated repository errors with helpful instructions."""
        print(f"\n{'=' * 70}")
        print(f"🔒 GATED REPOSITORY DETECTED")
        print(f"{'=' * 70}")
        print(f"The model '{self.model_name}' requires HuggingFace authentication.\n")
        
        print("📝 SOLUTION 1: Login to HuggingFace")
        print("-" * 70)
        print("Run this command in terminal:")
        print("   huggingface-cli login")
        print("\nThen paste your HuggingFace token (get it from https://huggingface.co/settings/tokens)")
        print("After logging in, request access to the model at:")
        print(f"   https://huggingface.co/{self.model_name}")
        
        print(f"\n📝 SOLUTION 2: Use Alternative Models (Recommended)")
        print("-" * 70)
        
        if self.model_name in ALTERNATIVE_HINDI_MODELS:
            alternatives = ALTERNATIVE_HINDI_MODELS[self.model_name]
            print("Try these alternative Hindi TTS models that work without authentication:\n")
            for alt in alternatives:
                print(f"   python {sys.argv[0]} -f {sys.argv[sys.argv.index('-f')+1]} -m {alt} -o {sys.argv[sys.argv.index('-o')+1]}")
            
            print(f"\n🎯 RECOMMENDED: Use facebook/mms-tts-hin (fast, reliable)")
            print(f"   python {sys.argv[0]} -f {sys.argv[sys.argv.index('-f')+1]} -m facebook/mms-tts-hin -o {sys.argv[sys.argv.index('-o')+1]}")
        
        print(f"\n📝 SOLUTION 3: Provide Token via Command Line")
        print("-" * 70)
        print(f"   python {sys.argv[0]} -f INPUT -m {self.model_name} --hf-token YOUR_TOKEN -o OUTPUT")
        
        print(f"\n{'=' * 70}\n")
        
        raise Exception(
            f"Cannot access gated model '{self.model_name}'.\n"
            f"Please authenticate with HuggingFace or use an alternative model (see above)."
        )
    
    def _load_bark(self):
        """Load Bark model with complete warning suppression."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.processor = BarkProcessor.from_pretrained(
                self.model_name,
                token=self.hf_token
            )
            self.model = BarkModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                token=self.hf_token
            ).to(self.device)
            
            if hasattr(self.model, 'generation_config'):
                if self.model.generation_config.pad_token_id is None:
                    self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
            
            if self.device == "cuda":
                try:
                    self.model = self.model.to_bettertransformer()
                    print("   ✅ Optimized for GPU")
                except:
                    pass
    
    def _load_vits(self):
        """Load VITS model."""
        print(f"   Language: {self.language}")
        try:
            self.tokenizer = VitsTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token
            )
            self.model = VitsModel.from_pretrained(
                self.model_name,
                token=self.hf_token
            ).to(self.device)
        except Exception as e:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    token=self.hf_token
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    token=self.hf_token
                ).to(self.device)
            except Exception as e2:
                raise Exception(f"Failed to load VITS model: {e}. AutoModel attempt: {e2}")
    
    def _load_speecht5(self):
        """Load SpeechT5 model."""
        self.processor = SpeechT5Processor.from_pretrained(
            self.model_name,
            token=self.hf_token
        )
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            self.model_name,
            token=self.hf_token
        ).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(self.device)
    
    def _load_coqui(self):
        """Load Coqui TTS model."""
        if not COQUI_AVAILABLE:
            raise ImportError(
                "Coqui TTS not installed.\n"
                "Install with: pip install TTS"
            )
        self.model = CoquiTTS(model_name=self.model_name, gpu=(self.device=="cuda"))
    
    def _load_ai4bharat(self):
        """Load AI4Bharat models with authentication support."""
        print(f"   Loading AI4Bharat model...")
        
        try:
            # Try loading with authentication
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                token=self.hf_token
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                token=self.hf_token
            ).to(self.device)
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            print("   ✅ Loaded with AutoModel")
            
        except Exception as e:
            # Check if it's an auth error
            error_str = str(e)
            if 'gated' in error_str.lower() or '401' in error_str or 'authenticated' in error_str.lower():
                raise  # Let the main handler deal with it
            
            # Otherwise try as VITS
            try:
                print(f"   Trying as VITS model...")
                self.tokenizer = VitsTokenizer.from_pretrained(
                    self.model_name,
                    token=self.hf_token
                )
                self.model = VitsModel.from_pretrained(
                    self.model_name,
                    token=self.hf_token
                ).to(self.device)
                print("   ✅ Loaded as VITS")
            except Exception as e2:
                raise Exception(
                    f"Failed to load AI4Bharat model.\n"
                    f"AutoModel error: {e}\n"
                    f"VITS error: {e2}"
                )
    
    def detect_language(self, text):
        """Detect text language."""
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = hindi_chars + english_chars
        if total_chars == 0:
            return "en"
        
        hindi_ratio = hindi_chars / total_chars
        return "hi" if hindi_ratio > 0.3 else "en"
    
    def generate_with_emotion(self, text, tone="neutral", sample_rate=24000):
        """Generate audio with emotional context."""
        if self.language == "auto":
            detected_lang = self.detect_language(text)
        else:
            detected_lang = self.language
        
        if self.model_type == "bark":
            return self._generate_bark(text, tone, detected_lang)
        elif self.model_type == "vits":
            return self._generate_vits(text, detected_lang)
        elif self.model_type == "speecht5":
            return self._generate_speecht5(text)
        elif self.model_type == "coqui":
            return self._generate_coqui(text, detected_lang)
        elif self.model_type == "ai4bharat":
            return self._generate_ai4bharat(text, tone, detected_lang)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _generate_bark(self, text, tone, language):
        """Generate audio using Bark - COMPLETELY WARNING-FREE."""
        voice_preset = self.bark_voice_presets.get(tone, 'v2/en_speaker_6')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            inputs = self.processor(
                text,
                voice_preset=voice_preset,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if 'input_ids' in inputs:
                attention_mask = torch.ones_like(inputs['input_ids'])
                inputs['attention_mask'] = attention_mask
            
            with torch.no_grad():
                import transformers
                old_verbosity = transformers.logging.get_verbosity()
                transformers.logging.set_verbosity_error()
                
                try:
                    speech_output = self.model.generate(
                        **inputs,
                        do_sample=True,
                        pad_token_id=self.model.generation_config.pad_token_id
                    )
                finally:
                    transformers.logging.set_verbosity(old_verbosity)
            
            audio_array = speech_output.cpu().numpy().squeeze()
        
        return audio_array, self.model.generation_config.sample_rate
    
    def _generate_vits(self, text, language):
        """Generate audio using VITS."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            input_ids = inputs['input_ids'].to(self.device)
            
            with torch.no_grad():
                output = self.model(input_ids)
            
            audio_array = output.waveform.cpu().numpy().squeeze()
            
        except Exception as e:
            try:
                inputs = self.processor(text, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    output = self.model.generate(**inputs)
                
                audio_array = output.cpu().numpy().squeeze()
            except:
                raise Exception(f"VITS generation failed: {e}")
        
        sample_rate = getattr(self.model.config, 'sampling_rate', 22050)
        return audio_array, sample_rate
    
    def _generate_speecht5(self, text):
        """Generate audio using SpeechT5."""
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        from datasets import load_dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=self.vocoder
            )
        
        audio_array = speech.cpu().numpy()
        return audio_array, 16000
    
    def _generate_coqui(self, text, language):
        """Generate audio using Coqui TTS."""
        audio_array = self.model.tts(text, language=language if language != "auto" else None)
        audio_array = np.array(audio_array)
        return audio_array, 22050
    
    def _generate_ai4bharat(self, text, tone, language):
        """Generate audio using AI4Bharat models."""
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Simple generation approach
                inputs = self.processor(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                if 'attention_mask' not in inputs and 'input_ids' in inputs:
                    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                
                with torch.no_grad():
                    if hasattr(self.model, 'generate'):
                        output = self.model.generate(**inputs, max_length=2048)
                        audio_array = output.cpu().numpy().squeeze()
                    else:
                        output = self.model(**inputs)
                        if hasattr(output, 'waveform'):
                            audio_array = output.waveform.cpu().numpy().squeeze()
                        elif hasattr(output, 'audio'):
                            audio_array = output.audio.cpu().numpy().squeeze()
                        else:
                            audio_array = output[0].cpu().numpy().squeeze()
                
                sample_rate = getattr(self.model.config, 'sampling_rate', 24000)
                return audio_array, sample_rate
                
        except Exception as e:
            raise Exception(f"AI4Bharat generation failed: {e}")


class HumanLikeTTSGenerator:
    """Main generator class."""
    
    def __init__(self, model_name, model_type="auto", device="cpu", output_dir=".", language="auto", hf_token=None):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.output_dir = Path(output_dir)
        self.language = language
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parser = TranscriptionParser()
        self.engine = MultilingualTTSEngine(model_name, model_type, device, language, hf_token)
    
    def generate_from_transcription(self, transcription_file):
        """Generate audio from transcription file."""
        print(f"\n{'=' * 70}")
        print(f"🎬 HUMAN-LIKE TTS GENERATION")
        print(f"{'=' * 70}")
        print(f"📄 Input: {transcription_file}")
        print(f"🤖 Model: {self.model_name} ({self.model_type})")
        print(f"🖥️ Device: {self.device}")
        print(f"🌐 Language: {self.language}")
        print(f"{'=' * 70}\n")
        
        transcription_path = Path(transcription_file)
        
        if transcription_path.suffix == '.json':
            with open(transcription_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = self._extract_text_from_json(data)
        else:
            with open(transcription_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        primary_lang = self.engine.detect_language(text)
        print(f"🔍 Detected primary language: {'Hindi' if primary_lang == 'hi' else 'English'}")
        
        print(f"\n📝 Parsing transcription...")
        segments = self.parser.parse(text)
        print(f"   Found {len(segments)} segments")
        
        print(f"\n🎵 Generating audio segments...")
        audio_segments = []
        
        start_time = time.time()
        
        for i, segment in enumerate(segments, 1):
            if segment['type'] == 'pause':
                duration_ms = int(segment['duration'] * 1000)
                silence = AudioSegment.silent(duration=duration_ms)
                audio_segments.append(silence)
                print(f"   [{i}/{len(segments)}] 🔇 Pause ({segment['duration']}s)")
            
            else:
                text = segment['text']
                tone = segment.get('tone', 'neutral')
                
                if not text.strip():
                    continue
                
                seg_lang = self.engine.detect_language(text)
                lang_label = "HI" if seg_lang == "hi" else "EN"
                
                display_text = text[:50] + "..." if len(text) > 50 else text
                print(f"   [{i}/{len(segments)}] 🎙️ [{lang_label}] ({tone}): {display_text}")
                
                try:
                    seg_start = time.time()
                    audio_array, sample_rate = self.engine.generate_with_emotion(text, tone)
                    seg_time = time.time() - seg_start
                    
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
                    audio_segments.append(AudioSegment.silent(duration=500))
                    continue
        
        print(f"\n🔗 Combining {len(audio_segments)} audio segments...")
        final_audio = AudioSegment.empty()
        for seg in audio_segments:
            final_audio += seg
        
        print(f"🎛️ Post-processing audio...")
        final_audio = self._post_process(final_audio)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"narration_{primary_lang}_{timestamp}.mp3"
        
        print(f"💾 Exporting to: {output_file}")
        final_audio.export(
            str(output_file),
            format="mp3",
            bitrate="192k",
            parameters=["-ar", "44100"]
        )
        
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
            if chapter.get('title'):
                text_parts.append(f"[PAUSE-SHORT] {chapter['title']} [PAUSE-MEDIUM]")
            
            for chunk in chapter.get('chunks', []):
                if chunk.get('narration'):
                    text_parts.append(chunk['narration'])
                    text_parts.append('[PAUSE-SHORT]')
        
        return '\n\n'.join(text_parts)
    
    def _post_process(self, audio):
        """Post-process audio for quality."""
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)
        
        audio = compress_dynamic_range(
            audio,
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0
        )
        
        audio = normalize(audio)
        return audio


def check_dependencies():
    """Check and report on dependencies."""
    issues = []
    
    if not DEPS_OK:
        issues.append(f"❌ Core dependencies missing: {IMPORT_ERROR}")
        issues.append("   Install: pip install torch transformers soundfile pydub numpy huggingface-hub")
    
    if not COQUI_AVAILABLE:
        issues.append("⚠️ Coqui TTS not available (optional)")
        issues.append("   Install: pip install TTS")
    
    return issues


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Multilingual TTS with HuggingFace Auth Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use Facebook MMS-TTS (no auth needed, recommended for Hindi)
  python tts.py -f transcription.txt -m facebook/mms-tts-hin -o output/
  
  # Use with HuggingFace token for gated models
  python tts.py -f hindi.txt -m ai4bharat/indic-parler-tts --hf-token YOUR_TOKEN -o output/
  
  # Bark without warnings
  python tts.py -f story.txt -m suno/bark -o output/
  
  # GPU acceleration
  python tts.py -f text.txt -m facebook/mms-tts-hin --device cuda -o output/

Recommended Models for Hindi (NO AUTH NEEDED):
  ✅ facebook/mms-tts-hin - Fast, reliable, no authentication
  ✅ microsoft/speecht5_tts - Good for English
  ✅ suno/bark - Best for emotions (English)

For Gated Models (like ai4bharat/indic-parler-tts):
  1. Login: huggingface-cli login
  2. Request access at: https://huggingface.co/MODEL_NAME
  3. Use --hf-token flag or login via CLI
        """
    )
    
    parser.add_argument('-f', '--file', required=True, help='Input transcription file')
    parser.add_argument('-m', '--model', required=True, help='TTS model name')
    parser.add_argument('-t', '--type', choices=['auto', 'bark', 'vits', 'speecht5', 'coqui', 'ai4bharat'],
                        default='auto', help='Model type (auto-detect)')
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device')
    parser.add_argument('--language', choices=['auto', 'en', 'hi'], default='auto', help='Language')
    parser.add_argument('--hf-token', help='HuggingFace token for gated repos')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies')
    
    args = parser.parse_args()
    
    if args.device == 'cpu' and torch.cuda.is_available():
        print("🔍 GPU detected! Consider using --device cuda for faster generation")
    elif args.device == 'cpu':
        print("🔍 No GPU detected, using CPU")
    
    if args.check_deps:
        print("\n📦 Checking dependencies...\n")
        issues = check_dependencies()
        
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("✅ All dependencies installed!")
        
        sys.exit(0)
    
    if not Path(args.file).exists():
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)
    
    if not DEPS_OK:
        print("❌ Error: Missing core dependencies")
        print(f"   {IMPORT_ERROR}")
        print("   Install: pip install torch transformers soundfile pydub numpy huggingface-hub")
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
            language=args.language,
            hf_token=args.hf_token
        )
        
        output_file = generator.generate_from_transcription(args.file)
        
        print(f"\n✅ Success! Audio saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Generation interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()