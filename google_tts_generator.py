#!/usr/bin/env python3
"""
Google Cloud Text-to-Speech Generator for Indian Languages
Supports Hindi, English, Bengali, Tamil, Telugu, Marathi and more

Author: Your Name
Email: sumitk87549@gmail.com
"""

import os
import sys
from typing import Optional, List, Dict, Any
from google.cloud import texttospeech
from google.oauth2 import service_account
from pydub import AudioSegment
import argparse
import json


class GoogleTTSGenerator:
    """
    Google Cloud Text-to-Speech Generator with comprehensive voice controls
    """
    
    # Indian language codes and their supported voices
    INDIAN_LANGUAGES = {
        'hindi': {
            'language_code': 'hi-IN',
            'voices': {
                'female': ['hi-IN-Wavenet-A', 'hi-IN-Wavenet-B', 'hi-IN-Wavenet-C'],
                'male': ['hi-IN-Wavenet-D']
            }
        },
        'english': {
            'language_code': 'en-IN',
            'voices': {
                'female': ['en-IN-Wavenet-A', 'en-IN-Wavenet-B'],
                'male': ['en-IN-Wavenet-C', 'en-IN-Wavenet-D']
            }
        },
        'bengali': {
            'language_code': 'bn-IN',
            'voices': {
                'female': ['bn-IN-Wavenet-A'],
                'male': ['bn-IN-Wavenet-B']
            }
        },
        'tamil': {
            'language_code': 'ta-IN',
            'voices': {
                'female': ['ta-IN-Wavenet-A'],
                'male': ['ta-IN-Wavenet-B']
            }
        },
        'telugu': {
            'language_code': 'te-IN',
            'voices': {
                'female': ['te-IN-Wavenet-A'],
                'male': ['te-IN-Wavenet-B']
            }
        },
        'marathi': {
            'language_code': 'mr-IN',
            'voices': {
                'female': ['mr-IN-Wavenet-A'],
                'male': ['mr-IN-Wavenet-B']
            }
        },
        'gujarati': {
            'language_code': 'gu-IN',
            'voices': {
                'female': ['gu-IN-Wavenet-A'],
                'male': ['gu-IN-Wavenet-B']
            }
        },
        'kannada': {
            'language_code': 'kn-IN',
            'voices': {
                'female': ['kn-IN-Wavenet-A'],
                'male': ['kn-IN-Wavenet-B']
            }
        },
        'malayalam': {
            'language_code': 'ml-IN',
            'voices': {
                'female': ['ml-IN-Wavenet-A'],
                'male': ['ml-IN-Wavenet-B']
            }
        },
        'punjabi': {
            'language_code': 'pa-IN',
            'voices': {
                'female': ['pa-IN-Wavenet-A'],
                'male': ['pa-IN-Wavenet-B']
            }
        },
        'odia': {
            'language_code': 'or-IN',
            'voices': {
                'female': ['or-IN-Wavenet-A'],
                'male': ['or-IN-Wavenet-B']
            }
        },
        'assamese': {
            'language_code': 'as-IN',
            'voices': {
                'female': ['as-IN-Wavenet-A'],
                'male': ['as-IN-Wavenet-B']
            }
        }
    }
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize the TTS Generator
        
        Args:
            credentials_path: Path to Google Cloud service account JSON file
        """
        self.client = None
        self.credentials_path = credentials_path
        
        try:
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.client = texttospeech.TextToSpeechClient(credentials=credentials)
            else:
                # Use default credentials (from environment or gcloud CLI)
                self.client = texttospeech.TextToSpeechClient()
            print("✓ Google Cloud TTS client initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing TTS client: {e}")
            sys.exit(1)
    
    def list_voices(self, language_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available voices for a specific language or all voices
        
        Args:
            language_code: Language code (e.g., 'hi-IN', 'en-IN')
        
        Returns:
            List of voice information
        """
        try:
            voices = self.client.list_voices(language_code=language_code)
            voice_list = []
            
            for voice in voices.voices:
                voice_info = {
                    'name': voice.name,
                    'language_codes': voice.language_codes,
                    'ssml_gender': texttospeech.SsmlVoiceGender(voice.ssml_gender).name,
                    'natural_sample_rate_hertz': voice.natural_sample_rate_hertz
                }
                voice_list.append(voice_info)
            
            return voice_list
        except Exception as e:
            print(f"✗ Error listing voices: {e}")
            return []
    
    def synthesize_speech(
        self,
        text: str,
        language: str = 'hindi',
        gender: str = 'female',
        voice_name: Optional[str] = None,
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        volume_gain_db: float = 0.0,
        sample_rate_hertz: int = 24000,
        audio_encoding: str = 'MP3',
        output_file: str = 'output.mp3'
    ) -> bool:
        """
        Synthesize speech from text with comprehensive voice controls
        
        Args:
            text: Text to convert to speech
            language: Target language (hindi, english, bengali, etc.)
            gender: Voice gender (male, female, neutral)
            voice_name: Specific voice name (overrides language and gender)
            speaking_rate: Speaking rate (0.25 to 4.0, default 1.0)
            pitch: Speaking pitch (-20.0 to 20.0 semitones, default 0.0)
            volume_gain_db: Volume gain in decibels (-96.0 to 16.0, default 0.0)
            sample_rate_hertz: Audio sample rate (8000 to 48000 Hz)
            audio_encoding: Audio format (MP3, OGG_OPUS, LINEAR16)
            output_file: Output audio file path
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate language
            language = language.lower()
            if language not in self.INDIAN_LANGUAGES:
                print(f"✗ Unsupported language: {language}")
                print(f"Supported languages: {', '.join(self.INDIAN_LANGUAGES.keys())}")
                return False
            
            # Determine voice selection
            if voice_name:
                # Use specified voice name
                selected_voice = voice_name
            else:
                # Auto-select voice based on language and gender
                gender = gender.lower()
                lang_info = self.INDIAN_LANGUAGES[language]
                
                if gender not in lang_info['voices']:
                    print(f"✗ Gender '{gender}' not available for {language}")
                    available_genders = list(lang_info['voices'].keys())
                    print(f"Available genders: {', '.join(available_genders)}")
                    return False
                
                # Use first available voice for the selected gender
                selected_voice = lang_info['voices'][gender][0]
            
            # Validate parameters
            speaking_rate = max(0.25, min(4.0, speaking_rate))
            pitch = max(-20.0, min(20.0, pitch))
            volume_gain_db = max(-96.0, min(16.0, volume_gain_db))
            sample_rate_hertz = max(8000, min(48000, sample_rate_hertz))
            
            # Map audio encoding string to enum
            encoding_map = {
                'MP3': texttospeech.AudioEncoding.MP3,
                'OGG_OPUS': texttospeech.AudioEncoding.OGG_OPUS,
                'LINEAR16': texttospeech.AudioEncoding.LINEAR16
            }
            
            if audio_encoding not in encoding_map:
                print(f"✗ Unsupported audio encoding: {audio_encoding}")
                print(f"Supported encodings: {', '.join(encoding_map.keys())}")
                return False
            
            # Map gender string to enum
            gender_map = {
                'male': texttospeech.SsmlVoiceGender.MALE,
                'female': texttospeech.SsmlVoiceGender.FEMALE,
                'neutral': texttospeech.SsmlVoiceGender.NEUTRAL
            }
            
            # Create synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Build voice selection parameters
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.INDIAN_LANGUAGES[language]['language_code'],
                name=selected_voice,
                ssml_gender=gender_map.get(gender.lower(), texttospeech.SsmlVoiceGender.NEUTRAL)
            )
            
            # Build audio configuration
            audio_config = texttospeech.AudioConfig(
                audio_encoding=encoding_map[audio_encoding],
                speaking_rate=speaking_rate,
                pitch=pitch,
                volume_gain_db=volume_gain_db,
                sample_rate_hertz=sample_rate_hertz
            )
            
            # Perform text-to-speech synthesis
            print(f"🔄 Synthesizing speech...")
            print(f"   Text: {text[:50]}{'...' if len(text) > 50 else ''}")
            print(f"   Language: {language}")
            print(f"   Voice: {selected_voice}")
            print(f"   Gender: {gender}")
            print(f"   Speaking Rate: {speaking_rate}")
            print(f"   Pitch: {pitch}")
            print(f"   Volume: {volume_gain_db} dB")
            
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Save the audio content to a file
            with open(output_file, 'wb') as out:
                out.write(response.audio_content)
            
            print(f"✓ Audio content written to '{output_file}'")
            return True
            
        except Exception as e:
            print(f"✗ Error synthesizing speech: {e}")
            return False
    
    def batch_synthesize(
        self,
        texts: List[str],
        language: str = 'hindi',
        gender: str = 'female',
        output_dir: str = 'output',
        **kwargs
    ) -> List[str]:
        """
        Synthesize multiple texts to separate files
        
        Args:
            texts: List of texts to synthesize
            language: Target language
            gender: Voice gender
            output_dir: Output directory for audio files
            **kwargs: Additional parameters for synthesize_speech
        
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        for i, text in enumerate(texts, 1):
            output_file = os.path.join(output_dir, f"output_{i:03d}.mp3")
            if self.synthesize_speech(text, language, gender, output_file=output_file, **kwargs):
                output_files.append(output_file)
        
        return output_files
    
    def merge_audio_files(self, audio_files: List[str], output_file: str = 'merged_output.mp3') -> bool:
        """
        Merge multiple audio files into one
        
        Args:
            audio_files: List of audio file paths
            output_file: Output merged file path
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not audio_files:
                print("✗ No audio files to merge")
                return False
            
            print(f"🔄 Merging {len(audio_files)} audio files...")
            
            # Load first audio file
            combined = AudioSegment.from_mp3(audio_files[0])
            
            # Append remaining audio files
            for audio_file in audio_files[1:]:
                audio = AudioSegment.from_mp3(audio_file)
                combined += audio
            
            # Export merged audio
            combined.export(output_file, format="mp3")
            print(f"✓ Merged audio saved to '{output_file}'")
            return True
            
        except Exception as e:
            print(f"✗ Error merging audio files: {e}")
            return False


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Google Cloud TTS Generator for Indian Languages')
    
    # Authentication
    parser.add_argument('--credentials', type=str, help='Path to Google Cloud service account JSON file')
    
    # Text input
    parser.add_argument('--text', type=str, help='Text to convert to speech')
    parser.add_argument('--file', type=str, help='Text file containing text to convert')
    
    # Language and voice
    parser.add_argument('--language', type=str, default='hindi', 
                       choices=list(GoogleTTSGenerator.INDIAN_LANGUAGES.keys()),
                       help='Target language')
    parser.add_argument('--gender', type=str, default='female', 
                       choices=['male', 'female', 'neutral'],
                       help='Voice gender')
    parser.add_argument('--voice', type=str, help='Specific voice name')
    
    # Voice parameters
    parser.add_argument('--rate', type=float, default=1.0, help='Speaking rate (0.25-4.0)')
    parser.add_argument('--pitch', type=float, default=0.0, help='Pitch (-20.0 to 20.0)')
    parser.add_argument('--volume', type=float, default=0.0, help='Volume gain in dB (-96.0 to 16.0)')
    parser.add_argument('--sample-rate', type=int, default=24000, help='Sample rate (8000-48000 Hz)')
    
    # Output
    parser.add_argument('--output', type=str, default='output.mp3', help='Output audio file')
    parser.add_argument('--format', type=str, default='MP3', 
                       choices=['MP3', 'OGG_OPUS', 'LINEAR16'],
                       help='Audio format')
    
    # Utility functions
    parser.add_argument('--list-voices', action='store_true', help='List available voices')
    parser.add_argument('--list-languages', action='store_true', help='List supported languages')
    
    args = parser.parse_args()
    
    # Initialize TTS generator
    tts = GoogleTTSGenerator(args.credentials)
    
    # List languages
    if args.list_languages:
        print("\n📋 Supported Indian Languages:")
        for lang, info in GoogleTTSGenerator.INDIAN_LANGUAGES.items():
            print(f"   {lang.title()}: {info['language_code']}")
        return
    
    # List voices
    if args.list_voices:
        language_code = GoogleTTSGenerator.INDIAN_LANGUAGES.get(args.language.lower(), {}).get('language_code')
        voices = tts.list_voices(language_code)
        
        print(f"\n🎤 Available Voices for {args.language.title()} ({language_code}):")
        for voice in voices:
            print(f"   {voice['name']} ({voice['ssml_gender']}) - {voice['natural_sample_rate_hertz']}Hz")
        return
    
    # Get text input
    text = args.text
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            print(f"✗ Error reading file: {e}")
            return
    
    if not text:
        print("✗ No text provided. Use --text or --file")
        parser.print_help()
        return
    
    # Synthesize speech
    success = tts.synthesize_speech(
        text=text,
        language=args.language,
        gender=args.gender,
        voice_name=args.voice,
        speaking_rate=args.rate,
        pitch=args.pitch,
        volume_gain_db=args.volume,
        sample_rate_hertz=args.sample_rate,
        audio_encoding=args.format,
        output_file=args.output
    )
    
    if success:
        print(f"\n🎉 TTS generation completed successfully!")
        print(f"📁 Output file: {args.output}")
    else:
        print(f"\n❌ TTS generation failed!")


if __name__ == "__main__":
    main()
