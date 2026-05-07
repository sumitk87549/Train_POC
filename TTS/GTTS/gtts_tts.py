#!/usr/bin/env python3
"""
gTTS Text-to-Speech Generator
Generates speech from text using Google Text-to-Speech API via gTTS library.
"""

from gtts import gTTS
import os
from pathlib import Path


# ==================== CONFIGURATION ====================
# Paste your text here
TEXT_TO_CONVERT = """
शर्लक होम्स के लिए वह हमेशा 'वही महिला' थीं। उनकी नज़र में वह अपनी जाति की सभी महिलाओं से श्रेष्ठ और सर्वोपरि थीं। 
"""

# Output file path (will be saved relative to this script)
OUTPUT_FILE = "output.mp3"

# Language code (common: en, hi, es, fr, de, it, ja, ko, zh, etc.)
LANGUAGE = "hi"

# Set to True for slower speech
SLOW_SPEED = False

# Top-level domain for accent (affects pronunciation)
# Options: 'com' (default US), 'co.in' (India), 'co.uk' (UK), 'ca' (Canada), 'com.au' (Australia)
TLD = "com"
# =======================================================


def generate_tts(
    text: str,
    output_file: str,
    lang: str = 'en',
    slow: bool = False,
    tld: str = 'com'
) -> str:
    """
    Generate TTS audio from text and save to file.
    
    Args:
        text: Input text to convert to speech
        output_file: Path to save the audio file (mp3 format)
        lang: Language code (e.g., 'en', 'hi', 'es', 'fr', 'de')
        slow: If True, speech will be slower
        tld: Top-level domain for Google Translate (affects accent)
              Options: 'com', 'co.in', 'co.uk', 'ca', 'com.au', etc.
    
    Returns:
        Path to the generated audio file
    
    Raises:
        Exception: If gTTS API fails or file save fails
    """
    try:
        print(f"Generating TTS for text: {text[:100]}...")
        print(f"Language: {lang}, Slow: {slow}, TLD: {tld}")
        
        # Create gTTS object
        tts = gTTS(
            text=text,
            lang=lang,
            slow=slow,
            tld=tld
        )
        
        # Ensure output directory exists
        output_path = Path(output_file)
        if not output_path.is_absolute():
            # Make path relative to script directory
            script_dir = Path(__file__).parent
            output_path = script_dir / output_file
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the audio file
        tts.save(str(output_path))
        
        print(f"✓ TTS audio saved to: {output_path.absolute()}")
        print(f"✓ File size: {output_path.stat().st_size / 1024:.2f} KB")
        return str(output_path.absolute())
        
    except Exception as e:
        print(f"✗ Error generating TTS: {str(e)}")
        print("Note: This might be a server-side error from Google TTS API.")
        raise


def main():
    # Clean up the text (remove extra whitespace)
    cleaned_text = TEXT_TO_CONVERT.strip()
    
    if not cleaned_text:
        print("✗ Error: TEXT_TO_CONVERT is empty. Please add text in the configuration section.")
        return
    
    try:
        generate_tts(
            text=cleaned_text,
            output_file=OUTPUT_FILE,
            lang=LANGUAGE,
            slow=SLOW_SPEED,
            tld=TLD
        )
        print("\n✓ TTS generation completed successfully!")
    except Exception as e:
        print(f"\n✗ Failed to generate TTS: {e}")


if __name__ == '__main__':
    main()
