#!/usr/bin/env python3
"""
Hindi TTS Generator using edge-tts
Generates continuous TTS audio from Hindi-Latin mixed text
"""

import asyncio
import edge_tts
import os
import sys
from pathlib import Path

# Hindi voice that handles Latin script well
VOICE = "hi-IN-SwaraNeural"  # Natural female Hindi voice
OUTPUT_DIR = Path("audio_output")
OUTPUT_FILE = OUTPUT_DIR / "hindi_story_tts.mp3"

async def generate_tts_from_file(text_file_path):
    """
    Generate TTS from a text file containing Hindi-Latin mixed content
    """
    try:
        # Read the text file
        print(f"📖 Reading text from: {text_file_path}")
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        if not text_content.strip():
            print("❌ Error: File is empty or contains no readable text")
            return False
        
        print(f"✅ Successfully read {len(text_content)} characters")
        
        # Create output directory if it doesn't exist
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Generate TTS
        print(f"🎙️ Generating TTS using voice: {VOICE}")
        print("⏳ This may take a few minutes depending on text length...")
        
        # Create edge-tts communicate object
        communicate = edge_tts.Communicate(text_content, VOICE)
        
        # Generate audio and save to file
        await communicate.save(str(OUTPUT_FILE))
        
        print(f"✅ TTS generation completed!")
        print(f"📁 Audio saved to: {OUTPUT_FILE}")
        
        # Get file size
        file_size = OUTPUT_FILE.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"📊 File size: {size_mb:.2f} MB")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Error: File not found - {text_file_path}")
        return False
    except Exception as e:
        print(f"❌ Error during TTS generation: {str(e)}")
        return False

async def main():
    """Main function"""
    print("🎵 Hindi TTS Generator")
    print("=" * 50)
    
    # Default file path
    default_file = "Translation/translation_hin_20260312_041204.txt"
    
    # Check if file exists
    if not os.path.exists(default_file):
        print(f"❌ Error: Default file not found - {default_file}")
        print("Please make sure the translation file exists in the Translation/ directory")
        sys.exit(1)
    
    # Generate TTS
    success = await generate_tts_from_file(default_file)
    
    if success:
        print("\n🎉 TTS generation completed successfully!")
        print(f"You can find the audio file at: {OUTPUT_FILE}")
        print("Play the file to hear the continuous Hindi story narration")
    else:
        print("\n❌ TTS generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
