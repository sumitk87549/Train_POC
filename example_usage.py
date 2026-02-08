#!/usr/bin/env python3
"""
Example usage of Google Cloud TTS Generator
"""

from google_tts_generator import GoogleTTSGenerator
import os

def main():
    """Example usage scenarios"""
    
    # Initialize TTS Generator (assumes service-account.json exists)
    # Or use environment variable GOOGLE_APPLICATION_CREDENTIALS
    tts = GoogleTTSGenerator('service-account.json')
    
    print("🎤 Google Cloud TTS Generator Examples\n")
    
    # Example 1: Basic Hindi TTS
    print("1. Basic Hindi TTS:")
    success = tts.synthesize_speech(
        text="नमस्ते दुनिया! मैं एक टेक्स्ट-टू-स्पीच सिस्टम हूं।",
        language="hindi",
        gender="female",
        output_file="examples/hindi_basic.mp3"
    )
    print(f"   Status: {'✓ Success' if success else '✗ Failed'}\n")
    
    # Example 2: English with custom parameters
    print("2. English with custom voice parameters:")
    success = tts.synthesize_speech(
        text="Hello! This is a demonstration of Google Cloud Text-to-Speech with custom parameters.",
        language="english",
        gender="male",
        speaking_rate=0.9,
        pitch=2.0,
        volume_gain_db=3.0,
        output_file="examples/english_custom.mp3"
    )
    print(f"   Status: {'✓ Success' if success else '✗ Failed'}\n")
    
    # Example 3: Bengali TTS
    print("3. Bengali TTS:")
    success = tts.synthesize_speech(
        text="হ্যালো! আমি গুগল ক্লাউড টেক্সট-টু-স্পিচ সিস্টেম।",
        language="bengali",
        gender="female",
        output_file="examples/bengali.mp3"
    )
    print(f"   Status: {'✓ Success' if success else '✗ Failed'}\n")
    
    # Example 4: Tamil TTS
    print("4. Tamil TTS:")
    success = tts.synthesize_speech(
        text="வணக்கம்! நான் கூகிள் கிளவுட் உரை-முதல்-பேச்சு அமைப்பு.",
        language="tamil",
        gender="female",
        output_file="examples/tamil.mp3"
    )
    print(f"   Status: {'✓ Success' if success else '✗ Failed'}\n")
    
    # Example 5: Telugu TTS
    print("5. Telugu TTS:")
    success = tts.synthesize_speech(
        text="హలో! నేను గూగుల్ క్లౌడ్ టెక్స్ట్-టు-స్పీచ్ సిస్టమ్.",
        language="telugu",
        gender="male",
        output_file="examples/telugu.mp3"
    )
    print(f"   Status: {'✓ Success' if success else '✗ Failed'}\n")
    
    # Example 6: Marathi TTS
    print("6. Marathi TTS:")
    success = tts.synthesize_speech(
        text="नमस्कार! मी गुगल क्लाउड टेक्स्ट-टू-स्पीच सिस्टम आहे.",
        language="marathi",
        gender="female",
        output_file="examples/marathi.mp3"
    )
    print(f"   Status: {'✓ Success' if success else '✗ Failed'}\n")
    
    # Example 7: Batch processing
    print("7. Batch Processing:")
    texts = [
        "यह पहला वाक्य है।",
        "यह दूसरा वाक्य है।",
        "यह तीसरा और अंतिम वाक्य है।"
    ]
    output_files = tts.batch_synthesize(
        texts=texts,
        language="hindi",
        gender="female",
        output_dir="examples/batch_output"
    )
    print(f"   Generated {len(output_files)} files in batch_output directory")
    
    # Example 8: Merge batch files
    if output_files:
        print("8. Merging batch files:")
        success = tts.merge_audio_files(output_files, "examples/merged_batch.mp3")
        print(f"   Status: {'✓ Success' if success else '✗ Failed'}\n")
    
    # Example 9: List available voices for Hindi
    print("9. Available voices for Hindi:")
    voices = tts.list_voices('hi-IN')
    for voice in voices:
        print(f"   {voice['name']} ({voice['ssml_gender']}) - {voice['natural_sample_rate_hertz']}Hz")
    print()
    
    # Example 10: Different audio formats
    print("10. Different audio formats:")
    formats = ['MP3', 'OGG_OPUS', 'LINEAR16']
    for fmt in formats:
        success = tts.synthesize_speech(
            text="Testing different audio formats.",
            language="english",
            gender="female",
            audio_encoding=fmt,
            output_file=f"examples/test_format.{fmt.lower().replace('_', '.')}"
        )
        print(f"   {fmt}: {'✓ Success' if success else '✗ Failed'}")
    
    print("\n🎉 All examples completed!")
    print("📁 Check the 'examples' directory for generated audio files.")

if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    main()
