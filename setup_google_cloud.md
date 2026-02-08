# Google Cloud TTS Setup Guide

## Prerequisites
- Google Cloud account with billing enabled
- Text-to-Speech API enabled
- Service account with TTS permissions

## Setup Instructions

### 1. Enable Text-to-Speech API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project (or create a new one)
3. Navigate to "APIs & Services" > "Library"
4. Search for "Cloud Text-to-Speech API"
5. Click "Enable"

### 2. Create Service Account
1. Go to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Enter a name (e.g., "tts-service-account")
4. Add role: "Cloud Text-to-Speech API User"
5. Click "Done"

### 3. Generate Service Account Key
1. Find your service account in the list
2. Click the three dots menu > "Manage keys"
3. Click "Add Key" > "Create new key"
4. Select "JSON" and click "Create"
5. Download the JSON file and save it securely

### 4. Set Up Authentication

#### Option A: Service Account Key File
Save the downloaded JSON file as `service-account.json` in your project directory.

#### Option B: Environment Variable
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account.json"
```

#### Option C: Google Cloud CLI
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
```python
from google_tts_generator import GoogleTTSGenerator

# Initialize with service account key
tts = GoogleTTSGenerator('service-account.json')

# Generate speech
tts.synthesize_speech(
    text="नमस्ते दुनिया!",
    language="hindi",
    gender="female",
    output_file="hindi_greeting.mp3"
)
```

### Command Line Usage
```bash
# Basic TTS generation
python google_tts_generator.py --text "Hello World" --language english --output hello.mp3

# Hindi TTS with custom voice parameters
python google_tts_generator.py --text "नमस्ते दुनिया" --language hindi --gender female --pitch 2.0 --rate 0.9 --output hindi_greeting.mp3

# List available voices for a language
python google_tts_generator.py --list-voices --language hindi

# List all supported languages
python google_tts_generator.py --list-languages
```

## Supported Languages
- Hindi (hi-IN)
- English India (en-IN)
- Bengali (bn-IN)
- Tamil (ta-IN)
- Telugu (te-IN)
- Marathi (mr-IN)
- Gujarati (gu-IN)
- Kannada (kn-IN)
- Malayalam (ml-IN)
- Punjabi (pa-IN)
- Odia (or-IN)
- Assamese (as-IN)

## Voice Parameters
- **speaking_rate**: 0.25 to 4.0 (default: 1.0)
- **pitch**: -20.0 to 20.0 semitones (default: 0.0)
- **volume_gain_db**: -96.0 to 16.0 dB (default: 0.0)
- **sample_rate_hertz**: 8000 to 48000 Hz (default: 24000)
- **audio_encoding**: MP3, OGG_OPUS, LINEAR16 (default: MP3)

## Advanced Features

### Batch Processing
```python
texts = ["First sentence", "Second sentence", "Third sentence"]
output_files = tts.batch_synthesize(texts, language="hindi", output_dir="batch_output")
```

### Audio Merging
```python
tts.merge_audio_files(output_files, "merged_output.mp3")
```

### Custom Voice Selection
```python
# Use specific voice name
tts.synthesize_speech(
    text="Custom voice test",
    voice_name="hi-IN-Wavenet-A",
    output_file="custom_voice.mp3"
)
```

## Troubleshooting

### Common Issues
1. **Authentication Error**: Ensure your service account has TTS permissions
2. **Quota Exceeded**: Check your Google Cloud quota limits
3. **Invalid Language Code**: Use the exact language codes specified above
4. **Network Issues**: Ensure you have internet connectivity

### Debug Mode
Enable verbose output by checking the console messages in the script.

## Cost Considerations
- Google Cloud TTS is priced per character
- Check current pricing at [Google Cloud TTS Pricing](https://cloud.google.com/text-to-speech/pricing)
- Monitor your usage in the Google Cloud Console

## Security Notes
- Never commit service account keys to version control
- Use environment variables in production
- Regularly rotate your service account keys
