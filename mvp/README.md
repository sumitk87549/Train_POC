# 📚 ReadLyte MVP

A Streamlit-based application for EPUB processing, translation, summarization, and text-to-speech generation.

## Quick Start

### 1. Prerequisites

- PostgreSQL running on `127.0.0.1:5432`
- Database `readlyte` created
- Python 3.8+
- Ollama installed (for translation/summarization)

### 2. Setup

```bash
# Navigate to MVP directory
cd /home/sumit/Desktop/Train_POC/mvp

# Install dependencies
pip install -r requirements.txt

# Initialize database
python db_utils.py
```

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Features

| Page | Description |
|------|-------------|
| 📤 Upload EPUB | Upload and process EPUB files |
| 📖 Books Library | Browse books and sections |
| 🌐 Translate | Translate to Hindi with model options |
| 📝 Summarize | Generate SHORT/MEDIUM/LONG summaries |
| 🎧 Listen | Generate and play audio |

## Model Recommendations

### Translation
- **Fast**: `qwen2.5:3b` (BASIC tier)
- **Quality**: `qwen2.5:7b` (INTERMEDIATE tier)
- **Best**: `qwen2.5:14b` (ADVANCED tier)

### Summarization
- **Fast**: `llama3.2:3b`
- **Quality**: `llama3.1:8b`

### Audio
- **Hindi**: `facebook/mms-tts-hin`
- **English**: `facebook/mms-tts-eng`
- **Expressive**: `suno/bark` (slow, needs GPU)

## Database

PostgreSQL schema includes:
- `books` - Book metadata
- `book_sections` - Segregated sections
- `translations` - Cached translations (per model/tier)
- `summaries` - Cached summaries (per model/tier/length)
- `audio_files` - Cached audio (stored as BYTEA)

## Troubleshooting

**Database connection error:**
```bash
createdb -U postgres readlyte
```

**Ollama not available:**
```bash
pip install ollama
ollama pull qwen2.5:3b
```

**Audio dependencies:**
```bash
pip install torch transformers soundfile pydub
```
