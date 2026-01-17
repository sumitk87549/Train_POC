"""
ReadLyte MVP - Main Streamlit Application
A complete solution for EPUB processing, translation, summarization, and audio generation
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="📚 ReadLyte MVP",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #cce5ff;
        border: 1px solid #99caff;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database on first run
try:
    from db_utils import init_database
    init_database()
except Exception as e:
    st.error(f"⚠️ Database connection error: {e}")
    st.info("💡 Make sure PostgreSQL is running and database 'readlyte' exists")
    st.code("createdb -U postgres readlyte", language="bash")

# Main page content
st.markdown('<div class="main-header">📚 ReadLyte MVP</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">EPUB Processing • Translation • Summarization • Audio Generation</div>', unsafe_allow_html=True)

st.divider()

# Feature overview
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload & Process")
    st.markdown("""
    - Upload EPUB files
    - Automatic cleaning & segregation
    - Section-based storage in database
    """)
    
    st.markdown("### 🌐 Translate")
    st.markdown("""
    - Hindi translation (more languages coming)
    - Multiple model options (Ollama/HuggingFace)
    - Quality tiers: BASIC, INTERMEDIATE, ADVANCED
    - Cached translations for fast retrieval
    """)

with col2:
    st.markdown("### 📝 Summarize")
    st.markdown("""
    - Context-aware summarization
    - Length control: SHORT, MEDIUM, LONG
    - Summarize originals or translations
    - Multiple model options
    """)
    
    st.markdown("### 🎧 Listen")
    st.markdown("""
    - Text-to-Speech generation
    - Hindi and English support
    - Multiple voice models
    - In-browser audio playback
    """)

st.divider()

# Quick start guide
st.markdown("### 🚀 Quick Start")

st.markdown("""
1. **Upload an EPUB** → Go to "📤 Upload EPUB" page
2. **Browse your library** → Go to "📖 Books Library" page  
3. **Translate sections** → Go to "🌐 Translate" page
4. **Generate summaries** → Go to "📝 Summarize" page
5. **Listen to audio** → Go to "🎧 Listen" page
""")

st.divider()

# Status check
st.markdown("### ⚙️ System Status")

col1, col2, col3 = st.columns(3)

with col1:
    # Check Ollama
    try:
        import ollama
        ollama.list()
        st.success("✅ Ollama Connected")
    except:
        st.warning("⚠️ Ollama not available")
        st.caption("Install: `pip install ollama`")

with col2:
    # Check database
    try:
        from db_utils import get_books
        books = get_books()
        st.success(f"✅ Database OK ({len(books)} books)")
    except Exception as e:
        st.error("❌ Database Error")
        st.caption(str(e)[:50])

with col3:
    # Check TTS
    try:
        import torch
        import soundfile
        st.success("✅ TTS Ready")
    except:
        st.warning("⚠️ TTS deps missing")
        st.caption("`pip install torch soundfile`")

st.divider()

# Hints and tips
with st.expander("💡 Tips & Hints"):
    st.markdown("""
    **Model Recommendations:**
    - **Fast translation**: `qwen2.5:3b` (BASIC tier)
    - **Quality translation**: `qwen2.5:7b` or `deepseek-r1:7b` (INTERMEDIATE tier)
    - **Best translation**: `qwen2.5:14b` (ADVANCED tier)
    
    **Summarization Tips:**
    - SHORT summaries: Quick overview (2-3 sentences)
    - MEDIUM summaries: Balanced coverage (4-6 sentences)
    - LONG summaries: Comprehensive (8-12 sentences)
    
    **Audio Quality:**
    - BASIC: `facebook/mms-tts-hin` - Fast, good quality
    - INTERMEDIATE: `suno/bark-small` - Better prosody
    - ADVANCED: `suno/bark` - Commercial quality (requires GPU)
    
    **Caching:**
    - Translations, summaries, and audio are cached in the database
    - Regenerating with different models/tiers creates new entries
    - Same model+tier reuses existing cached content
    """)

# Footer
st.divider()
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>ReadLyte MVP v1.0 | Built with Streamlit | Use sidebar to navigate</div>",
    unsafe_allow_html=True
)
