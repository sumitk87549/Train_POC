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

# Performance optimization - cache heavy operations
@st.cache_resource
def init_db():
    """Initialize database (cached)"""
    from db_utils import init_database
    init_database()
    return True

# Custom CSS for better styling and performance
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
        transition: all 0.3s ease;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    /* Lightweight - minimal animations */
    * {
        transition: opacity 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database on first run (cached for performance)
try:
    init_db()
except Exception as e:
    st.error(f"⚠️ Database connection error: {e}")
    st.info("💡 Make sure PostgreSQL is running and database 'readlyte' exists")
    st.code("createdb -U postgres readlyte", language="bash")

# Performance notice
st.info("💡 **NEW**: Live progress display, thinking process streaming, and quick presets added!")

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
with st.expander("💡 New Features & Tips"):
    st.markdown("""
    **🆕 What's New:**
    - **Live Progress Display**: See generation happen in real-time
    - **Thinking Process**: Watch reasoning models think (deepseek-r1)
    - **Quick Presets**: Fast/Balanced/Quality buttons for easy testing
    - **Smart Suggestions**: Tooltips explain what each parameter does
    - **Time Estimates**: Know how long each operation will take
    
    **🚀 Quick Start:**
    1. Upload an EPUB → **Fast preset** → Small section
    2. Test translation → Use **qwen2.5:3b** (BASIC tier)
    3. Try different models/parameters to see differences
    
    **Model Recommendations:**
    - **Fast testing**: `qwen2.5:3b` (BASIC) - 30-60s per chunk
    - **Quality translation**: `qwen2.5:7b` (INTERMEDIATE) - 60-120s
    - **Best translation**: `qwen2.5:14b` or `deepseek-r1:7b` (ADVANCED) - 2-5min
    - **See thinking**: `deepseek-r1` models show reasoning process!
    
    **Summarization Tips:**
    - **SHORT**: 2-3 sentences, quick scan
    - **MEDIUM**: 4-6 sentences, balanced (recommended)
    - **LONG**: 8-12 sentences, comprehensive
    - Lower temperature (0.1-0.2) = more factual
    
    **Audio Quality:**
    - **BASIC**: `facebook/mms-tts-hin` - Fast Hindi (~10s per paragraph)
    - **INTERMEDIATE**: `suno/bark-small` - Better prosody (~30s)  
    - **ADVANCED**: `suno/bark` - Best quality (~60s, needs GPU)
    
    **Performance Tips:**
    - 📦 **Caching**: All generations cached in DB
    - ⚡ **Browser**: Lightweight UI, models run server-side
    - 🔄 **Live Updates**: Progress shown without page refreshes
    - 💾 **Same settings**: Fast retrieval from cache
    
    **Testing Different Parameters:**
    - Use **Quick Presets** to try Fast/Balanced/Quality
    - Change **Temperature** to see creativity vs consistency
    - Try **Chunk Size** to balance speed vs context
    - Compare **Models** side-by-side using same section
    """)

# Footer
st.divider()
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>ReadLyte MVP v1.0 | Built with Streamlit | Use sidebar to navigate</div>",
    unsafe_allow_html=True
)
