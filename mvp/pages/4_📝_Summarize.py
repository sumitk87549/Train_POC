"""
Page 4: Summarization
Summarize book sections or translations with model options
"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_utils import (
    get_books, get_sections, get_section, 
    get_summary, save_summary, get_section_summaries,
    get_section_translations, get_translation
)
from summary_engine import (
    summarize_text, get_length_info, get_available_models, 
    OLLAMA_AVAILABLE, MODEL_RECOMMENDATIONS
)

st.set_page_config(page_title="📝 Summarize", page_icon="📝", layout="wide")

st.title("📝 Summarize Section")
st.markdown("Generate summaries of book sections or translations with customizable options.")

st.divider()

# ===== SIDEBAR OPTIONS =====
with st.sidebar:
    st.markdown("### ⚙️ Summary Options")
    
    # Provider
    provider = st.radio(
        "Provider",
        ["ollama", "huggingface"],
        help="Ollama is recommended for local inference"
    )
    
    # Model
    st.markdown("**Model**")
    available_models = get_available_models(provider)
    
    model = st.selectbox(
        "Model Name",
        available_models if available_models else MODEL_RECOMMENDATIONS["FAST"],
        help="Recommended: llama3.1:8b or qwen2.5:7b"
    )
    
    custom_model = st.text_input("Or enter custom model name")
    if custom_model:
        model = custom_model
    
    st.divider()
    
    # Tier
    tier = st.selectbox(
        "Quality Tier",
        ["BASIC", "INTERMEDIATE", "ADVANCED"],
        index=1,
        help="Higher tiers produce more analytical summaries"
    )
    
    # Length
    st.markdown("**Summary Length**")
    length_info = get_length_info()
    
    length_type = st.selectbox(
        "Length",
        list(length_info.keys()),
        index=1,
        format_func=lambda x: f"{x} - {length_info[x]['target']}"
    )
    
    st.caption(length_info[length_type]['description'])
    
    st.divider()
    st.markdown("### 💡 Tips")
    st.caption("""
    - **SHORT**: Quick overview (5-10%)
    - **MEDIUM**: Balanced (15-20%)
    - **LONG**: Comprehensive (25-35%)
    - Can summarize original or translation
    """)

# ===== MAIN CONTENT =====

# Book/Section selection
col1, col2 = st.columns(2)

with col1:
    books = get_books()
    book_options = {f"{b['title'][:40]}... ({b['section_count']} sections)": b['id'] for b in books}
    
    if not book_options:
        st.warning("No books in library. Upload an EPUB first!")
        st.stop()
    
    # Check for pre-selected section
    preselected_section_id = st.session_state.get('summarize_section_id')
    default_book_idx = 0
    
    if preselected_section_id:
        section = get_section(preselected_section_id)
        if section:
            for i, (name, bid) in enumerate(book_options.items()):
                if bid == section['book_id']:
                    default_book_idx = i
                    break
    
    selected_book = st.selectbox("📚 Select Book", list(book_options.keys()), index=default_book_idx)
    book_id = book_options[selected_book]

with col2:
    sections = get_sections(book_id)
    section_options = {f"{s['section_number']}. {s['section_title'][:30]}... ({s['word_count']} words)": s['id'] 
                       for s in sections}
    
    if not section_options:
        st.warning("No sections in this book.")
        st.stop()
    
    default_section_idx = 0
    if preselected_section_id:
        for i, (name, sid) in enumerate(section_options.items()):
            if sid == preselected_section_id:
                default_section_idx = i
                break
        if 'summarize_section_id' in st.session_state:
            del st.session_state['summarize_section_id']
    
    selected_section = st.selectbox("📑 Select Section", list(section_options.keys()), index=default_section_idx)
    section_id = section_options[selected_section]

# Source selection
st.divider()

section = get_section(section_id)
translations = get_section_translations(section_id)

source_options = ["Original Text"]
source_map = {"Original Text": ("original", None)}

for t in translations:
    label = f"Translation ({t['model_name']} - {t['tier']})"
    source_options.append(label)
    source_map[label] = ("translation", t['id'])

# Check for pre-selected source
preselected_source = st.session_state.get('summarize_source', 'original')
default_source_idx = 0
if preselected_source == "translation" and len(source_options) > 1:
    default_source_idx = 1

source_choice = st.selectbox("📄 Summarize", source_options, index=default_source_idx)
source_type, source_id = source_map[source_choice]

if section:
    st.divider()
    
    # Get source text
    if source_type == "original":
        source_text = section['content']
        st.markdown("### 📄 Original Text")
    else:
        # Get translation text
        trans = next((t for t in translations if t['id'] == source_id), None)
        if trans:
            source_text = trans['translated_text']
            st.markdown(f"### 🌐 Translation ({trans['model_name']})")
        else:
            st.error("Translation not found")
            st.stop()
    
    word_count = len(source_text.split())
    st.caption(f"Words: {word_count}")
    
    with st.expander("View Source Text"):
        st.text(source_text[:2000] + ("..." if len(source_text) > 2000 else ""))
    
    st.divider()
    
    # Check for existing summary
    existing = get_summary(section_id, tier, length_type, model, source_type, source_id)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### 📝 Summary ({length_type})")
    
    with col2:
        if existing:
            st.success("📦 Cached")
        else:
            st.info("⚡ New generation")
    
    # Generate button
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        
        if existing:
            st.success("✅ Loaded from cache!")
            summary = existing['summary_text']
        else:
            if not OLLAMA_AVAILABLE and provider == "ollama":
                st.error("Ollama not available. Install with: pip install ollama")
                st.stop()
            
            with st.spinner(f"Summarizing with {model} ({tier} tier, {length_type})..."):
                try:
                    start_time = time.time()
                    
                    summary = summarize_text(
                        text=source_text,
                        model=model,
                        tier=tier,
                        length_type=length_type,
                        provider=provider
                    )
                    
                    elapsed = time.time() - start_time
                    
                    # Save to database
                    save_summary(section_id, tier, length_type, model, summary, source_type, source_id)
                    
                    st.success(f"✅ Summary complete! ({elapsed:.1f}s)")
                    
                except Exception as e:
                    st.error(f"❌ Summarization failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
        
        # Display summary
        st.divider()
        
        st.markdown("**Summary:**")
        st.markdown(summary)
        
        # Stats
        summary_words = len(summary.split())
        compression = (summary_words / word_count * 100) if word_count > 0 else 0
        
        st.caption(f"Original: {word_count} words → Summary: {summary_words} words ({compression:.1f}% of original)")
        
        # Actions
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎧 Listen to Summary"):
                st.session_state.audio_section_id = section_id
                st.session_state.audio_source = "summary"
                st.switch_page("pages/5_🎧_Listen.py")
        
        with col2:
            st.download_button(
                "💾 Download",
                summary,
                file_name=f"{section['section_title']}_summary_{length_type}.txt",
                mime="text/plain"
            )

    # Show existing summaries
    st.divider()
    with st.expander("📚 Previous Summaries"):
        prev_summaries = get_section_summaries(section_id)
        if prev_summaries:
            for s in prev_summaries:
                st.markdown(f"- **{s['length_type']}** | {s['tier']} | {s['model_name']} | {s['source_type']} | {s['created_at']}")
        else:
            st.caption("No previous summaries for this section.")
