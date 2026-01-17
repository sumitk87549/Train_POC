"""
Page 3: Translation
Translate book sections to Hindi with model options
"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_utils import (
    get_books, get_sections, get_section, 
    get_translation, save_translation, get_section_translations
)
from translation_engine import (
    translate_text, get_available_models, MODEL_TIERS, OLLAMA_AVAILABLE
)

st.set_page_config(page_title="🌐 Translate", page_icon="🌐", layout="wide")

st.title("🌐 Translate Section")
st.markdown("Translate book sections to Hindi with customizable model options.")

st.divider()

# ===== SIDEBAR OPTIONS =====
with st.sidebar:
    st.markdown("### ⚙️ Translation Options")
    
    # Provider
    provider = st.radio(
        "Provider",
        ["ollama", "huggingface"],
        help="Ollama is recommended for local inference"
    )
    
    # Model
    st.markdown("**Model**")
    available_models = get_available_models(provider)
    
    # Show tier recommendations
    tier_choice = st.selectbox(
        "Quality Tier",
        ["BASIC", "INTERMEDIATE", "ADVANCED"],
        help="Higher tiers use more detailed prompts for better quality"
    )
    
    # Model recommendations based on tier
    if tier_choice == "BASIC":
        recommended = MODEL_TIERS.get("FAST", ["qwen2.5:3b"])
    elif tier_choice == "INTERMEDIATE":
        recommended = MODEL_TIERS.get("BALANCED", ["qwen2.5:7b"])
    else:
        recommended = MODEL_TIERS.get("QUALITY", ["qwen2.5:14b"])
    
    model = st.selectbox(
        "Model Name",
        available_models if available_models else recommended,
        help=f"Recommended for {tier_choice}: {recommended[0] if recommended else 'qwen2.5:3b'}"
    )
    
    # Allow custom model
    custom_model = st.text_input("Or enter custom model name")
    if custom_model:
        model = custom_model
    
    # Temperature
    temperature = st.slider(
        "Temperature",
        0.1, 1.0, 0.3, 0.1,
        help="Lower = more consistent, Higher = more creative"
    )
    
    # Language (currently only Hindi)
    language = st.selectbox(
        "Target Language",
        ["hindi"],
        help="More languages coming soon!"
    )
    
    st.divider()
    st.markdown("### 💡 Tips")
    st.caption("""
    - **BASIC**: Fast, good for simple text
    - **INTERMEDIATE**: Better quality, preserves more nuance
    - **ADVANCED**: Best quality, longer time
    - Cached translations load instantly
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
    preselected_section_id = st.session_state.get('translate_section_id')
    default_book_idx = 0
    
    if preselected_section_id:
        # Find which book this section belongs to
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
    
    # Find default section index
    default_section_idx = 0
    if preselected_section_id:
        for i, (name, sid) in enumerate(section_options.items()):
            if sid == preselected_section_id:
                default_section_idx = i
                break
        # Clear the preselection
        del st.session_state['translate_section_id']
    
    selected_section = st.selectbox("📑 Select Section", list(section_options.keys()), index=default_section_idx)
    section_id = section_options[selected_section]

st.divider()

# Load section content
section = get_section(section_id)

if section:
    # Display original text
    st.markdown("### 📄 Original Text")
    st.caption(f"Words: {section['word_count']} | Section: {section['section_title']}")
    
    with st.expander("View Original Text", expanded=True):
        st.text(section['content'][:2000] + ("..." if len(section['content']) > 2000 else ""))
    
    st.divider()
    
    # Check for existing translation
    existing = get_translation(section_id, tier_choice, model, language)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🌐 Hindi Translation")
    
    with col2:
        if existing:
            st.success("📦 Cached")
        else:
            st.info("⚡ New generation")
    
    # Generate button
    if st.button("🚀 Generate Translation", type="primary", use_container_width=True):
        
        if existing:
            # Load from cache
            st.success("✅ Loaded from cache!")
            translation = existing['translated_text']
        else:
            # Generate new
            if not OLLAMA_AVAILABLE and provider == "ollama":
                st.error("Ollama not available. Install with: pip install ollama")
                st.stop()
            
            with st.spinner(f"Translating with {model} ({tier_choice} tier)..."):
                try:
                    start_time = time.time()
                    
                    translation = translate_text(
                        text=section['content'],
                        model=model,
                        tier=tier_choice,
                        provider=provider,
                        language=language,
                        temperature=temperature
                    )
                    
                    elapsed = time.time() - start_time
                    
                    # Save to database
                    save_translation(section_id, tier_choice, model, translation, language)
                    
                    st.success(f"✅ Translation complete! ({elapsed:.1f}s)")
                    
                except Exception as e:
                    st.error(f"❌ Translation failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
        
        # Display translation
        st.divider()
        
        # Side by side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original (English)**")
            st.text_area("", section['content'], height=400, disabled=True, key="orig")
        
        with col2:
            st.markdown("**Translation (Hindi)**")
            st.text_area("", translation, height=400, disabled=True, key="trans")
        
        # Stats
        orig_words = len(section['content'].split())
        trans_chars = len(translation)
        
        st.caption(f"Original: {orig_words} words | Translation: {trans_chars} characters")
        
        # Actions
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Summarize Translation"):
                st.session_state.summarize_section_id = section_id
                st.session_state.summarize_source = "translation"
                st.switch_page("pages/4_📝_Summarize.py")
        
        with col2:
            if st.button("🎧 Listen to Translation"):
                st.session_state.audio_section_id = section_id
                st.session_state.audio_source = "translation"
                st.switch_page("pages/5_🎧_Listen.py")
        
        with col3:
            st.download_button(
                "💾 Download",
                translation,
                file_name=f"{section['section_title']}_hindi.txt",
                mime="text/plain"
            )

    # Show existing translations for this section
    st.divider()
    with st.expander("📚 Previous Translations"):
        prev_translations = get_section_translations(section_id)
        if prev_translations:
            for t in prev_translations:
                st.markdown(f"- **{t['tier']}** | {t['model_name']} | {t['language']} | {t['created_at']}")
        else:
            st.caption("No previous translations for this section.")
