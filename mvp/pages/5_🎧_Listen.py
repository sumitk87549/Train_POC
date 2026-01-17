"""
Page 5: Audio Generation
Generate and play audio for book sections, translations, or summaries
"""

import streamlit as st
from pathlib import Path
import sys
import time
import base64

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_utils import (
    get_books, get_sections, get_section, 
    get_audio, save_audio, get_section_audio,
    get_section_translations, get_section_summaries
)
from audio_engine import (
    generate_audio, get_model_info, check_dependencies, 
    get_available_models, TTS_MODELS
)

st.set_page_config(page_title="🎧 Listen", page_icon="🎧", layout="wide")

st.title("🎧 Listen to Audio")
st.markdown("Generate and play audio for book sections, translations, or summaries.")

# Check dependencies
deps = check_dependencies()
if not deps['all_ok']:
    st.error("⚠️ Audio dependencies not fully installed!")
    st.code("pip install torch soundfile pydub transformers", language="bash")
    st.warning("Some features may not work without all dependencies.")

st.divider()

# ===== SIDEBAR OPTIONS =====
with st.sidebar:
    st.markdown("### ⚙️ Audio Options")
    
    # Provider
    provider = st.radio(
        "Provider",
        ["huggingface"],
        help="HuggingFace VITS/Bark models"
    )
    
    # Tier
    tier = st.selectbox(
        "Quality Tier",
        ["BASIC", "INTERMEDIATE", "ADVANCED"],
        help="Higher tiers = better quality, slower"
    )
    
    # Model
    st.markdown("**TTS Model**")
    
    available_models = get_available_models(tier, provider)
    
    model = st.selectbox(
        "Model",
        available_models if available_models else ["facebook/mms-tts-hin"],
        help="BASIC: fast VITS, ADVANCED: Bark (slow but better)"
    )
    
    custom_model = st.text_input("Or enter custom model")
    if custom_model:
        model = custom_model
    
    st.divider()
    
    # Model info
    st.markdown("### 📊 Model Info")
    models_info = TTS_MODELS.get(tier, {}).get(provider, [])
    for m in models_info:
        if m['name'] == model:
            st.caption(f"**{m['name']}**")
            st.caption(f"Language: {m['lang']}")
            st.caption(m['description'])
    
    st.divider()
    st.markdown("### 💡 Tips")
    st.caption("""
    - **facebook/mms-tts-hin**: Fast Hindi
    - **facebook/mms-tts-eng**: Fast English
    - **suno/bark**: Slow but expressive
    - Audio is cached in database
    - Long texts are chunked automatically
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
    
    preselected_section_id = st.session_state.get('audio_section_id')
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
        if 'audio_section_id' in st.session_state:
            del st.session_state['audio_section_id']
    
    selected_section = st.selectbox("📑 Select Section", list(section_options.keys()), index=default_section_idx)
    section_id = section_options[selected_section]

# Source selection
st.divider()

section = get_section(section_id)
translations = get_section_translations(section_id)
summaries = get_section_summaries(section_id)

source_options = ["Original Text"]
source_map = {"Original Text": ("original", None, section['content'] if section else "")}

for t in translations:
    label = f"Translation ({t['model_name'][:20]})"
    source_options.append(label)
    source_map[label] = ("translation", t['id'], t['translated_text'])

for s in summaries:
    label = f"Summary ({s['length_type']} - {s['model_name'][:15]})"
    source_options.append(label)
    source_map[label] = ("summary", s['id'], s['summary_text'])

preselected_source = st.session_state.get('audio_source', 'original')
default_source_idx = 0
if preselected_source == "translation" and len(translations) > 0:
    default_source_idx = 1
elif preselected_source == "summary" and len(summaries) > 0:
    default_source_idx = 1 + len(translations)

source_choice = st.selectbox("🔊 Generate Audio For", source_options, index=min(default_source_idx, len(source_options)-1))
source_type, source_id, source_text = source_map[source_choice]

if section:
    st.divider()
    
    # Display source
    st.markdown(f"### 📄 {source_choice}")
    
    word_count = len(source_text.split())
    char_count = len(source_text)
    
    st.caption(f"Words: {word_count} | Characters: {char_count}")
    
    # Estimate duration (rough: 150 words per minute)
    estimated_minutes = word_count / 150
    st.caption(f"Estimated audio duration: ~{estimated_minutes:.1f} minutes")
    
    with st.expander("View Source Text"):
        st.text(source_text[:1500] + ("..." if len(source_text) > 1500 else ""))
    
    st.divider()
    
    # Check for existing audio
    existing = get_audio(section_id, source_type, tier, model, source_id)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎧 Audio Player")
    
    with col2:
        if existing:
            st.success("📦 Cached")
        else:
            st.info("⚡ New generation")
    
    # Warning for long texts
    if word_count > 500:
        st.warning(f"⚠️ Long text ({word_count} words). Generation may take several minutes.")
    
    # Generate button
    if st.button("🚀 Generate Audio", type="primary", use_container_width=True):
        
        if existing and existing.get('audio_data'):
            st.success("✅ Loaded from cache!")
            audio_data = bytes(existing['audio_data'])
        else:
            if not deps['all_ok']:
                st.error("Audio dependencies not installed!")
                st.stop()
            
            with st.spinner(f"Generating audio with {model} ({tier} tier)... This may take a while."):
                try:
                    start_time = time.time()
                    
                    # Limit text for demo (full generation can be very slow)
                    text_to_speak = source_text
                    if word_count > 300:
                        st.info("💡 For demo purposes, limiting to first 300 words. Full generation available in production.")
                        words = source_text.split()[:300]
                        text_to_speak = ' '.join(words)
                    
                    audio_data = generate_audio(
                        text=text_to_speak,
                        model=model,
                        tier=tier,
                        provider=provider
                    )
                    
                    elapsed = time.time() - start_time
                    
                    # Save to database
                    save_audio(
                        section_id=section_id,
                        source_type=source_type,
                        tier=tier,
                        model_name=model,
                        audio_data=audio_data,
                        source_id=source_id
                    )
                    
                    st.success(f"✅ Audio generated! ({elapsed:.1f}s)")
                    
                except Exception as e:
                    st.error(f"❌ Audio generation failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
        
        # Display audio player
        st.divider()
        
        if audio_data:
            # Create audio player
            st.audio(audio_data, format="audio/wav")
            
            # Download button
            st.download_button(
                "💾 Download Audio",
                audio_data,
                file_name=f"{section['section_title']}_{source_type}.wav",
                mime="audio/wav"
            )
            
            # Audio info
            audio_size = len(audio_data) / 1024
            st.caption(f"Audio size: {audio_size:.1f} KB")
        else:
            st.warning("No audio data available")

    # Show existing audio files
    st.divider()
    with st.expander("📚 Previous Audio Files"):
        prev_audio = get_section_audio(section_id)
        if prev_audio:
            for a in prev_audio:
                st.markdown(f"- **{a['source_type']}** | {a['tier']} | {a['model_name']} | {a['created_at']}")
        else:
            st.caption("No previous audio for this section.")

# Dependency check section
st.divider()
with st.expander("🔧 System Check"):
    st.markdown("**TTS Dependencies:**")
    for dep, status in deps.items():
        if dep != 'all_ok':
            icon = "✅" if status else "❌"
            st.markdown(f"- {icon} {dep}")
