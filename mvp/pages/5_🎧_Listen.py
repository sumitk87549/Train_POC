"""
Page 5: Audio Generation - ENHANCED with Progress & Model Guidance
Generate and play audio with live progress and helpful tips
"""

import streamlit as st
from pathlib import Path
import sys
import time

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

st.title("🎧 Generate Audio")
st.markdown("Text-to-speech with live progress")

# Check dependencies
deps = check_dependencies()
if not deps['all_ok']:
    st.warning("⚠️ Some audio dependencies missing. Audio generation may be limited.")
    with st.expander("🔧 Install Missing Dependencies"):
        st.code("pip install torch soundfile pydub transformers", language="bash")

st.divider()

# ===== SIDEBAR OPTIONS =====
with st.sidebar:
    st.markdown("### ⚙️ Audio Options")
    
    # Quick presets
    st.markdown("**🚀 Quick Select**")
    preset_col1, preset_col2 = st.columns(2)
    
    with preset_col1:
        if st.button("⚡ Fast Hindi", use_container_width=True):
            st.session_state.audio_preset = {"provider": "huggingface", "tier": "BASIC", "model": "facebook/mms-tts-hin"}
    with preset_col2:
        if st.button("💎 Quality", use_container_width=True):
            st.session_state.audio_preset = {"provider": "huggingface", "tier": "INTERMEDIATE", "model": "suno/bark-small"}
    
    st.divider()
    
    # Provider
    provider = st.radio(
        "Provider",
        ["huggingface"],
        help="HuggingFace VITS/Bark models"
    )
    
    # Tier with detailed info
    tier_info = {
        "BASIC": "⚡ Fast VITS models - Good quality, quick generation",
        "INTERMEDIATE": "⚖️ Bark small - Better prosody, slower",
        "ADVANCED": "💎 Bark full - Best quality, very slow, GPU recommended"
    }
    
    tier = st.selectbox(
        "Quality Tier",
        ["BASIC", "INTERMEDIATE", "ADVANCED"],
        index=0 if 'audio_preset' not in st.session_state else 0,
        help="Higher tier = better prosody but much slower"
    )
    
    st.info(tier_info[tier])
    
    st.divider()
    
    # Model selection with info
    st.markdown("**🎙️ TTS Model**")
    
    available_models = get_available_models(tier, provider)
    
    model = st.selectbox(
        "Select Model",
        available_models if available_models else ["facebook/mms-tts-hin"],
        help="Choose based on your language and quality needs"
    )
    
    custom_model = st.text_input("Or custom model")
    if custom_model:
        model = custom_model
    
    # Model info display
    models_info = TTS_MODELS.get(tier, {}).get(provider, [])
    for m in models_info:
        if m['name'] == model:
            st.success(f"**Language:** {m['lang']}")
            st.caption(m['description'])
            if 'quality' in m:
                st.caption(f"💡 {m['quality']}")
    
    st.divider()
    
    # Advanced settings
    with st.expander("🔧 Advanced"):
        limit_chars = st.number_input(
            "Limit characters (0 = no limit)",
            0, 5000, 1000, 100,
            help="For testing, limit text length. 0 = generate full audio."
        )
        
        st.caption("💡 Limiting text speeds up generation for testing. Full generation can take several minutes for long text.")
    
    # Detailed tips
    with st.expander("💡 Model Guide"):
        st.markdown("""
        **BASIC Tier (Fast):**
        - **facebook/mms-tts-hin**: Hindi, ~10s per paragraph
        - **facebook/mms-tts-eng**: English, ~10s per paragraph
        - Best for: Quick testing, previews
        
        **INTERMEDIATE Tier:**
        - **suno/bark-small**: Multilingual, emotional, ~30s per paragraph
        - Best for: Good quality without long wait
        
        **ADVANCED Tier:**
        - **suno/bark**: Best quality, ~60s+ per paragraph
        - Needs: GPU recommended
        - Best for: Final production audio
        
        **Time Estimates:**
        - 100 words ≈ 15-60s depending on tier
        - 500 words ≈ 1-5 min depending on tier
        - 1000+ words ≈ 5-30+ min depending on tier
        
        💡 **Pro Tip**: Use BASIC for testing, ADVANCED for final
        """)

# ===== MAIN CONTENT =====

# Book/Section selection
col1, col2 = st.columns(2)

with col1:
    books = get_books()
    book_options = {f"{b['title'][:40]}... ({b['section_count']} sections)": b['id'] for b in books}
    
    if not book_options:
        st.warning("📚 No books in library!")
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
        st.warning("No sections.")
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

source_choice = st.selectbox("🔊 Audio Source", source_options, index=min(default_source_idx, len(source_options)-1))
source_type, source_id, source_text = source_map[source_choice]

if section:
    # Apply limit if set
    if limit_chars > 0 and len(source_text) > limit_chars:
        original_length = len(source_text)
        source_text = source_text[:limit_chars]
        st.warning(f"⚠️ Limited to first {limit_chars} chars (original: {original_length} chars)")
    
    word_count = len(source_text.split())
    char_count = len(source_text)
    
    # Estimate based on tier and model
    if tier == "BASIC":
        est_seconds = word_count / 10  # ~10 words per second
    elif tier == "INTERMEDIATE":
        est_seconds = word_count / 5   # ~5 words per second
    else:
        est_seconds = word_count / 2   # ~2 words per second
    
    est_minutes = est_seconds / 60
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words", f"{word_count:,}")
    with col2:
        st.metric("Characters", f"{char_count:,}")
    with col3:
        if est_minutes < 1:
            st.metric("Est. Time", f"~{est_seconds:.0f}s")
        else:
            st.metric("Est. Time", f"~{est_minutes:.1f} min")
    
    # Preview
    with st.expander("📄 View Source Text"):
        st.text_area("", source_text[:1000] + ("..." if len(source_text) > 1000 else ""), height=150, disabled=True)
    
    st.divider()
    
    # Check for existing
    existing = get_audio(section_id, source_type, tier, model, source_id)
    
    if existing and existing.get('audio_data'):
        st.success("📦 Found cached audio!")
    
    # Generate button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        generate_btn = st.button(
            "🚀 Generate Audio" if not existing else "🔄 Regenerate",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if existing and existing.get('audio_data'):
            load_btn = st.button("📦 Load Cached", use_container_width=True)
        else:
            load_btn = False
    
    # Handle load
    if load_btn and existing:
        st.session_state.current_audio = bytes(existing['audio_data'])
        st.session_state.audio_loaded = True
        st.rerun()
    
    # Handle generate
    if generate_btn:
        if not deps['all_ok']:
            st.error("❌ Audio dependencies not installed!")
            st.stop()
        
        # Progress
        status_container = st.status("🚀 Generating audio...", expanded=True)
        progress_bar = st.progress(0)
        
        try:
            start_time = time.time()
            
            with status_container:
                st.write(f"🔊 Source: {source_choice}")
                st.write(f"🎙️ Model: {model}")
                st.write(f"🎯 Tier: {tier}")
                st.write(f"📊 Text: {word_count} words, {char_count} chars")
                
                progress_bar.progress(0.1)
                time.sleep(0.3)
                
                st.write("🔄 Loading model...")
                progress_bar.progress(0.2)
                
                st.write("🎵 Generating audio...")
                
                # Generate
                audio_data = generate_audio(
                    text=source_text,
                    model=model,
                    tier=tier,
                    provider=provider
                )
                
                progress_bar.progress(0.9)
                
                # Save
                save_audio(
                    section_id=section_id,
                    source_type=source_type,
                    tier=tier,
                    model_name=model,
                    audio_data=audio_data,
                    source_id=source_id
                )
                
                progress_bar.progress(1.0)
                elapsed = time.time() - start_time
                
                st.write(f"✅ Complete in {elapsed:.1f}s!")
                
                st.session_state.current_audio = audio_data
                st.session_state.audio_loaded = True
        
        except Exception as e:
            st.error(f"❌ Failed: {str(e)}")
            import traceback
            with st.expander("🔍 Details"):
                st.code(traceback.format_exc())
            st.stop()
    
    # Display player
    if st.session_state.get('audio_loaded'):
        st.divider()
        st.markdown("### 🎧 Audio Player")
        
        audio_data = st.session_state.current_audio
        
        # Player
        st.audio(audio_data, format="audio/wav")
        
        # Download
        st.download_button(
            "💾 Download Audio",
            audio_data,
            file_name=f"{section['section_title']}_{source_type}.wav",
            mime="audio/wav",
            use_container_width=True
        )
        
        # Info
        audio_size = len(audio_data) / 1024
        st.caption(f"Size: {audio_size:.1f} KB")

# History
st.divider()
with st.expander("📚 Audio History"):
    prev_audio = get_section_audio(section_id)
    if prev_audio:
        for a in prev_audio:
            st.markdown(f"- **{a['source_type']}** | {a['tier']} | {a['model_name']} | {a['created_at']}")
    else:
        st.caption("No previous audio.")

# Dependency status
with st.expander("🔧 System Status"):
    for dep, status in deps.items():
        if dep != 'all_ok':
            icon = "✅" if status else "❌"
            st.markdown(f"{icon} **{dep}**")
