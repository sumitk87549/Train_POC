"""
Page 4: Summarization - ENHANCED with Progress & Better UX
Generate summaries with live progress and helpful parameter guidance
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
st.markdown("Generate context-aware summaries with live progress")

st.divider()

# ===== SIDEBAR OPTIONS =====
with st.sidebar:
    st.markdown("### ⚙️ Summary Options")
    
    # Quick Presets
    st.markdown("**🚀 Quick Testing**")
    preset_col1, preset_col2 = st.columns(2)
    
    with preset_col1:
        if st.button("⚡ Fast Test", use_container_width=True):
            st.session_state.sum_preset = {"tier": "BASIC", "length": "SHORT", "model_type": "FAST"}
    with preset_col2:
        if st.button("💎 Quality", use_container_width=True):
            st.session_state.sum_preset = {"tier": "ADVANCED", "length": "LONG", "model_type": "QUALITY"}
    
    st.divider()
    
    # Provider
    provider = st.radio(
        "Provider",
        ["ollama", "huggingface"],
        help="💡 Ollama recommended for local inference"
    )
    
    # Tier with detailed help
    tier_descriptions = {
        "BASIC": "📊 Factual summary - captures main points clearly",
        "INTERMEDIATE": "🎯 Analytical summary - identifies themes and connections (Recommended)",
        "ADVANCED": "🎓 Publication-quality - deep analysis with subtext and significance"
    }
    
    tier = st.selectbox(
        "Quality Tier",
        ["BASIC", "INTERMEDIATE", "ADVANCED"],
        index=1 if 'sum_preset' not in st.session_state else 
              (0 if st.session_state.sum_preset["tier"] == "BASIC" else 1),
        help="Higher tier = more analytical & nuanced understanding"
    )
    
    st.info(tier_descriptions[tier])
    
    st.divider()
    
    # Length with visual guide
    st.markdown("**📏 Summary Length**")
    length_info = get_length_info()
    
    # Visual length selector
    length_type = st.select_slider(
        "Length",
        options=["SHORT", "MEDIUM", "LONG"],
        value="MEDIUM" if 'sum_preset' not in st.session_state else st.session_state.sum_preset.get("length", "MEDIUM"),
        help="Slide to choose summary detail level"
    )
    
    # Show length details
    selected_info = length_info[length_type]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Target", selected_info['target'])
    with col2:
        st.metric("Ratio", selected_info['ratio'])
    
    st.caption(selected_info['description'])
    
    st.divider()
    
    # Model selection
    st.markdown("**🤖 Model**")
    
    recommended = MODEL_RECOMMENDATIONS.get(
        st.session_state.sum_preset.get("model_type", "QUALITY") if 'sum_preset' in st.session_state else "QUALITY",
        ["llama3.1:8b"]
    )
    
    available_models = get_available_models(provider)
    
    model = st.selectbox(
        "Select Model",
        available_models if available_models else recommended,
        help=f"💡 Recommended: {recommended[0]}"
    )
    
    # Model info
    if "llama3.2:3b" in model:
        st.info("⚡ Fast - Good for testing (~15-30s)")
    elif "llama3.1:8b" in model or "qwen2.5:7b" in model:
        st.success("⚖️ Excellent quality - Balanced speed/quality (~30-60s)")
    
    custom_model = st.text_input("Or custom model")
    if custom_model:
        model = custom_model
    
    st.divider()
    
    # Advanced settings
    with st.expander("🔧 Advanced"):
        temperature = st.slider(
            "Temperature",
            0.1, 0.5, 0.2, 0.05,
            help="📊 Lower = more factual\n📊 Higher = more creative"
        )
        
        st.caption("💡 Lower temperature (0.1-0.2) recommended for summaries to maintain factual accuracy")
    
    # Tips
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        **Length Guidelines:**
        - **SHORT** (5-10%): Quick overview for scanning
        - **MEDIUM** (15-20%): Best for general understanding
        - **LONG** (25-35%): Comprehensive, preserves nuance
        
        **Tier Advice:**
        - **BASIC**: Straightforward facts (news, reports)
        - **INTERMEDIATE**: Thematic analysis (most books)
        - **ADVANCED**: Literary analysis (classics, philosophy)
        
        **For Best Results:**
        - Use MEDIUM length + INTERMEDIATE tier
        - llama3.1:8b or qwen2.5:7b
        - Temperature 0.2
        """)

# ===== MAIN CONTENT =====

# Book/Section selection
col1, col2 = st.columns(2)

with col1:
    books = get_books()
    book_options = {f"{b['title'][:40]}... ({b['section_count']} sections)": b['id'] for b in books}
    
    if not book_options:
        st.warning("📚 No books in library. Upload an EPUB first!")
        st.stop()
    
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
    label = f"Translation ({t['model_name'][:20]} - {t['tier']})"
    source_options.append(label)
    source_map[label] = ("translation", t['id'])

preselected_source = st.session_state.get('summarize_source', 'original')
default_source_idx = 0
if preselected_source == "translation" and len(source_options) > 1:
    default_source_idx = 1

source_choice = st.selectbox("📄 Source to Summarize", source_options, index=default_source_idx)
source_type, source_id = source_map[source_choice]

if section:
    # Get source text
    if source_type == "original":
        source_text = section['content']
    else:
        trans = next((t for t in translations if t['id'] == source_id), None)
        if trans:
            source_text = trans['translated_text']
        else:
            st.error("Translation not found")
            st.stop()
    
    word_count = len(source_text.split())
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Source Words", f"{word_count:,}")
    with col2:
        # Estimate based on length type
        ratio_map = {"SHORT": 0.075, "MEDIUM": 0.175, "LONG": 0.30}
        expected_words = int(word_count * ratio_map[length_type])
        st.metric("Expected Summary", f"~{expected_words} words")
    with col3:
        estimated_time = 20 if tier == "BASIC" else (35 if tier == "INTERMEDIATE" else 50)
        st.metric("Est. Time", f"~{estimated_time}s")
    
    with st.expander("📄 View Source Text"):
        st.text_area("", source_text[:1500] + ("..." if len(source_text) > 1500 else ""), height=200, disabled=True)
    
    st.divider()
    
    # Check for existing summary
    existing = get_summary(section_id, tier, length_type, model, source_type, source_id)
    
    if existing:
        st.success("📦 Found cached summary!")
    
    # Generate button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        generate_btn = st.button(
            "🚀 Generate Summary" if not existing else "🔄 Regenerate",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if existing:
            load_btn = st.button("📦 Load Cached", use_container_width=True)
        else:
            load_btn = False
    
    # Handle load
    if load_btn and existing:
        st.session_state.current_summary = existing['summary_text']
        st.session_state.summary_loaded = True
        st.rerun()
    
    # Handle generate
    if generate_btn:
        if not OLLAMA_AVAILABLE and provider == "ollama":
            st.error("❌ Ollama not available. Install: pip install ollama")
            st.stop()
        
        # Progress display
        progress_container = st.status("🚀 Generating summary...", expanded=True)
        progress_bar = st.progress(0)
        summary_preview = st.empty()
        
        try:
            start_time = time.time()
            
            with progress_container:
                st.write(f"📝 Source: {source_choice}")
                st.write(f"📏 Length: {length_type} ({length_info[length_type]['ratio']})")
                st.write(f"🎯 Tier: {tier}")
                st.write(f"🤖 Model: {model}")
                
                progress_bar.progress(0.1)
                time.sleep(0.3)
                
                st.write("🔄 Processing...")
                
                # Generate
                summary = summarize_text(
                    text=source_text,
                    model=model,
                    tier=tier,
                    length_type=length_type,
                    provider=provider,
                    temperature=temperature
                )
                
                progress_bar.progress(0.9)
                
                # Save
                save_summary(section_id, tier, length_type, model, summary, source_type, source_id)
                
                progress_bar.progress(1.0)
                elapsed = time.time() - start_time
                
                st.write(f"✅ Complete in {elapsed:.1f}s!")
                
                st.session_state.current_summary = summary
                st.session_state.summary_loaded = True
        
        except Exception as e:
            st.error(f"❌ Failed: {str(e)}")
            import traceback
            with st.expander("🔍 Details"):
                st.code(traceback.format_exc())
            st.stop()
    
    # Display result
    if st.session_state.get('summary_loaded'):
        st.divider()
        st.markdown("### ✅ Summary")
        
        st.markdown(st.session_state.current_summary)
        
        # Stats
        summary_words = len(st.session_state.current_summary.split())
        compression = (summary_words / word_count * 100) if word_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original", f"{word_count} words")
        with col2:
            st.metric("Summary", f"{summary_words} words")
        with col3:
            st.metric("Compression", f"{compression:.1f}%")
        
        # Actions
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎧 Listen to Summary", use_container_width=True):
                st.session_state.audio_section_id = section_id
                st.session_state.audio_source = "summary"
                st.switch_page("pages/5_🎧_Listen.py")
        
        with col2:
            st.download_button(
                "💾 Download",
                st.session_state.current_summary,
                file_name=f"{section['section_title']}_summary_{length_type}.txt",
                mime="text/plain",
                use_container_width=True
            )

# History
st.divider()
with st.expander("📚 Summary History"):
    prev_summaries = get_section_summaries(section_id)
    if prev_summaries:
        for s in prev_summaries:
            st.markdown(f"- **{s['length_type']}** | {s['tier']} | {s['model_name']} | {s['created_at']}")
    else:
        st.caption("No previous summaries.")
