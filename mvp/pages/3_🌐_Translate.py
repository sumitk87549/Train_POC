"""
Page 3: Translation - ENHANCED with Streaming & Progress Display
Translate book sections to Hindi with live progress and model insights
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
    translate_text, get_available_models, MODEL_TIERS, OLLAMA_AVAILABLE,
    _translate_ollama_stream, TRANSLATION_PROMPTS
)

st.set_page_config(page_title="🌐 Translate", page_icon="🌐", layout="wide")

st.title("🌐 Translate Section")
st.markdown("Translate book sections to Hindi with live progress display")

# Performance optimization
if 'translation_cache' not in st.session_state:
    st.session_state.translation_cache = {}

st.divider()

# ===== SIDEBAR OPTIONS =====
with st.sidebar:
    st.markdown("### ⚙️ Translation Options")
    
    # Quick Presets
    st.markdown("**🚀 Quick Presets**")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    with preset_col1:
        if st.button("⚡ Fast", use_container_width=True, help="Quick test, good quality"):
            st.session_state.preset = "FAST"
    with preset_col2:
        if st.button("⚖️ Balanced", use_container_width=True, help="Best balance"):
            st.session_state.preset = "BALANCED"
    with preset_col3:
        if st.button("💎 Quality", use_container_width=True, help="Slow, best quality"):
            st.session_state.preset = "QUALITY"
    
    st.divider()
    
    # Provider
    provider = st.radio(
        "Provider",
        ["ollama", "huggingface"],
        help="💡 Ollama recommended for local inference with live progress"
    )
    
    # Tier with explanations
    tier_help = {
        "BASIC": "⚡ Fast, concise prompts. Use for quick tests or simple text.",
        "INTERMEDIATE": "⚖️ Balanced. Better context preservation. Recommended for most use cases.",
        "ADVANCED": "💎 Comprehensive prompts. Best quality but slower. Use for important translations."
    }
    
    tier_choice = st.selectbox(
        "Quality Tier",
        ["BASIC", "INTERMEDIATE", "ADVANCED"],
        index=1 if 'preset' not in st.session_state else (0 if st.session_state.preset == "FAST" else 1),
        help="Higher tiers = more detailed prompts → better quality → slower"
    )
    
    st.info(tier_help[tier_choice])
    
    # Model selection with smart recommendations
    st.markdown("**🤖 Model Selection**")
    
    # Get preset-based recommendation
    if 'preset' in st.session_state:
        if st.session_state.preset == "FAST":
            recommended_tier = "FAST"
        elif st.session_state.preset == "BALANCED":
            recommended_tier = "BALANCED"
        else:
            recommended_tier = "QUALITY"
    else:
        if tier_choice == "BASIC":
            recommended_tier = "FAST"
        elif tier_choice == "INTERMEDIATE":
            recommended_tier = "BALANCED"
        else:
            recommended_tier = "QUALITY"
    
    recommended_models = MODEL_TIERS.get(recommended_tier, ["qwen2.5:3b"])
    available_models = get_available_models(provider)
    
    model = st.selectbox(
        "Model",
        available_models if available_models else recommended_models,
        index=0,
        help=f"💡 Recommended: {recommended_models[0]}"
    )
    
    # Show model info
    if "deepseek-r1" in model:
        st.success("🧠 Reasoning model detected! You'll see the thinking process.")
    elif "qwen2.5:3b" in model:
        st.info("⚡ Fast model - Great for testing (~30-60s per chunk)")
    elif "qwen2.5:7b" in model:
        st.info("⚖️ Balanced - Excellent quality (~60-120s per chunk)")
    elif "qwen2.5:14b" in model:
        st.info("💎 High quality - Best results (~120-300s per chunk)")
    
    custom_model = st.text_input("Or custom model", help="Enter any Ollama model name")
    if custom_model:
        model = custom_model
    
    st.divider()
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        temperature = st.slider(
            "Temperature",
            0.1, 1.0, 0.3, 0.1,
            help="📊 Lower (0.1-0.3) = consistent, factual\n📊 Higher (0.7-1.0) = creative, varied"
        )
        
        chunk_words = st.number_input(
            "Chunk Size (words)",
            100, 1000, 350, 50,
            help="💡 Smaller chunks = faster feedback, more API calls\n💡 Larger chunks = better context, fewer calls"
        )
        
        show_thinking = st.checkbox(
            "Show Thinking Process",
            value=True,
            help="Display model's reasoning for deepseek-r1 and similar models"
        )
    
    # Language (prepared for future)
    language = st.selectbox(
        "Target Language",
        ["hindi"],
        help="🚧 More languages coming soon!"
    )
    
    st.divider()
    
    # Tips
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        **For Testing:**
        - Use Fast preset with qwen2.5:3b
        - Try small sections first
        - Temperature 0.3 is safe
        
        **For Production:**
        - Use Balanced/Quality preset
        - qwen2.5:7b or deepseek-r1:7b
        - Temperature 0.2-0.4
        
        **Understanding Tiers:**
        - BASIC: "Translate this to Hindi"
        - INTERMEDIATE: "Translate preserving context and tone"
        - ADVANCED: "Publish-quality literary translation"
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
    
    # Check for pre-selected section
    preselected_section_id = st.session_state.get('translate_section_id')
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
        if 'translate_section_id' in st.session_state:
            del st.session_state['translate_section_id']
    
    selected_section = st.selectbox("📑 Select Section", list(section_options.keys()), index=default_section_idx)
    section_id = section_options[selected_section]

st.divider()

# Load section content
section = get_section(section_id)

if section:
    # Display info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words", f"{section['word_count']:,}")
    with col2:
        estimated_time = section['word_count'] / 350 * 45  # Rough estimate
        st.metric("Est. Time", f"~{estimated_time:.0f}s")
    with col3:
        chunks_est = (section['word_count'] // chunk_words) + 1
        st.metric("Chunks", chunks_est)
    
    # Original text preview
    with st.expander("📄 View Original Text", expanded=False):
        st.text_area("", section['content'], height=200, disabled=True, key="orig_preview")
    
    st.divider()
    
    # Check for existing translation
    existing = get_translation(section_id, tier_choice, model, language)
    
    if existing:
        st.success("📦 Found cached translation! Click below to load or regenerate.")
    
    # Generate button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        generate_btn = st.button(
            "🚀 Translate Now" if not existing else "🔄 Regenerate Translation",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if existing:
            load_btn = st.button("📦 Load Cached", use_container_width=True)
        else:
            load_btn = False
    
    # Handle load cached
    if load_btn and existing:
        st.session_state.current_translation = existing['translated_text']
        st.session_state.translation_loaded = True
        st.rerun()
    
    # Handle generate
    if generate_btn:
        if not OLLAMA_AVAILABLE and provider == "ollama":
            st.error("❌ Ollama not available. Install with: pip install ollama")
            st.stop()
        
        # Create containers for live updates
        status_container = st.status("🚀 Starting translation...", expanded=True)
        thinking_container = st.empty()
        translation_container = st.empty()
        progress_bar = st.progress(0)
        
        try:
            start_time = time.time()
            
            with status_container:
                st.write("📝 Preparing prompt...")
                prompts = TRANSLATION_PROMPTS.get(tier_choice, TRANSLATION_PROMPTS["BASIC"])
                user_prompt = prompts["user"].format(text=section['content'])
                
                st.write(f"🤖 Using model: {model}")
                st.write(f"🎯 Tier: {tier_choice}")
                st.write(f"🌡️ Temperature: {temperature}")
                
                time.sleep(0.5)
                st.write("▶️ Starting generation...")
                
                # Stream the translation
                import ollama
                
                thinking_text = ""
                translation_text = ""
                full_text = ""
                in_thinking = False
                
                stream = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompts["system"]},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": temperature,
                        "num_ctx": 8192
                    },
                    stream=True
                )
                
                token_count = 0
                
                for chunk in stream:
                    if 'message' in chunk and 'content' in chunk['message']:
                        token = chunk['message']['content']
                        full_text += token
                        token_count += 1
                        
                        # Update progress
                        progress = min(token_count / 500, 0.99)  # Rough estimate
                        progress_bar.progress(progress)
                        
                        # Detect thinking tags
                        if '<think>' in token:
                            in_thinking = True
                            token = token.replace('<think>', '')
                            st.write("🧠 **Model is reasoning...**")
                        
                        if '</think>' in token:
                            in_thinking = False
                            token = token.replace('</think>', '')
                            st.write("✅ **Thinking complete, generating translation...**")
                        
                        # Display appropriately
                        if in_thinking and show_thinking:
                            thinking_text += token
                            thinking_container.text_area(
                                "🧠 Thinking Process",
                                thinking_text,
                                height=150,
                                key=f"think_{token_count}"
                            )
                        elif not in_thinking:
                            translation_text += token
                            translation_container.text_area(
                                "🌐 Translation (Live)",
                                translation_text,
                                height=300,
                                key=f"trans_{token_count}"
                            )
                
                progress_bar.progress(1.0)
                elapsed = time.time() - start_time
                
                # Clean translation
                from translation_engine import clean_translation
                final_translation = clean_translation(translation_text)
                
                # Save to database
                save_translation(section_id, tier_choice, model, final_translation, language)
                
                st.success(f"✅ Translation complete in {elapsed:.1f}s!")
                st.session_state.current_translation = final_translation
                st.session_state.translation_loaded = True
                
        except Exception as e:
            st.error(f"❌ Translation failed: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
            st.stop()
    
    # Display final translation
    if st.session_state.get('translation_loaded'):
        st.divider()
        st.markdown("### ✅ Translation Complete")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original (English)**")
            st.text_area("", section['content'], height=400, disabled=True, key="final_orig")
        
        with col2:
            st.markdown("**Translation (Hindi)**")
            st.text_area("", st.session_state.current_translation, height=400, disabled=True, key="final_trans")
        
        # Stats
        orig_words = len(section['content'].split())
        trans_chars = len(st.session_state.current_translation)
        
        st.caption(f"Original: {orig_words} words | Translation: {trans_chars} characters")
        
        # Actions
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Summarize Translation", use_container_width=True):
                st.session_state.summarize_section_id = section_id
                st.session_state.summarize_source = "translation"
                st.switch_page("pages/4_📝_Summarize.py")
        
        with col2:
            if st.button("🎧 Listen to Translation", use_container_width=True):
                st.session_state.audio_section_id = section_id
                st.session_state.audio_source = "translation"
                st.switch_page("pages/5_🎧_Listen.py")
        
        with col3:
            st.download_button(
                "💾 Download",
                st.session_state.current_translation,
                file_name=f"{section['section_title']}_hindi.txt",
                mime="text/plain",
                use_container_width=True
            )

# Show translation history
st.divider()
with st.expander("📚 Translation History"):
    prev_translations = get_section_translations(section_id)
    if prev_translations:
        for t in prev_translations:
            st.markdown(f"- **{t['tier']}** | {t['model_name']} | {t['language']} | {t['created_at']}")
    else:
        st.caption("No previous translations for this section.")
