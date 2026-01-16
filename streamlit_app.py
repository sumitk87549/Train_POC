
import streamlit as st
import os
import sys
from processor import BookProcessor

# Fix for PyTorch/Streamlit compatibility issue
# Disable Streamlit's file watcher to prevent PyTorch compatibility issues
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Also try to monkey patch torch._classes if available
try:
    import torch
    # Monkey patch to prevent Streamlit from examining torch._classes
    if hasattr(torch, '_classes'):
        torch._classes.__path__ = None
except ImportError:
    pass

st.set_page_config(page_title="Book Processor AI", layout="wide")

# Initialize processor
if 'processor' not in st.session_state:
    st.session_state.processor = BookProcessor()

st.title("📚 AI Book Processor - Advanced Workbench")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Smart Process", "My Library"], index=0)

if page == "Smart Process":
    st.header("🧪 Advanced Processing Workbench")
    
    # 1. File Upload
    uploaded_file = st.file_uploader("Upload Book (PDF/EPUB)", type=["pdf", "epub"])
    
    # 2. Configuration Grid
    with st.expander("🛠️  Model & Parameter Configuration", expanded=True):
        col_prov, col_mod = st.columns([1, 2])
        
        with col_prov:
            text_provider = st.selectbox("Text Provider", ["ollama", "huggingface"])
            
        with col_mod:
            # Dynamic Model Loading
            if text_provider == "ollama":
                available_models = st.session_state.processor.get_ollama_models()
                # Suggestions
                suggestions = ["qwen2.5:3b", "deepseek-r1:1.5b", "deepseek-r1:7b", "llama3.2:3b"]
                # Merge unique
                all_models = list(set(suggestions + available_models))
                all_models.sort()
                
                text_model = st.selectbox(
                    "Select Model (Supports 'Thinking' models like deepseek-r1)", 
                    all_models, 
                    index=all_models.index("qwen2.5:3b") if "qwen2.5:3b" in all_models else 0
                )
            else:
                text_model = st.text_input("HuggingFace Model ID", value="meta-llama/Llama-3.2-3B-Instruct")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            chunk_size = st.number_input("Chunk Size (Words)", min_value=100, max_value=5000, value=2000, step=100)
        with col_p2:
            temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.3)
        with col_p3:
            tier = st.selectbox("Processing Tier", ["BASIC", "INTERMEDIATE", "ADVANCED"])

        st.subheader("🔊 Audio Settings")
        c_au1, c_au2 = st.columns(2)
        with c_au1:
             audio_provider = st.selectbox("Audio Provider", ["huggingface", "coqui"])
        with c_au2:
             # Basic mapping suggestion
             default_audio = "facebook/mms-tts-hin"
             audio_model = st.text_input("Audio Model", value=default_audio)
             
        action_options = st.multiselect(
            "Actions to Perform",
            ["Summarize", "Translate (to Hindi)", "Generate Audio"],
            default=["Summarize", "Translate (to Hindi)", "Generate Audio"]
        )

    # 3. Processing
    if st.button("🚀 Start Processing Experiment") and uploaded_file:
        try:
            ext = uploaded_file.name.split('.')[-1].lower()
            temp_filename = f"temp.{ext}"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            proc = st.session_state.processor
            text = ""
            
            # --- Status Container ---
            status_container = st.container()
            with status_container:
                st.info("📖 Reading & Cleaning File...")
                if ext == "pdf":
                    text = proc.extract_text_from_pdf(temp_filename)
                elif ext == "epub":
                    text = proc.extract_text_from_epub(temp_filename)
                
                cleaned_text = proc.clean_text(text)
                st.write(f"Refined Text Length: {len(cleaned_text)} characters")
                
                final_summary = ""
                final_translation = ""
                audio_data = None
                
                # --- Summarization Loop ---
                if "Summarize" in action_options:
                    st.divider()
                    st.subheader("📝 Generating Summary")
                    
                    summary_box = st.empty()
                    thinking_expander = None
                    thinking_text = ""
                    current_content = ""
                    
                    for event in proc.stream_summary(
                        cleaned_text, 
                        model=text_model, 
                        provider=text_provider,
                        chunk_size=chunk_size,
                        temperature=temperature
                    ):
                        if event["type"] == "section_header":
                            st.caption(event["content"])
                        elif event["type"] == "thinking":
                            if thinking_expander is None:
                                thinking_expander = st.status("🧠 Model Thinking...", expanded=True)
                            thinking_text += event["content"]
                            thinking_expander.code(thinking_text, language=None)
                        elif event["type"] == "thinking_done":
                            if thinking_expander:
                                thinking_expander.update(label="Thinking Complete", state="complete", expanded=False)
                                thinking_expander = None # Reset for next chunk if any
                        elif event["type"] == "content":
                            current_content += event["content"]
                            summary_box.markdown(current_content + "▌")
                        elif event["type"] == "error":
                            st.error(event["content"])
                            
                    final_summary = current_content
                    summary_box.markdown(final_summary) # Final render without cursor

                # --- Translation Loop ---
                if "Translate (to Hindi)" in action_options:
                    st.divider()
                    st.subheader("🈯 Generating Translation")
                    
                    trans_box = st.empty()
                    thinking_expander = None
                    thinking_text = ""
                    current_content = ""
                    
                    for event in proc.stream_translation(
                        cleaned_text, 
                        model=text_model, 
                        provider=text_provider,
                        chunk_size=(chunk_size // 4), # Translations usually need smaller chunks
                        temperature=temperature
                    ):
                        if event["type"] == "section_header":
                            st.caption(event["content"])
                        elif event["type"] == "thinking":
                            if thinking_expander is None:
                                thinking_expander = st.status("🧠 Model Thinking...", expanded=True)
                            thinking_text += event["content"]
                            thinking_expander.code(thinking_text, language=None)
                        elif event["type"] == "thinking_done":
                            if thinking_expander:
                                thinking_expander.update(label="Thinking Complete", state="complete", expanded=False)
                                thinking_expander = None
                        elif event["type"] == "content":
                            current_content += event["content"]
                            trans_box.markdown(current_content + "▌")
                        elif event["type"] == "error":
                            st.error(event["content"])

                    final_translation = current_content
                    trans_box.markdown(final_translation)

                # --- Audio Generation ---
                if "Generate Audio" in action_options:
                    st.divider()
                    st.subheader("🎧 Generating Audio")
                    st.caption("Using generated translation if available, otherwise summary or text.")
                    
                    with st.spinner("Synthesizing..."):
                        text_for = final_translation if (final_translation and "Translate (to Hindi)" in action_options) else (final_summary if final_summary else cleaned_text)
                        
                        if text_for:
                             audio_data = proc.generate_audio_from_text(text_for, model=audio_model, provider=audio_provider)
                             if audio_data:
                                 st.success("Audio Ready!")
                                 st.audio(audio_data, format="audio/mp3")
                             else:
                                 st.error("Audio Generation Failed.")
                        else:
                             st.warning("No text content available to generate audio.")

                # --- DB Save ---
                st.divider()
                with st.spinner("Saving results to Database..."):
                     proc.save_to_db(uploaded_file.name, uploaded_file.name, cleaned_text, final_summary, final_translation, audio_data)
                     st.success(f"✅ Saved '{uploaded_file.name}' to Library!")
                     
            os.remove(temp_filename)

        except Exception as e:
            st.error(f"Processing Error: {e}")
            import traceback
            st.code(traceback.format_exc())

elif page == "My Library":
    st.header("📚 My Processed Library")
    
    proc = st.session_state.processor
    books = proc.get_all_books()
    
    if not books:
        st.info("Library is empty. Process a book to get started!")
    else:
        for book in books:
            book_id, title, original_filename, created_at = book
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.subheader(title)
                c1.caption(f"File: {original_filename} | Processed: {created_at}")
                if c2.button("Open Reader", key=f"read_{book_id}"):
                    st.session_state.current_book = book_id
                    st.rerun()
    
    # Detail View
    if 'current_book' in st.session_state:
        st.divider()
        book_details = proc.get_book_details(st.session_state.current_book)
        if book_details:
             title, cleaned_text, summary, translation, audio_data = book_details
             st.markdown(f"## 📖 Reading: {title}")
             
             t1, t2, t3, t4 = st.tabs(["📝 Summary", "🈯 Translation", "📄 Original Text", "🎧 Audio"])
             
             with t1: st.markdown(summary if summary else "*No summary available*")
             with t2: st.markdown(translation if translation else "*No translation available*")
             with t3: st.text_area("Full Text", cleaned_text, height=400)
             with t4: 
                 if audio_data:
                     st.audio(audio_data, format="audio/mp3")
                     st.download_button("Download MP3", audio_data, file_name=f"{title}.mp3", mime="audio/mp3")
                 else:
                     st.info("No audio available")

        if st.button("Close Reader"):
            del st.session_state.current_book
            st.rerun()
