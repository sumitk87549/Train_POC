"""
Page 1: EPUB Upload and Processing
Upload EPUB files, clean, segregate, and store sections in database
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_utils import save_book, save_section, get_books
from epub_processor import process_epub, EPUB_AVAILABLE

st.set_page_config(page_title="📤 Upload EPUB", page_icon="📤", layout="wide")

st.title("📤 Upload EPUB")
st.markdown("Upload an EPUB file to clean, segregate, and store in the database.")

if not EPUB_AVAILABLE:
    st.error("⚠️ EPUB processing dependencies not installed!")
    st.code("pip install ebooklib beautifulsoup4 lxml", language="bash")
    st.stop()

st.divider()

# Upload section
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose an EPUB file",
        type=['epub'],
        help="Upload an EPUB book file to process"
    )

with col2:
    st.markdown("### Manual Override")
    manual_title = st.text_input("Title (optional)", help="Override extracted title")
    manual_author = st.text_input("Author (optional)", help="Override extracted author")

if uploaded_file:
    st.divider()
    
    # File info
    st.markdown(f"**File:** `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")
    
    # Process button
    if st.button("🚀 Process EPUB", type="primary", use_container_width=True):
        
        with st.spinner("Processing EPUB..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            try:
                # Process the EPUB
                progress_bar = st.progress(0, text="Extracting content...")
                
                title, author, sections = process_epub(tmp_path)
                
                # Use manual overrides if provided
                if manual_title:
                    title = manual_title
                if manual_author:
                    author = manual_author
                
                progress_bar.progress(30, text="Saving book to database...")
                
                # Save to database
                book_id = save_book(title, author, uploaded_file.name)
                
                progress_bar.progress(50, text="Saving sections...")
                
                # Save sections
                for i, section in enumerate(sections):
                    save_section(
                        book_id=book_id,
                        section_number=section["number"],
                        section_title=section["title"],
                        content=section["content"]
                    )
                    progress = 50 + int((i + 1) / len(sections) * 50)
                    progress_bar.progress(progress, text=f"Saving section {i+1}/{len(sections)}...")
                
                progress_bar.progress(100, text="Complete!")
                
                # Success message
                st.success(f"✅ Book processed successfully!")
                
                # Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Title", title[:30] + "..." if len(title) > 30 else title)
                with col2:
                    st.metric("Author", author[:20] + "..." if len(author) > 20 else author)
                with col3:
                    st.metric("Sections", len(sections))
                
                # Section preview
                st.markdown("### 📑 Sections Preview")
                
                for section in sections[:5]:  # Show first 5
                    with st.expander(f"**{section['title']}** ({section['word_count']} words)"):
                        preview = section['content'][:500] + "..." if len(section['content']) > 500 else section['content']
                        st.text(preview)
                
                if len(sections) > 5:
                    st.info(f"... and {len(sections) - 5} more sections")
                
                # Next steps
                st.divider()
                st.markdown("### ➡️ Next Steps")
                st.markdown("""
                - Go to **📖 Books Library** to browse your books
                - Select sections to **🌐 Translate** or **📝 Summarize**
                - Generate **🎧 Audio** for any content
                """)
                
            except Exception as e:
                st.error(f"❌ Error processing EPUB: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            
            finally:
                # Cleanup temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

else:
    # No file uploaded - show existing books
    st.divider()
    st.markdown("### 📚 Existing Books in Library")
    
    try:
        books = get_books()
        if books:
            for book in books[:5]:
                st.markdown(f"- **{book['title']}** by {book['author'] or 'Unknown'} ({book['section_count']} sections)")
            if len(books) > 5:
                st.info(f"... and {len(books) - 5} more books")
        else:
            st.info("No books in library yet. Upload an EPUB to get started!")
    except Exception as e:
        st.warning(f"Could not load books: {e}")

# Help section
st.divider()
with st.expander("💡 Help & Tips"):
    st.markdown("""
    **Supported Formats:**
    - EPUB files (.epub)
    - Works best with Project Gutenberg books
    
    **Processing includes:**
    - Extracting text from EPUB structure
    - Cleaning Gutenberg headers/footers
    - Detecting chapter/section boundaries
    - Segregating into distinct sections
    
    **Section Detection:**
    - Chapters (Chapter I, Chapter 1, etc.)
    - Parts and Books
    - Prologue, Epilogue, Preface
    - Acts and Scenes (for plays)
    
    **Tips:**
    - If auto-detection doesn't find the right title/author, use manual override
    - Large books may take a few seconds to process
    - Each section is stored separately for individual processing
    """)
