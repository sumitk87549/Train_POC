"""
Page 2: Books Library
Browse all books and their sections stored in the database
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_utils import get_books, get_sections, get_section, delete_book

st.set_page_config(page_title="📖 Books Library", page_icon="📖", layout="wide")

st.title("📖 Books Library")
st.markdown("Browse all books and sections stored in the database.")

st.divider()

# Load books
try:
    books = get_books()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

if not books:
    st.info("📚 No books in library yet!")
    st.markdown("Go to **📤 Upload EPUB** to add your first book.")
    st.stop()

# Book selector
st.markdown(f"### 📚 {len(books)} Books Available")

# Display books in grid
cols = st.columns(3)

for i, book in enumerate(books):
    with cols[i % 3]:
        with st.container(border=True):
            st.markdown(f"#### 📕 {book['title'][:40]}{'...' if len(book['title']) > 40 else ''}")
            st.caption(f"✍️ {book['author'] or 'Unknown Author'}")
            st.caption(f"📑 {book['section_count']} sections")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📖 View", key=f"view_{book['id']}", use_container_width=True):
                    st.session_state.selected_book_id = book['id']
            with col2:
                if st.button("🗑️", key=f"del_{book['id']}", use_container_width=True):
                    st.session_state.delete_book_id = book['id']

# Handle delete confirmation
if 'delete_book_id' in st.session_state:
    book_id = st.session_state.delete_book_id
    book = next((b for b in books if b['id'] == book_id), None)
    if book:
        st.warning(f"⚠️ Delete '{book['title']}'? This will remove all sections, translations, summaries, and audio!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Delete", type="primary"):
                delete_book(book_id)
                del st.session_state.delete_book_id
                st.rerun()
        with col2:
            if st.button("❌ Cancel"):
                del st.session_state.delete_book_id
                st.rerun()

st.divider()

# Selected book details
if 'selected_book_id' in st.session_state:
    book_id = st.session_state.selected_book_id
    book = next((b for b in books if b['id'] == book_id), None)
    
    if book:
        st.markdown(f"## 📖 {book['title']}")
        st.markdown(f"**Author:** {book['author'] or 'Unknown'}")
        st.markdown(f"**File:** {book['filename'] or 'N/A'}")
        
        st.divider()
        
        # Load sections
        sections = get_sections(book_id)
        
        st.markdown(f"### 📑 {len(sections)} Sections")
        
        # Section list
        for section in sections:
            with st.expander(f"**{section['section_number']}. {section['section_title']}** ({section['word_count']} words)"):
                # Content preview
                content = section['content']
                if len(content) > 1000:
                    st.text(content[:1000] + "...")
                    st.caption(f"... {len(content) - 1000} more characters")
                else:
                    st.text(content)
                
                # Action buttons
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("🌐 Translate", key=f"trans_{section['id']}"):
                        st.session_state.translate_section_id = section['id']
                        st.switch_page("pages/3_🌐_Translate.py")
                
                with col2:
                    if st.button("📝 Summarize", key=f"sum_{section['id']}"):
                        st.session_state.summarize_section_id = section['id']
                        st.switch_page("pages/4_📝_Summarize.py")
                
                with col3:
                    if st.button("🎧 Listen", key=f"audio_{section['id']}"):
                        st.session_state.audio_section_id = section['id']
                        st.switch_page("pages/5_🎧_Listen.py")
                
                with col4:
                    if st.button("📋 Copy", key=f"copy_{section['id']}"):
                        st.code(content, language=None)
        
        # Stats
        st.divider()
        total_words = sum(s['word_count'] for s in sections)
        st.markdown(f"**Total words:** {total_words:,}")

else:
    st.info("👆 Click **View** on a book above to see its sections")

# Tips
st.divider()
with st.expander("💡 Tips"):
    st.markdown("""
    **Navigation:**
    - Click **View** to see book sections
    - Click on section expanders to read content
    - Use action buttons to translate, summarize, or generate audio
    
    **Actions:**
    - 🌐 **Translate** - Translate section to Hindi
    - 📝 **Summarize** - Generate summary of section
    - 🎧 **Listen** - Generate audio for section
    - 📋 **Copy** - Copy full text
    
    **Deleting:**
    - Deleting a book removes all associated data
    - This includes all translations, summaries, and audio files
    """)
