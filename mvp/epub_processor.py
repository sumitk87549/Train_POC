"""
EPUB Processing Module for ReadLyte MVP
Extracts, cleans, and segregates EPUB content into sections
"""

import re
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import shutil

# EPUB / HTML parsing
try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False
    print("⚠️ Install epub dependencies: pip install ebooklib beautifulsoup4 lxml")


# Section patterns from original script
SECTION_PATTERNS = [
    r'^\s*PROLOGUE\b', r'^\s*EPILOGUE\b', r'^\s*PREFACE\b', r'^\s*FOREWORD\b', r'^\s*INTRODUCTION\b',
    r"^\s*AUTHOR['\\u2019]?S NOTE\b", r'^\s*AUTHOR NOTE\b', r'^\s*ACKNOWLEDG(E)?MENTS?\b',
    r'^\s*CONTENTS\b', r'^\s*INDEX\b', r'^\s*GLOSSARY\b', r'^\s*REFERENCES\b', r'^\s*BIBLIOGRAPHY\b',
    r'^\s*APPENDIX\b', r'^\s*THE END\b',
    r'^\s*CHAPTER\b', r'^\s*BOOK\b', r'^\s*PART\b',
    r'^\s*CHAPTER\s+[IVXLCDM]+\b', r'^\s*CHAPTER\s+\d+\b',
    r'^\s*PART\s+[IVXLCDM\d]+\b', r'^\s*BOOK\s+[IVXLCDM\d]+\b',
    r'^\s*STAVE\s+[IVXLCDM\w]+\b',
    r'^\s*ACT\s+[IVXLCDM]+\b', r'^\s*ACT\s+\d+\b',
    r'^\s*SCENE\s+[IVXLCDM]+\b', r'^\s*SCENE\s+\d+\b',
]
SECTION_REGEX = re.compile("|".join("(" + p + ")" for p in SECTION_PATTERNS), flags=re.IGNORECASE)


def extract_metadata(book) -> Tuple[str, str]:
    """Extract title and author from EPUB metadata."""
    title = "Unknown Title"
    author = "Unknown Author"
    
    try:
        title_meta = book.get_metadata('DC', 'title')
        if title_meta:
            title = title_meta[0][0]
    except:
        pass
    
    try:
        author_meta = book.get_metadata('DC', 'creator')
        if author_meta:
            author = author_meta[0][0]
    except:
        pass
    
    return title, author


def html_to_text_blocks(html: str) -> List[str]:
    """Convert HTML to clean text blocks."""
    soup = BeautifulSoup(html, "lxml")
    blocks = []
    
    # Handle <pre> blocks (Gutenberg style)
    for pre in soup.find_all('pre'):
        text = pre.get_text()
        parts = re.split(r'\n\s*\n', text)
        for p in parts:
            p = p.strip()
            if p:
                p = re.sub(r'\s+', ' ', p)
                blocks.append(p)
    
    # Handle normal elements
    for el in soup.find_all(['h1','h2','h3','h4','h5','h6','p','div']):
        if el.find_parent('pre'):
            continue
        txt = el.get_text(separator=" ", strip=True)
        if txt:
            txt = re.sub(r'\s+', ' ', txt).strip()
            if txt:
                blocks.append(txt)
    
    # Fallback: get body text
    if not blocks:
        body = soup.body
        if body:
            body_text = body.get_text("\n", strip=True)
            parts = re.split(r'\n\s*\n', body_text)
            for p in parts:
                p = p.strip()
                if p:
                    blocks.append(re.sub(r'\s+', ' ', p))
    
    return blocks


def is_section_heading(text: str) -> Optional[str]:
    """Check if text is a section heading."""
    t = (text or "").strip()
    if not t or len(t) < 3:
        return None
    
    # Reject scene breaks
    if re.match(r'^[\s*\-_·•—–=~#]+$', t):
        return None
    
    t_clean = re.sub(r'^[\W_]+', '', t).strip()
    if not t_clean:
        return None
    
    if SECTION_REGEX.search(t_clean):
        return t_clean.upper()
    
    return None


def clean_gutenberg_content(blocks: List[str]) -> List[str]:
    """Remove Project Gutenberg header/footer content."""
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "Start of the Project Gutenberg EBook",
    ]
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
    ]
    
    start_idx = 0
    end_idx = len(blocks)
    
    for i, block in enumerate(blocks):
        block_lower = block.lower()
        if any(m.lower() in block_lower for m in start_markers):
            start_idx = i + 1
            break
    
    for i in range(start_idx, len(blocks)):
        block_lower = blocks[i].lower()
        if any(m.lower() in block_lower for m in end_markers):
            end_idx = i
            break
    
    content_blocks = blocks[start_idx:end_idx]
    
    # Filter junk lines
    junk_patterns = [
        r'^Produced by', r'^Title:', r'^Author:', r'^Release Date:',
        r'^Language:', r'^Character set', r'^Project Gutenberg',
    ]
    
    filtered = []
    for block in content_blocks:
        is_junk = False
        for pattern in junk_patterns:
            if re.match(pattern, block, re.IGNORECASE):
                is_junk = True
                break
        if not is_junk:
            filtered.append(block)
    
    return filtered if filtered else content_blocks


def split_into_sections(blocks: List[str]) -> List[Tuple[str, str]]:
    """Split text blocks into sections with titles."""
    sections = []
    current_title = "OPENING"
    current_content = []
    
    for block in blocks:
        heading = is_section_heading(block)
        if heading:
            # Save current section
            if current_content:
                content = "\n\n".join(current_content)
                sections.append((current_title, content))
            current_title = heading
            current_content = []
        else:
            current_content.append(block)
    
    # Save last section
    if current_content:
        content = "\n\n".join(current_content)
        sections.append((current_title, content))
    
    return sections


def process_epub(epub_path: str) -> Tuple[str, str, List[Dict]]:
    """
    Process an EPUB file and return title, author, and sections.
    
    Returns:
        (title, author, sections) where sections is a list of dicts:
        [{"title": "CHAPTER 1", "content": "...", "word_count": 123}, ...]
    """
    if not EPUB_AVAILABLE:
        raise ImportError("ebooklib not installed. Run: pip install ebooklib beautifulsoup4 lxml")
    
    book = epub.read_epub(str(epub_path))
    title, author = extract_metadata(book)
    
    # Extract all text
    all_blocks = []
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    
    for item in items:
        try:
            html = item.get_content().decode('utf-8', errors='ignore')
        except:
            html = item.get_content().decode('latin-1', errors='ignore')
        
        blocks = html_to_text_blocks(html)
        all_blocks.extend(blocks)
    
    # Clean Gutenberg content
    cleaned_blocks = clean_gutenberg_content(all_blocks)
    
    # Split into sections
    raw_sections = split_into_sections(cleaned_blocks)
    
    # Format output
    sections = []
    for i, (sec_title, content) in enumerate(raw_sections, 1):
        word_count = len(content.split()) if content else 0
        # Skip very short sections (likely metadata)
        if word_count > 20:
            sections.append({
                "number": i,
                "title": sec_title,
                "content": content,
                "word_count": word_count
            })
    
    return title, author, sections


def parse_segregated_file(file_path: str) -> List[Dict]:
    """Parse a file with ===SECTION TITLE=== markers."""
    sections = []
    current_title = None
    current_content = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            
            # Check for section marker
            if line.startswith('===') and line.endswith('==='):
                # Save previous section
                if current_title is not None and current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        sections.append({
                            "number": len(sections) + 1,
                            "title": current_title,
                            "content": content,
                            "word_count": len(content.split())
                        })
                
                # Start new section
                current_title = line.strip('=').strip()
                current_content = []
            else:
                current_content.append(line)
    
    # Save last section
    if current_title is not None and current_content:
        content = "\n".join(current_content).strip()
        if content:
            sections.append({
                "number": len(sections) + 1,
                "title": current_title,
                "content": content,
                "word_count": len(content.split())
            })
    
    return sections


if __name__ == "__main__":
    # Test with a sample EPUB
    import sys
    if len(sys.argv) > 1:
        epub_file = sys.argv[1]
        print(f"📖 Processing: {epub_file}")
        title, author, sections = process_epub(epub_file)
        print(f"📚 Title: {title}")
        print(f"✍️ Author: {author}")
        print(f"📑 Sections: {len(sections)}")
        for sec in sections[:5]:
            print(f"  - {sec['title']}: {sec['word_count']} words")
    else:
        print("Usage: python epub_processor.py <epub_file>")
