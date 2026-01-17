#!/usr/bin/env python3
"""
universal_book_cleaner.py

Clean and segregate both EPUB and PDF files into well-structured, chapter-wise text files.
Automatically detects file type and applies appropriate processing.

Usage:
    python universal_book_cleaner.py --input-dir ./books --output-dir ./PROCESSED/1_clean --csv books.csv

Supports: .epub and .pdf files
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import re
from pathlib import Path
from difflib import SequenceMatcher
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional

# EPUB / HTML parsing
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# PDF parsing
import PyPDF2
import pdfplumber

from tqdm import tqdm

# Optional fast ratio
try:
    import Levenshtein
    def similarity(a,b): return Levenshtein.ratio(a or "", b or "")
except Exception:
    def similarity(a,b): return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

# -----------------------
# Configuration
# -----------------------
LOG_FILE = "universal_book_cleaner.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")])
logger = logging.getLogger("bookcleaner")

DEFAULT_CATALOG = [
    ("A Christmas Carol", "Charles Dickens"),
    ("Aesop's Fables", "Aesop"),
    ("Alice's Adventures in Wonderland", "Lewis Carroll"),
    ("Crime and Punishment", "Fyodor Dostoyevsky"),
    ("Dracula", "Bram Stoker"),
    ("Frankenstein", "Mary Wollstonecraft Shelley"),
    ("Pride and Prejudice", "Jane Austen"),
    ("The Great Gatsby", "F. Scott Fitzgerald"),
]

SECTION_PATTERNS = [
    r'^\s*PROLOGUE\b', r'^\s*EPILOGUE\b', r'^\s*PREFACE\b', r'^\s*FOREWORD\b', r'^\s*INTRODUCTION\b',
    r"^\s*AUTHOR['\u2019]?S NOTE\b", r'^\s*ACKNOWLEDG(E)?MENTS?\b',
    r'^\s*CONTENTS\b', r'^\s*INDEX\b', r'^\s*GLOSSARY\b', r'^\s*REFERENCES\b', r'^\s*BIBLIOGRAPHY\b',
    r'^\s*APPENDIX\b', r'^\s*THE END\b',
    r'^\s*CHAPTER\b', r'^\s*BOOK\b', r'^\s*PART\b',
    r'^\s*CHAPTER\s+[IVXLCDM]+\b', r'^\s*CHAPTER\s+\d+\b',
    r'^\s*PART\s+[IVXLCDM\d]+\b', r'^\s*BOOK\s+[IVXLCDM\d]+\b',
    r'^\s*STAVE\s+[IVXLCDM\w]+\b',
    r'^\s*ACT\s+[IVXLCDM]+\b', r'^\s*ACT\s+\d+\b',
    r'^\s*SCENE\s+[IVXLCDM]+\b', r'^\s*SCENE\s+\d+\b',
    r'^\s*FABLE\s+[IVXLCDM\d]+\b', r'^\s*TALE\s+[IVXLCDM\d]+\b',
]
SECTION_REGEX = re.compile("|".join("(" + p + ")" for p in SECTION_PATTERNS), flags=re.IGNORECASE)
CHAPTER_FALLBACK_REGEX = re.compile(r'^\s*(CHAPTER|CH\.|BOOK|PART|STAVE|ACT|SCENE|VOLUME)\b.*', flags=re.IGNORECASE)
SCENE_BREAK_REGEX = re.compile(r'^[\s*\-_·•–—=~#]+$')
CHAPTER_TITLE_PATTERN = re.compile(r'^(CHAPTER|BOOK|PART)\s+([IVXLCDM\d]+)\b', re.I)

# -----------------------
# File Discovery
# -----------------------
def find_best_book_file(title: str, author: Optional[str], book_dir: Path) -> Optional[Tuple[Path, str]]:
    """Find best matching EPUB or PDF file. Returns (path, type) or None."""
    epub_files = list(book_dir.glob("**/*.epub"))
    pdf_files = list(book_dir.glob("**/*.pdf"))
    all_files = [(f, "epub") for f in epub_files] + [(f, "pdf") for f in pdf_files]

    if not all_files:
        return None

    best = None
    best_score = 0.0
    best_type = None
    query = f"{title} {author or ''}".strip().lower()

    for f, ftype in all_files:
        name = f.stem.lower()
        s = similarity(query, name)
        tokens = [t for t in re.split(r'\W+', title.lower()) if len(t) > 2]
        token_score = sum(1 for t in tokens if t in name) / max(1, len(tokens))
        score = max(s, token_score * 0.9)
        if score > best_score:
            best_score = score
            best = f
            best_type = ftype

    if best_score < 0.25:
        logger.debug("No good match (score=%.3f) for '%s'", best_score, title)
        return None

    logger.debug("Best match for '%s' -> %s [%s] (score=%.3f)", title, best, best_type, best_score)
    return (best, best_type)

# -----------------------
# EPUB Processing
# -----------------------
def html_to_text_blocks(html: str) -> List[Tuple[str,str]]:
    """Convert HTML to (tag, text) blocks, handling <pre> tags."""
    soup = BeautifulSoup(html, "lxml")
    blocks: List[Tuple[str,str]] = []

    for pre in soup.find_all('pre'):
        text = pre.get_text()
        parts = re.split(r'\n\s*\n', text)
        for p in parts:
            p = p.strip()
            if p:
                p2 = re.sub(r'[ \t\r\f\v]+', ' ', p)
                p2 = re.sub(r'\n+', ' ', p2).strip()
                blocks.append(('pre', p2))

    for code in soup.find_all('code'):
        text = code.get_text()
        if text and len(text.strip()) > 0:
            p2 = re.sub(r'\s+', ' ', text).strip()
            blocks.append(('code', p2))

    for el in soup.find_all(['h1','h2','h3','h4','h5','h6','p','div','section','span']):
        if el.find_parent('pre'):
            continue
        txt = el.get_text(separator=" ", strip=True)
        if not txt:
            continue
        txt2 = re.sub(r'\s+', ' ', txt).strip()
        if txt2:
            blocks.append((el.name, txt2))

    if not blocks:
        body = soup.body if soup.body else soup
        body_text = body.get_text("\n", strip=True)
        parts = re.split(r'\n\s*\n', body_text)
        for p in parts:
            p2 = p.strip()
            if p2:
                p2 = re.sub(r'\s+', ' ', p2)
                blocks.append(('text', p2))
    return blocks

def is_audio_epub(epub_path: Path) -> bool:
    """Detect audio-only EPUBs."""
    file_size = epub_path.stat().st_size
    if file_size > 200000:
        return False

    try:
        book = epub.read_epub(str(epub_path))
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

        audio_keywords = ['librivox', '.mp3', '.ogg', 'audio', 'audiobook']
        text_content = ""
        for item in items[:3]:
            try:
                html = item.get_content().decode('utf-8', errors='ignore')
                text_content += html.lower()
            except:
                pass

        audio_score = sum(1 for kw in audio_keywords if kw in text_content)
        if file_size < 50000 and audio_score >= 2:
            return True
        return False
    except Exception:
        return False

def gather_text_from_epub(epub_path: Path) -> List[str]:
    """Extract text blocks from EPUB."""
    if is_audio_epub(epub_path):
        logger.warning(f"Audio EPUB detected: {epub_path.name}")
        return [f"[AUDIO EPUB - No text content available]"]

    book = epub.read_epub(str(epub_path))
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    blocks: List[str] = []

    for item in items:
        try:
            html = item.get_content().decode('utf-8', errors='ignore')
        except Exception:
            html = item.get_content().decode('latin-1', errors='ignore')
        pairs = html_to_text_blocks(html)
        for tag, txt in pairs:
            txt2 = re.sub(r'\s+', ' ', txt).strip()
            if txt2:
                blocks.append(txt2)

    if len(blocks) <= 3 and items:
        try:
            html = items[0].get_content().decode('utf-8', errors='ignore')
        except Exception:
            html = items[0].get_content().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(html, "lxml")
        body = soup.body
        if body:
            joined = body.get_text("\n", strip=True)
            parts = re.split(r'\n\s*\n', joined)
            new_blocks = []
            for p in parts:
                p2 = p.strip()
                if p2:
                    p2 = re.sub(r'\s+', ' ', p2)
                    new_blocks.append(p2)
            if len(new_blocks) > len(blocks):
                blocks = new_blocks

    return clean_gutenberg_content(blocks)

# -----------------------
# PDF Processing
# -----------------------
def gather_text_from_pdf(pdf_path: Path) -> List[str]:
    """Extract text blocks from PDF using multiple methods for best results."""
    blocks: List[str] = []

    # Method 1: Try pdfplumber first (better for formatted text)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Split into paragraphs
                    paragraphs = re.split(r'\n\s*\n', text)
                    for para in paragraphs:
                        # Clean up hyphenation and line breaks
                        para = re.sub(r'-\s*\n\s*', '', para)  # Remove hyphenation
                        para = re.sub(r'\n', ' ', para)  # Convert line breaks to spaces
                        para = re.sub(r'\s+', ' ', para).strip()
                        if para and len(para) > 20:  # Filter out very short fragments
                            blocks.append(para)
        logger.info(f"Extracted {len(blocks)} blocks from PDF using pdfplumber")
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}, trying PyPDF2")

    # Method 2: Fallback to PyPDF2 if pdfplumber failed or got few results
    if len(blocks) < 10:
        blocks = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        paragraphs = re.split(r'\n\s*\n', text)
                        for para in paragraphs:
                            para = re.sub(r'-\s*\n\s*', '', para)
                            para = re.sub(r'\n', ' ', para)
                            para = re.sub(r'\s+', ' ', para).strip()
                            if para and len(para) > 20:
                                blocks.append(para)
            logger.info(f"Extracted {len(blocks)} blocks from PDF using PyPDF2")
        except Exception as e:
            logger.error(f"PyPDF2 extraction also failed: {e}")
            return []

    # Clean common PDF artifacts
    cleaned_blocks = []
    page_number_pattern = re.compile(r'^\s*\d+\s*$')  # Just a number (page numbers)
    header_footer_pattern = re.compile(r'^.{1,50}$')  # Very short lines (often headers/footers)

    for block in blocks:
        # Skip page numbers
        if page_number_pattern.match(block):
            continue
        # Skip very short blocks unless they look like headings
        if len(block) < 30 and not is_section_heading(block):
            continue
        cleaned_blocks.append(block)

    # Apply Gutenberg cleaning (works for many public domain PDFs)
    return clean_gutenberg_content(cleaned_blocks)

# -----------------------
# Common Cleaning
# -----------------------
def clean_gutenberg_content(blocks: List[str]) -> List[str]:
    """Remove Gutenberg headers/footers and metadata."""
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
        block_clean = block.strip()
        if any(m.lower() in block_clean.lower() for m in start_markers):
            start_idx = i + 1
            break
        if re.search(r'\*{3,}\s*START\s+(?:OF\s+)?(?:THIS|THE)\s+PROJECT\s+GUTENBERG', block_clean, re.IGNORECASE):
            start_idx = i + 1
            break

    for i in range(start_idx, len(blocks)):
        block_clean = blocks[i].strip()
        if any(m.lower() in block_clean.lower() for m in end_markers):
            end_idx = i
            break
        if re.search(r'\*{3,}\s*END\s+(?:OF\s+)?(?:THIS|THE)\s+PROJECT\s+GUTENBERG', block_clean, re.IGNORECASE):
            end_idx = i
            break

    content_blocks = blocks[start_idx:end_idx]
    if not content_blocks and len(blocks) > 0:
        content_blocks = blocks

    # Filter metadata
    junk_patterns = [
        r'^(Title|Author|Release Date|Language|Credits|Encoding|Produced by|Copyright|Note)\s*:',
        r'^Produced by',
        r'Project Gutenberg',
        r'www\.gutenberg\.org',
    ]

    final_blocks = []
    for block in content_blocks:
        is_junk = False
        for pattern in junk_patterns:
            if re.match(pattern, block.strip(), re.IGNORECASE):
                is_junk = True
                break
        if not is_junk:
            final_blocks.append(block)

    return final_blocks

# -----------------------
# Section Detection
# -----------------------
def is_section_heading(text: str) -> Optional[str]:
    """Detect if text is a section heading."""
    t = (text or "").strip()
    if not t or len(t) < 3:
        return None

    if SCENE_BREAK_REGEX.match(t):
        return None
    if re.match(r'^[*\-_·•–—=~#\s]+$', t):
        return None

    t_clean = re.sub(r'^[\W_]+', '', t).strip()
    if not t_clean or re.sub(r'[A-Za-z0-9]', '', t_clean).strip() == t_clean.strip():
        return None

    if SECTION_REGEX.search(t_clean):
        return t_clean.upper()
    if CHAPTER_FALLBACK_REGEX.match(t_clean):
        return t_clean.upper()

    # Anthology titles (short ALL-CAPS starting with "THE")
    if len(t_clean) < 80 and t_clean.isupper() and re.match(r"^THE\s+[A-Z][A-Z\s\-,';]+$", t_clean):
        return t_clean

    return None

def split_into_sections(blocks: List[str]) -> List[Tuple[str, List[str]]]:
    """Split blocks into sections with enhanced chapter title capture."""
    sections: List[Tuple[str, List[str]]] = []
    current_title = None
    current_pars: List[str] = []
    skip_next = False

    def flush():
        nonlocal current_title, current_pars
        if current_title is None and current_pars:
            current_title = "OPENING_CREDITS"
        if current_pars:
            sections.append((current_title or "UNKNOWN", current_pars.copy()))
        current_title = None
        current_pars = []

    for idx, block in enumerate(blocks):
        if skip_next:
            skip_next = False
            continue

        heading = is_section_heading(block)
        if heading:
            flush()
            # Check for chapter title on next line
            if CHAPTER_TITLE_PATTERN.match(heading) and idx + 1 < len(blocks):
                potential_title = blocks[idx + 1].strip()
                if (len(potential_title) < 100 and
                        not is_section_heading(potential_title) and
                        (potential_title.isupper() or potential_title.istitle() or len(potential_title.split()) <= 10)):
                    if not potential_title.endswith('.') or len(potential_title) < 60:
                        heading = f"{heading}: {potential_title.upper()}"
                        skip_next = True

            current_title = re.sub(r'\s+', ' ', heading).strip()
            continue
        current_pars.append(block)

    flush()
    return sections

def write_output_file(out_dir: Path, book_title: str, author: Optional[str], sections: List[Tuple[str,List[str]]]) -> Path:
    """Write sections in Audible-style format."""
    safe_name = re.sub(r'[\\/:"*?<>|]+', '', book_title).strip()
    filename = f"{safe_name}__Cleaned.txt"
    outpath = out_dir / filename

    with open(outpath, "w", encoding="utf-8") as fh:
        filtered_sections = []
        for sec_title, paras in sections:
            if not paras:
                continue
            if sec_title == "OPENING_CREDITS" and len(paras) <= 3:
                combined = " ".join(paras).lower()
                if book_title.lower() in combined or (author and author.lower() in combined):
                    continue
            filtered_sections.append((sec_title, paras))

        for sec_title, paras in filtered_sections:
            title_line = re.sub(r'\s+', ' ', sec_title).strip() or "UNTITLED SECTION"
            fh.write(f"==={title_line}===\n\n")
            fh.write("\n\n".join(paras).strip() + "\n\n")

    return outpath

# -----------------------
# Main Processing
# -----------------------
def process_book(title: str, author: Optional[str], book_dir: Path, out_dir: Path) -> Dict:
    """Process a book (EPUB or PDF)."""
    logger.info("Processing: %s — %s", title, author or "")
    result = {"title": title, "author": author, "found_file": None, "file_type": None,
              "output_file": None, "sections": [], "warnings": []}

    book_match = find_best_book_file(title, author, book_dir)
    if book_match is None:
        warning = f"No matching file found for '{title}'"
        logger.warning(warning)
        result["warnings"].append(warning)
        return result

    book_file, file_type = book_match
    result["found_file"] = str(book_file)
    result["file_type"] = file_type

    try:
        if file_type == "epub":
            blocks = gather_text_from_epub(book_file)
        elif file_type == "pdf":
            blocks = gather_text_from_pdf(book_file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.exception("Failed to parse %s: %s", book_file, e)
        result["warnings"].append(f"Parse error: {e}")
        return result

    if not blocks:
        result["warnings"].append("No text blocks extracted")
        return result

    sections = split_into_sections(blocks)

    # Fallback splitting if needed
    if len(sections) <= 1:
        joined = "\n\n".join(blocks)
        parts = re.split(r'(?m)^\s*(CHAPTER\s+[IVXLCDM\d]+|CHAPTER\b)', joined, flags=re.IGNORECASE)
        if len(parts) > 3:
            sections = [("CHAPTER_FALLBACK", [p.strip()]) for p in parts if p.strip()]

    out_path = write_output_file(out_dir, title, author, sections)
    result["output_file"] = str(out_path)
    result["sections"] = [{"title": s[0], "paragraphs": len(s[1])} for s in sections]
    logger.info("Wrote: %s", out_path)

    return result

def load_csv_mapping(csv_path: Path) -> List[Tuple[str,str]]:
    """Load title,author pairs from CSV."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            parts = [p.strip() for p in ln.split(",", 1)]
            if len(parts) == 1:
                rows.append((parts[0], ""))
            else:
                rows.append((parts[0], parts[1]))
    return rows

def main():
    parser = argparse.ArgumentParser(description="Clean and segregate EPUB and PDF files")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory with EPUB/PDF files")
    parser.add_argument("--output-dir", "-o", default="./PROCESSED/1_clean", help="Output directory")
    parser.add_argument("--csv", "-c", help="CSV file with Title,Author pairs")
    args = parser.parse_args()

    book_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_csv_mapping(Path(args.csv)) if args.csv else DEFAULT_CATALOG

    results = []
    for title, author in tqdm(catalog, desc="Processing books"):
        try:
            res = process_book(title, author, book_dir, out_dir)
            results.append(res)
        except Exception as e:
            logger.exception("Error processing %s: %s", title, e)
            results.append({"title": title, "author": author, "error": str(e)})

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(book_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "results": results
    }

    with open(out_dir / "processing_report.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    logger.info("Complete! Report: %s", out_dir / "processing_report.json")

if __name__ == "__main__":
    main()