#!/usr/bin/env python3
"""
clean_and_segregate_epubs_fixed.py

Minimal, safe upgrade of your cleaning script to correctly extract EPUBs that store
their content inside <pre> blocks (Project Gutenberg style) or other non-<p> tags.

Usage (same as before):
    python clean_and_segregate_epubs_fixed.py --input-dir ./epubs --output-dir ./PROCESSED/1_clean --csv books.csv
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
from tqdm import tqdm

# Optional fast ratio
try:
    import Levenshtein
    def similarity(a,b): return Levenshtein.ratio(a or "", b or "")
except Exception:
    def similarity(a,b): return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

# -----------------------
# Configuration (unchanged)
# -----------------------
LOG_FILE = "clean_and_segregate.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")])
logger = logging.getLogger("cleanseg")

DEFAULT_CATALOG = [
    ("A Christmas Carol", "Charles Dickens"),
    ("Aesop's Fables", "Aesop"),
    ("Alice's Adventures in Wonderland", "Lewis Carroll"),
    ("A Room in a View", "E. M. (Edward Morgan) Forster"),
    ("Crime and Punishment", "Fyodor Dostoyevsky"),
    ("Dracula", "Bram Stoker"),
    ("Frankenstein", "Mary Wollstonecraft Shelley"),
    ("Grimm's Fairy Tales", "Jacob Grimm, Wilhelm Grimm"),
    ("Jane Eyre", "Charlotte Brontë"),
    ("Little Women", "Louisa May Alcott"),
    ("Middlemarch", "George Eliot"),
    ("Pride and Prejudice", "Jane Austen"),
    ("Romeo and Juliet", "William Shakespeare"),
    ("The Brothers Karamazov", "Fyodor Dostoyevsky"),
    ("The Complete Works of William Shakespeare", "William Shakespeare"),
    ("The Count of Monte Cristo", "Alexandre Dumas"),
    ("The Great Gatsby", "F. Scott (Francis Scott) Fitzgerald"),
    ("The King in Yellow", "Robert W. (Robert William) Chambers"),
    ("The Picture of Dorian Gray", "Oscar Wilde"),
    ("The Strange Case of Dr. Jekyll and Mr. Hyde", "Robert Louis Stevenson"),
    ("Wuthering Heights", "Emily Brontë"),
]

SECTION_PATTERNS = [
    # Standard book sections
    r'^\s*PROLOGUE\b', r'^\s*EPILOGUE\b', r'^\s*PREFACE\b', r'^\s*FOREWORD\b', r'^\s*INTRODUCTION\b',
    r"^\s*AUTHOR['\u2019]?S NOTE\b", r'^\s*AUTHOR NOTE\b', r'^\s*ACKNOWLEDG(E)?MENTS?\b', r'^\s*ACKNOWLEDGMENT\b',
    r'^\s*CONTENTS\b', r'^\s*INDEX\b', r'^\s*GLOSSARY\b', r'^\s*REFERENCES\b', r'^\s*BIBLIOGRAPHY\b',
    r'^\s*APPENDIX\b', r'^\s*THE END\b',
    # Chapters, Books, Parts
    r'^\s*CHAPTER\b', r'^\s*BOOK\b', r'^\s*PART\b',
    r'^\s*CHAPTER\s+[IVXLCDM]+\b', r'^\s*CHAPTER\s+\d+\b',
    r'^\s*PART\s+[IVXLCDM\d]+\b', r'^\s*BOOK\s+[IVXLCDM\d]+\b',
    r'^\s*BOOK\s+[IVXLCDM]+\b',
    # A Christmas Carol uses STAVE
    r'^\s*STAVE\s+[IVXLCDM\w]+\b', r'^\s*STAVE\s+ONE\b', r'^\s*STAVE\s+TWO\b', 
    r'^\s*STAVE\s+THREE\b', r'^\s*STAVE\s+FOUR\b', r'^\s*STAVE\s+FIVE\b',
    # Plays - Act/Scene structure
    r'^\s*ACT\s+[IVXLCDM]+\b', r'^\s*ACT\s+\d+\b',
    r'^\s*SCENE\s+[IVXLCDM]+\b', r'^\s*SCENE\s+\d+\b',
    # Anthologies - Fables, Tales, Stories
    r'^\s*FABLE\s+[IVXLCDM\d]+\b', r'^\s*TALE\s+[IVXLCDM\d]+\b', r'^\s*STORY\s+[IVXLCDM\d]+\b',
    # Complete Works / Series markers
    r'^\s*NOVEL\s*:', r'^\s*WORK\s*:', r'^\s*VOLUME\s+[IVXLCDM\d]+\b',
]
SECTION_REGEX = re.compile("|".join("(" + p + ")" for p in SECTION_PATTERNS), flags=re.IGNORECASE)
CHAPTER_FALLBACK_REGEX = re.compile(r'^\s*(CHAPTER|CH\.|BOOK|PART|STAVE|ACT|SCENE|VOLUME)\b.*', flags=re.IGNORECASE)

# Anthology title pattern - for Grimm's Fairy Tales, Aesop, etc.
# Matches short ALL-CAPS lines starting with "THE" that look like story titles
ANTHOLOGY_TITLE_REGEX = re.compile(r"^THE\s+[A-Z][A-Z\s\-,';]+$")
# Standard Gutenberg "Produced by" etc patterns to skip if they appear at start
JUNK_PATTERNS = [
    r'^Produced by',
    r'^End of the Project Gutenberg',
    r'^End of Project Gutenberg',
    r'^This file should be named',
    r'^Project Gutenberg',
]
# Exclude "CONTENTS" or "INDEX" if user wants *only* main content?
# The user said "creating segregation properly and section headings are proper and only main-content of books are there not useless junk data".
# We will keep CONTENTS for now as it is often part of the book structure, but we definitely want to kill the license.

SEP_LINE = "=" * 25

# -----------------------
# Helpers (unchanged except extraction)
# -----------------------
def find_best_epub_file(title: str, author: Optional[str], epub_dir: Path) -> Optional[Path]:
    files = list(epub_dir.glob("**/*.epub"))
    if not files:
        return None
    best = None
    best_score = 0.0
    query = f"{title} {author or ''}".strip().lower()
    for f in files:
        name = f.stem.lower()
        s = similarity(query, name)
        tokens = [t for t in re.split(r'\W+', title.lower()) if len(t) > 2]
        token_score = sum(1 for t in tokens if t in name) / max(1, len(tokens))
        score = max(s, token_score * 0.9)
        if score > best_score:
            best_score = score
            best = f
    if best_score < 0.25:
        logger.debug("No good match (best_score=%.3f) for '%s' in %s", best_score, title, epub_dir)
        return None
    logger.debug("Best match for '%s' -> %s (score=%.3f)", title, best, best_score)
    return best

# ---------  KEY CHANGE ----------
def html_to_text_blocks(html: str) -> List[Tuple[str,str]]:
    """
    Convert an HTML document to a list of (tag, text) blocks.
    Key change: include and specially handle <pre> and <code> elements (Project Gutenberg style).
    """
    soup = BeautifulSoup(html, "lxml")
    blocks: List[Tuple[str,str]] = []

    # 1) Handle <pre> specially: split by blank lines into paragraphs
    # This addresses Gutenberg-style EPUBs where content is inside a single <pre>.
    for pre in soup.find_all('pre'):
        text = pre.get_text()
        # split into paragraphs on blank lines
        parts = re.split(r'\n\s*\n', text)
        for p in parts:
            p = p.strip()
            if p:
                # collapse multiple whitespace into single space for consistency
                p2 = re.sub(r'[ \t\r\f\v]+', ' ', p)
                # keep line breaks inside paragraphs as spaces
                p2 = re.sub(r'\n+', ' ', p2).strip()
                blocks.append(('pre', p2))

    # 2) Also consider code blocks (rare) similarly
    for code in soup.find_all('code'):
        text = code.get_text()
        if text and len(text.strip()) > 0:
            p2 = re.sub(r'\s+', ' ', text).strip()
            blocks.append(('code', p2))

    # 3) Now gather normal headers/paragraphs/divs etc. but avoid duplicating <pre> content:
    # We'll select structural tags and add them in document order skipping pre/code already processed.
    for el in soup.find_all(['h1','h2','h3','h4','h5','h6','p','div','section','span']):
        # ignore elements that are within a pre (to avoid duplicate content)
        if el.find_parent('pre'):
            continue
        txt = el.get_text(separator=" ", strip=True)
        if not txt:
            continue
        # normalize whitespace
        txt2 = re.sub(r'\s+', ' ', txt).strip()
        if txt2:
            blocks.append((el.name, txt2))

    # 4) Fallback: if no blocks found, return entire stripped body text (split on blank lines)
    if not blocks:
        body = soup.body
        if body:
            body_text = body.get_text("\n", strip=True)
        else:
            body_text = soup.get_text("\n", strip=True)
        # split on blank lines
        parts = re.split(r'\n\s*\n', body_text)
        for p in parts:
            p2 = p.strip()
            if p2:
                p2 = re.sub(r'\s+', ' ', p2)
                blocks.append(('text', p2))
    return blocks
# ---------  END KEY CHANGE ----------

# Regex for scene break patterns - asterisks, dashes, dots used as separators
SCENE_BREAK_REGEX = re.compile(r'^[\s*\-_·•—–=~#]+$')

def is_section_heading(text: str) -> Optional[str]:
    """
    Conservative detection of section headings.

    Rules:
    - Reject scene break patterns like '* * * * *' or '-----'.
    - Reject lines that contain no alphanumeric characters.
    - Accept only if SECTION_REGEX matches (explicit structural keywords).
    - Or accept if CHAPTER_FALLBACK_REGEX matches.
    - Or accept if ANTHOLOGY_TITLE_REGEX matches (for fairy tales, fables, etc.)
    """
    t = (text or "").strip()
    if not t:
        return None

    # Reject scene break patterns first (e.g., "* * * * *", "-----", "* * *", "—")
    if SCENE_BREAK_REGEX.match(t):
        return None
    # Also reject if only asterisks/dashes/spaces after any transformation
    if re.match(r'^[*\-_·•—–=~#\s]+$', t):
        return None

    # Reject very short (likely noise)
    if len(t) < 3:
        return None

    # Remove leading decorative punctuation (like '======' or '***') for matching, but keep core text
    t_clean = re.sub(r'^[\W_]+', '', t).strip()

    # If after removing leading punctuation there's nothing meaningful, bail out
    if not t_clean:
        return None

    # Reject lines consisting only of punctuation/symbols (e.g., "*****", "----", "*** * * *")
    if re.sub(r'[A-Za-z0-9]', '', t_clean).strip() == t_clean.strip():
        return None

    # Strong signal: explicit SECTION patterns (PREFACE, INDEX, CHAPTER, APPENDIX, GLOSSARY, etc.)
    if SECTION_REGEX.search(t_clean):
        return t_clean.upper()

    # Strong fallback: lines that begin with CHAPTER / CH. / BOOK / PART / STAVE / ACT / SCENE etc.
    if CHAPTER_FALLBACK_REGEX.match(t_clean):
        return t_clean.upper()

    # Anthology detection: Short ALL-CAPS lines starting with "THE" (story titles in Grimm's, etc.)
    # Only match if line is reasonable length (<80) and ALL CAPS
    if len(t_clean) < 80 and t_clean.isupper() and ANTHOLOGY_TITLE_REGEX.match(t_clean):
        return t_clean

    # No other heuristics — return None (not a heading)
    return None

def is_audio_epub(epub_path: Path) -> bool:
    """
    Detect audio-only EPUBs (LibriVox, etc.) that contain no readable text.
    These files typically contain MP3 manifests or audio links instead of book content.
    
    Signs of audio EPUB:
    - Very small file size (<50KB)
    - Contains 'audio', 'mp3', 'librivox', 'ogg' keywords prominently
    - No substantial text content
    """
    # Check file size - audio manifests are typically very small
    file_size = epub_path.stat().st_size
    if file_size > 200000:  # >200KB is likely a real text EPUB
        return False
    
    try:
        book = epub.read_epub(str(epub_path))
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        
        # Check for audio keywords in content
        audio_keywords = ['librivox', '.mp3', '.ogg', 'audio', 'audiobook', 'recording', 'reader:']
        text_content = ""
        for item in items[:3]:  # Check first few items
            try:
                html = item.get_content().decode('utf-8', errors='ignore')
                text_content += html.lower()
            except:
                pass
        
        audio_score = sum(1 for kw in audio_keywords if kw in text_content)
        
        # If file is small AND has audio keywords, it's likely audio-only
        if file_size < 50000 and audio_score >= 2:
            return True
        
        return False
    except Exception as e:
        logger.debug(f"Error checking audio EPUB: {e}")
        return False

def gather_text_from_epub(epub_path: Path) -> List[str]:
    """
    Return a list of normalized text blocks extracted from EPUB.
    Additional safe fallback: if items produce very few blocks, attempt to extract body text and split on blank lines.
    """
    # Check if this is an audio-only EPUB
    if is_audio_epub(epub_path):
        logger.warning(f"AUDIO EPUB DETECTED: {epub_path.name} appears to be an audio-only EPUB (LibriVox/audiobook manifest). No text content will be extracted.")
        return [f"[AUDIO EPUB - No text content available. This file appears to be an audio manifest for {epub_path.stem}]"]
    
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

    # If extraction produced very few blocks, try a final body-text fallback (split by blank lines)
    if len(blocks) <= 3:
        logger.debug("Very few blocks extracted (%d). Applying body-text fallback for %s", len(blocks), epub_path.name)
        # try the first document's raw body text
        if items:
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
                    logger.debug("Body-text fallback produced %d blocks (was %d)", len(new_blocks), len(blocks))
                    blocks = new_blocks


    # Clean Project Gutenberg headers/footers
    cleaned_blocks = clean_gutenberg_content(blocks)
    return cleaned_blocks

def clean_gutenberg_content(blocks: List[str]) -> List[str]:
    """
    Remove Project Gutenberg header and footer content using markers and text heuristics.
    """
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
        "Start of the Project Gutenberg EBook",
        "Start of this Project Gutenberg EBook",
    ]
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of this Project Gutenberg EBook",
    ]

    # Find start and end indices
    start_idx = 0
    end_idx = len(blocks)

    # Search for start marker
    for i, block in enumerate(blocks):
        if any(m.lower() in block.lower() for m in start_markers):
            start_idx = i + 1
            break

    # Search for end marker after start_idx
    for i in range(start_idx, len(blocks)):
        block = blocks[i]
        if any(m.lower() in block.lower() for m in end_markers):
            end_idx = i
            break

    content_blocks = blocks[start_idx:end_idx]

    # Safety fallback: if empty, use original
    if not content_blocks and len(blocks) > 0:
        content_blocks = blocks

    # --- Secondary Filtering Pass (Heuristic) ---
    # Remove common metadata lines that often appear *after* the start marker or if marker was missed.
    # We will remove blocks from the START until we hit something that looks like real text.

    final_blocks = []

    junk_starts = [
        "Produced by",
        "Title:",
        "Author:",
        "Release Date:",
        "Language:",
        "Character set",
        "Encoding:",
        "Credits:",
        "E-text prepared by",
        "Project Gutenberg",
        "The Project Gutenberg",
        "*** START",
    ]

    junk_contains = [
        "Project Gutenberg License",
        "Distributed Proofreaders",
        "http://gutenberg.org",
        "www.gutenberg.org",
    ]

    # 1. Trim junk from the start
    content_start_offset = 0
    for i, block in enumerate(content_blocks):
        b_clean = block.strip()
        is_junk = False

        # Check start patterns
        for p in junk_starts:
            if b_clean.lower().startswith(p.lower()):
                is_junk = True
                break

        # Check contains patterns
        if not is_junk:
            for p in junk_contains:
                if p.lower() in b_clean.lower():
                    is_junk = True
                    break

        # Regex check for metadata with variable spacing (Title :, Author :, etc.)
        if not is_junk:
            if re.match(r'^(Title|Author|Release Date|Language|Credits|Character set|Encoding|Produced by|Copyright|Note|Audio performance)\s*[:]', b_clean, re.IGNORECASE):
                is_junk = True

        if is_junk:
            content_start_offset = i + 1
        else:
            # If we hit a block that is NOT junk, we might still be in the header zone (e.g. Title line followed by Metadata).
            # But if we blindly skip everything until "Chapter 1", we lose the Title.
            # Instead of stopping here, we will filter this block out later if it matches "Strict Junk".
            # But "Trim" implies contiguous removal.
            # Let's STOP trimming here, but apply a STRICT filter to the remaining blocks.
            break

    # 2. Trim junk from the end (license remainder)
    content_end_offset = len(content_blocks)
    for i in range(len(content_blocks) - 1, -1, -1):
        block = content_blocks[i]
        b_clean = block.strip()
        is_junk = False
        # Check start patterns
        for p in junk_starts:
            if b_clean.lower().startswith(p.lower()):
                is_junk = True
                break
        if not is_junk:
            for p in junk_contains:
                if p.lower() in b_clean.lower():
                    is_junk = True
                    break
        if not is_junk:
            if re.match(r'^(Title|Author|Release Date|Language|Credits|Character set|Encoding|Produced by|Copyright|Note)\s*[:]', b_clean, re.IGNORECASE):
                is_junk = True

        if is_junk:
            content_end_offset = i
        else:
            break

    final_subset = content_blocks[content_start_offset:content_end_offset]

    # 3. Filter any REMAINING blocks that are explicitly strictly junk (Metadata lines that survived trim)
    really_final_blocks = []

    # Expanded Strict Junk Patterns
    strict_junk_starts = junk_starts + [
        "Copyright",
        "Audio performance",
        "This is an audio eBook",
        "It is available as a series",
    ]

    for block in final_subset:
        b_clean = block.strip()
        is_junk = False

        # Regex check (Applied to ALL blocks now)
        if re.match(r'^(Title|Author|Release Date|Language|Credits|Character set|Encoding|Produced by|Copyright|Note|Audio performance)\s*[:]', b_clean, re.IGNORECASE):
            is_junk = True

        if not is_junk:
            for p in strict_junk_starts:
                if b_clean.lower().startswith(p.lower()):
                    is_junk = True
                    break

        if not is_junk:
            # Special check for .mp3 lists
            if ".mp3" in b_clean.lower() and re.search(r'\d+-\d+\.mp3', b_clean):
                is_junk = True

        # Special check for strict Gutenberg/License in middle
        if not is_junk:
            if "project gutenberg" in b_clean.lower() and ("license" in b_clean.lower() or "ebook" in b_clean.lower()):
                is_junk = True

        if not is_junk:
            really_final_blocks.append(block)

    return really_final_blocks

# Section type classification
FRONT_MATTER_PATTERNS = re.compile(r'^(PREFACE|FOREWORD|INTRODUCTION|DEDICATION|AUTHOR\'?S?\s*NOTE|ACKNOWLEDGMENTS?|CONTENTS)\b', re.I)
BACK_MATTER_PATTERNS = re.compile(r'^(APPENDIX|GLOSSARY|INDEX|BIBLIOGRAPHY|NOTES|REFERENCES|AFTERWORD)\b', re.I)
CHAPTER_TITLE_PATTERN = re.compile(r'^(CHAPTER|BOOK|PART)\s+([IVXLCDM\d]+)\b', re.I)

def get_section_type(title: str) -> str:
    """Classify section as FRONT_MATTER, CHAPTER, or BACK_MATTER."""
    if FRONT_MATTER_PATTERNS.search(title):
        return "FRONT_MATTER"
    if BACK_MATTER_PATTERNS.search(title):
        return "BACK_MATTER"
    return "CHAPTER"

def split_into_sections(blocks: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Split normalized text blocks into sections using is_section_heading().
    Enhanced to capture chapter titles from following lines (e.g., "CHAPTER I" + "MR. SHERLOCK HOLMES").
    """
    sections: List[Tuple[str, List[str]]] = []
    current_title = None
    current_pars: List[str] = []
    skip_next = False

    def flush():
        nonlocal current_title, current_pars
        if current_title is None and current_pars:
            # Rename MAIN_TEXT to OPENING_CREDITS for content before first chapter
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
            # Check if this is a chapter/book/part heading that might have a title on next line
            if CHAPTER_TITLE_PATTERN.match(heading) and idx + 1 < len(blocks):
                potential_title = blocks[idx + 1].strip()
                # Title should be short (< 100 chars), mostly uppercase or title case, and not a paragraph
                if (len(potential_title) < 100 and 
                    not is_section_heading(potential_title) and
                    (potential_title.isupper() or potential_title.istitle() or 
                     len(potential_title.split()) <= 10)):
                    # Check it's not starting a paragraph (no period at end, not too long)
                    if not potential_title.endswith('.') or len(potential_title) < 60:
                        heading = f"{heading}: {potential_title.upper()}"
                        skip_next = True
            
            current_title = re.sub(r'\s+', ' ', heading).strip()
            continue
        # otherwise, keep appending to current paragraph list
        current_pars.append(block)

    flush()
    return sections

def pretty_section_title(title: str) -> str:
    t = title.strip()
    t = re.sub(r'[_\-\s]{2,}', ' ', t)
    if not t:
        return "UNTITLED SECTION"
    return t

def write_output_file(out_dir: Path, book_title: str, author: Optional[str], sections: List[Tuple[str,List[str]]]) -> Path:
    """Write sections using Audible-style format: ===SECTION TITLE==="""
    safe_name = re.sub(r'[\\/:"*?<>|]+', '', book_title).strip()
    filename = f"{safe_name}__Cleaned_segregated_final.txt"
    outpath = out_dir / filename
    with open(outpath, "w", encoding="utf-8") as fh:
        # Skip OPENING_CREDITS if it only contains title/author info (< 3 paragraphs)
        filtered_sections = []
        for sec_title, paras in sections:
            # Skip empty sections
            if not paras:
                continue
            # Optionally skip very short opening credits (just title/author)
            if sec_title == "OPENING_CREDITS" and len(paras) <= 3:
                # Check if it's just book metadata
                combined = " ".join(paras).lower()
                if book_title.lower() in combined or (author and author.lower() in combined):
                    continue  # Skip this, it's just title/author info
            filtered_sections.append((sec_title, paras))
        
        # Write sections in Audible-style format
        for sec_title, paras in filtered_sections:
            title_line = pretty_section_title(sec_title)
            fh.write(f"==={title_line}===\n\n")
            fh.write("\n\n".join(paras).strip() + "\n\n")
    return outpath

def load_csv_mapping(csv_path: Path) -> List[Tuple[str,str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            parts = [p.strip() for p in ln.split(",",1)]
            if len(parts) == 1:
                rows.append((parts[0], ""))
            else:
                rows.append((parts[0], parts[1]))
    return rows

def process_book(title: str, author: Optional[str], epub_dir: Path, out_dir: Path) -> Dict:
    logger.info("Processing book: %s — %s", title, author or "")
    result = {"title": title, "author": author, "found_epub": None, "output_file": None, "sections": [], "warnings": []}
    epub_file = find_best_epub_file(title, author, epub_dir)
    if epub_file is None:
        warning = f"No EPUB match found for '{title}' in {epub_dir}"
        logger.warning(warning)
        result["warnings"].append(warning)
        return result
    result["found_epub"] = str(epub_file)
    try:
        blocks = gather_text_from_epub(epub_file)
    except Exception as e:
        logger.exception("Failed to parse EPUB %s: %s", epub_file, e)
        result["warnings"].append(f"EPUB parse error: {e}")
        return result
    if not blocks:
        warning = "No text blocks extracted"
        result["warnings"].append(warning)
        logger.warning(warning)
        return result
    sections = split_into_sections(blocks)
    if len(sections) <= 1:
        # fallback as before: try splitting by CHAPTER tokens inside the joined text
        fallback_sections = []
        joined = "\n\n".join(blocks)
        parts = re.split(r'(?m)^\s*(CHAPTER\s+[IVXLCDM\d]+|CHAPTER\b|Chapter\s+\d+|CHAPTER\s+\w+)\b.*', joined, flags=re.IGNORECASE)
        if len(parts) > 3:
            for p in parts:
                p2 = p.strip()
                if p2:
                    fallback_sections.append(("CHAPTER_FALLBACK", [p2]))
            sections = fallback_sections
            logger.info("Fallback chapter-splitting produced %d sections", len(sections))
        else:
            logger.info("Fallback splitting not applied; keeping single-section output")
    final_sections = []
    for sec_title, paras in sections:
        st = sec_title or ""
        if st in ("START","UNKNOWN", None):
            if paras and len(paras) > 0:
                guess = paras[0].strip()
                if len(guess.split()) <= 8 and guess == guess.upper():
                    st = guess.upper()
                else:
                    st = "MAIN_TEXT"
        final_sections.append((st, paras))
    out_path = write_output_file(out_dir, title, author, final_sections)
    result["output_file"] = str(out_path)
    result["sections"] = [{"title": s[0], "paragraphs": len(s[1])} for s in final_sections]
    logger.info("Wrote cleaned segregated file: %s", out_path)
    return result

def main():
    p = argparse.ArgumentParser(description="Clean and segregate EPUBs into sectioned plain text files.")
    p.add_argument("--input-dir", "-i", required=True, help="Directory containing EPUB files")
    p.add_argument("--output-dir", "-o", required=False, default="./PROCESSED/1_clean", help="Directory to write cleaned files")
    p.add_argument("--csv", "-c", required=False, help="Optional CSV file with Title,Author list (one per line)")
    args = p.parse_args()

    epub_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        catalog = load_csv_mapping(Path(args.csv))
    else:
        catalog = DEFAULT_CATALOG

    results = []
    for title, author in tqdm(catalog, desc="Books"):
        try:
            res = process_book(title, author, epub_dir, out_dir)
            results.append(res)
        except Exception as e:
            logger.exception("Unhandled error processing %s: %s", title, e)
            results.append({"title": title, "author": author, "error": str(e)})

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
        "input_dir": str(epub_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "results": results
    }
    with open(out_dir / "processing_report.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    logger.info("Processing complete. Summary written to %s", out_dir / "processing_report.json")

if __name__ == "__main__":
    main()
