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
from datetime import datetime
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
    r'^\s*PROLOGUE\b', r'^\s*EPILOGUE\b', r'^\s*PREFACE\b', r'^\s*FOREWORD\b', r'^\s*INTRODUCTION\b',
    r'^\s*AUTHOR[’\']?S NOTE\b', r'^\s*AUTHOR NOTE\b', r'^\s*ACKNOWLEDG(E)?MENTS?\b', r'^\s*ACKNOWLEDGMENT\b',
    r'^\s*CONTENTS\b', r'^\s*INDEX\b', r'^\s*GLOSSARY\b', r'^\s*REFERENCES\b', r'^\s*BIBLIOGRAPHY\b',
    r'^\s*APPENDIX\b', r'^\s*CHAPTER\b', r'^\s*BOOK\b', r'^\s*PART\b',
    r'^\s*CHAPTER\s+[IVXLCDM]+\b', r'^\s*CHAPTER\s+\d+\b',
    r'^\s*PART\s+[IVXLCDM\d]+\b', r'^\s*BOOK\s+[IVXLCDM\d]+\b',
    r'^\s*BOOK\s+[IVXLCDM]+\b', r'^\s*THE END\b'
]
SECTION_REGEX = re.compile("|".join("(" + p + ")" for p in SECTION_PATTERNS), flags=re.IGNORECASE)
CHAPTER_FALLBACK_REGEX = re.compile(r'^\s*(CHAPTER|CH\.|BOOK|PART)\b.*', flags=re.IGNORECASE)
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

def is_section_heading(text: str) -> Optional[str]:
    t = text.strip()
    if not t:
        return None
    t_clean = re.sub(r'^[\W_]+', '', t)
    m = SECTION_REGEX.search(t_clean)
    if m:
        return t_clean.upper()
    if len(t_clean) < 120 and (t_clean.upper() == t_clean and len(t_clean.split()) <= 8):
        return t_clean.upper()
    if CHAPTER_FALLBACK_REGEX.match(t_clean):
        return t_clean.upper()
    return None

def gather_text_from_epub(epub_path: Path) -> List[str]:
    """
    Return a list of normalized text blocks extracted from EPUB.
    Additional safe fallback: if items produce very few blocks, attempt to extract body text and split on blank lines.
    """
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

    return blocks

def split_into_sections(blocks: List[str]) -> List[Tuple[str, List[str]]]:
    sections: List[Tuple[str, List[str]]] = []
    current_title = None
    current_pars: List[str] = []

    def flush():
        nonlocal current_title, current_pars
        if current_title is None and current_pars:
            current_title = "START"
        if current_pars:
            sections.append((current_title or "UNKNOWN", current_pars.copy()))
        current_title = None
        current_pars = []

    for block in blocks:
        heading = is_section_heading(block)
        if heading:
            flush()
            heading_clean = re.sub(r'\s+', ' ', heading).strip()
            current_title = heading_clean
            continue
        if len(block.split()) <= 8 and block == block.upper() and len(block) > 0:
            flush()
            current_title = block.upper()
            continue
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
    safe_name = re.sub(r'[\\/:"*?<>|]+', '', book_title).strip()
    filename = f"{safe_name}__Cleaned_segregated_final.txt"
    outpath = out_dir / filename
    with open(outpath, "w", encoding="utf-8") as fh:
        fh.write(f"=== {book_title} — {author or ''} ===\n")
        fh.write(f"Processed at: {datetime.utcnow().isoformat()}Z\n\n")
        for sec_title, paras in sections:
            title_line = pretty_section_title(sec_title)
            fh.write(SEP_LINE + "\n")
            fh.write(f"====== {title_line} ======\n")
            fh.write(SEP_LINE + "\n")
            fh.write("\n".join(paras).strip() + "\n\n")
            fh.write(SEP_LINE + "\n\n")
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
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_dir": str(epub_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "results": results
    }
    with open(out_dir / "processing_report.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    logger.info("Processing complete. Summary written to %s", out_dir / "processing_report.json")

if __name__ == "__main__":
    main()

