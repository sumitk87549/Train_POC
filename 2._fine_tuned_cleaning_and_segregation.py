#!/usr/bin/env python3
"""
Enhanced EPUB Cleaning and Segregation Script

This script provides advanced cleaning and segregation of EPUB files into structured text format.
It builds upon the existing implementation with improved section detection, better text cleaning,
and more robust handling of diverse EPUB formats.

Key Features:
- Advanced EPUB extraction with error handling
- Enhanced section detection using regex patterns and context analysis
- Comprehensive text cleaning for common EPUB artifacts
- Structured output format matching the target specification
- Parallel processing for improved performance
- Detailed logging and progress tracking

Usage:
    python 2._fine_tuned_cleaning_and_segregation.py --input-dir ./book/EPUB --output-dir ./PROCESSED
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import re
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

# EPUB / HTML parsing
try:
    from ebooklib import epub
    EBOOKLIB_AVAILABLE = True
except ImportError:
    EBOOKLIB_AVAILABLE = False
    print("Warning: ebooklib not available, falling back to zipfile extraction")

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("Warning: BeautifulSoup not available, HTML parsing will be limited")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available, progress bars will be disabled")

# Optional fast ratio
try:
    import Levenshtein
    def similarity(a, b) -> float:
        return Levenshtein.ratio((a or "").lower(), (b or "").lower())
except Exception:
    def similarity(a, b) -> float:
        return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

# -----------------------
# Configuration
# -----------------------
LOG_FILE = "enhanced_cleaning_segregation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("enhanced_cleanseg")

# Enhanced section heading regex patterns (order matters: more specific first)
SECTION_PATTERNS = [
    # Standard section types
    r'^\s*PROLOGUE\b', r'^\s*EPILOGUE\b', r'^\s*PREFACE\b', r'^\s*FOREWORD\b',
    r'^\s*INTRODUCTION\b', r'^\s*AFTERWORD\b', r'^\s*CONCLUSION\b',
    r'^\s*AUTHOR[’\']?S NOTE\b', r'^\s*AUTHOR NOTE\b', r'^\s*TRANSLATOR[’\']?S NOTE\b',
    r'^\s*ACKNOWLEDG(E)?MENTS?\b', r'^\s*ACKNOWLEDGMENT\b',
    r'^\s*CONTENTS\b', r'^\s*TABLE OF CONTENTS\b', r'^\s*INDEX\b', r'^\s*GLOSSARY\b',
    r'^\s*REFERENCES\b', r'^\s*BIBLIOGRAPHY\b', r'^\s*NOTES\b', r'^\s*ENDNOTES\b',
    r'^\s*APPENDIX\b', r'^\s*APPENDICES\b',
    # Chapter patterns - specific to avoid false positives
    r'^\s*CHAPTER\b', r'^\s*CH\.\b', r'^\s*BOOK\b', r'^\s*PART\b', r'^\s*SECTION\b',
    r'^\s*VOLUME\b', r'^\s*ACT\b', r'^\s*SCENE\b', r'^\s*STAVE\b',
    # Roman numerals chapters e.g. "CHAPTER I", "Chapter IV", and variants with numbers
    r'^\s*CHAPTER\s+[IVXLCDM]+\b', r'^\s*CHAPTER\s+\d+\b',
    r'^\s*PART\s+[IVXLCDM\d]+\b', r'^\s*BOOK\s+[IVXLCDM\d]+\b',
    r'^\s*SECTION\s+[IVXLCDM\d]+\b', r'^\s*ACT\s+[IVXLCDM\d]+\b',
    r'^\s*STAVE\s+[A-Z]+\b', r'^\s*STAVE\s+\d+\b',
    # Some Gutenberg style headings like "CHAPTER I. — TITLE"
    r'^\s*CHAPTER\s+[IVXLCDM]+\.\s*—.*', r'^\s*CHAPTER\s+\d+\.\s*—.*',
    r'^\s*THE END\b', r'^\s*END\b',
    # Additional common patterns
    r'^\s*DEDICATION\b', r'^\s*ABOUT THE AUTHOR\b', r'^\s*ABOUT THE BOOK\b',
    r'^\s*COPYRIGHT\b', r'^\s*PUBLISHER[’\']?S NOTE\b'
]
SECTION_REGEX = re.compile("|".join("(" + p + ")" for p in SECTION_PATTERNS), flags=re.IGNORECASE)

# Generic chapter heading detection (fallback)
CHAPTER_FALLBACK_REGEX = re.compile(
    r'^\s*(CHAPTER|CH\.?|BOOK|PART|SECTION|VOLUME|ACT|SCENE|STAVE)\b.*',
    flags=re.IGNORECASE
)

# Visual separator used in output
SEP_LINE = "=" * 25

# Common EPUB artifacts to remove
EPUB_ARTIFACTS = [
    r'\[Illustration.*?\]',  # Illustration placeholders
    r'\[Image.*?\]',         # Image placeholders
    r'\[.*?\]',              # Other bracket placeholders
    r'\*\*\*START.*?\*\*\*', # Project Gutenberg headers
    r'\*\*\*END.*?\*\*\*',   # Project Gutenberg footers
    r'This ebook is for the use of anyone anywhere.*?END OF THE PROJECT GUTENBERG',
    r'Project Gutenberg.*?Electronic Edition',
    r'Produced by.*?distributed proofreaders',
    r'Transcriber[’\']?s Note:.*?\n',
    r'Copyright.*?\n',
    r'All rights reserved.*?\n',
    r'ISBN.*?\n',
    r'http[s]?://\S+',      # URLs
    r'www\.\S+',            # Web addresses
    r'[^\S\n]{2,}',         # Excessive whitespace
    r'\n{3,}',              # Multiple empty lines
    r'[ \t]+$',             # Trailing whitespace
    r'^\s*$',               # Empty lines
    # Enhanced Project Gutenberg patterns
    r'The Project Gutenberg eBook of .*?\n.*?\n',
    r'Updated editions will replace the previous one.*?\n',
    r'Creating the works from print editions.*?\n',
    r'Various characteristics of each ebook.*?\n',
    r'Click on any of the filenumbers below.*?\n',
    r'THE FULL PROJECT GUTENBERG LICENSE.*?\n',
    r'Section \d+\..*?\n',
    r'1\.[A-E]\..*?\n',
    r'2\..*?\n',
    r'3\..*?\n',
    r'4\..*?\n',
    r'5\..*?\n',
    r'END OF THE PROJECT GUTENBERG.*?\n',
    r'End of the Project Gutenberg.*?\n',
    r'End of Project Gutenberg.*?\n',
    # More aggressive license removal
    r'be renamed\. States without permission.*?electronic works in your possession\.',
    r'Start: full license please read this.*?\n',
    r'Full project gutenberg™ license.*?\n',
    r'General terms of use.*?\n',
    r'Trademark license.*?\n',
    r'Electronic work.*?\n',
    r'Project gutenberg™.*?\n',
    r' redistribute.*?\n',
    r'royalties.*?\n',
    r' trademark.*?\n',
    r'license.*?\n',
    r'agreement.*?\n',
    r'permission.*?\n',
    r'copying.*?\n',
    r'distributing.*?\n',
]

# Common text cleaning rules
TEXT_CLEANING_RULES = [
    (r'[“”]', '"'),         # Smart quotes to straight quotes
    (r'[‘’]', "'"),         # Smart apostrophes to straight
    (r'[–—]', '-'),         # Em dashes and en dashes to hyphen
    (r'[…]', '...'),        # Ellipsis normalization
    # (r'\s+', ' '),          # Multiple spaces to single space - REMOVED: this merges lines!
    (r'(\w) - (\w)', r'\1-\2'),  # Fix hyphenated words split by line breaks
]

# -----------------------
# Helpers
# -----------------------
def find_best_epub_file(title: str, author: Optional[str], epub_dir: Path) -> Optional[Path]:
    """Find the best-matching .epub file in epub_dir for the requested title/author using filename similarity."""
    files = list(epub_dir.glob("**/*.epub"))
    if not files:
        return None

    best = None
    best_score = 0.0
    query = f"{title} {author or ''}".strip().lower()

    for f in files:
        name = f.stem.lower()
        s = similarity(query, name)
        # also check if title tokens appear in filename
        tokens = [t for t in re.split(r'\W+', title.lower()) if len(t) > 2]
        token_score = sum(1 for t in tokens if t in name) / max(1, len(tokens))
        score = max(s, token_score * 0.9)

        if score > best_score:
            best_score = score
            best = f

    # require a minimum match to avoid false positives
    if best_score < 0.25:
        logger.debug("No good match (best_score=%.3f) for '%s' in %s", best_score, title, epub_dir)
        return None

    logger.debug("Best match for '%s' -> %s (score=%.3f)", title, best, best_score)
    return best

def extract_text_from_epub(epub_path: Path) -> str:
    """
    Extract text content from EPUB file using multiple methods.
    First try ebooklib, then fallback to zipfile extraction.
    """
    try:
        if EBOOKLIB_AVAILABLE:
            book = epub.read_epub(str(epub_path))
            text_content = []

            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                try:
                    content = item.get_content().decode('utf-8', errors='ignore')
                except Exception:
                    content = item.get_content().decode('latin-1', errors='ignore')

                if BEAUTIFULSOUP_AVAILABLE:
                    soup = BeautifulSoup(content, "lxml")
                    text = soup.get_text(separator="\n", strip=True)
                else:
                    # Simple HTML tag removal
                    text = re.sub(r'<[^>]+>', ' ', content)

                text_content.append(text)

            return "\n\n".join(text_content)

    except Exception as e:
        logger.warning("ebooklib extraction failed: %s, trying zipfile method", e)

    # Fallback: direct ZIP extraction
    try:
        with zipfile.ZipFile(epub_path, 'r') as zip_ref:
            text_content = []
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith(('.html', '.htm', '.xhtml', '.xml')):
                    try:
                        with zip_ref.open(file_info) as f:
                            content = f.read().decode('utf-8', errors='ignore')
                            if BEAUTIFULSOUP_AVAILABLE:
                                soup = BeautifulSoup(content, "lxml")
                                text = soup.get_text(separator="\n", strip=True)
                            else:
                                text = re.sub(r'<[^>]+>', ' ', content.decode('utf-8', errors='ignore'))
                            text_content.append(text)
                    except Exception as inner_e:
                        logger.debug("Failed to process %s: %s", file_info.filename, inner_e)
                        continue

            return "\n\n".join(text_content)

    except Exception as e:
        logger.error("Failed to extract text from EPUB %s: %s", epub_path, e)
        return ""

def clean_text_content(text: str) -> str:
    """
    Clean extracted text by removing EPUB artifacts and normalizing formatting.
    """
    if not text:
        return ""

    # First, remove large blocks of license/legalese text
    # Look for Project Gutenberg license blocks
    gutenberg_start = text.find('START: FULL LICENSE')
    if gutenberg_start != -1:
        gutenberg_end = text.find('END OF THE PROJECT GUTENBERG', gutenberg_start)
        if gutenberg_end != -1:
            text = text[:gutenberg_start] + text[gutenberg_end + len('END OF THE PROJECT GUTENBERG'):]
    
    # Remove other common license blocks
    license_blocks = [
        ('This ebook is for the use of anyone anywhere', 'END OF THE PROJECT GUTENBERG'),
        ('The Project Gutenberg eBook of', 'END OF THE PROJECT GUTENBERG'),
        ('be renamed. States without permission', 'electronic works in your possession.'),
        ('Updated editions will replace the previous one', 'Foundation.'),
        ('Creating the works from print editions', 'Foundation.'),
        ('Please check the Project Gutenberg web pages', 'volunteer support.'),
        ('Royalty payments must be paid', 'generations to come.'),
        ('1.F.', 'generations to come.'),
    ]
    
    for start_marker, end_marker in license_blocks:
        start_idx = text.find(start_marker)
        if start_idx != -1:
            end_idx = text.find(end_marker, start_idx)
            if end_idx != -1:
                text = text[:start_idx] + text[end_idx + len(end_marker):]

    # Remove common EPUB artifacts
    for pattern in EPUB_ARTIFACTS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    # Apply text cleaning rules
    for pattern, replacement in TEXT_CLEANING_RULES:
        text = re.sub(pattern, replacement, text)

    # Additional cleaning - be more careful with paragraph structure
    text = text.strip()
    # Don't normalize paragraph spacing too aggressively - keep separate paragraphs
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Reduce multiple empty lines to double
    text = re.sub(r'[ \t]+\n', '\n', text)   # Remove trailing spaces before newlines
    text = re.sub(r'\n[ \t]+', '\n', text)   # Remove leading spaces after newlines
    
    return text

def is_section_heading(text: str) -> Optional[str]:
    """
    If the text looks like a major section heading, return canonicalized heading; else None.
    """
    if not text or not text.strip():
        return None

    t = text.strip()

    # Remove surrounding punctuation and normalize whitespace
    t_clean = re.sub(r'^[^\w]+', '', t)
    t_clean = re.sub(r'\s+', ' ', t_clean).strip()

    # Direct match with enhanced patterns
    m = SECTION_REGEX.search(t_clean)
    if m:
        return t_clean.upper()

    # More lenient chapter detection - look for "CHAPTER" anywhere in the line
    if re.search(r'\bCHAPTER\b', t_clean, re.IGNORECASE):
        return t_clean.upper()
    
    # Look for chapter numbers like "CHAPTER 1", "CHAPTER I", etc.
    if re.search(r'\bCHAPTER\s+[\dIVXLCDM]+\b', t_clean, re.IGNORECASE):
        return t_clean.upper()
    
    # Look for standalone chapter numbers
    if re.match(r'^\s*[\dIVXLCDM]+\.(.*)', t_clean):
        return t_clean.upper()

    # Fallback for lines that are all caps and short (likely headings)
    if (len(t_clean) < 120 and
        t_clean.upper() == t_clean and
        len(t_clean.split()) <= 8 and
        len(t_clean) > 2):
        return t_clean.upper()

    # Fallback pattern with chapter-like structures
    if CHAPTER_FALLBACK_REGEX.match(t_clean):
        return t_clean.upper()

    return None

def split_into_sections(text: str) -> List[Tuple[str, List[str]]]:
    """
    Split cleaned text into sections based on detected headings.
    Returns list of (section_title, paragraphs) tuples.
    """
    if not text:
        return []

    # Split text into lines first, then group into paragraphs
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Group consecutive lines into paragraphs, but also check for headings within lines
    paragraphs = []
    current_para = []
    
    for line in lines:
        # Check if this line is a heading first
        heading = is_section_heading(line)
        if heading:
            # Flush any current paragraph
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
            # Add heading as its own paragraph
            paragraphs.append(line)
            current_para = []
        elif not current_para:
            current_para.append(line)
        elif len(current_para[-1]) < 200 and line and not line.endswith(('.', '!', '?', ':', ';')):
            # Likely continuation of a short line
            current_para.append(' ' + line)
        else:
            # New paragraph
            paragraphs.append(' '.join(current_para))
            current_para = [line]
    
    if current_para:
        paragraphs.append(' '.join(current_para))

    sections: List[Tuple[str, List[str]]] = []
    current_title = None
    current_pars: List[str] = []

    def flush_section():
        nonlocal current_title, current_pars
        if current_pars:
            if current_title is None:
                current_title = "MAIN_CONTENT"
            sections.append((current_title, current_pars.copy()))
        current_title = None
        current_pars = []

    for para in paragraphs:
        heading = is_section_heading(para)
        if heading:
            # New section starts here
            flush_section()
            current_title = heading
            continue

        # Heuristic: if paragraph is very short (<6 words) in all caps, treat as heading
        if (len(para.split()) <= 6 and
            para == para.upper() and
            len(para) > 2 and
            len(para) < 60):
            flush_section()
            current_title = para.upper()
            continue

        # Add to current section
        current_pars.append(para)

    # Final flush
    flush_section()

    return sections

def enhance_section_detection(sections: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    """
    Apply additional heuristics to improve section detection and merging.
    """
    if not sections:
        return []

    enhanced_sections = []
    i = 0

    while i < len(sections):
        title, paras = sections[i]

        # Merge consecutive sections with similar content (likely split by mistake)
        if (i < len(sections) - 1 and
            title == sections[i+1][0] and
            len(paras) < 3 and len(sections[i+1][1]) < 3):
            # Merge small consecutive sections with same title
            merged_paras = paras + sections[i+1][1]
            enhanced_sections.append((title, merged_paras))
            i += 2
            continue

        # Improve generic section titles
        if title in ("START", "UNKNOWN", "MAIN_CONTENT") and paras:
            first_para = paras[0].strip()
            # Check if first paragraph looks like a heading
            if (len(first_para.split()) <= 8 and
                first_para == first_para.upper() and
                len(first_para) > 3):
                enhanced_sections.append((first_para.upper(), paras[1:]))
            else:
                enhanced_sections.append((title, paras))
        else:
            enhanced_sections.append((title, paras))

        i += 1

    return enhanced_sections

def write_output_file(out_dir: Path, book_title: str, author: Optional[str],
                     sections: List[Tuple[str, List[str]]]) -> Path:
    """
    Write cleaned and segregated text to output file in the target format.
    """
    safe_name = re.sub(r'[\\/:"*?<>|]+', '', book_title).strip()
    filename = f"{safe_name}__Cleaned_segregated_final.txt"
    outpath = out_dir / filename

    with open(outpath, "w", encoding="utf-8") as fh:
        # Write header
        fh.write(f"=== {book_title}")
        if author:
            fh.write(f" — {author}")
        fh.write(" ===\n")
        fh.write(f"Processed at: {datetime.utcnow().isoformat()}Z\n\n")

        # Write sections
        for sec_title, paras in sections:
            # Clean section title for output
            title_line = sec_title.strip().upper()
            title_line = re.sub(r'[_\-\s]{2,}', ' ', title_line)

            # Write section header
            fh.write(SEP_LINE + "\n")
            fh.write(f"====== {title_line} ======\n")
            fh.write(SEP_LINE + "\n")

            # Write section content
            if paras:
                fh.write("\n".join(paras).strip() + "\n")

            fh.write(SEP_LINE + "\n\n")

    return outpath

def process_single_book(epub_path: Path, out_dir: Path) -> Dict:
    """
    Process a single EPUB file and return processing results.
    """
    result = {
        "epub_file": str(epub_path),
        "output_file": None,
        "sections": [],
        "warnings": [],
        "error": None
    }

    try:
        # Extract book metadata from filename
        filename = epub_path.stem
        # Remove common patterns to get clean title
        title = re.sub(r' - [A-Za-z, ]+$', '', filename)
        title = re.sub(r'[\(\[].*?[\)\]]', '', title).strip()

        # Try to extract author from filename pattern "Title - Author"
        author_match = re.search(r' - (.+)$', filename)
        author = author_match.group(1) if author_match else None

        logger.info("Processing: %s (Title: %s, Author: %s)", epub_path.name, title, author)

        # Extract text from EPUB
        raw_text = extract_text_from_epub(epub_path)
        if not raw_text:
            warning = "No text extracted from EPUB"
            result["warnings"].append(warning)
            logger.warning(warning)
            return result

        # Clean the extracted text
        cleaned_text = clean_text_content(raw_text)
        if not cleaned_text:
            warning = "No text remaining after cleaning"
            result["warnings"].append(warning)
            logger.warning(warning)
            return result

        # Split into sections
        sections = split_into_sections(cleaned_text)
        if not sections:
            warning = "No sections detected"
            result["warnings"].append(warning)
            logger.warning(warning)
            return result

        # Enhance section detection
        enhanced_sections = enhance_section_detection(sections)

        # Write output file
        out_path = write_output_file(out_dir, title, author, enhanced_sections)
        result["output_file"] = str(out_path)
        result["sections"] = [{"title": s[0], "paragraphs": len(s[1])} for s in enhanced_sections]

        logger.info("Successfully processed: %s -> %s sections", epub_path.name, len(enhanced_sections))

    except Exception as e:
        error_msg = f"Error processing {epub_path}: {str(e)}"
        result["error"] = error_msg
        logger.error(error_msg)
        logger.exception("Detailed error processing %s", epub_path)

    return result

def process_all_epubs(input_dir: Path, output_dir: Path, max_workers: int = 4) -> List[Dict]:
    """
    Process all EPUB files in the input directory using parallel processing.
    """
    epub_files = list(input_dir.glob("**/*.epub"))
    if not epub_files:
        logger.warning("No EPUB files found in %s", input_dir)
        return []

    logger.info("Found %d EPUB files to process", len(epub_files))

    results = []
    processed_count = 0
    success_count = 0

    # Use progress bar if available
    if TQDM_AVAILABLE:
        progress_bar = tqdm(epub_files, desc="Processing EPUBs", unit="book")
    else:
        progress_bar = epub_files

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_epub = {
            executor.submit(process_single_book, epub_file, output_dir): epub_file
            for epub_file in progress_bar
        }

        # Process completed tasks
        for future in as_completed(future_to_epub):
            epub_file = future_to_epub[future]
            try:
                result = future.result()
                results.append(result)
                processed_count += 1

                if result["output_file"]:
                    success_count += 1

                # Update progress description
                if TQDM_AVAILABLE:
                    progress_bar.set_postfix({
                        "processed": processed_count,
                        "success": success_count,
                        "current": epub_file.name
                    })

            except Exception as e:
                error_result = {
                    "epub_file": str(epub_file),
                    "error": f"Processing failed: {str(e)}",
                    "warnings": [],
                    "sections": []
                }
                results.append(error_result)
                logger.error("Exception in processing %s: %s", epub_file, e)

    if TQDM_AVAILABLE:
        progress_bar.close()

    logger.info("Processing complete: %d processed, %d successful", processed_count, success_count)
    return results

def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced EPUB cleaning and segregation tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing EPUB files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory to write cleaned files"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of worker threads for parallel processing"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Process all EPUB files
    start_time = datetime.now()
    results = process_all_epubs(input_dir, output_dir, args.workers)
    end_time = datetime.now()

    # Write processing report
    report = {
        "script_version": "2.0",
        "created_at": end_time.isoformat() + "Z",
        "processing_time_seconds": (end_time - start_time).total_seconds(),
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "total_files_processed": len(results),
        "successful_processing": sum(1 for r in results if r.get("output_file")),
        "files_with_errors": sum(1 for r in results if r.get("error")),
        "files_with_warnings": sum(1 for r in results if r.get("warnings")),
        "results": results
    }

    report_file = output_dir / "enhanced_processing_report.json"
    with open(report_file, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    logger.info("Processing report written to: %s", report_file)
    logger.info("Enhanced cleaning and segregation completed successfully!")

    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except Exception as e:
        logger.exception("Fatal error in main execution")
        exit(1)
