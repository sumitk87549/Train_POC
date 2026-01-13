#!/usr/bin/env python3
"""
auto_download_catalog.py

Auto-download a curated catalog of public-domain books (EPUB preferred, then PDF).
Saves EPUBs to ./book/EPUB/ and PDFs to ./book/PDF/.
Writes a provenance JSON for each downloaded file.
Writes titles that could not be downloaded (as EPUB/PDF) to REMAINING.txt.

Dependencies (recommended):
    pip install requests tqdm ebooklib

ebooklib is optional; if present the script will attempt to convert plain text -> EPUB.
"""

from __future__ import annotations
import requests
import json
import time
import re
import sys
import os
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import quote_plus, urljoin
from typing import Optional, Dict, Tuple
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# Optional import for text->epub conversion
try:
    from ebooklib import epub
    EPUBLIB_AVAILABLE = True
except Exception:
    EPUBLIB_AVAILABLE = False

# -------------------------
# Configuration
# -------------------------
USER_AGENT = "AudioCook-BookFetcher/1.0 (+https://example.com) Python-requests"
GUTENDEX_SEARCH = "https://gutendex.com/books"
ARCHIVE_ADV_SEARCH = "https://archive.org/advancedsearch.php"
ARCHIVE_METADATA = "https://archive.org/metadata/"
DEFAULT_FORMAT_PRIORITY = ["application/epub+zip", "application/pdf", "text/plain; charset=utf-8", "text/plain"]
SIMILARITY_THRESHOLD = 0.55  # fuzzy match threshold (lower to be permissive)
POLITE_DELAY = 1.0  # seconds between network calls

OUT_ROOT = Path("book")
OUT_EPUB = OUT_ROOT / "EPUB"
OUT_PDF = OUT_ROOT / "PDF"
OUT_OTHER = OUT_ROOT / "OTHER"
LOG_FILE = "download_books_verbose.log"
REMAINING_FILE = "REMAINING.txt"

# ensure directories
for d in (OUT_EPUB, OUT_PDF, OUT_OTHER):
    d.mkdir(parents=True, exist_ok=True)

# Logging basic setup
import logging
logger = logging.getLogger("book_fetcher")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# create HTTP session with retries
def create_session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

session = create_session()

# -------------------------
# Catalog (exact list requested)
# -------------------------
CATALOG = [
    # Must-have fiction (global classics)
    ("Pride and Prejudice", "Jane Austen"),
    ("Frankenstein", "Mary Shelley"),
    ("Moby-Dick", "Herman Melville"),
    ("Dracula", "Bram Stoker"),
    ("The Strange Case of Dr. Jekyll and Mr. Hyde", "R. L. Stevenson"),
    ("Alice’s Adventures in Wonderland", "Lewis Carroll"),
    ("Jane Eyre", "Charlotte Brontë"),
    ("Wuthering Heights", "Emily Brontë"),
    ("The Picture of Dorian Gray", "Oscar Wilde"),
    ("The Count of Monte Cristo", "Alexandre Dumas"),
    ("Crime and Punishment", "Fyodor Dostoyevsky"),
    ("The Brothers Karamazov", "Fyodor Dostoyevsky"),
    ("Little Women", "Louisa May Alcott"),
    ("A Room with a View", "E. M. Forster"),
    ("Middlemarch", "George Eliot"),
    ("The King in Yellow", "Robert W. Chambers"),
    ("The Great Gatsby", "F. Scott Fitzgerald"),
    # Plays / poetry
    ("Romeo and Juliet", "William Shakespeare"),
    ("Complete Works of William Shakespeare", "William Shakespeare"),
    # Seasonal
    ("A Christmas Carol", "Charles Dickens"),
    # Tier A (India) - Hindi / Indian works (Devanagari included)
    ("गोदान", "Munshi Premchand"),
    ("निर्मला", "Munshi Premchand"),
    ("गबन", "Munshi Premchand"),
    ("प्रेमचंद की सर्वश्रेष्ठ कहानियां", "Munshi Premchand"),
    # Tier B collections
    ("Sherlock Holmes (Complete)", "Arthur Conan Doyle"),
    ("Mark Twain (Collected)", "Mark Twain"),
    ("H. G. Wells (Collected)", "H. G. Wells"),
    ("Jules Verne (Collected)", "Jules Verne"),
    ("Edgar Allan Poe (Complete)", "Edgar Allan Poe"),
    ("Grimm's Fairy Tales", "Jacob Grimm & Wilhelm Grimm"),
    ("Aesop's Fables", "Aesop"),
    ("Panchatantra", "Unknown"),
    ("Hitopadesha", "Unknown"),
]

# -------------------------
# Utilities
# -------------------------
def sanitize_filename(s: str) -> str:
    s = s.strip()
    # keep unicode letters, numbers, spaces, hyphens, parentheses, commas, dots, underscores
    s = re.sub(r'[\\/*?:"<>|]', "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

def polite_sleep():
    time.sleep(POLITE_DELAY)

# -------------------------
# Gutendex (Project Gutenberg mirror) search & download
# -------------------------
def search_gutendex(title: str, author: Optional[str] = None, pages: int = 4) -> Optional[Dict]:
    logger.debug("Gutendex search: %s (author=%s)", title, author)
    params = {"search": f"{title} {author or ''}".strip()}
    best = None
    best_score = 0.0
    for page in range(1, pages+1):
        params["page"] = page
        try:
            r = session.get(GUTENDEX_SEARCH, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.debug("Gutendex request failed: %s", e)
            break
        results = data.get("results", [])
        for item in results:
            cand_title = item.get("title", "")
            cand_authors = ", ".join([a.get("name", "") for a in item.get("authors", [])])
            score = max(similarity(title, cand_title), 0.0)
            if author:
                score = max(score, 0.75*similarity(title, cand_title) + 0.25*similarity(author, cand_authors))
            if score > best_score:
                best_score = score
                best = item
        if best_score >= 0.9:
            break
        if not data.get("next"):
            break
        polite_sleep()
    if best and best_score >= SIMILARITY_THRESHOLD:
        logger.info("Gutendex matched: %s (score=%.3f)", best.get("title"), best_score)
        return best
    logger.debug("Gutendex no good match for: %s", title)
    return None

def choose_gutendex_format(book_entry: dict) -> Optional[Tuple[str,str]]:
    fmts = book_entry.get("formats", {})
    # prefer epub, then pdf, then text
    for pref in DEFAULT_FORMAT_PRIORITY:
        for key, url in fmts.items():
            if key and key.lower().startswith(pref.split(";")[0].lower()):
                return key, url
    # fallback: pick any http link
    for k, v in fmts.items():
        if isinstance(v, str) and v.startswith("http"):
            return k, v
    return None

# -------------------------
# Internet Archive search & download
# -------------------------
def search_internet_archive(title: str, author: Optional[str] = None, rows: int = 8) -> Optional[Dict]:
    logger.debug("Internet Archive search: %s (author=%s)", title, author)
    q_parts = []
    if title:
        q_parts.append(f'title:("{title}")')
    if author:
        q_parts.append(f'creator:("{author}")')
    q = " AND ".join(q_parts) if q_parts else f'title:("{title}")'
    params = {"q": q, "fl": "identifier,title,creator", "rows": rows, "output": "json"}
    try:
        r = session.get(ARCHIVE_ADV_SEARCH, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.debug("Internet Archive search failed: %s", e)
        return None
    docs = data.get("response", {}).get("docs", [])
    best = None
    best_score = 0.0
    for d in docs:
        cand_title = d.get("title", "")
        cand_authors = ", ".join(d.get("creator") or [])
        score = max(similarity(title, cand_title), 0.0)
        if author:
            score = max(score, 0.8*similarity(title, cand_title) + 0.2*similarity(author, cand_authors))
        if score > best_score:
            best_score = score
            best = d
    if best and best_score >= SIMILARITY_THRESHOLD:
        logger.info("Internet Archive matched: %s (score=%.3f)", best.get("title"), best_score)
        return best
    logger.debug("Internet Archive no good match for: %s", title)
    return None

def archive_files_for_identifier(identifier: str) -> Dict[str, str]:
    """Return map mime -> download_url for an archive.org identifier metadata."""
    logger.debug("Fetching archive metadata for: %s", identifier)
    url = urljoin(ARCHIVE_METADATA, quote_plus(identifier))
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.debug("Archive metadata failed: %s", e)
        return {}
    files = data.get("files", [])
    out = {}
    for f in files:
        fname = f.get("name")
        fmt = f.get("format") or f.get("mime_type") or ""
        if not fname:
            continue
        dl = f"https://archive.org/download/{identifier}/{quote_plus(fname)}"
        out_key = fmt or fname
        out[out_key] = dl
    # also try the 'original' 'downloads' if present (some metadata pages include direct download links)
    return out

# -------------------------
# Download helpers
# -------------------------
def download_stream(url: str, outpath: Path, timeout: int = 90) -> bool:
    logger.debug("Downloading: %s -> %s", url, outpath)
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = outpath.with_suffix(outpath.suffix + ".part")
            total = int(r.headers.get("Content-Length", 0) or 0)
            with open(tmp, "wb") as fh:
                with tqdm(total=total if total>0 else None, unit="B", unit_scale=True, desc=outpath.name, leave=False) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
                            pbar.update(len(chunk))
            tmp.rename(outpath)
        logger.info("Saved file: %s", outpath)
        return True
    except Exception as e:
        logger.warning("Download failed: %s (%s)", url, e)
        try:
            if outpath.exists():
                outpath.unlink()
        except Exception:
            pass
        return False

def write_provenance(outpath: Path, meta: dict):
    try:
        p = outpath.with_suffix(outpath.suffix + ".provenance.json")
        meta = dict(meta)
        meta["downloaded_at"] = datetime.utcnow().isoformat() + "Z"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        logger.debug("Wrote provenance: %s", p)
    except Exception as e:
        logger.debug("Could not write provenance: %s", e)

# -------------------------
# Simple text -> EPUB conversion (if ebooklib available)
# -------------------------
def text_to_epub(text: str, title: str, author: Optional[str], outpath: Path) -> bool:
    if not EPUBLIB_AVAILABLE:
        logger.debug("ebooklib not available; skipping text->epub conversion for: %s", title)
        return False
    try:
        book = epub.EpubBook()
        book.set_identifier(re.sub(r"\W+", "", title)[:32] or "id")
        book.set_title(title)
        book.add_author(author or "Unknown")
        # simple single-chapter EPUB
        chapter = epub.EpubHtml(title=title, file_name="chap_1.xhtml", lang="en")
        # Escape HTML-unsafe characters minimally
        html_body = "<h1>{}</h1><pre>{}</pre>".format(
            title, (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        )
        chapter.content = html_body.encode("utf-8")
        book.add_item(chapter)
        book.toc = (epub.Link("chap_1.xhtml", title, "chap_1"),)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        style = "body { font-family: Arial, sans-serif; }"
        nav_css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
        book.add_item(nav_css)
        book.spine = ["nav", chapter]
        epub.write_epub(str(outpath), book)
        logger.info("Converted text to EPUB: %s", outpath)
        return True
    except Exception as e:
        logger.warning("Failed to convert text -> EPUB: %s", e)
        return False

# -------------------------
# Main per-title processing
# -------------------------
def process_title(title: str, author: Optional[str]=None) -> Tuple[str, dict]:
    """
    Returns (status, metadata)
        status in {"downloaded_epub", "downloaded_pdf", "converted_text_to_epub", "not_found", "only_txt_saved"}
    """
    logger.info("Processing: %s %s", title, f"({author})" if author else "")
    # 1) Gutendex
    polite_sleep()
    try:
        gut = search_gutendex(title, author)
    except Exception as e:
        logger.debug("Gutendex search error: %s", e)
        gut = None

    if gut:
        fmt = choose_gutendex_format(gut)
        if fmt:
            mime, url = fmt
            # choose extension
            if "epub" in mime:
                ext = ".epub"
                outpath = OUT_EPUB / sanitize_filename(f"{gut.get('title')} - {','.join([a.get('name') for a in gut.get('authors', [])])}{ext}")
                ok = download_stream(url, outpath)
                if ok:
                    write_provenance(outpath, {"source":"gutendex","gutendex_entry":gut, "selected_format":mime, "source_url": url})
                    return "downloaded_epub", {"path": str(outpath)}
            if "pdf" in mime:
                ext = ".pdf"
                outpath = OUT_PDF / sanitize_filename(f"{gut.get('title')} - {','.join([a.get('name') for a in gut.get('authors', [])])}{ext}")
                ok = download_stream(url, outpath)
                if ok:
                    write_provenance(outpath, {"source":"gutendex","gutendex_entry":gut, "selected_format":mime, "source_url": url})
                    return "downloaded_pdf", {"path": str(outpath)}
            # if text only
            if "text" in mime or url.endswith(".txt"):
                try:
                    r = session.get(url, timeout=30)
                    r.raise_for_status()
                    text = r.text
                    # attempt conversion to EPUB if possible
                    sanitized = sanitize_filename(f"{gut.get('title')} - {','.join([a.get('name') for a in gut.get('authors', [])])}")
                    out_epub_path = OUT_EPUB / (sanitized + ".epub")
                    if text_to_epub(text, gut.get("title"), author or ", ".join([a.get('name') for a in gut.get('authors', [])]), out_epub_path):
                        write_provenance(out_epub_path, {"source":"gutendex","gutendex_entry":gut,"note":"converted text->epub","text_source":url})
                        return "converted_text_to_epub", {"path": str(out_epub_path)}
                    else:
                        out_txt = OUT_OTHER / (sanitized + ".txt")
                        with open(out_txt, "w", encoding="utf-8") as fh:
                            fh.write(text)
                        write_provenance(out_txt, {"source":"gutendex","gutendex_entry":gut,"text_source":url})
                        return "only_txt_saved", {"path": str(out_txt)}
                except Exception as e:
                    logger.debug("Failed to retrieve Gutendex text: %s", e)

    # 2) Internet Archive fallback
    polite_sleep()
    try:
        ia = search_internet_archive(title, author)
    except Exception as e:
        logger.debug("Archive search error: %s", e)
        ia = None

    if ia:
        ident = ia.get("identifier")
        files_map = archive_files_for_identifier(ident)
        # choose best
        chosen_url = None
        chosen_mime = None
        for pref in DEFAULT_FORMAT_PRIORITY:
            for k, v in files_map.items():
                if pref.split(";")[0].lower() in k.lower() or pref.split(";")[0].lower() in (v.lower() if isinstance(v, str) else ""):
                    chosen_url = v
                    chosen_mime = k
                    break
            if chosen_url:
                break
        # last-resort: any url that ends with .epub or .pdf
        if not chosen_url:
            for k, v in files_map.items():
                if isinstance(v, str) and v.lower().endswith(".epub"):
                    chosen_url, chosen_mime = v, k
                    break
            if not chosen_url:
                for k, v in files_map.items():
                    if isinstance(v, str) and v.lower().endswith(".pdf"):
                        chosen_url, chosen_mime = v, k
                        break
        if chosen_url:
            # choose destination based on extension
            dest_ext = Path(quote_plus(chosen_url)).suffix.lower()
            if dest_ext == ".epub" or (".epub" in (chosen_mime or "").lower()):
                filename = sanitize_filename(f"{ia.get('title')} - {','.join(ia.get('creator') or [])}.epub")
                outpath = OUT_EPUB / filename
                ok = download_stream(chosen_url, outpath)
                if ok:
                    write_provenance(outpath, {"source":"internet_archive","archive_identifier":ident,"selected_format":chosen_mime,"source_url":chosen_url})
                    return "downloaded_epub", {"path": str(outpath)}
            if dest_ext == ".pdf" or (".pdf" in (chosen_mime or "").lower()):
                filename = sanitize_filename(f"{ia.get('title')} - {','.join(ia.get('creator') or [])}.pdf")
                outpath = OUT_PDF / filename
                ok = download_stream(chosen_url, outpath)
                if ok:
                    write_provenance(outpath, {"source":"internet_archive","archive_identifier":ident,"selected_format":chosen_mime,"source_url":chosen_url})
                    return "downloaded_pdf", {"path": str(outpath)}
            # if text available try conversion
            # try to find text/plain
            for k, v in files_map.items():
                if "plain" in k.lower() or (isinstance(k, str) and k.lower().endswith(".txt")):
                    try:
                        r = session.get(v, timeout=30)
                        r.raise_for_status()
                        text = r.text
                        sanitized = sanitize_filename(f"{ia.get('title')} - {','.join(ia.get('creator') or [])}")
                        out_epub_path = OUT_EPUB / (sanitized + ".epub")
                        if text_to_epub(text, ia.get('title') or title, author or (ia.get('creator')[0] if ia.get('creator') else None), out_epub_path):
                            write_provenance(out_epub_path, {"source":"internet_archive","archive_identifier":ident,"note":"converted text->epub","text_source":v})
                            return "converted_text_to_epub", {"path": str(out_epub_path)}
                        else:
                            out_txt = OUT_OTHER / (sanitized + ".txt")
                            with open(out_txt, "w", encoding="utf-8") as fh:
                                fh.write(text)
                            write_provenance(out_txt, {"source":"internet_archive","archive_identifier":ident,"text_source":v})
                            return "only_txt_saved", {"path": str(out_txt)}
                    except Exception:
                        continue

    # If we reach here: not found as EPUB/PDF, no conversion succeeded
    logger.info("Could not obtain EPUB/PDF for: %s", title)
    return "not_found", {}

# -------------------------
# Orchestration
# -------------------------
def main():
    remaining = []
    summary = []
    logger.info("Starting catalog download: %d titles", len(CATALOG))
    for title, author in CATALOG:
        status, meta = process_title(title, author)
        summary.append({"title": title, "author": author, "status": status, "meta": meta})
        if status == "not_found":
            remaining.append(f"{title} - {author or ''}".strip())
        # polite pause
        polite_sleep()

    # write remaining file
    if remaining:
        with open(REMAINING_FILE, "w", encoding="utf-8") as fh:
            fh.write("\n".join(remaining))
        logger.info("Wrote REMAINING.txt with %d items", len(remaining))
    else:
        try:
            os.remove(REMAINING_FILE)
        except Exception:
            pass
        logger.info("All items processed; REMAINING.txt not created (or removed)")

    # write summary json
    sum_path = OUT_ROOT / "download_summary.json"
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump({"created_at": datetime.utcnow().isoformat()+"Z", "summary": summary}, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote summary: %s", sum_path)

if __name__ == "__main__":
    main()
