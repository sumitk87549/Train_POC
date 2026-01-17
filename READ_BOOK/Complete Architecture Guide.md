# Universal Book Cleaner - Complete Architecture Guide

## 🎯 **Script Purpose & Flow**

This script takes messy EPUB/PDF files and produces clean, chapter-segregated text files perfect for:
- **Audio generation** (TTS/audiobook creation)
- **Translation** (chapter-by-chapter)
- **Summarization** (section-wise summaries)
- **Reading** (clean, organized content)

---

## 📊 **High-Level Flow Diagram**

```
INPUT: books.csv + book_dir/
  ↓
1. DISCOVERY: Find matching .epub or .pdf files
  ↓
2. EXTRACTION: Pull raw text blocks from file
  ├─ EPUB: Parse HTML/XML structure
  └─ PDF: Extract text from pages
  ↓
3. CLEANING: Remove Gutenberg headers, metadata, junk
  ↓
4. SEGMENTATION: Split into chapters/sections
  ↓
5. FORMATTING: Write ===CHAPTER TITLE=== format
  ↓
OUTPUT: BookTitle__Cleaned.txt + processing_report.json
```

---

## 🔧 **Section-by-Section Breakdown**

### **1. Configuration Block (Lines 20-75)**

```python
SECTION_PATTERNS = [
    r'^\s*CHAPTER\b', 
    r'^\s*PROLOGUE\b',
    # ... etc
]
```

**What it does:**
- Defines regex patterns to recognize chapter/section headings
- `^\s*` = start of line with optional whitespace
- `\b` = word boundary (prevents matching "CHAPTERS" when looking for "CHAPTER")

**How to improve:**
- Add patterns for your specific books (e.g., "Letter I", "Day 1", "Episode")
- Test with actual chapter headings from your library
- Use `re.VERBOSE` for complex patterns with comments

**Example improvement:**
```python
# Add these for different book styles:
r'^\s*LETTER\s+[IVXLCDM\d]+\b',  # Epistolary novels
r'^\s*DAY\s+\d+\b',               # Diary format
r'^\s*\d+\.\s+[A-Z]',             # "1. THE BEGINNING"
```

---

### **2. File Discovery (Lines 77-105)**

```python
def find_best_book_file(title, author, book_dir):
    # Searches for .epub and .pdf files
    # Scores them by similarity to title+author
    # Returns best match
```

**What it does:**
1. Globs all `.epub` and `.pdf` files recursively
2. Computes similarity score using:
   - **Levenshtein distance** (if installed) - character-level matching
   - **Token matching** - word-level matching (e.g., "Pride" in "Pride_and_Prejudice.epub")
3. Returns file with highest score (>0.25 threshold)

**Why this matters:**
- Files might be named `Pride_Prejudice_JaneAusten.epub` or `1342-0.pdf` (Gutenberg ID)
- Need fuzzy matching to handle variations

**How to improve:**
```python
# Add weighted scoring for better matches
def advanced_scoring(title, author, filename):
    score = 0.0
    
    # Exact title match = highest weight
    if title.lower() in filename.lower():
        score += 0.5
    
    # Author name match
    if author and author.lower() in filename.lower():
        score += 0.3
    
    # Fuzzy similarity
    score += similarity(title, filename) * 0.2
    
    return score
```

---

### **3. EPUB Processing (Lines 107-210)**

#### **3a. HTML to Text Blocks (Lines 107-142)**

```python
def html_to_text_blocks(html):
    # Handles <pre>, <code>, <p>, <h1-h6>, etc.
```

**The Problem:**
- Project Gutenberg EPUBs store text in `<pre>` tags (plain text format)
- Modern EPUBs use `<p>` tags for paragraphs
- Need to handle BOTH

**The Solution:**
1. **Extract `<pre>` content first** - split by blank lines (`\n\s*\n`)
2. **Extract structural tags** - headings, paragraphs, divs
3. **Normalize whitespace** - collapse multiple spaces, remove line breaks
4. **Fallback** - if nothing found, grab body text

**Key line explained:**
```python
para = re.sub(r'-\s*\n\s*', '', para)  # Remove hyphenation
# "beauti-\nful" → "beautiful"
```

**How to improve:**
- Add support for `<blockquote>`, `<aside>` (for special sections)
- Preserve emphasis markers (italic/bold) if needed for audio inflection
- Handle poetry/verse formatting (line breaks matter!)

---

#### **3b. Audio EPUB Detection (Lines 144-168)**

```python
def is_audio_epub(epub_path):
    # Detects LibriVox audiobook manifests
```

**Why this exists:**
- Some "EPUBs" are just playlists of MP3 files
- Would waste time trying to extract text
- Early detection saves processing

**How it works:**
- Small file size (<200KB) = suspicious
- Contains keywords like "librivox", ".mp3" = audio manifest
- Returns warning instead of trying to extract

---

#### **3c. Main EPUB Extraction (Lines 170-210)**

```python
def gather_text_from_epub(epub_path):
    book = epub.read_epub(str(epub_path))
    items = book.get_items_of_type(ITEM_DOCUMENT)
    # Process each HTML document in EPUB
```

**The EPUB structure:**
```
book.epub
├── chapter1.xhtml
├── chapter2.xhtml
├── chapter3.xhtml
└── styles.css
```

**What it does:**
1. Opens EPUB (it's a ZIP file internally)
2. Gets all `ITEM_DOCUMENT` files (HTML/XHTML content)
3. Decodes HTML → extracts text blocks
4. Concatenates all blocks from all chapters
5. Applies Gutenberg cleaning

**Fallback strategy:**
```python
if len(blocks) <= 3 and items:
    # Very few blocks? Try raw body text extraction
```

This catches EPUBs with unusual structure.

---

### **4. PDF Processing (Lines 212-255)**

```python
def gather_text_from_pdf(pdf_path):
    # Dual-method extraction for robustness
```

**The Challenge:**
PDFs are notoriously difficult because:
- Text might be images (scanned books) - not extractable without OCR
- Layout is visual, not semantic (no "paragraph" concept)
- Headers/footers on every page
- Column layouts, text boxes, etc.

**Two-Method Approach:**

**Method 1: pdfplumber (Primary)**
```python
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
```
- Better at preserving layout
- Handles tables, columns better
- More accurate text positioning

**Method 2: PyPDF2 (Fallback)**
```python
pdf_reader = PyPDF2.PdfReader(file)
for page in pdf_reader.pages:
    text = page.extract_text()
```
- Faster, simpler
- Works when pdfplumber fails
- Less layout-aware

**Cleaning Operations:**
```python
# Fix hyphenation: "beau-\ntiful" → "beautiful"
para = re.sub(r'-\s*\n\s*', '', para)

# Join broken lines: "This is a\nsentence" → "This is a sentence"
para = re.sub(r'\n', ' ', para)

# Normalize spaces: "too    many" → "too many"
para = re.sub(r'\s+', ' ', para).strip()
```

**Artifact removal:**
```python
page_number_pattern = re.compile(r'^\s*\d+\s*$')
# Skips blocks that are just "42" (page numbers)
```

**How to improve:**
```python
# Detect and remove running headers
def is_running_header(block, prev_block):
    # If same text appears on multiple pages = header
    return similarity(block, prev_block) > 0.9

# Detect column layout
def merge_columns(blocks):
    # Group blocks by Y-position
    # Merge left + right column text
    pass

# OCR fallback for scanned PDFs
import pytesseract
from pdf2image import convert_from_path

def extract_with_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text
```

---

### **5. Gutenberg Cleaning (Lines 257-305)**

```python
def clean_gutenberg_content(blocks):
    # Remove license text, metadata
```

**What Project Gutenberg adds:**
```
*** START OF THIS PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***

Title: Pride and Prejudice
Author: Jane Austen
Release Date: June 1998

[ACTUAL BOOK CONTENT]

*** END OF THIS PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***

[Long legal text about licensing]
```

**The cleaning strategy:**

**Step 1: Find boundaries**
```python
for i, block in enumerate(blocks):
    if "START OF" in block and "PROJECT GUTENBERG" in block:
        start_idx = i + 1  # Start AFTER marker
    if "END OF" in block and "PROJECT GUTENBERG" in block:
        end_idx = i  # End BEFORE marker
```

**Step 2: Filter metadata**
```python
junk_patterns = [
    r'^Title\s*:',
    r'^Author\s*:',
    r'^Produced by',
    r'www\.gutenberg\.org',
]
# Remove any block matching these
```

**Why regex instead of simple string matching:**
```python
# These all need to be caught:
"Title: Pride and Prejudice"
"Title:Pride and Prejudice"
"Title :  Pride and Prejudice"
# Regex: r'^Title\s*:' handles all variations
```

**How to improve:**
```python
# Add more publishers
PUBLISHER_PATTERNS = [
    r'Standard Ebooks',
    r'ManyBooks\.net',
    r'Feedbooks',
    r'Internet Archive',
]

# Detect copyright pages
def is_copyright_page(text):
    copyright_indicators = [
        'all rights reserved',
        'copyright ©',
        '© \d{4}',
        'isbn',
    ]
    return any(re.search(p, text, re.I) for p in copyright_indicators)
```

---

### **6. Section Detection (Lines 307-330)**

```python
def is_section_heading(text):
    # Determines if a block is a chapter heading
```

**This is CRITICAL for your use case** because:
- Bad detection = chapters merged or split wrongly
- Affects audio generation (pause between chapters)
- Affects translation (chapter boundaries matter)

**The Detection Logic:**

**Step 1: Reject noise**
```python
if SCENE_BREAK_REGEX.match(t):
    return None  # "* * * *" is NOT a heading
```

**Step 2: Clean the text**
```python
t_clean = re.sub(r'^[\W_]+', '', t).strip()
# "===CHAPTER 1===" → "CHAPTER 1"
```

**Step 3: Pattern matching**
```python
if SECTION_REGEX.search(t_clean):
    return t_clean.upper()
# Matches: "Chapter 1", "PROLOGUE", "PART II"

if CHAPTER_FALLBACK_REGEX.match(t_clean):
    return t_clean.upper()
# Matches: "Chapter One", "CH. 5"
```

**Step 4: Anthology detection**
```python
if t_clean.isupper() and re.match(r"^THE\s+[A-Z]", t_clean):
    return t_clean
# Matches: "THE UGLY DUCKLING" (Grimm's Fairy Tales)
```

**Common false positives to avoid:**
```python
# These should NOT be detected as headings:
"He went to CHAPTER 9 of the book"  # Mid-sentence
"VERY IMPORTANT NOTE"                # Emphasis, not heading
"THE END."                           # Unless at actual end
```

**How to improve:**
```python
def is_section_heading_enhanced(text, context):
    """
    context = {
        'prev_block': previous text block,
        'next_block': next text block,
        'position': index in book (0.0-1.0)
    }
    """
    # Check surrounding context
    if context['prev_block'] and len(context['prev_block']) < 50:
        # Previous block was short = likely heading too
        # This might be a subheading, not main chapter
        return None
    
    # Check position in book
    if context['position'] < 0.05:
        # First 5% of book = likely front matter
        if "DEDICATION" in text or "FOREWORD" in text:
            return text.upper()
    
    # Check text characteristics
    words = text.split()
    if len(words) > 15:
        # Too long to be a heading
        return None
    
    if text.endswith('.') and len(words) > 5:
        # Sentence-like = probably not heading
        return None
    
    # Use ML model for ambiguous cases
    # return ml_heading_classifier.predict(text)
```

---

### **7. Section Splitting (Lines 332-370)**

```python
def split_into_sections(blocks):
    # Groups paragraphs under chapter headings
```

**The Algorithm:**

```python
sections = []
current_title = None
current_pars = []

for block in blocks:
    if is_heading(block):
        # Flush current section
        sections.append((current_title, current_pars))
        
        # Start new section
        current_title = block
        current_pars = []
    else:
        # Accumulate paragraphs
        current_pars.append(block)
```

**Enhanced Title Capture:**
```python
# Detects this pattern:
# "CHAPTER I"          ← heading
# "THE BEGINNING"      ← chapter title
# "It was a dark..."   ← content

if CHAPTER_TITLE_PATTERN.match(heading):
    potential_title = blocks[idx + 1]
    if looks_like_title(potential_title):
        heading = f"{heading}: {potential_title}"
        skip_next = True  # Don't process title as paragraph
```

**How to improve:**
```python
def split_with_hierarchy(blocks):
    """
    Create hierarchical structure:
    BOOK I
      ├─ PART 1
      │   ├─ CHAPTER 1
      │   └─ CHAPTER 2
      └─ PART 2
          └─ CHAPTER 3
    """
    hierarchy_levels = {
        'BOOK': 1,
        'PART': 2,
        'CHAPTER': 3,
        'SCENE': 4,
    }
    
    # Build tree structure
    # Useful for navigation in audio player
```

**Fallback for poorly formatted books:**
```python
if len(sections) <= 1:
    # No headings detected? Try splitting on whitespace
    joined = "\n\n".join(blocks)
    parts = re.split(r'(?m)^\s*CHAPTER\b', joined)
    # This catches inline headings
```

---

### **8. Output Writing (Lines 372-395)**

```python
def write_output_file(out_dir, book_title, author, sections):
    # Writes ===CHAPTER=== format
```

**Output Format:**
```
===PROLOGUE===

First paragraph of prologue.

Second paragraph of prologue.

===CHAPTER I: THE BEGINNING===

First paragraph of chapter.

Second paragraph.
```

**Why this format:**
- Easy to parse later (split on `===`)
- Human-readable
- Preserves structure
- Compatible with most TTS systems

**Filtering:**
```python
if sec_title == "OPENING_CREDITS" and len(paras) <= 3:
    # Skip short opening sections
    # (just title/author info)
```

**How to improve for your use case:**

```python
def write_output_for_audio(sections):
    """
    Add metadata for TTS systems
    """
    output = {
        'metadata': {
            'title': book_title,
            'author': author,
            'chapters': []
        },
        'sections': []
    }
    
    for idx, (title, paragraphs) in enumerate(sections):
        chapter_data = {
            'id': f'ch_{idx:03d}',
            'title': title,
            'word_count': sum(len(p.split()) for p in paragraphs),
            'estimated_duration_minutes': calculate_duration(paragraphs),
            'text': '\n\n'.join(paragraphs),
            'ssml': generate_ssml(title, paragraphs),  # For advanced TTS
        }
        output['sections'].append(chapter_data)
    
    return output

def generate_ssml(title, paragraphs):
    """Speech Synthesis Markup Language for better TTS"""
    ssml = f'<speak>'
    ssml += f'<prosody rate="slow" pitch="+10%">{title}</prosody>'
    ssml += '<break time="1s"/>'
    for para in paragraphs:
        ssml += f'<p>{para}</p>'
        ssml += '<break time="500ms"/>'
    ssml += '</speak>'
    return ssml
```

---

## 🎯 **Critical Improvements for State-of-the-Art**

### **1. Advanced Chapter Detection**
```python
# Use machine learning for ambiguous cases
from transformers import pipeline

classifier = pipeline("text-classification", 
                     model="chapter-heading-classifier")

def ml_heading_detection(text):
    result = classifier(text)
    return result['label'] == 'HEADING' and result['score'] > 0.9
```

### **2. Content Quality Checks**
```python
def validate_section_quality(sections):
    for title, paras in sections:
        # Check for missing content
        if len(paras) == 0:
            warnings.append(f"Empty section: {title}")
        
        # Check for too-short chapters
        word_count = sum(len(p.split()) for p in paras)
        if word_count < 100:
            warnings.append(f"Suspiciously short: {title} ({word_count} words)")
        
        # Check for repeated text
        if has_duplicates(paras):
            warnings.append(f"Duplicate content in: {title}")
```

### **3. OCR for Scanned PDFs**
```python
def extract_with_ocr(pdf_path):
    """For scanned/image-based PDFs"""
    from pdf2image import convert_from_path
    import pytesseract
    
    images = convert_from_path(pdf_path, dpi=300)
    blocks = []
    
    for page_num, image in enumerate(images):
        # Preprocess for better OCR
        image = enhance_image(image)
        
        # Extract with confidence scoring
        text = pytesseract.image_to_string(image)
        conf = pytesseract.image_to_data(image, output_type='dict')
        
        # Filter low-confidence text
        filtered_text = filter_by_confidence(text, conf)
        blocks.extend(split_to_paragraphs(filtered_text))
    
    return blocks
```

### **4. Language-Specific Handling**
```python
def detect_and_process_language(text):
    from langdetect import detect
    
    lang = detect(text)
    
    if lang == 'fr':
        # French: handle guillemets « »
        return process_french_quotes(text)
    elif lang == 'de':
        # German: handle ß, umlauts
        return process_german_text(text)
    # etc.
```

### **5. Dialogue Detection for Audio**
```python
def mark_dialogue(paragraphs):
    """
    Detect dialogue for different TTS voices
    Useful for audio generation
    """
    processed = []
    for para in paragraphs:
        if para.strip().startswith('"') or para.strip().startswith("'"):
            processed.append({
                'text': para,
                'type': 'dialogue',
                'voice': 'character'
            })
        else:
            processed.append({
                'text': para,
                'type': 'narrative',
                'voice': 'narrator'
            })
    return processed
```

---

## 🔍 **Testing & Validation Strategy**

### **1. Unit Tests**
```python
def test_section_detection():
    # True positives
    assert is_section_heading("CHAPTER 1")
    assert is_section_heading("PROLOGUE")
    
    # True negatives
    assert not is_section_heading("He went to chapter 1")
    assert not is_section_heading("* * * * *")

def test_gutenberg_cleaning():
    sample = [
        "*** START OF PROJECT GUTENBERG ***",
        "Title: Test Book",
        "Chapter 1",
        "Actual content here.",
        "*** END OF PROJECT GUTENBERG ***"
    ]
    cleaned = clean_gutenberg_content(sample)
    assert len(cleaned) == 2
    assert "Actual content" in cleaned[1]
```

### **2. Integration Tests**
```python
def test_end_to_end():
    # Process a known good book
    result = process_book(
        "Pride and Prejudice",
        "Jane Austen",
        Path("./test_books"),
        Path("./test_output")
    )
    
    # Validate output
    assert result['sections'] > 50  # Should have ~60 chapters
    assert result['warnings'] == []
    
    # Check output file
    output_text = Path(result['output_file']).read_text()
    assert "===CHAPTER" in output_text
```

---

## 📈 **Performance Optimization**

```python
# For large libraries (1000+ books)
from concurrent.futures import ProcessPoolExecutor

def process_library_parallel(catalog, book_dir, out_dir):
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_book, title, author, book_dir, out_dir)
            for title, author in catalog
        ]
        results = [f.result() for f in futures]
    return results
```

---

## 🎤 **Recommendations for Your Use Case**

Since you're doing **translation, summarization, and audio**:

1. **Preserve formatting markers**
   - Keep *italic* markers → `_text_`
   - Keep **bold** markers → `**text**`
   - TTS can use these for emphasis

2. **Add sentence boundaries**
   ```python
   import nltk
   sentences = nltk.sent_tokenize(paragraph)
   # Better for translation quality
   ```

3. **Metadata for each section**
   ```python
   {
       'chapter_id': 'ch_001',
       'title': 'Chapter 1',
       'word_count': 2500,
       'estimated_reading_time': '10 min',
       'estimated_audio_duration': '15 min',
       'paragraphs': [...],
       'language': 'en',
       'genre': 'fiction'
   }
   ```

4. **Quality scoring**
   ```python
   def calculate_quality_score(section):
       score = 100
       
       # Penalize unusual characteristics
       if avg_sentence_length(section) < 5:
           score -= 20  # Choppy text
       if has_encoding_errors(section):
           score -= 30  # Corrupted text
       if is_mostly_numbers(section):
           score -= 40  # Tables/metadata
       
       return score
   ```

Would you like me to dive deeper into any specific part or show you how to implement advanced features like ML-based heading detection or OCR integration?