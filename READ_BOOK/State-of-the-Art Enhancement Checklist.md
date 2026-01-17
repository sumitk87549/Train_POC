# 🚀 State-of-the-Art Enhancement Checklist

## ✅ **Phase 1: Quick Wins (1-2 hours)**

### 1. **Better Chapter Title Capture**
**Current issue:** Misses descriptive chapter titles like "Chapter 1: The Storm"

**Fix:**
```python
def enhanced_chapter_capture(blocks):
    """Captures multi-line chapter headings"""
    i = 0
    while i < len(blocks):
        block = blocks[i]
        heading = is_section_heading(block)
        
        if heading and CHAPTER_TITLE_PATTERN.match(heading):
            # Look ahead for title (next 1-2 lines)
            potential_titles = []
            for j in range(i+1, min(i+3, len(blocks))):
                next_block = blocks[j].strip()
                
                # Check if it's a title
                if (len(next_block) < 100 and 
                    len(next_block.split()) <= 12 and
                    not is_section_heading(next_block) and
                    not next_block[0].islower()):  # Starts with capital
                    
                    potential_titles.append(next_block)
                else:
                    break  # Hit content, stop looking
            
            if potential_titles:
                full_title = f"{heading}: {' '.join(potential_titles)}"
                return full_title, i + len(potential_titles) + 1
        
        i += 1
```

### 2. **Add Sentence Segmentation**
**Why:** Better for translation (translate sentence-by-sentence) and TTS (natural pauses)

```python
import nltk
nltk.download('punkt')

def segment_paragraph_to_sentences(paragraph):
    """Split paragraphs into sentences for better processing"""
    sentences = nltk.sent_tokenize(paragraph)
    return sentences

# Update write_output_file to include sentences
def write_with_sentences(sections):
    output = []
    for title, paragraphs in sections:
        section_data = {
            'title': title,
            'paragraphs': []
        }
        for para in paragraphs:
            section_data['paragraphs'].append({
                'text': para,
                'sentences': segment_paragraph_to_sentences(para)
            })
        output.append(section_data)
    return output
```

### 3. **Remove Repeated Headers/Footers (PDFs)**
**Current issue:** PDFs repeat "Chapter 1" and page numbers on every page

```python
def remove_repeated_elements(blocks):
    """Remove headers/footers that appear multiple times"""
    from collections import Counter
    
    # Count block frequencies
    block_counts = Counter(blocks)
    
    # Blocks appearing 5+ times are likely headers/footers
    repeated = {block for block, count in block_counts.items() 
                if count >= 5 and len(block) < 100}
    
    # Filter out repeated elements
    filtered = [b for b in blocks if b not in repeated]
    
    logger.info(f"Removed {len(repeated)} repeated header/footer elements")
    return filtered
```

---

## 🎯 **Phase 2: Audio Generation Optimization (2-4 hours)**

### 4. **Add Pause Markers for TTS**
```python
def add_tts_markers(sections):
    """Add SSML markers for natural audio pauses"""
    for title, paragraphs in sections:
        enhanced_paras = []
        for para in paragraphs:
            # Add pause after punctuation
            para = re.sub(r'([.!?])\s+', r'\1<break time="500ms"/> ', para)
            
            # Add pause for em-dashes (longer thoughts)
            para = re.sub(r'—', '<break time="300ms"/>—<break time="300ms"/>', para)
            
            # Add pause for dialogue
            if para.strip().startswith('"'):
                para = '<break time="400ms"/>' + para
            
            enhanced_paras.append(para)
        
        yield title, enhanced_paras
```

### 5. **Detect and Mark Dialogue**
**Why:** Use different voice/tone in TTS for narrator vs. characters

```python
def detect_dialogue_blocks(paragraphs):
    """Mark dialogue vs. narrative for voice switching"""
    processed = []
    
    for para in paragraphs:
        # Check various quote styles
        is_dialogue = (
            para.strip().startswith('"') or
            para.strip().startswith("'") or
            para.strip().startswith('"') or  # Smart quotes
            para.strip().startswith(''')
        )
        
        # Check for speech tags
        has_speech_tag = any(tag in para.lower() for tag in 
                           ['said', 'asked', 'replied', 'shouted', 'whispered'])
        
        processed.append({
            'text': para,
            'type': 'dialogue' if is_dialogue or has_speech_tag else 'narrative',
            'tts_voice': 'character' if is_dialogue else 'narrator'
        })
    
    return processed
```

### 6. **Estimate Audio Duration**
```python
def estimate_audio_duration(text, wpm=150):
    """
    Estimate TTS audio duration
    Average speaking rate: 150 words/minute
    """
    words = len(text.split())
    minutes = words / wpm
    
    # Add time for pauses at punctuation
    pause_time = text.count('.') * 0.5  # 500ms per period
    pause_time += text.count(',') * 0.25  # 250ms per comma
    pause_time += text.count('!') * 0.6  # Longer for emphasis
    pause_time += text.count('?') * 0.6
    
    total_seconds = (minutes * 60) + pause_time
    return {
        'minutes': int(total_seconds // 60),
        'seconds': int(total_seconds % 60),
        'total_seconds': total_seconds
    }

# Add to section output
section_data['audio_duration'] = estimate_audio_duration(section_text)
```

---

## 📚 **Phase 3: Advanced Quality (4-8 hours)**

### 7. **OCR Support for Scanned PDFs**
```python
def is_scanned_pdf(pdf_path):
    """Detect if PDF is scanned (images) vs. text"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page_text = pdf.pages[0].extract_text()
            return len(first_page_text.strip()) < 50
    except:
        return True

def extract_with_ocr(pdf_path):
    """Extract text from scanned PDFs using OCR"""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        
        images = convert_from_path(pdf_path, dpi=300)
        blocks = []
        
        for i, image in enumerate(images):
            logger.info(f"OCR processing page {i+1}/{len(images)}")
            text = pytesseract.image_to_string(image, lang='eng')
            
            # Split into paragraphs
            paragraphs = re.split(r'\n\s*\n', text)
            for para in paragraphs:
                para = re.sub(r'\s+', ' ', para).strip()
                if para and len(para) > 20:
                    blocks.append(para)
        
        return blocks
    except ImportError:
        logger.error("OCR requires: pip install pdf2image pytesseract")
        return []

# Update gather_text_from_pdf
def gather_text_from_pdf_enhanced(pdf_path):
    blocks = gather_text_from_pdf(pdf_path)  # Try normal extraction
    
    if len(blocks) < 10:
        logger.warning("Few blocks extracted, trying OCR...")
        blocks = extract_with_ocr(pdf_path)
    
    return blocks
```

### 8. **Content Validation & Quality Scoring**
```python
def validate_section_content(title, paragraphs):
    """Quality check for extracted sections"""
    issues = []
    
    # Check for empty sections
    if len(paragraphs) == 0:
        issues.append(f"CRITICAL: Empty section '{title}'")
    
    # Check for suspiciously short chapters
    total_words = sum(len(p.split()) for p in paragraphs)
    if total_words < 100 and 'CHAPTER' in title:
        issues.append(f"WARNING: Very short chapter '{title}' ({total_words} words)")
    
    # Check for encoding errors
    for i, para in enumerate(paragraphs):
        if '�' in para or '\x00' in para:
            issues.append(f"ERROR: Encoding issue in '{title}' paragraph {i}")
    
    # Check for duplicate content
    if len(paragraphs) != len(set(paragraphs)):
        issues.append(f"WARNING: Duplicate paragraphs in '{title}'")
    
    # Check for unusual character ratios
    for para in paragraphs:
        if len(para) > 50:
            alpha_ratio = sum(c.isalpha() for c in para) / len(para)
            if alpha_ratio < 0.5:
                issues.append(f"WARNING: Low text ratio in '{title}' (might be table/metadata)")
    
    return issues

# Use in process_book
all_issues = []
for title, paragraphs in sections:
    issues = validate_section_content(title, paragraphs)
    all_issues.extend(issues)

result['quality_issues'] = all_issues
```

### 9. **Smart Hyphenation Fixing**
**Problem:** PDFs break words across lines: "beau-\ntiful" or "exam-\nple"

```python
def fix_hyphenation_advanced(text):
    """
    Smart hyphenation fixing with dictionary validation
    """
    # Simple case: word-\nword
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Advanced: validate with dictionary
    try:
        import enchant
        d = enchant.Dict("en_US")
        
        # Find all hyphenated words
        hyphenated = re.findall(r'(\w+)-\s*\n\s*(\w+)', text)
        for part1, part2 in hyphenated:
            combined = part1 + part2
            if d.check(combined):  # Valid word
                text = text.replace(f"{part1}-\n{part2}", combined)
            else:
                # Keep hyphenation (might be compound word)
                text = text.replace(f"{part1}-\n{part2}", f"{part1}-{part2}")
    except ImportError:
        logger.warning("Install pyenchant for better hyphenation: pip install pyenchant")
    
    return text
```

### 10. **Chapter Number Normalization**
**Problem:** "Chapter I", "Chapter 1", "Ch. One" all mean the same thing

```python
def normalize_chapter_number(title):
    """Convert all chapter numbers to consistent format"""
    
    # Roman numerals to numbers
    roman_map = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
        'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
        'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
        # ... add more as needed
    }
    
    # Extract number from title
    match = re.search(r'CHAPTER\s+([IVXLCDM]+|\d+|ONE|TWO|THREE|FOUR|FIVE)', title, re.I)
    if match:
        num_str = match.group(1).upper()
        
        # Convert to integer
        if num_str in roman_map:
            num = roman_map[num_str]
        elif num_str.isdigit():
            num = int(num_str)
        else:
            # Word numbers
            word_map = {'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5}
            num = word_map.get(num_str, 0)
        
        # Reconstruct with normalized number
        rest = title.split(match.group(0), 1)[1] if len(title.split(match.group(0))) > 1 else ''
        return f"CHAPTER {num}{rest}"
    
    return title
```

---

## 🌍 **Phase 4: Multi-Language Support (Optional)**

### 11. **Language Detection**
```python
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # Consistent results

def detect_book_language(blocks):
    """Detect the language of the book"""
    # Sample first 1000 words
    sample_text = ' '.join(blocks[:10])
    
    try:
        lang = detect(sample_text)
        logger.info(f"Detected language: {lang}")
        return lang
    except:
        return 'en'  # Default to English

# Language-specific processing
def process_language_specific(blocks, lang):
    if lang == 'fr':
        # French quotes: « text »
        blocks = [re.sub(r'«\s*', '"', b) for b in blocks]
        blocks = [re.sub(r'\s*»', '"', b) for b in blocks]
    elif lang == 'de':
        # German: „text"
        blocks = [re.sub(r'„', '"', b) for b in blocks]
    # Add more languages...
    
    return blocks
```

---

## 📊 **Phase 5: Output Formats for Your Pipeline**

### 12. **JSON Output for Translation/Audio Pipeline**
```python
def write_json_output(out_dir, book_title, author, sections):
    """
    Structured JSON for downstream processing
    """
    output = {
        'metadata': {
            'title': book_title,
            'author': author,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'total_sections': len(sections),
            'total_words': sum(len(' '.join(p).split()) for _, p in sections),
        },
        'sections': []
    }
    
    for idx, (title, paragraphs) in enumerate(sections):
        # Prepare sentences for translation
        sentences = []
        for para in paragraphs:
            sentences.extend(nltk.sent_tokenize(para))
        
        section_data = {
            'id': f'section_{idx:03d}',
            'title': normalize_chapter_number(title),
            'word_count': len(' '.join(paragraphs).split()),
            'audio_estimate': estimate_audio_duration(' '.join(paragraphs)),
            
            # Full text
            'text': '\n\n'.join(paragraphs),
            
            # Structured data
            'paragraphs': paragraphs,
            'sentences': sentences,
            
            # Dialogue markers
            'dialogue_blocks': detect_dialogue_blocks(paragraphs),
            
            # Ready for translation
            'translation_units': sentences,  # Translate sentence-by-sentence
        }
        output['sections'].append(section_data)
    
    # Write JSON
    json_path = out_dir / f"{book_title}__structured.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    return json_path
```

---

## 🧪 **Testing Your Improvements**

### Test Cases to Run
```python
# Test 1: Well-formatted book (baseline)
test_books = [
    ("Pride and Prejudice", "Jane Austen"),  # Should get ~60 chapters
    ("Dracula", "Bram Stoker"),              # Has letters/diary format
    ("Romeo and Juliet", "Shakespeare"),     # Has ACT/SCENE structure
]

# Test 2: Edge cases
edge_cases = [
    ("Grimm's Fairy Tales", "Grimm"),        # Anthology (100+ stories)
    ("Alice in Wonderland", "Carroll"),      # Short chapters
    ("Crime and Punishment", "Dostoyevsky"), # Long chapters, parts
]

# Test 3: Problem books
problem_cases = [
    # Books with known issues
    ("The Complete Works of Shakespeare", "Shakespeare"),  # Huge file
    # Add books you've had trouble with
]

# Run tests
for title, author in test_books:
    result = process_book(title, author, book_dir, out_dir)
    assert len(result['sections']) > 5, f"Too few sections for {title}"
    assert result['warnings'] == [], f"Warnings for {title}: {result['warnings']}"
```

---

## 📈 **Performance Monitoring**

### Add Metrics
```python
import time

def process_book_with_metrics(title, author, book_dir, out_dir):
    start_time = time.time()
    
    result = process_book(title, author, book_dir, out_dir)
    
    # Add performance metrics
    result['metrics'] = {
        'processing_time_seconds': time.time() - start_time,
        'blocks_processed': result.get('blocks_processed', 0),
        'blocks_per_second': result.get('blocks_processed', 0) / max(1, time.time() - start_time)
    }
    
    return result
```

---

## 🎯 **Priority Order for Your Use Case**

### Must-Have (Do First):
1. ✅ Better chapter title capture (#1)
2. ✅ Sentence segmentation (#2)
3. ✅ Audio duration estimation (#6)
4. ✅ JSON output format (#12)

### Should-Have (Do Next):
5. ✅ Dialogue detection (#5)
6. ✅ Content validation (#8)
7. ✅ Remove repeated elements (#3)

### Nice-to-Have (If Time):
8. ✅ OCR support (#7)
9. ✅ Language detection (#11)
10. ✅ Advanced hyphenation (#9)

---

## 🔧 **Quick Integration Guide**

To integrate these improvements:

```python
# In process_book function:
def process_book_enhanced(title, author, book_dir, out_dir):
    # ... existing code ...
    
    # Add after getting blocks:
    blocks = remove_repeated_elements(blocks)  # #3
    
    # Add after splitting sections:
    for i, (sec_title, paras) in enumerate(sections):
        sections[i] = (normalize_chapter_number(sec_title), paras)  # #10
    
    # Add before writing:
    validated = []
    for title, paras in sections:
        issues = validate_section_content(title, paras)  # #8
        if issues:
            result['warnings'].extend(issues)
        validated.append((title, paras))
    
    # Write both formats:
    write_output_file(out_dir, book_title, author, validated)  # Text
    write_json_output(out_dir, book_title, author, validated)  # JSON (#12)
```

---

## 📝 **Next Steps**

1. **Start with Phase 1** (1-2 hours) - immediate improvements
2. **Test on 3-5 books** from your library
3. **Implement Phase 2** if doing audio generation
4. **Iterate based on actual problems** you encounter

Would you like me to help implement any specific enhancement?