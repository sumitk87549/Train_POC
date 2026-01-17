# 📚 Regex & Pattern Matching Quick Reference

## 🎯 **Chapter Detection Patterns Explained**

### **Basic Chapter Patterns**

```python
# Pattern: r'^\s*CHAPTER\b'
# Explanation:
#   ^       = Start of line
#   \s*     = Zero or more whitespace
#   CHAPTER = Literal text "CHAPTER"
#   \b      = Word boundary (prevents matching "CHAPTERS")

# Matches:
✅ "CHAPTER"
✅ "  CHAPTER"
✅ "CHAPTER 1"
❌ "CHAPTERS"  # Plural
❌ "In CHAPTER 5"  # Mid-sentence
```

### **Numbered Chapters**

```python
# Pattern: r'^\s*CHAPTER\s+(\d+)\b'
# Explanation:
#   \s+     = One or more whitespace (required space)
#   (\d+)   = Capture group: one or more digits
#   \b      = Word boundary

# Matches:
✅ "CHAPTER 1"
✅ "CHAPTER 42"
✅ "CHAPTER  999"  # Extra spaces OK
❌ "CHAPTER1"  # No space
❌ "CHAPTER"  # No number
```

### **Roman Numerals**

```python
# Pattern: r'^\s*CHAPTER\s+([IVXLCDM]+)\b'
# Explanation:
#   [IVXLCDM]+  = One or more Roman numeral characters

# Matches:
✅ "CHAPTER I"
✅ "CHAPTER XIV"
✅ "CHAPTER XXIII"
❌ "CHAPTER ABC"  # Not Roman numerals
```

### **Optional Decorations**

```python
# Pattern: r'^[\s*=-]*CHAPTER'
# Explanation:
#   [\s*=-]*  = Zero or more of: space, asterisk, equals, dash

# Matches:
✅ "CHAPTER 1"
✅ "***CHAPTER 1***"
✅ "=== CHAPTER 1 ==="
✅ "- - CHAPTER 1 - -"
```

---

## 🔍 **Common Issues & Solutions**

### **Issue 1: False Positive - Scene Breaks**

```python
# Problem: Detecting "*****" as a chapter
Text: "* * * * *"

# Solution: Reject pure punctuation
if re.match(r'^[*\-_·•–—=~#\s]+$', text):
    return None  # Not a heading

# Test cases:
"* * * * *"      → None ✅
"CHAPTER 1"      → "CHAPTER 1" ✅
"====="          → None ✅
```

### **Issue 2: False Positive - Emphasis**

```python
# Problem: Detecting "VERY IMPORTANT" as a heading
Text: "THIS IS VERY IMPORTANT TO NOTE"

# Solution: Require structural keywords
if SECTION_REGEX.search(text):
    return text  # Has CHAPTER, PROLOGUE, etc.
else:
    return None  # Just emphasized text

# Test cases:
"IMPORTANT NOTE"        → None ✅
"AUTHOR'S NOTE"         → "AUTHOR'S NOTE" ✅
"CHAPTER NOTE"          → "CHAPTER NOTE" ✅
```

### **Issue 3: Mid-Sentence References**

```python
# Problem: "He read CHAPTER 5 yesterday"
Text: "He read CHAPTER 5 yesterday"

# Solution: Require start of line (^)
Pattern: r'^\s*CHAPTER'  # ^ prevents mid-sentence matches

# Test cases:
"CHAPTER 5"                → Match ✅
"He read CHAPTER 5"        → No match ✅
"  CHAPTER 5"              → Match ✅
```

### **Issue 4: Chapter Titles on Next Line**

```python
# Problem:
Line 1: "CHAPTER I"
Line 2: "THE BEGINNING"
Line 3: "It was a dark..."

# Current: Only captures "CHAPTER I"
# Desired: "CHAPTER I: THE BEGINNING"

# Solution:
if CHAPTER_TITLE_PATTERN.match(heading) and idx + 1 < len(blocks):
    next_line = blocks[idx + 1].strip()
    
    # Check if next line is a title
    if (len(next_line) < 100 and           # Not too long
        len(next_line.split()) <= 10 and   # 10 words max
        not is_section_heading(next_line) and  # Not another heading
        next_line[0].isupper()):           # Starts with capital
        
        heading = f"{heading}: {next_line.upper()}"
        skip_next = True  # Don't process title as paragraph
```

### **Issue 5: Hyphenation Across Lines**

```python
# Problem in PDFs:
"The beauti-
ful sunset"

# Should be: "The beautiful sunset"

# Solution:
text = re.sub(r'-\s*\n\s*', '', text)

# Explanation:
#   -       = Literal hyphen
#   \s*     = Optional whitespace
#   \n      = Newline
#   \s*     = Optional whitespace
# Replace with: nothing (empty string)

# Test cases:
"beauti-\nful"    → "beautiful" ✅
"well-known"      → "well-known" ✅ (keep valid hyphens)
```

---

## 🎨 **Pattern Templates**

### **Template 1: Match Anything at Start**
```python
r'^\s*YOUR_KEYWORD\b'

# Use for: PROLOGUE, EPILOGUE, FOREWORD, etc.
# Examples:
r'^\s*PROLOGUE\b'
r'^\s*FOREWORD\b'
r'^\s*INTRODUCTION\b'
```

### **Template 2: Match with Number**
```python
r'^\s*YOUR_KEYWORD\s+(\d+)\b'

# Use for: numbered chapters, parts, acts
# Examples:
r'^\s*CHAPTER\s+(\d+)\b'
r'^\s*PART\s+(\d+)\b'
r'^\s*ACT\s+(\d+)\b'
```

### **Template 3: Match with Roman/Arabic**
```python
r'^\s*YOUR_KEYWORD\s+([IVXLCDM\d]+)\b'

# Use for: flexible numbering
# Examples:
r'^\s*BOOK\s+([IVXLCDM\d]+)\b'  # BOOK I or BOOK 1
r'^\s*SCENE\s+([IVXLCDM\d]+)\b'
```

### **Template 4: Match Possessive Forms**
```python
r"^\s*AUTHOR['\u2019]?S NOTE\b"

# Explanation:
#   ['\u2019]?  = Optional apostrophe or smart quote
#   ?           = Zero or one (makes it optional)

# Matches:
✅ "AUTHOR'S NOTE"    # Straight quote
✅ "AUTHORS NOTE"     # No apostrophe
✅ "AUTHOR'S NOTE"   # Smart quote (unicode)
```

---

## 🧹 **Cleaning Patterns**

### **Gutenberg Markers**

```python
# Start markers
r'\*{3,}\s*START\s+(?:OF\s+)?(?:THIS|THE)\s+PROJECT\s+GUTENBERG'

# Explanation:
#   \*{3,}      = Three or more asterisks
#   \s*         = Optional whitespace
#   (?:OF\s+)?  = Optional "OF " (non-capturing group)
#   (?:THIS|THE) = Either "THIS" or "THE"

# Matches:
✅ "*** START OF THIS PROJECT GUTENBERG EBOOK"
✅ "***START OF THE PROJECT GUTENBERG EBOOK"
✅ "*****START OF PROJECT GUTENBERG EBOOK"
✅ "*** START THIS PROJECT GUTENBERG EBOOK"
```

### **Metadata Lines**

```python
# Pattern for title/author metadata
r'^(Title|Author|Release Date|Language)\s*:'

# Explanation:
#   ^           = Start of line
#   (...)       = Group of alternatives
#   \s*         = Optional whitespace
#   :           = Literal colon

# Matches:
✅ "Title: Pride and Prejudice"
✅ "Author:Jane Austen"
✅ "Release Date : 1998"
❌ "The Title: is important"  # Not at start
```

### **URLs and Domains**

```python
# Pattern for Gutenberg URLs
r'www\.gutenberg\.org'

# Explanation:
#   \.  = Escaped dot (literal period)

# Also matches:
- "http://www.gutenberg.org"
- "Visit www.gutenberg.org for more"
- "www.gutenberg.org/ebooks/1342"
```

---

## 🔢 **Number Conversion Patterns**

### **Roman to Arabic**

```python
roman_values = {
    'I': 1, 'V': 5, 'X': 10, 'L': 50,
    'C': 100, 'D': 500, 'M': 1000
}

def roman_to_int(roman):
    """Convert Roman numeral to integer"""
    result = 0
    prev_value = 0
    
    for char in reversed(roman):
        value = roman_values[char]
        if value < prev_value:
            result -= value  # IV = 5-1 = 4
        else:
            result += value
        prev_value = value
    
    return result

# Test:
roman_to_int("XIV")   → 14 ✅
roman_to_int("XLII")  → 42 ✅
roman_to_int("MCMXC") → 1990 ✅
```

### **Word to Number**

```python
word_to_num = {
    'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5,
    'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9, 'TEN': 10,
    'ELEVEN': 11, 'TWELVE': 12, 'THIRTEEN': 13,
    'TWENTY': 20, 'THIRTY': 30, 'FORTY': 40, 'FIFTY': 50
}

# Pattern to extract:
r'CHAPTER\s+(ONE|TWO|THREE|FOUR|FIVE)\b'
```

---

## 🎭 **Dialogue Detection**

### **Pattern 1: Standard Quotes**

```python
# Pattern: r'^\s*["\'"]'
# Explanation:
#   ^\s*   = Start of line + optional whitespace
#   [...]  = Character class
#   "      = Straight double quote
#   '      = Straight single quote
#   "      = Smart double quote (unicode)
#   '      = Smart single quote (unicode)

# Matches:
✅ "Hello," she said.
✅ 'Stop!' he yelled.
✅ "Smart quotes" work too.
❌ He said "hello"  # Mid-sentence
```

### **Pattern 2: Speech Tags**

```python
speech_verbs = [
    'said', 'asked', 'replied', 'answered',
    'whispered', 'shouted', 'yelled', 'cried',
    'muttered', 'exclaimed', 'demanded'
]

# Pattern:
r'\b(' + '|'.join(speech_verbs) + r')\b'

# Usage:
if re.search(pattern, paragraph.lower()):
    is_dialogue = True
```

---

## 🧪 **Testing Your Patterns**

### **Test Framework**

```python
def test_pattern(pattern, test_cases):
    """Test regex pattern against cases"""
    regex = re.compile(pattern, re.IGNORECASE)
    
    for text, should_match in test_cases:
        matches = bool(regex.search(text))
        status = "✅" if matches == should_match else "❌"
        print(f"{status} '{text}' -> {matches} (expected {should_match})")

# Example usage:
test_cases = [
    ("CHAPTER 1", True),
    ("CHAPTER I", True),
    ("Chapter One", True),
    ("CHAPTERS", False),
    ("He read CHAPTER 5", False),
    ("* * * * *", False),
]

test_pattern(r'^\s*CHAPTER\b', test_cases)
```

---

## 📋 **Common Regex Symbols Cheat Sheet**

```
^       Start of line/string
$       End of line/string
.       Any character (except newline)
*       0 or more of previous
+       1 or more of previous
?       0 or 1 of previous (optional)
{n}     Exactly n of previous
{n,}    n or more of previous
{n,m}   Between n and m of previous

\s      Whitespace (space, tab, newline)
\S      Non-whitespace
\d      Digit (0-9)
\D      Non-digit
\w      Word character (a-z, A-Z, 0-9, _)
\W      Non-word character
\b      Word boundary

[]      Character class
[^]     Negated character class
()      Capture group
(?:)    Non-capturing group
|       OR operator

\       Escape special character
```

---

## 🎯 **Priority Patterns for Your Use Case**

### **Must-Have Patterns:**

1. **Chapter detection** - Core functionality
   ```python
   r'^\s*CHAPTER\s+[IVXLCDM\d]+\b'
   ```

2. **Remove Gutenberg** - Clean public domain books
   ```python
   r'\*{3,}\s*(?:START|END)\s+(?:OF\s+)?(?:THIS|THE)\s+PROJECT\s+GUTENBERG'
   ```

3. **Fix hyphenation** - PDF line breaks
   ```python
   r'-\s*\n\s*'
   ```

### **Should-Have Patterns:**

4. **Dialogue detection** - For voice switching in TTS
   ```python
   r'^\s*["\'"]' 
   ```

5. **Page numbers** - Remove from PDFs
   ```python
   r'^\s*\d+\s*$'
   ```

6. **Metadata lines** - Remove title/author
   ```python
   r'^(Title|Author|Release Date)\s*:'
   ```

---

## 🔧 **Debugging Regex**

### **Online Tools:**
- https://regex101.com - Best for testing and explanation
- https://regexr.com - Visual regex tester
- https://pythex.org - Python-specific tester

### **In Python:**

```python
import re

# Test pattern
pattern = r'^\s*CHAPTER\s+(\d+)\b'
text = "CHAPTER 42"

# Find matches
match = re.search(pattern, text, re.IGNORECASE)
if match:
    print(f"Matched: {match.group(0)}")  # Full match
    print(f"Captured: {match.group(1)}") # First group (42)
    
# Find all
all_matches = re.findall(pattern, "CHAPTER 1\nCHAPTER 2", re.MULTILINE)
print(all_matches)  # ['1', '2']
```

---

## 💡 **Pro Tips**

1. **Always use raw strings** for regex: `r'...'`
2. **Test on real data** from your books
3. **Start simple**, add complexity as needed
4. **Use verbose mode** for complex patterns:
   ```python
   pattern = re.compile(r'''
       ^\s*              # Start of line, optional whitespace
       CHAPTER           # Literal word CHAPTER
       \s+               # Required whitespace
       ([IVXLCDM\d]+)    # Capture: Roman or Arabic numeral
       \b                # Word boundary
   ''', re.VERBOSE | re.IGNORECASE)
   ```

5. **Capture groups carefully** - use `(?:...)` for non-capturing
6. **Unicode matters** - use `\u2019` for smart quotes

Would you like me to explain any specific pattern in more detail or help you create a custom pattern for your books?