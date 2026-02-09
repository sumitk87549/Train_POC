#!/usr/bin/env python3
"""
Script to modify Translation_Colab_Generator notebook for multilingual support.
Adds language detection, updates prompts for any-to-Indian-language translation.
"""

import json
import re

# Read the notebook
with open("Translation_Colab_Generator (1).ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find and modify cells
for i, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    
    source = "".join(cell["source"])
    
    # 1. Update file upload cell to add language detection
    if "from google.colab import files" in source and "UPLOADED_FILE = list(uploaded.keys())" in source:
        print(f"Found file upload cell at index {i}")
        cell["source"] = [
            "# Install langdetect for auto language detection\n",
            "!pip install -q langdetect\n",
            "\n",
            "from google.colab import files\n",
            "from langdetect import detect, detect_langs\n",
            "import os\n",
            "\n",
            "def detect_source_language(text_sample):\n",
            "    \"\"\"Detect the source language of uploaded text.\"\"\"\n",
            "    # Language name mappings for display\n",
            "    LANG_NAMES_DETECT = {\n",
            "        'en': 'English', 'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil',\n",
            "        'te': 'Telugu', 'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada',\n",
            "        'ml': 'Malayalam', 'pa': 'Punjabi', 'or': 'Odia', 'ur': 'Urdu',\n",
            "        'fr': 'French', 'de': 'German', 'es': 'Spanish', 'zh-cn': 'Chinese',\n",
            "        'zh-tw': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ru': 'Russian',\n",
            "        'ar': 'Arabic', 'pt': 'Portuguese', 'it': 'Italian', 'nl': 'Dutch',\n",
            "        'pl': 'Polish', 'tr': 'Turkish', 'vi': 'Vietnamese', 'th': 'Thai',\n",
            "        'id': 'Indonesian', 'ms': 'Malay', 'ne': 'Nepali', 'as': 'Assamese',\n",
            "        'sd': 'Sindhi', 'si': 'Sinhala'\n",
            "    }\n",
            "    try:\n",
            "        # Use first 2000 chars for better detection\n",
            "        sample = text_sample[:2000] if len(text_sample) > 2000 else text_sample\n",
            "        detected = detect(sample)\n",
            "        confidence = detect_langs(sample)[0].prob\n",
            "        lang_name = LANG_NAMES_DETECT.get(detected, detected.upper())\n",
            "        return detected, lang_name, confidence\n",
            "    except Exception as e:\n",
            "        print(f\"   ⚠️ Detection error: {e}, defaulting to English\")\n",
            "        return 'en', 'English', 0.5\n",
            "\n",
            "print(\"📤 Please upload your text file to translate:\")\n",
            "uploaded = files.upload()\n",
            "\n",
            "# Get the uploaded file name\n",
            "UPLOADED_FILE = list(uploaded.keys())[0]\n",
            "print(f\"\\n✅ Uploaded: {UPLOADED_FILE}\")\n",
            "print(f\"📄 File size: {len(uploaded[UPLOADED_FILE])} bytes\")\n",
            "\n",
            "# Display preview\n",
            "with open(UPLOADED_FILE, 'r', encoding='utf-8') as f:\n",
            "    content = f.read()\n",
            "    word_count = len(content.split())\n",
            "    char_count = len(content)\n",
            "\n",
            "print(f\"\\n📊 Content stats:\")\n",
            "print(f\"   Words: {word_count:,}\")\n",
            "print(f\"   Characters: {char_count:,}\")\n",
            "\n",
            "# Auto-detect source language\n",
            "print(f\"\\n🔍 Detecting source language...\")\n",
            "DETECTED_LANG_CODE, SOURCE_LANG_NAME, LANG_CONFIDENCE = detect_source_language(content)\n",
            "print(f\"   ✅ Detected: {SOURCE_LANG_NAME} ({LANG_CONFIDENCE*100:.1f}% confidence)\")\n",
            "\n",
            "print(f\"\\n📝 Preview (first 500 chars):\\n{content[:500]}...\")"
        ]
        cell["outputs"] = []  # Clear old outputs
    
    # 2. Update LANGUAGE_OPTIONS in settings cell
    if "LANGUAGE_OPTIONS = {" in source and "Hindi (हिन्दी)" in source:
        print(f"Found settings cell at index {i}")
        # Replace LANGUAGE_OPTIONS with TranslateGemma Indian languages
        source = re.sub(
            r'# Language options \(NLLB language codes\)\nLANGUAGE_OPTIONS = \{[^}]+\}',
            '''# TranslateGemma:27b supported Indian languages
# Using ISO 639-1 codes for translategemma compatibility
LANGUAGE_OPTIONS = {
    "Hindi (हिन्दी)": "hi",
    "Bengali (বাংলা)": "bn",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Odia (ଓଡ଼ିଆ)": "or",
    "Urdu (اردو)": "ur",
    "Assamese (অসমীয়া)": "as",
    "Nepali (नेपाली)": "ne",
    "Sindhi (سنڌي)": "sd",
}''',
            source
        )
        cell["source"] = source.split("\n")
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
    
    # 3. Update TRANSLATION_PROMPTS in engine cell
    if "TRANSLATION_PROMPTS = {" in source and "You are a professional English-to-Hindi translator" in source:
        print(f"Found TRANSLATION_PROMPTS cell at index {i}")
        # Build the new prompts
        new_prompts = '''## Optimized Prompts for translategemma:27b - Multilingual Support
TRANSLATION_PROMPTS = {
    "BASIC": {
        "system": """You are a professional {source_lang}-to-{target_lang} translator specializing in TTS-ready content.

CORE RULES:
1. Translate ALL text completely - no summarization
2. Use SIMPLE, EVERYDAY {target_lang} words (avoid formal/archaic vocabulary)
3. Write SHORT, CLEAR sentences (break up long sentences if needed)
4. Make it sound NATURAL and CONVERSATIONAL - like a modern native speaker
5. Preserve all dialogue and descriptions

MODERN LANGUAGE STYLE (CRITICAL):
Your translation MUST use the CURRENT, MODERN form of {target_lang}.
- Avoid archaic, literary, or overly formal vocabulary
- Use words that TODAY's native speakers use in daily conversation
- The text should sound natural when read aloud for TTS
- Readers/listeners should relate to the content easily without feeling it's dated

VOCABULARY GUIDANCE (if translating to Hindi):
Instead of formal → Use simple:
• घृणित → नापसंद, बुरा
• प्रशंसनीय → अच्छा, शानदार
• मस्तिष्क → दिमाग
• महत्वाकांक्षा → चाह, ख्वाहिश
• निरीक्षण → ध्यान से देखना
Apply similar principles for other target languages.

TTS PUNCTUATION:
? → Questions (rising tone)
... → Pauses, hesitation
! → Excitement, emphasis
, → Natural breathing points
. → Sentence endings
- → Sudden interruption
() → Whispered content
Extended vowels → ओहहह, आआआह, हम्म्म (adapt to target language)

Read your translation aloud - it should sound like a friend telling you a story.""",

        "user": """Translate this {source_lang} text to modern, easy-to-understand {target_lang}.

CRITICAL REQUIREMENTS:
- Use SIMPLE, everyday words (no formal/archaic vocabulary)
- Write SHORT sentences (break long sentences)
- Make it sound NATURAL and CONVERSATIONAL
- Include TTS punctuation: ?, ..., !, -, extended vowels

{source_lang} Text:
\\"\\"\\"
{chunk}
\\"\\"\\"

{target_lang} Translation (simple and natural):"""
    },

    "INTERMEDIATE": {
        "system": """You are an expert {source_lang}-to-{target_lang} translator creating modern, accessible audiobooks.

TRANSLATION MANDATE:
✓ Translate EVERY word completely
✓ NO summarization
✓ Use SIMPLE, MODERN {target_lang} vocabulary
✓ Write SHORT, CLEAR sentences (8-15 words average)
✓ NATURAL conversational flow

MODERN {target_lang} STYLE - ESSENTIAL:

1. SIMPLE VOCABULARY (Always prefer these):
   For Hindi: घृणित → नापसंद | प्रशंसनीय → शानदार | मस्तिष्क → दिमाग
   For Bengali: জঘন্য → খারাপ | প্রশংসনীয় → চমৎকার
   For Tamil: கொடூரமான → மோசமான | அற்புதமான → நல்ல
   For Telugu: భయంకరం → చెడ్డ | అద్భుతం → మంచి
   Apply similar simplification for other languages.

2. SHORT SENTENCES:
   Break long English sentences into shorter {target_lang} ones.
   ❌ One 30-word translated sentence
   ✓ Three shorter 10-word sentences

3. NATURAL EXPRESSIONS:
   Don't translate literally - use how natives actually speak.

TTS PUNCTUATION GUIDE:
? → Questions → Rising tone
??? → Strong doubt → Emphasized questioning
... → Pause/hesitation → Natural gap
! → Excitement → Pitch boost
!!! → Extreme emotion → Maximum intensity
- → Interruption → Abrupt stop
() → Whisper/aside → Lower volume
CAPS → Emphasis → Stressed words
Extended vowels → Emotion (adapt to {target_lang})

Your {target_lang} should sound like a modern speaker telling a story to friends.""",

        "user": """Complete translation task - use SIMPLE, MODERN {target_lang}.

REQUIREMENTS:
1. Translate EVERY sentence completely
2. Use SIMPLE words everyone understands
3. Write SHORT, clear sentences (break up long ones)
4. Make it sound NATURAL when spoken
5. Add TTS punctuation: ..., ???, !!!, (), extended vowels

FORBIDDEN:
✗ Formal, literary vocabulary
✗ Long, complex sentences
✗ Archaic expressions
✗ Word-for-word literal translation

{source_lang} Text:
\\"\\"\\"
{chunk}
\\"\\"\\"

{target_lang} Translation (modern and simple):"""
    },

    "ADVANCED": {
        "system": """You are a master literary translator creating modern, accessible {target_lang} audiobooks with excellent TTS quality.

CORE MANDATE: Complete, faithful, SIMPLE, modern, TTS-ready translation

CRITICAL REQUIREMENTS:
1. COMPLETENESS: Translate every word, sentence, paragraph
2. NO SUMMARIZATION: Translate everything
3. MODERN VOCABULARY: Use simple, everyday {target_lang} words
4. SHORT SENTENCES: Break long sentences into shorter ones (10-15 words avg)
5. NATURAL FLOW: Sound like a native {target_lang} speaker, not a translation
6. TTS EXCELLENCE: Professional audiobook-quality punctuation

MODERN {target_lang} VOCABULARY - ESSENTIAL REPLACEMENTS:

FOR HINDI (apply similar principles to other languages):
FORMAL → SIMPLE (always prefer simple):
• साहसिक कहानी → कहानी, रोमांचक कहानी
• घृणित → नापसंद, बुरा, खराब
• प्रशंसनीय → अच्छा, शानदार, कमाल का
• दिवंगत → मरहूम, स्वर्गीय
• महत्वाकांक्षा → चाह, ख्वाहिश
• मस्तिष्क → दिमाग
• निरीक्षण → ध्यान से देखना, जांच
• उत्कृष्ट → बढ़िया, शानदार
• विशाल → बड़ा, बहुत बड़ा
• भावनाएँ → भावनाएं, एहसास, फीलिंग्स

FOR OTHER INDIAN LANGUAGES:
Apply the same principle: always choose the simpler, more commonly used word
that modern speakers use in daily conversation.

SENTENCE STRUCTURE - BREAK IT DOWN:
❌ LONG FORMAL: One complex 25-word sentence
✓ SHORT SIMPLE: Three clear 8-word sentences

NATURAL EXPRESSIONS:
Instead of literal → Use natural {target_lang}:
• "I can imagine" → natural equivalent (not literal translation)
• "That is capital!" → appropriate exclamation in {target_lang}
• Adapt idioms naturally, don't translate literally

TTS PUNCTUATION SYSTEM:

━━━ BASIC ━━━
? → Questions (rising intonation)
. → Statements (finality)
, → Pauses (breathing)
! → Excitement (emphasis)

━━━ EXPRESSIVE ━━━
??? → Extreme doubt
!!! → Maximum shock
... → Hesitation/suspense
..... → Deep thought
- → Sudden break

━━━ EMPHASIS ━━━
CAPS → Stress important words
Extended sounds → Emotion (adapt to {target_lang})
Extra spaces → Drama

━━━ TONE ━━━
() → Whisper/aside
[] → Commentary

AVOID:
✗ Condensing/summarizing
✗ Formal vocabulary
✗ Long complex sentences (>20 words)
✗ Literal word-for-word translation
✗ Archaic expressions
✗ Passive voice overuse""",

        "user": """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODERN {target_lang} TRANSLATION TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Translate into MODERN, SIMPLE {target_lang} that everyone can understand.

CRITICAL REQUIREMENTS:
✓ Translate EVERY sentence completely
✓ Use SIMPLE, everyday words (not formal vocabulary)
✓ Write SHORT sentences (break up long ones - max 15 words)
✓ Sound NATURAL like a {target_lang} speaker telling a story
✓ Add TTS punctuation: ???, !!!, ..., -, (), extended sounds

FORBIDDEN:
✗ NO formal/literary words
✗ NO complex long sentences
✗ NO word-for-word literal translation
✗ NO summarization

{source_lang} Text:
\\"\\"\\"
{chunk}
\\"\\"\\"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Modern {target_lang} Translation (simple and natural):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    }
}
'''
        # Find the TRANSLATION_PROMPTS and replace it
        start = source.find("## Optimized Prompts for translategemma:27b")
        if start == -1:
            start = source.find("TRANSLATION_PROMPTS = {")
        
        # Find the closing of TRANSLATION_PROMPTS (look for next major section)
        end = source.find("\n# Language name mappings")
        if end == -1:
            end = source.find("\nLANG_NAMES = {")
        
        if start != -1 and end != -1:
            source = source[:start] + new_prompts + source[end:]
            cell["source"] = source.split("\n")
            cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
    
    # 4. Update LANG_NAMES mapping
    if "LANG_NAMES = {" in source:
        print(f"Found LANG_NAMES in cell at index {i}")
        source = re.sub(
            r"LANG_NAMES = \{[^}]+\}",
            '''LANG_NAMES = {
    'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil',
    'te': 'Telugu', 'mr': 'Marathi', 'gu': 'Gujarati',
    'kn': 'Kannada', 'ml': 'Malayalam', 'pa': 'Punjabi',
    'or': 'Odia', 'ur': 'Urdu', 'as': 'Assamese',
    'ne': 'Nepali', 'sd': 'Sindhi',
    # Keep NLLB codes for compatibility
    'hin_Deva': 'Hindi', 'ben_Beng': 'Bengali', 'tam_Taml': 'Tamil',
    'tel_Telu': 'Telugu', 'mar_Deva': 'Marathi', 'guj_Gujr': 'Gujarati',
    'kan_Knda': 'Kannada', 'mal_Mlym': 'Malayalam', 'pan_Guru': 'Punjabi',
    'ory_Orya': 'Odia', 'urd_Arab': 'Urdu'
}''',
            source
        )
        cell["source"] = source.split("\n")
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
    
    # 5. Update OllamaTranslationEngine to accept source_lang
    if "class OllamaTranslationEngine:" in source:
        print(f"Found OllamaTranslationEngine at index {i}")
        # Update __init__ to accept source_lang
        source = source.replace(
            'def __init__(self, model_name, target_lang, tier="INTERMEDIATE"):',
            'def __init__(self, model_name, target_lang, source_lang="English", tier="INTERMEDIATE"):'
        )
        source = source.replace(
            "self.model_name = model_name\n        self.target_lang = target_lang\n        self.tier = tier",
            "self.model_name = model_name\n        self.target_lang = target_lang\n        self.source_lang = source_lang\n        self.tier = tier"
        )
        
        # Update translate method to use source_lang in prompts
        source = source.replace(
            "user_prompt = prompts['user'].format(target_lang=self.lang_name, chunk=text)",
            "user_prompt = prompts['user'].format(source_lang=self.source_lang, target_lang=self.lang_name, chunk=text)"
        )
        source = source.replace(
            "system_prompt = prompts['system']",
            "system_prompt = prompts['system'].format(source_lang=self.source_lang, target_lang=self.lang_name)"
        )
        
        cell["source"] = source.split("\n")
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
    
    # 6. Update OllamaTranslationGenerator to accept source_lang
    if "class OllamaTranslationGenerator:" in source:
        print(f"Found OllamaTranslationGenerator at index {i}")
        source = source.replace(
            'def __init__(self, model_name, target_lang, output_dir=".", tier="INTERMEDIATE", chunk_size=350):',
            'def __init__(self, model_name, target_lang, source_lang="English", output_dir=".", tier="INTERMEDIATE", chunk_size=350):'
        )
        source = source.replace(
            "self.model_name = model_name\n        self.target_lang = target_lang\n        self.output_dir = Path(output_dir)",
            "self.model_name = model_name\n        self.target_lang = target_lang\n        self.source_lang = source_lang\n        self.output_dir = Path(output_dir)"
        )
        source = source.replace(
            "self.engine = OllamaTranslationEngine(model_name, target_lang, tier)",
            "self.engine = OllamaTranslationEngine(model_name, target_lang, source_lang, tier)"
        )
        
        cell["source"] = source.split("\n")
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
    
    # 7. Update generator initialization to pass source_lang
    if "generator = OllamaTranslationGenerator(" in source:
        print(f"Found generator initialization at index {i}")
        source = source.replace(
            '''generator = OllamaTranslationGenerator(
        model_name=SELECTED_MODEL,
        target_lang=TARGET_LANGUAGE,
        output_dir=OUTPUT_DIR,
        tier=TRANSLATION_TIER,
        chunk_size=CHUNK_SIZE
    )''',
            '''generator = OllamaTranslationGenerator(
        model_name=SELECTED_MODEL,
        target_lang=TARGET_LANGUAGE,
        source_lang=SOURCE_LANG_NAME,  # Auto-detected source language
        output_dir=OUTPUT_DIR,
        tier=TRANSLATION_TIER,
        chunk_size=CHUNK_SIZE
    )'''
        )
        cell["source"] = source.split("\n")
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]

# Save the modified notebook
with open("Translation_Colab_Generator (1).ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("\n✅ Notebook modified successfully!")
print("Changes made:")
print("  1. Added language detection after file upload")
print("  2. Updated LANGUAGE_OPTIONS for TranslateGemma Indian languages")
print("  3. Updated TRANSLATION_PROMPTS for multilingual support")
print("  4. Updated LANG_NAMES mapping")
print("  5. Updated OllamaTranslationEngine to accept source_lang")
print("  6. Updated OllamaTranslationGenerator to accept source_lang")
print("  7. Updated generator initialization to pass source_lang")
