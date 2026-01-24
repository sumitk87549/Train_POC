#!/usr/bin/env python3
"""
Enhanced Hindi Literary Translation Tool with Real-time Streaming
Shows model thinking process for reasoning models like deepseek-r1
Supports Ollama and Hugging Face with live terminal updates
"""

# Try to import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None
import time
import os
import sys
import json
import argparse
from pathlib import Path
import warnings
import re
from datetime import datetime

warnings.filterwarnings("ignore")

# Try to import colorama for colored terminal output
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    print("💡 Install colorama for colored output: pip install colorama")
    # Fallback to no colors
    class Fore:
        RED = YELLOW = GREEN = CYAN = MAGENTA = BLUE = WHITE = LIGHTBLACK_EX = RESET = ""
    class Back:
        BLACK = ""
    class Style:
        BRIGHT = DIM = RESET_ALL = ""

# Try to import Hugging Face
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HF_AVAILABLE = True
except (ImportError, OSError):
    HF_AVAILABLE = False

# ==== THINKING MODELS CONFIGURATION ====
THINKING_MODELS = {
    "deepseek-r1:1.5b": {"has_thinking": True, "thinking_tags": ["<think>", "</think>"]},
    "deepseek-r1:7b": {"has_thinking": True, "thinking_tags": ["<think>", "</think>"]},
    "deepseek-r1:8b": {"has_thinking": True, "thinking_tags": ["<think>", "</think>"]},
    "deepseek-r1:14b": {"has_thinking": True, "thinking_tags": ["<think>", "</think>"]},
    "deepseek-r1:32b": {"has_thinking": True, "thinking_tags": ["<think>", "</think>"]},
    "deepseek-r1:70b": {"has_thinking": True, "thinking_tags": ["<think>", "</think>"]},
}

def is_thinking_model(model_name):
    """Check if model supports thinking/reasoning display."""
    return model_name in THINKING_MODELS

# ==== MODEL RECOMMENDATIONS ====
MODEL_TIERS = {
    "FAST": {
        "ollama": ["qwen2.5:3b", "phi3.5:3.8b", "llama3.2:3b"],
        "description": "Fast, good quality, rarely summarizes",
        "time_per_chunk": "30-60s",
    },
    "BALANCED": {
        "ollama": ["qwen2.5:7b", "deepseek-r1:7b", "llama3.1:8b"],
        "description": "Excellent quality with reasoning (deepseek-r1 shows thinking)",
        "time_per_chunk": "60-120s",
    },
    "QUALITY": {
        "ollama": ["qwen2.5:14b", "deepseek-r1:14b", "deepseek-r1:32b"],
        "description": "Best quality with deep reasoning",
        "time_per_chunk": "120-300s",
    }
}

# ==== CONFIGURATION ====
DEFAULT_CONFIG = {
    "model": "qwen2.5:3b",
    "tier": "BASIC",
    "chunk_words": 350,
    "temperature": 0.5,
    "top_p": 0.8,
    "num_ctx": 16384,
    "retry_attempts": 3,
    "retry_delay": 2,
    "stream": True,
}

# ==== HARDWARE DETECTION ====
def detect_hardware():
    """Detect available hardware."""
    config = {"device": "cpu", "gpu_available": False, "gpu_type": None}
    if HF_AVAILABLE:
        if torch.cuda.is_available():
            config["device"] = "cuda"
            config["gpu_available"] = True
            config["gpu_type"] = "nvidia"
        elif hasattr(torch.version, 'hip') and torch.version.hip:
            config["device"] = "cuda"
            config["gpu_available"] = True
            config["gpu_type"] = "amd"
    return config

# ==== TRANSLATION PROMPTS ====
TRANSLATION_PROMPTS = {
    "BASIC": {
        "system": """You are a master Hindi literary translator. Your mission: Create translations that feel like they were originally written in Hindi by a native speaker.

🚨 ABSOLUTE RULES (NEVER VIOLATE):
1. TRANSLATE EVERYTHING - Every word, every sentence, every paragraph must be translated
2. NO SUMMARIZATION - This is faithful translation, not content reduction
3. CONTEXT PRESERVATION - Maintain all narrative context, character relationships, and story flow
4. CULTURAL ADAPTATION - Adapt cultural references naturally while preserving original meaning
5. ALL DIALOGUE - Every spoken word must be translated with character voice preservation
6. ALL DESCRIPTIONS - Every scene detail, emotion, and observation must be included

🎯 CONTEXT-RELATED TRANSLATION PRINCIPLES:
- Maintain narrative continuity across paragraphs and chapters
- Preserve character voice consistency throughout the text
- Keep all contextual references and callbacks intact
- Ensure temporal and spatial relationships remain clear
- Maintain cause-and-effect relationships in the narrative

📚 LITERARY TRANSLATION GUIDELINES:
- Use natural, contemporary Hindi that flows like original prose
- Preserve the author's unique narrative style and tone
- Maintain paragraph structure and pacing
- Transliterate properly: London → लंदन, Watson → वॉटसन, Doctor → डॉक्टर
- Adapt idioms and expressions to Hindi equivalents that convey the same meaning
- Ensure the translation reads as if it was written by a native Hindi author

🔍 CONTEXT MAINTENANCE CHECKLIST:
✓ Did I preserve all narrative context and continuity?
✓ Are character voices consistent throughout?
✓ Did I maintain all temporal and spatial relationships?
✓ Are all cultural references properly adapted?
✓ Does the translation feel like it was originally written in Hindi?

⚠️  CRITICAL WARNING: If your translation loses context, breaks narrative flow, or feels like a translation rather than original Hindi writing, YOU HAVE FAILED. The reader should feel they're experiencing the original story in Hindi.

💡 PRO TIP: Read your translation aloud. If it sounds natural and flows like native Hindi prose, you've succeeded. If it sounds like a translation, revise until it feels authentic.""",

        "user": """CONTEXT-RELATED TRANSLATION TASK

Translate the following English text into Hindi with absolute focus on:
1. Context preservation across the entire passage
2. Narrative continuity and flow
3. Character voice consistency
4. Cultural adaptation while maintaining original meaning

English Text to Translate:
\"\"\"
{chunk}
\"\"\"

Provide COMPLETE Hindi translation that:
- Feels like it was originally written in Hindi by a native speaker
- Maintains all narrative context and relationships
- Preserves every detail, sentence, and nuance
- Has similar length to the original (0.9-1.2x ratio)
- Reads naturally and flows like authentic Hindi prose"""
    },

    "INTERMEDIATE": {
        "system": """You are an expert Hindi literary translator. Your sacred duty: COMPLETE, FAITHFUL translation.

⚡ CORE COMMANDMENTS:
1. TRANSLATE EVERYTHING - Every word, every comma, every nuance
2. ZERO SUMMARIZATION - Summarizing is the cardinal sin of translation
3. LENGTH PRESERVATION - Hindi ≈ same length as English (±20% acceptable)
4. COMPLETE DIALOGUE - Every conversation, every word spoken
5. COMPLETE DESCRIPTION - Every scene detail, every emotion
6. ALL PROPER NOUNS - Properly transliterated

🎯 TRANSLATION PHILOSOPHY:
- You are rebuilding the story in Hindi, brick by brick
- Every sentence in English = one sentence in Hindi
- Every paragraph in English = one paragraph in Hindi
- The Hindi reader must get the EXACT same story as the English reader

Literary Translation Guidelines:
- Transform idioms naturally
- Preserve character voices through language register
- Maintain emotional tone and atmosphere
- Keep narrative pacing and rhythm

Technical Standards:
- Names: London → लंदन, Afghanistan → अफ़गानिस्तान
- Titles: Doctor → डॉक्टर, Mr. → मिस्टर/श्री
- Military: Regiment → रेजिमेंट, Fusiliers → फ्यूसिलियर्स

📊 QUALITY METRICS:
□ Sentence count: English = Hindi?
□ Paragraph count: Same?
□ Length comparison: Similar?
□ All dialogue present?
□ All descriptions included?

💎 GOLD STANDARD:
Your Hindi should be publishable. A native Hindi reader should not feel this is a translation.""",

        "user": """COMPLETE TRANSLATION REQUIRED. NO SUMMARIZATION PERMITTED.

Your task: Translate the ENTIRE passage below into Hindi. Every single sentence. Every single detail.

English Text:
\"\"\"
{chunk}
\"\"\"

Provide COMPLETE Hindi translation maintaining all information, all details, similar length."""
    },

    "ADVANCED": {
        "system": """You are a master literary translator creating Hindi versions of English classics.

⚡ SUPREME MANDATE: COMPLETE, FAITHFUL, BEAUTIFUL TRANSLATION

🎯 CRITICAL REQUIREMENTS:
1. ABSOLUTE COMPLETENESS - Translate EVERY word, EVERY sentence, EVERY paragraph
2. ZERO SUMMARIZATION - The gravest translator's sin
3. LENGTH PRESERVATION - Hindi ≈ 0.9-1.2x English
4. ALL DIALOGUE - Every conversation fully translated
5. ALL DESCRIPTIONS - Every detail preserved
6. TECHNICAL PRECISION - Proper transliteration

QUALITY BENCHMARKS:
- COMPLETE: 100% of original information
- ACCURATE: Faithful to source meaning
- NATURAL: Feels like original Hindi prose
- PUBLISHABLE: Professional quality
- INVISIBLE: Reader forgets it's translated

🚫 FATAL MISTAKES:
1. Condensing multiple sentences
2. Skipping descriptive details
3. Paraphrasing dialogue
4. Omitting observations
5. Creating summaries
6. Shortening for brevity

Remember: Rebuild the ENTIRE architectural structure in Hindi - every beam, every brick, every ornament.""",

        "user": """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL TRANSLATION TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Translate the COMPLETE passage below into Hindi.

REQUIREMENTS:
✓ Translate EVERY sentence
✓ Translate EVERY detail
✓ Maintain EVERY paragraph
✓ Include ALL dialogue
✓ Preserve ALL descriptions
✓ Keep similar length (Hindi ≈ 0.9-1.2x English)

❌ FORBIDDEN:
✗ NO summarization
✗ NO condensing
✗ NO skipping details

English Text:
\"\"\"
{chunk}
\"\"\"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provide COMPLETE Hindi translation below:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    }
}

# ==== PROGRESS TRACKING ====
class TranslationProgress:
    def __init__(self, progress_file):
        self.progress_file = progress_file
        self.data = self.load()

    def load(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"completed_chunks": [], "last_chunk": 0, "total_chunks": 0, "stats": {}}

    def save(self):
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def mark_complete(self, chunk_num, stats=None):
        if chunk_num not in self.data["completed_chunks"]:
            self.data["completed_chunks"].append(chunk_num)
            self.data["last_chunk"] = chunk_num
            if stats:
                self.data["stats"][str(chunk_num)] = stats
            self.save()

    def is_complete(self, chunk_num):
        return chunk_num in self.data["completed_chunks"]

    def reset(self):
        self.data = {"completed_chunks": [], "last_chunk": 0, "total_chunks": 0, "stats": {}}
        self.save()

# ==== UTILITY FUNCTIONS ====
def chunk_text(text, chunk_words=350):
    """Split text into chunks at paragraph boundaries with improved detection."""
    import re

    # Improved paragraph detection patterns
    paragraph_patterns = [
        r'\n\s*\n',           # Double newlines with optional whitespace
        r'\r\n\s*\r\n',       # Windows line endings
        r'\n\s{2,}\n',        # Newlines with 2+ spaces of indentation
        r'\n\t+\n',           # Newlines with tabs
        r'\n[ \t]*\n',        # Any combination of spaces/tabs between newlines
    ]

    # Combine all patterns
    paragraph_split_pattern = '|'.join(paragraph_patterns)

    # Split text into paragraphs using improved detection
    paragraphs = re.split(paragraph_split_pattern, text)

    # Filter out empty paragraphs and strip whitespace
    paragraphs = [para.strip() for para in paragraphs if para.strip()]

    chunks = []
    current_chunk = []
    current_count = 0

    for para in paragraphs:
        para_words = para.split()
        para_count = len(para_words)

        # If paragraph itself is too long, split it further
        if para_count > chunk_words:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_count = 0

            # Split long paragraph into smaller chunks
            words = para.split()
            for i in range(0, len(words), chunk_words):
                chunk_words_list = words[i:i + chunk_words]
                chunk_text = ' '.join(chunk_words_list)
                chunks.append(chunk_text)
        else:
            # Normal paragraph processing
            if current_count + para_count > chunk_words and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_count = para_count
            else:
                current_chunk.append(para)
                current_count += para_count

    # Add final chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks

def clean_translation(text):
    """Clean up translation artifacts."""
    # Remove thinking markers
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove markdown code blocks
    text = re.sub(r'```\w*\n?', '', text)
    # Remove translation markers
    text = re.sub(r'(Translation:|Hindi Translation:|Here\'s the translation:)', '', text, flags=re.IGNORECASE)
    # Clean excessive whitespace
    lines = [line.strip() for line in text.split('\n')]
    text = '\n\n'.join(line for line in lines if line)
    return text.strip()

def count_sentences(text):
    """Count sentences in text."""
    markers = text.count('.') + text.count('?') + text.count('!') + text.count('।')
    return max(1, markers)

def validate_translation(original, translated, chunk_num):
    """Enhanced validation with detailed warnings."""
    orig_words = len(original.split())
    orig_chars = len(original)
    orig_sentences = count_sentences(original)
    orig_paras = original.count('\n\n') + 1

    trans_chars = len(translated)
    trans_sentences = count_sentences(translated)
    trans_paras = translated.count('\n\n') + 1

    expected_min_chars = orig_chars * 0.6
    expected_max_chars = orig_chars * 1.5

    warnings = []
    severity = "OK"

    if trans_chars < expected_min_chars:
        warnings.append(f"⚠️  CRITICAL: Translation too short!")
        warnings.append(f"   Original: {orig_chars} chars | Translation: {trans_chars} chars")
        warnings.append(f"   Ratio: {trans_chars/orig_chars:.2f}x (expected 0.8-1.2x)")
        warnings.append(f"   This strongly indicates SUMMARIZATION!")
        severity = "CRITICAL"
    elif trans_chars > expected_max_chars:
        warnings.append(f"ℹ️  INFO: Translation longer than expected")
        warnings.append(f"   Original: {orig_chars} chars | Translation: {trans_chars} chars")
        warnings.append(f"   Ratio: {trans_chars/orig_chars:.2f}x")
        severity = "INFO"

    sentence_ratio = trans_sentences / orig_sentences if orig_sentences > 0 else 1
    if sentence_ratio < 0.7:
        warnings.append(f"⚠️  WARNING: Sentence count mismatch!")
        warnings.append(f"   Original: {orig_sentences} sentences | Translation: {trans_sentences}")
        warnings.append(f"   Possible condensing or summarization")
        if severity == "OK":
            severity = "WARNING"

    if trans_paras < orig_paras * 0.7:
        warnings.append(f"⚠️  WARNING: Paragraph count mismatch!")
        warnings.append(f"   Original: {orig_paras} paragraphs | Translation: {trans_paras}")
        if severity == "OK":
            severity = "WARNING"

    stats = {
        "orig_chars": orig_chars,
        "trans_chars": trans_chars,
        "ratio": trans_chars/orig_chars if orig_chars > 0 else 0,
        "orig_sentences": orig_sentences,
        "trans_sentences": trans_sentences,
        "orig_paras": orig_paras,
        "trans_paras": trans_paras,
        "severity": severity
    }

    return warnings, stats

# ==== STREAMING DISPLAY ====
class StreamingDisplay:
    """Handle real-time display of model output with thinking process."""

    def __init__(self, is_thinking_model=False):
        self.is_thinking_model = is_thinking_model
        self.in_thinking = False
        self.thinking_buffer = ""
        self.translation_buffer = ""
        self.thinking_lines_shown = 0
        self.last_update_time = time.time()
        self.chars_generated = 0
        self.words_generated = 0

    def process_token(self, token):
        """Process incoming token and handle thinking/translation separation."""
        # Check for thinking tags
        if '<think>' in token:
            self.in_thinking = True
            self._display_thinking_start()
            token = token.replace('<think>', '')

        if '</think>' in token:
            self.in_thinking = False
            token = token.replace('</think>', '')
            if self.thinking_buffer:
                self._display_thinking_summary()
                self.thinking_buffer = ""
            return

        # Route token to appropriate buffer
        if self.in_thinking:
            self.thinking_buffer += token
            self._display_thinking_token(token)
        else:
            self.translation_buffer += token
            self._display_translation_token(token)

    def _display_thinking_start(self):
        """Display indicator when thinking starts."""
        if COLORS_AVAILABLE:
            print(f"\n{Fore.MAGENTA}🧠 THINKING STARTED{Style.RESET_ALL}")
            print(f"{Fore.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
        else:
            print(f"\n🧠 THINKING STARTED")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def _display_thinking_token(self, token):
        """Display thinking process in real-time with better visibility."""
        if not COLORS_AVAILABLE:
            # Fallback to plain text with newlines
            if '\n' in token:
                print(token, end='', flush=True)
            else:
                print(token, end='', flush=True)
            return

        # Show thinking with better visibility - use blue color instead of dimmed
        # Add newlines for better separation
        if '\n' in token:
            # When there's a newline, print with extra spacing
            print(f"\n{Fore.BLUE}🤔 {token}{Style.RESET_ALL}", end='', flush=True)
        else:
            # For regular tokens, show in blue
            print(f"{Fore.BLUE}{token}{Style.RESET_ALL}", end='', flush=True)

    def _display_translation_token(self, token):
        """Display translation in real-time with stats."""
        self.chars_generated += len(token)
        if ' ' in token or '\n' in token:
            self.words_generated += token.count(' ') + token.count('\n')

        # Show translation in green
        if COLORS_AVAILABLE:
            print(f"{Fore.GREEN}{token}{Style.RESET_ALL}", end='', flush=True)
        else:
            print(token, end='', flush=True)

        # Update stats periodically (every 0.5 seconds)
        current_time = time.time()
        if current_time - self.last_update_time > 0.5:
            self._show_inline_stats()
            self.last_update_time = current_time

    def _display_thinking_summary(self):
        """Show summary of thinking process with better visibility."""
        if not self.thinking_buffer:
            return

        # Count thinking length
        thinking_words = len(self.thinking_buffer.split())
        thinking_lines = self.thinking_buffer.count('\n') + 1

        if COLORS_AVAILABLE:
            print(f"\n{Fore.MAGENTA}🎯 THINKING COMPLETED{Style.RESET_ALL}")
            print(f"{Fore.CYAN}💭 Reasoning Summary: {thinking_words} words, {thinking_lines} lines{Style.RESET_ALL}")
            print(f"{Fore.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
        else:
            print(f"\n🎯 THINKING COMPLETED")
            print(f"💭 Reasoning Summary: {thinking_words} words, {thinking_lines} lines")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def _show_inline_stats(self):
        """Show inline statistics during generation."""
        # Move cursor to show stats on same line
        stats_line = f" [{self.chars_generated} chars, ~{self.words_generated} words]"

        if COLORS_AVAILABLE:
            print(f"{Fore.YELLOW}{stats_line}{Style.RESET_ALL}", end='\r', flush=True)

    def get_translation(self):
        """Get the final cleaned translation."""
        return self.translation_buffer

    def finalize(self):
        """Show final stats."""
        print()  # New line after generation
        if COLORS_AVAILABLE:
            print(f"{Fore.CYAN}✓ Generated: {self.chars_generated} chars, ~{self.words_generated} words{Style.RESET_ALL}")
        else:
            print(f"✓ Generated: {self.chars_generated} chars, ~{self.words_generated} words")

# ==== MODEL PROVIDER WITH STREAMING ====
class ModelProvider:
    def __init__(self, provider_type, model_name, device="cpu"):
        self.provider_type = provider_type
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_thinking = is_thinking_model(model_name)

    def load_model(self):
        if self.provider_type == "ollama":
            return self._validate_ollama()
        elif self.provider_type == "huggingface":
            return self._load_huggingface()

    def _validate_ollama(self):
        if not OLLAMA_AVAILABLE:
            return False
        try:
            ollama.show(self.model_name)
            return True
        except:
            return False

    def _load_huggingface(self):
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face not available")

        print(f"📥 Loading HuggingFace model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=4096,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        print("✅ Model loaded")
        return True

    def translate_streaming(self, system_prompt, user_prompt, temperature, top_p, num_ctx):
        """Translate with streaming support."""
        if self.provider_type == "ollama":
            return self._translate_ollama_streaming(system_prompt, user_prompt, temperature, top_p, num_ctx)
        else:
            return self._translate_huggingface(system_prompt, user_prompt, temperature, top_p)

    def _translate_ollama_streaming(self, system_prompt, user_prompt, temperature, top_p, num_ctx):
        """Translate using Ollama with real-time streaming."""
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama not available")

        # Initialize streaming display
        display = StreamingDisplay(is_thinking_model=self.is_thinking)

        try:
            # Show streaming header
            if self.is_thinking:
                if COLORS_AVAILABLE:
                    print(f"\n{Fore.MAGENTA}🧠 Reasoning Model Detected - Showing thinking process...{Style.RESET_ALL}")
                else:
                    print(f"\n🧠 Reasoning Model Detected - Showing thinking process...")

            if COLORS_AVAILABLE:
                print(f"{Fore.CYAN}▶ Streaming translation...{Style.RESET_ALL}\n")
            else:
                print(f"▶ Streaming translation...\n")

            # Stream the response
            stream = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_ctx": num_ctx
                },
                stream=True
            )

            # Process stream
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    token = chunk['message']['content']
                    display.process_token(token)

            # Finalize display
            display.finalize()

            # Return cleaned translation
            return display.get_translation()

        except Exception as e:
            print(f"\n{Fore.RED}❌ Streaming error: {str(e)}{Style.RESET_ALL}")
            raise

    def _translate_huggingface(self, system_prompt, user_prompt, temperature, top_p):
        """Translate using Hugging Face (non-streaming fallback)."""
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

        print(f"\n🤖 Generating translation...")

        response = self.pipeline(
            full_prompt,
            max_new_tokens=4096,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

        generated = response[0]['generated_text']
        return generated.split("Assistant:")[-1].strip()

# ==== TRANSLATION ENGINE ====
def translate_chunk(chunk, chunk_num, total_chunks, config, prompts, provider):
    """Translate chunk with streaming and enhanced validation."""

    # Print header
    print(f"\n{'='*70}")
    if COLORS_AVAILABLE:
        print(f"{Fore.YELLOW}{Style.BRIGHT}📄 CHUNK {chunk_num}/{total_chunks}{Style.RESET_ALL}")
    else:
        print(f"📄 CHUNK {chunk_num}/{total_chunks}")
    print(f"{'='*70}")

    # Show input stats
    orig_words = len(chunk.split())
    orig_chars = len(chunk)
    orig_sentences = count_sentences(chunk)
    orig_paras = chunk.count('\n\n') + 1

    if COLORS_AVAILABLE:
        print(f"{Fore.CYAN}📝 Input Stats:{Style.RESET_ALL}")
        print(f"   {orig_chars:,} chars | {orig_words:,} words")
        print(f"   {orig_sentences} sentences | {orig_paras} paragraphs")
    else:
        print(f"📝 Input Stats:")
        print(f"   {orig_chars:,} chars | {orig_words:,} words")
        print(f"   {orig_sentences} sentences | {orig_paras} paragraphs")

    for attempt in range(config['retry_attempts']):
        try:
            start_time = time.time()

            user_prompt = prompts["user"].format(chunk=chunk)

            if COLORS_AVAILABLE:
                print(f"\n{Fore.GREEN}🤖 Translating (attempt {attempt + 1}/{config['retry_attempts']})...{Style.RESET_ALL}")
            else:
                print(f"\n🤖 Translating (attempt {attempt + 1}/{config['retry_attempts']})...")

            # Use streaming translation
            translated = provider.translate_streaming(
                prompts["system"],
                user_prompt,
                config['temperature'],
                config['top_p'],
                config['num_ctx']
            )

            # Clean translation
            translated = clean_translation(translated)
            elapsed = time.time() - start_time

            # Show completion stats
            print()
            if COLORS_AVAILABLE:
                print(f"{Fore.GREEN}✅ Translation completed in {elapsed:.1f}s{Style.RESET_ALL}")
                print(f"{Fore.CYAN}📊 Output: {len(translated):,} chars | {count_sentences(translated)} sentences{Style.RESET_ALL}")
                print(f"{Fore.CYAN}📈 Ratio: {len(translated)/orig_chars:.2f}x{Style.RESET_ALL}")
            else:
                print(f"✅ Translation completed in {elapsed:.1f}s")
                print(f"📊 Output: {len(translated):,} chars | {count_sentences(translated)} sentences")
                print(f"📈 Ratio: {len(translated)/orig_chars:.2f}x")

            # Validate
            warnings, stats = validate_translation(chunk, translated, chunk_num)

            if warnings:
                print(f"\n{'='*70}")
                for warning in warnings:
                    if COLORS_AVAILABLE:
                        print(f"{Fore.RED}{warning}{Style.RESET_ALL}")
                    else:
                        print(warning)
                print(f"{'='*70}")
            else:
                if COLORS_AVAILABLE:
                    print(f"\n{Fore.GREEN}✅ Quality check: PASSED{Style.RESET_ALL}")
                else:
                    print(f"\n✅ Quality check: PASSED")

            return translated, stats

        except Exception as e:
            if COLORS_AVAILABLE:
                print(f"{Fore.RED}❌ Error: {str(e)}{Style.RESET_ALL}")
            else:
                print(f"❌ Error: {str(e)}")

            if attempt < config['retry_attempts'] - 1:
                wait = config['retry_delay'] * (attempt + 1)
                print(f"⏳ Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    return None, None

# ==== MAIN ====
def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Hindi Translation with Real-time Streaming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast testing
  python translate.py input.txt -ol -m qwen2.5:3b -t BASIC
  
  # With reasoning model (shows thinking process)
  python translate.py input.txt -ol -m deepseek-r1:7b -t INTERMEDIATE
  
  # Best quality with streaming
  python translate.py input.txt -ol -m qwen2.5:14b -t ADVANCED
  
  # List models
  python translate.py --list-models
        """
    )

    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument('-ol', '--ollama', action='store_true', help='Use Ollama')
    provider_group.add_argument('-hf', '--huggingface', action='store_true', help='Use Hugging Face')

    parser.add_argument('input_file', nargs='?', help='Input text file')
    parser.add_argument('-o', '--output', default='output_hi.txt', help='Output file')
    parser.add_argument('-m', '--model', help='Model name')
    parser.add_argument('-t', '--tier', choices=['BASIC', 'INTERMEDIATE', 'ADVANCED'],
                        default='BASIC', help='Translation tier')
    parser.add_argument('--chunk-words', type=int, default=350, help='Words per chunk')
    parser.add_argument('--temperature', type=float, default=0.4, help='Temperature')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--reset', action='store_true', help='Reset progress')
    parser.add_argument('--list-models', action='store_true', help='List recommended models')

    args = parser.parse_args()

    # List models
    if args.list_models:
        print("🤖 Recommended Models by Tier:\n")
        for tier, info in MODEL_TIERS.items():
            print(f"{'='*70}")
            print(f"⚡ {tier} - {info['description']}")
            print(f"   Time: {info['time_per_chunk']} per chunk")
            print(f"\n📦 Ollama models:")
            for model in info['ollama']:
                thinking_marker = " 🧠" if is_thinking_model(model) else ""
                print(f"   • {model}{thinking_marker}")
            print()

        print("💡 Models with 🧠 show reasoning/thinking process in real-time!")
        return

    # Validate
    if not args.input_file:
        parser.print_help()
        sys.exit(1)

    if not Path(args.input_file).exists():
        print(f"❌ File not found: {args.input_file}")
        sys.exit(1)

    # Determine provider
    provider_type = "ollama" if args.ollama or not args.huggingface else "huggingface"

    # Default model
    if not args.model:
        args.model = "qwen2.5:3b" if provider_type == "ollama" else "meta-llama/Llama-3.2-3B-Instruct"

    # Config
    config = {
        'temperature': args.temperature,
        'top_p': 0.75,
        'num_ctx': 8192,
        'retry_attempts': 3,
        'retry_delay': 2,
        'chunk_words': args.chunk_words
    }

    # Hardware
    hardware = detect_hardware()

    # Initialize provider
    print(f"🔍 Initializing {provider_type} provider...")
    provider = ModelProvider(provider_type, args.model, hardware['device'])

    if not provider.load_model():
        print(f"❌ Model not available: {args.model}")
        if provider_type == "ollama":
            print(f"💡 Install: ollama pull {args.model}")
        sys.exit(1)

    print(f"✅ Ready\n")

    # Check if thinking model
    if provider.is_thinking:
        if COLORS_AVAILABLE:
            print(f"{Fore.MAGENTA}{Style.BRIGHT}🧠 REASONING MODEL DETECTED!{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}   You'll see the model's thinking process in real-time{Style.RESET_ALL}\n")
        else:
            print(f"🧠 REASONING MODEL DETECTED!")
            print(f"   You'll see the model's thinking process in real-time\n")

    # Print config
    print("=" * 70)
    if COLORS_AVAILABLE:
        print(f"{Fore.CYAN}{Style.BRIGHT}🚀 ENHANCED HINDI TRANSLATION WITH STREAMING{Style.RESET_ALL}")
    else:
        print("🚀 ENHANCED HINDI TRANSLATION WITH STREAMING")
    print("=" * 70)
    print(f"📖 Input:       {args.input_file}")
    print(f"💾 Output:      {args.output}")
    print(f"🤖 Provider:    {provider_type}")
    print(f"🤖 Model:       {args.model}")
    print(f"🎯 Tier:        {args.tier}")
    print(f"🖥️  Device:      {hardware['device']}")
    print(f"📦 Chunk size:  {config['chunk_words']} words")
    print(f"🌡️  Temperature: {config['temperature']}")
    print(f"⚡ Streaming:   ENABLED")
    print("=" * 70)

    # Read input
    print(f"\n📖 Reading input...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Clean markers
    lines = text.split('\n')
    cleaned = [l for l in lines if not (l.strip().startswith('===') and l.strip().endswith('==='))]
    text = '\n'.join(cleaned).strip()

    word_count = len(text.split())
    char_count = len(text)
    print(f"📊 Total: {char_count:,} chars, {word_count:,} words")

    # Chunk
    print(f"\n📦 Creating chunks...")
    chunks = chunk_text(text, config['chunk_words'])
    print(f"✅ Created {len(chunks)} chunks")

    # Progress
    progress_file = f"{args.output}.progress.json"
    progress = TranslationProgress(progress_file)

    if args.reset:
        progress.reset()
        print("🔄 Progress reset")

    # Prompts
    prompts = TRANSLATION_PROMPTS[args.tier]

    # Translate
    print("\n" + "=" * 70)
    if COLORS_AVAILABLE:
        print(f"{Fore.GREEN}{Style.BRIGHT}🎯 STARTING TRANSLATION{Style.RESET_ALL}")
    else:
        print("🎯 STARTING TRANSLATION")
    print("=" * 70)

    start_time = time.time()
    total_stats = {
        "total_orig_chars": 0,
        "total_trans_chars": 0,
        "warnings": 0,
        "critical": 0
    }

    mode = 'a' if (args.resume and progress.data['last_chunk'] > 0) else 'w'

    if mode == 'a':
        print(f"📄 Resuming from chunk {progress.data['last_chunk'] + 1}")

    try:
        with open(args.output, mode, encoding='utf-8') as out:
            for i, chunk in enumerate(chunks, 1):
                if progress.is_complete(i):
                    print(f"\n⏭️  Chunk {i}/{len(chunks)} - Already done")
                    continue

                translated, stats = translate_chunk(chunk, i, len(chunks), config, prompts, provider)

                if translated:
                    out.write(translated + "\n\n")
                    out.flush()

                    progress.mark_complete(i, stats)

                    # Update totals
                    total_stats["total_orig_chars"] += stats["orig_chars"]
                    total_stats["total_trans_chars"] += stats["trans_chars"]
                    if stats["severity"] in ["WARNING", "CRITICAL"]:
                        total_stats["warnings"] += 1
                    if stats["severity"] == "CRITICAL":
                        total_stats["critical"] += 1

                    # Progress
                    elapsed = time.time() - start_time
                    avg = elapsed / i
                    remaining = len(chunks) - i
                    eta = remaining * avg

                    print(f"\n📈 Progress: {i}/{len(chunks)} ({i/len(chunks)*100:.1f}%)")
                    print(f"⏱️  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

        # Summary
        total_time = time.time() - start_time
        overall_ratio = total_stats["total_trans_chars"] / total_stats["total_orig_chars"] if total_stats["total_orig_chars"] > 0 else 0

        print("\n" + "=" * 70)
        if COLORS_AVAILABLE:
            print(f"{Fore.GREEN}{Style.BRIGHT}🎉 TRANSLATION COMPLETE!{Style.RESET_ALL}")
        else:
            print("🎉 TRANSLATION COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Time:       {total_time/60:.1f} minutes")
        print(f"📦 Chunks:     {len(chunks)}")
        print(f"⚡ Avg/chunk:  {total_time/len(chunks):.1f}s")
        print(f"📝 Input:      {total_stats['total_orig_chars']:,} chars")
        print(f"📝 Output:     {total_stats['total_trans_chars']:,} chars")
        print(f"📊 Ratio:      {overall_ratio:.2f}x")
        print(f"⚠️  Warnings:   {total_stats['warnings']}")
        print(f"🚨 Critical:   {total_stats['critical']}")
        print(f"💾 Output:     {args.output}")
        print("=" * 70)

        # Clean up
        if os.path.exists(progress_file):
            os.remove(progress_file)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
