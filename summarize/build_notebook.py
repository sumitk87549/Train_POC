#!/usr/bin/env python3
"""Generate the upgraded summary_generator_colab.ipynb notebook."""
import json, textwrap

def md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code_from_str(s):
    """Build a code cell from a plain string (auto-splits to lines)."""
    lines = []
    for line in s.split('\n'):
        lines.append(line + '\n')
    # Remove trailing newline from last line
    if lines and lines[-1] == '\n':
        lines = lines[:-1]
    elif lines:
        lines[-1] = lines[-1].rstrip('\n')
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

cells = []

# ─── CELL 1: Title ───
cells.append(md([
    "# Book Summarization Studio — Commercial Grade\n",
    "\n",
    "**Generate publication-ready summaries from books in multiple creative formats.**\n",
    "\n",
    "This notebook:\n",
    "1. Installs Ollama and downloads the best summarization model\n",
    "2. Uploads your book/text file\n",
    "3. Splits text into **overlapping, sentence-boundary-aware chunks**\n",
    "4. Generates summaries in **9 formats** — from quick overviews to Instagram captions, YouTube scripts, blog posts, and more\n",
    "5. Downloads the polished output ready to post/publish\n",
    "\n",
    "---"
]))

# ─── CELL 2 ───
cells.append(md(["## Step 1: Install Dependencies & Setup Ollama"]))

# ─── CELL 3: Install ───
cells.append(code_from_str(r'''# Install required packages
!pip install -q ollama requests ipywidgets colorama

import subprocess, time, os, sys

print("Installing Ollama...")
!apt-get update -qq && apt-get install -y -qq zstd > /dev/null 2>&1
!curl -fsSL https://ollama.com/install.sh | sh

print("\nStarting Ollama server...")
os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'
subprocess.Popen(['/usr/local/bin/ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)

try:
    import ollama
    ollama.list()
    print("[OK] Ollama server is running!")
except Exception as e:
    print(f"[WARN] Server may not be ready: {e}")

# Color support
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore: RED=GREEN=YELLOW=CYAN=MAGENTA=RESET=''
    class Style: BRIGHT=RESET_ALL=''

print("[OK] All dependencies ready!")'''))

# ─── CELL 4 ───
cells.append(md([
    "## Step 2: Download Summarization Model\n",
    "\n",
    "**Recommended models for book summarization:**\n",
    "\n",
    "| Model | Strengths | VRAM |\n",
    "|---|---|---|\n",
    "| `gemma3:27b` | Best creative writing & prose quality, 128K context | ~18GB |\n",
    "| `gemma3:12b` | Great all-rounder, fits T4 GPU | ~8GB |\n",
    "| `qwen3:14b` | Strong reasoning, multilingual, Hindi | ~10GB |\n",
    "| `llama3.1:8b` | Fast, 128K context, good for large books | ~6GB |"
]))

# ─── CELL 5: Model selection ───
cells.append(code_from_str(r'''import ipywidgets as widgets
from IPython.display import display
import ollama

MODEL_PRESETS = {
    "gemma3:27b  (Best Creative Quality)": "gemma3:27b",
    "gemma3:12b  (T4 GPU Friendly)": "gemma3:12b",
    "qwen3:14b   (Reasoning + Hindi)": "qwen3:14b",
    "llama3.1:8b (Fast + Large Context)": "llama3.1:8b",
}

model_dropdown = widgets.Dropdown(
    options=list(MODEL_PRESETS.keys()),
    value="gemma3:12b  (T4 GPU Friendly)",
    description='Model:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='450px')
)
display(model_dropdown)
print("Select a model, then run the next cell to download it.")'''))

# ─── CELL 6: Pull model ───
cells.append(code_from_str(r'''selected_model = MODEL_PRESETS[model_dropdown.value]
print(f"Downloading {selected_model}...")
try:
    current_digest = ''
    for progress in ollama.pull(selected_model, stream=True):
        digest = progress.get('digest', '')
        if digest != current_digest and current_digest: print()
        current_digest = digest
        status = progress.get('status', '')
        if 'completed' in progress and 'total' in progress:
            pct = (progress['completed'] / progress['total'] * 100) if progress['total'] > 0 else 0
            print(f"\r   {status}: {pct:.1f}%", end='', flush=True)
        else:
            print(f"\r   {status}", end='', flush=True)
    print(f"\n[OK] Model '{selected_model}' ready!")
except Exception as e:
    print(f"[ERROR] {e}")'''))

# ─── CELL 7 ───
cells.append(md([
    "## Step 3: Summarization Engine\n",
    "Defines all summary types, prompts, chunking logic, and generation."
]))

# ─── CELL 8: Type specs ───
cells.append(code_from_str(r'''import re, time, json, warnings
from collections import Counter
warnings.filterwarnings('ignore')

# ═══════════ SUMMARY TYPE SPECIFICATIONS ═══════════

SUMMARY_TYPES = {
    # Traditional lengths
    "SHORT": {
        "label": "Short Overview",
        "chunk_target": "2-3 sentences",
        "final_words": "150-300",
        "temperature": 0.2,
        "description": "Brief factual overview of key points"
    },
    "MEDIUM": {
        "label": "Medium Summary",
        "chunk_target": "4-6 sentences",
        "final_words": "400-700",
        "temperature": 0.2,
        "description": "Balanced summary covering main ideas"
    },
    "LONG": {
        "label": "Long Detailed",
        "chunk_target": "8-12 sentences",
        "final_words": "800-1500",
        "temperature": 0.2,
        "description": "Comprehensive summary preserving nuance"
    },
    # Creative / Commercial
    "INSTAGRAM": {
        "label": "Instagram Caption",
        "chunk_target": "2-3 key moments or emotions",
        "final_words": "150-200",
        "temperature": 0.7,
        "description": "Hook + caption + hashtags, ready to post"
    },
    "YOUTUBE_TTS": {
        "label": "YouTube / TTS Script",
        "chunk_target": "Key narrative beats with emotional texture",
        "final_words": "500-800",
        "temperature": 0.5,
        "description": "Narration script optimized for voiceover/TTS audio"
    },
    "TWITTER_THREAD": {
        "label": "Twitter/X Thread",
        "chunk_target": "1-2 tweetable insights",
        "final_words": "400-600",
        "temperature": 0.65,
        "description": "Numbered thread (8-15 tweets), each <=280 chars"
    },
    "BLOG_POST": {
        "label": "Blog Post / Article",
        "chunk_target": "6-8 sentences with analysis",
        "final_words": "800-1200",
        "temperature": 0.5,
        "description": "SEO-friendly article with headings, ready to publish"
    },
    "NEWSLETTER": {
        "label": "Email Newsletter",
        "chunk_target": "3-4 sentences, conversational",
        "final_words": "300-500",
        "temperature": 0.6,
        "description": "Email-friendly digest, personal and engaging tone"
    },
    "PODCAST_SCRIPT": {
        "label": "Podcast / Audio Script",
        "chunk_target": "Key talking points with transitions",
        "final_words": "600-1000",
        "temperature": 0.55,
        "description": "Conversational narration script for podcast episodes"
    },
}

print(f"[OK] {len(SUMMARY_TYPES)} summary types loaded!")'''))

# ─── CELL 9: Prompts ───
# Build prompts cell source as a raw Python string to avoid quoting hell
prompts_source = '''# ═══════════ PROMPT TEMPLATES PER SUMMARY TYPE ═══════════

def _p(text):
    """Dedent prompt text."""
    return text.strip()

PROMPTS = {
    "SHORT": {
        "system": "You are a precise summarizer. Output ONLY the summary text, no explanations.",
        "chunk": _p("""Summarize this text in {chunk_target}. Be factual and concise.

PREVIOUS CONTEXT:
{context}

OVERLAP FROM PREVIOUS SECTION:
{overlap}

TEXT:
{chunk}

Summary:"""),
        "final": _p("""Combine these chunk summaries into ONE flowing narrative of {final_words} words.
Remove all repetition. Maintain chronological order. Output ONLY the summary.

CHUNK SUMMARIES:
{summaries}

Final summary:""")
    },
    "MEDIUM": {
        "system": "You are an analytical summarizer. Capture ideas and their connections. Output ONLY the summary.",
        "chunk": _p("""Summarize this text in {chunk_target}. Capture main ideas, themes, and key details.

PREVIOUS CONTEXT:
{context}

OVERLAP FROM PREVIOUS SECTION:
{overlap}

TEXT:
{chunk}

Summary:"""),
        "final": _p("""Synthesize these chunk summaries into ONE cohesive narrative of {final_words} words.
Show how ideas develop and connect. Remove redundancy. Output ONLY the summary.

CHUNK SUMMARIES:
{summaries}

Final summary:""")
    },
    "LONG": {
        "system": "You are a senior literary analyst creating publication-quality summaries. Output ONLY the summary.",
        "chunk": _p("""Create a detailed analytical summary in {chunk_target}. Capture content, subtext, and significance.

PREVIOUS CONTEXT:
{context}

OVERLAP FROM PREVIOUS SECTION:
{overlap}

TEXT:
{chunk}

Analysis:"""),
        "final": _p("""Synthesize a comprehensive, publication-quality summary of {final_words} words.
Preserve the work's intellectual architecture. Show how ideas interconnect.
Write with polish of a published analysis. Output ONLY the summary.

CHUNK SUMMARIES:
{summaries}

Final synthesis:""")
    },
    "INSTAGRAM": {
        "system": _p("""You are a viral social media content creator specializing in book content.
You write scroll-stopping Instagram captions that make people want to read the book.
Output ONLY the Instagram caption, nothing else."""),
        "chunk": _p("""Extract the most emotionally compelling moment or insight from this text.
Find what would make someone stop scrolling. Keep it to {chunk_target}.

PREVIOUS CONTEXT:
{context}

OVERLAP:
{overlap}

TEXT:
{chunk}

Key moment:"""),
        "final": _p("""Create an Instagram caption from these book highlights.

FORMAT (follow EXACTLY):
[Opening hook - 1 punchy line that stops the scroll]

[2-3 short paragraphs telling the story's most gripping moments]

[Call to action - "Have you read this?" or "Tag someone who needs this book"]

[15-20 relevant hashtags on a new line, e.g. #BookReview #MustRead #BookTok etc.]

TARGET: {final_words} words (excluding hashtags).
TONE: Conversational, emotional, hook-driven.
Use emojis sparingly (2-3 max).

BOOK HIGHLIGHTS:
{summaries}

Instagram caption:""")
    },
    "YOUTUBE_TTS": {
        "system": _p("""You are a professional audiobook narrator and YouTube content creator.
You write scripts that sound natural when read aloud by TTS or a human narrator.
Output ONLY the narration script, nothing else."""),
        "chunk": _p("""Extract the key narrative beats from this text for audio narration.
Focus on {chunk_target}. Write as if telling a story to a listener.

PREVIOUS CONTEXT:
{context}

OVERLAP:
{overlap}

TEXT:
{chunk}

Narration notes:"""),
        "final": _p("""Create a YouTube narration script from these story beats.

RULES FOR TTS/AUDIO:
- Write in a warm, storytelling voice ("Imagine this..." / "Picture a world where...")
- Use short sentences. Vary sentence length for rhythm.
- Add natural pauses: use "..." for dramatic pauses, commas for breathing room
- NO bullet points, NO headings, NO markdown - just flowing narration
- Include an engaging opening hook (first 10 seconds are critical)
- End with a thought-provoking closing line
- Avoid words that sound awkward in TTS (acronyms, URLs, special characters)

TARGET: {final_words} words.
TONE: Warm, engaging, like a friend telling you about an amazing book.

STORY BEATS:
{summaries}

Narration script:""")
    },
    "TWITTER_THREAD": {
        "system": _p("""You are a viral Twitter/X thread writer who distills books into addictive threads.
Each tweet must be self-contained and compelling. Output ONLY the thread."""),
        "chunk": _p("""Extract 1-2 tweetable insights from this text. Each must be a standalone hook.

PREVIOUS CONTEXT:
{context}

OVERLAP:
{overlap}

TEXT:
{chunk}

Tweet-worthy insights:"""),
        "final": _p("""Create a Twitter/X thread from these insights.

FORMAT:
1/ [Hook tweet - the most provocative claim or question from the book]

2/ [Context or setup]

3/ through 12/ [Key insights, one per tweet]

[Final tweet] If this thread resonated, follow for more book breakdowns.

RULES:
- Each tweet MUST be under 280 characters
- Start tweet 1/ with a bold hook that makes people click "Show more"
- Use line breaks within tweets for readability
- 8-15 tweets total
- No hashtags except in the final tweet (2-3 max)

TARGET: {final_words} words total.

INSIGHTS:
{summaries}

Thread:""")
    },
    "BLOG_POST": {
        "system": _p("""You are a professional book reviewer and content writer.
You write engaging, SEO-optimized articles that readers love and search engines rank.
Output ONLY the article, nothing else."""),
        "chunk": _p("""Analyze this section for a book review article. Extract {chunk_target}.
Note themes, character arcs, writing style, and quotable moments.

PREVIOUS CONTEXT:
{context}

OVERLAP:
{overlap}

TEXT:
{chunk}

Section analysis:"""),
        "final": _p("""Write a blog post / book review article from these section analyses.

FORMAT:
# [Compelling Title]

## Introduction
[Hook the reader. Why should they care about this book?]

## [2-4 thematic sections with descriptive headings]
[Analysis with specific examples from the text]

## Final Verdict
[Who should read this? What makes it special?]

TARGET: {final_words} words.
TONE: Authoritative but accessible. Like a trusted book-lover friend.

SECTION ANALYSES:
{summaries}

Article:""")
    },
    "NEWSLETTER": {
        "system": _p("""You are a beloved newsletter writer with thousands of subscribers.
Your tone is personal, warm, and insightful - like a letter from a well-read friend.
Output ONLY the newsletter content, nothing else."""),
        "chunk": _p("""Find the most interesting insight or story from this text.
Extract {chunk_target}. Think about what your newsletter subscribers would find valuable.

PREVIOUS CONTEXT:
{context}

OVERLAP:
{overlap}

TEXT:
{chunk}

Key insight:"""),
        "final": _p("""Write a newsletter digest about this book from these insights.

FORMAT:
Subject: [Intriguing one-liner about the book]

Hey there,

[Personal opening - why you picked up this book]

[2-3 paragraphs on the best insights, told conversationally]

[One key takeaway the reader can apply today]

Until next time,
[Sign off]

P.S. [One fun or surprising fact from the book]

TARGET: {final_words} words.
TONE: Personal, warm, like writing to a friend.

KEY INSIGHTS:
{summaries}

Newsletter:""")
    },
    "PODCAST_SCRIPT": {
        "system": _p("""You are a podcast host known for making books come alive through conversation.
You write scripts that sound natural and engaging when spoken aloud.
Output ONLY the script, nothing else."""),
        "chunk": _p("""Extract key talking points from this text for a podcast episode.
Focus on {chunk_target}. Think about what would spark interesting discussion.

PREVIOUS CONTEXT:
{context}

OVERLAP:
{overlap}

TEXT:
{chunk}

Talking points:"""),
        "final": _p("""Write a podcast episode script about this book.

FORMAT:
[INTRO]
"Hey everyone, welcome back! Today we're diving into [book]..."

[SEGMENT 1-3: Key themes or chapters]
"So here's what really got me..."
"And this is where it gets interesting..."

[CLOSING]
"So my final thoughts on this one..."

RULES:
- Conversational, not academic
- Use rhetorical questions to engage listeners
- Include transition phrases ("Now here's the thing...", "But wait...")
- Vary energy - some parts reflective, some excited
- No stage directions, just natural speech

TARGET: {final_words} words.
TONE: Enthusiastic but thoughtful, like talking to a friend at a coffee shop.

TALKING POINTS:
{summaries}

Podcast script:""")
    },
}

print(f"[OK] {len(PROMPTS)} prompt templates loaded!")'''

cells.append(code_from_str(prompts_source))

# ─── CELL 10: Chunking + generation ───
engine_source = r'''# ═══════════ SENTENCE-BOUNDARY-AWARE CHUNKING WITH OVERLAP ═══════════

def chunk_text_with_overlap(text, chunk_words=400, overlap_words=80):
    """Split text into overlapping chunks at sentence boundaries.
    Returns list of dicts: {text, overlap_prefix}.
    """
    # Split into sentences (Hindi + English punctuation)
    sentences = re.split(r'(?<=[\u0964\u0965.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        if current_word_count + s_words > chunk_words and current_sentences:
            chunks.append(' '.join(current_sentences))
            # Keep overlap sentences from the end
            overlap_sents = []
            overlap_wc = 0
            for s in reversed(current_sentences):
                overlap_wc += len(s.split())
                if overlap_wc > overlap_words:
                    break
                overlap_sents.insert(0, s)
            current_sentences = overlap_sents + [sentence]
            current_word_count = sum(len(s.split()) for s in current_sentences)
        else:
            current_sentences.append(sentence)
            current_word_count += s_words

    if current_sentences:
        chunks.append(' '.join(current_sentences))

    # Build overlap data
    result = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlap = ''
        else:
            prev = chunks[i-1]
            prev_words = prev.split()
            overlap = ' '.join(prev_words[-overlap_words:]) if len(prev_words) > overlap_words else prev
        result.append({'text': chunk, 'overlap_prefix': overlap})

    return result


def clean_output(text):
    """Remove thinking tags, code blocks, and meta-prefixes."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'^(Summary:|Analysis:|Here is.*?:|Here\'s.*?:)\s*', '', text, flags=re.IGNORECASE)
    # Remove Chinese characters (hallucination guard)
    text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf]+', '', text)
    return text.strip()


def generate_summary(model, summary_type, chunks, max_tokens_chunk=800, max_tokens_final=3000):
    """Generate a summary using the specified type and model."""
    spec = SUMMARY_TYPES[summary_type]
    prompts = PROMPTS[summary_type]
    temp = spec['temperature']

    context = ''
    chunk_summaries = []
    start_time = time.time()

    for i, chunk_data in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] Processing chunk...", end=' ')

        chunk_prompt = prompts['chunk'].format(
            chunk=chunk_data['text'],
            context=context[-2000:] if context else '(Start of text)',
            overlap=chunk_data.get('overlap_prefix', '') or '(None)',
            chunk_target=spec['chunk_target']
        )

        result = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': prompts['system']},
                {'role': 'user', 'content': chunk_prompt}
            ],
            options={
                'temperature': temp,
                'num_ctx': 8192,
                'num_predict': max_tokens_chunk
            }
        )
        chunk_result = clean_output(result['message']['content'])
        chunk_summaries.append(chunk_result)
        context += '\n\n' + chunk_result
        wc = len(chunk_result.split())
        print(f"done ({wc} words)")

    # Final synthesis
    print(f"  Generating final {spec['label']}...")
    final_prompt = prompts['final'].format(
        summaries='\n\n---\n\n'.join(chunk_summaries),
        final_words=spec['final_words']
    )

    final_result = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': prompts['system']},
            {'role': 'user', 'content': final_prompt}
        ],
        options={
            'temperature': temp,
            'num_ctx': 8192,
            'num_predict': max_tokens_final
        }
    )
    final_text = clean_output(final_result['message']['content'])

    elapsed = time.time() - start_time
    return final_text, elapsed, chunk_summaries


print('[OK] Chunking + generation engine loaded!')'''

cells.append(code_from_str(engine_source))

# ─── CELL 11 ───
cells.append(md(["## Step 4: Upload Your Text File"]))

# ─── CELL 12: Upload ───
cells.append(code_from_str(r'''from google.colab import files

print("Upload your text file (.txt):")
uploaded = files.upload()

if uploaded:
    uploaded_filename = list(uploaded.keys())[0]
    input_text = uploaded[uploaded_filename].decode('utf-8')
    word_count = len(input_text.split())
    print(f"\n[OK] Uploaded: {uploaded_filename}")
    print(f"   Words: {word_count:,}")
    print(f"   Preview: {input_text[:300]}...")
else:
    print("[WARN] No file uploaded.")'''))

# ─── CELL 13 ───
cells.append(md([
    "## Step 5: Choose Summary Type & Generate\n",
    "Select your output format and configure chunking parameters."
]))

# ─── CELL 14: Configure ───
cells.append(code_from_str(r'''import ipywidgets as widgets
from IPython.display import display

# Summary type dropdown
type_options = [(f"{v['label']} — {v['description']}", k) for k, v in SUMMARY_TYPES.items()]
type_dropdown = widgets.Dropdown(
    options=type_options,
    value='MEDIUM',
    description='Output Type:',
    style={'description_width': '120px'},
    layout=widgets.Layout(width='600px')
)

chunk_slider = widgets.IntSlider(
    value=500, min=200, max=1000, step=50,
    description='Chunk Words:',
    style={'description_width': '120px'},
    layout=widgets.Layout(width='400px')
)

overlap_slider = widgets.IntSlider(
    value=100, min=0, max=250, step=25,
    description='Overlap Words:',
    style={'description_width': '120px'},
    layout=widgets.Layout(width='400px')
)

print("Configure your summary:")
print("=" * 50)
display(type_dropdown)
display(chunk_slider)
display(overlap_slider)
print("\nOverlap = how many words from previous chunk are passed as context.")
print("Higher overlap = better continuity, especially for long books.")
print("\nRun the next cell to generate!")'''))

# ─── CELL 15: Run generation ───
cells.append(code_from_str(r'''if 'input_text' not in dir() or not input_text:
    print('[WARN] Upload a file first (Step 4)!')
else:
    summary_type = type_dropdown.value
    spec = SUMMARY_TYPES[summary_type]

    print(f"{'='*60}")
    print(f"  Generating: {spec['label']}")
    print(f"  Model: {selected_model}")
    print(f"  Temperature: {spec['temperature']}")
    print(f"  Target: {spec['final_words']} words")
    print(f"{'='*60}\n")

    # Chunk with overlap
    chunks = chunk_text_with_overlap(input_text, chunk_slider.value, overlap_slider.value)
    print(f"Text: {len(input_text.split()):,} words -> {len(chunks)} chunks (overlap: {overlap_slider.value} words)\n")

    # Generate
    final_summary, elapsed, chunk_summaries = generate_summary(
        selected_model, summary_type, chunks
    )

    final_wc = len(final_summary.split())
    original_wc = len(input_text.split())

    print(f"\n{'='*60}")
    print(f"  COMPLETE!")
    print(f"  Original: {original_wc:,} words")
    print(f"  Output: {final_wc:,} words ({final_wc/original_wc*100:.1f}%)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")
    print(final_summary)'''))

# ─── CELL 16 ───
cells.append(md(["## Step 6: Download Result"]))

# ─── CELL 17: Download ───
cells.append(code_from_str(r'''from google.colab import files
from datetime import datetime

if 'final_summary' not in dir() or not final_summary:
    print('[WARN] Generate a summary first (Step 5)!')
else:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = uploaded_filename.rsplit('.', 1)[0] if 'uploaded_filename' in dir() else 'document'
    stype = summary_type.lower()
    output_filename = f"{base}_{stype}_{timestamp}.txt"

    spec = SUMMARY_TYPES[summary_type]
    header = f"""{'='*60}
BOOK SUMMARIZATION STUDIO — {spec['label'].upper()}
{'='*60}
Source: {uploaded_filename if 'uploaded_filename' in dir() else 'Unknown'}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {selected_model}
Type: {spec['label']} ({spec['description']})
Original: {len(input_text.split()):,} words
Output: {len(final_summary.split()):,} words
{'='*60}

"""

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(header + final_summary + '\n')

    print(f"[OK] Saved: {output_filename}")
    files.download(output_filename)
    print("[OK] Download started!")'''))

# ─── CELL 18: Reference ───
cells.append(md([
    "## Troubleshooting & Reference\n",
    "\n",
    "### Summary Types\n",
    "| Type | Words | Temp | Best For |\n",
    "|---|---|---|---|\n",
    "| Short | 150-300 | 0.2 | Quick overviews, abstracts |\n",
    "| Medium | 400-700 | 0.2 | Balanced summaries |\n",
    "| Long | 800-1500 | 0.2 | Detailed comprehension |\n",
    "| Instagram | 150-200 | 0.7 | Social media captions |\n",
    "| YouTube/TTS | 500-800 | 0.5 | Voiceover narration |\n",
    "| Twitter Thread | 400-600 | 0.65 | Viral threads |\n",
    "| Blog Post | 800-1200 | 0.5 | Website publishing |\n",
    "| Newsletter | 300-500 | 0.6 | Email digests |\n",
    "| Podcast Script | 600-1000 | 0.55 | Audio content |\n",
    "\n",
    "### Tips\n",
    "- **Creative types** (Instagram, Twitter) use higher temperature for engaging prose\n",
    "- **Factual types** (Short, Medium, Long) use low temperature for accuracy\n",
    "- **Overlap** helps the model maintain context across chunk boundaries — use 100-150 for most books\n",
    "- **For very long books** (100k+ words): use chunk size 400, overlap 80\n",
    "- **For Hindi text**: `qwen3:14b` handles Hindi/Devanagari better than other models\n",
    "- **For best creative quality**: `gemma3:27b` (needs A100 GPU) or `gemma3:12b` (T4 compatible)"
]))

# ─── Build notebook ───
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

output_path = "/home/sumit/Documents/GitHub/Train_POC/summarize/summary_generator_colab.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook written to {output_path}")
print(f"     Cells: {len(cells)}")
