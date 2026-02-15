#!/usr/bin/env python3
"""
Creates the TTS_Thinking_Optimizer.ipynb notebook.
Uses Ollama thinking models with visible reasoning process.
"""

import json

def make_cell(cell_type, source_lines, metadata=None):
    """Helper to create a notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source_lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

notebook = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

cells = notebook["cells"]

# ==============================================================================
# Cell 0: Title Markdown
# ==============================================================================
cells.append(make_cell("markdown", [
    "# TTS Text Optimizer with Thinking Models\n",
    "\n",
    "**Format translated text for optimal DesiVocal.com TTS output using reasoning-capable LLMs.**\n",
    "\n",
    "This notebook:\n",
    "1. **Installs Ollama** and downloads a thinking model (deepseek-r1, qwen3, magistral, etc.)\n",
    "2. **Shows the model's reasoning process** in real-time as it analyzes your text\n",
    "3. **Formats text** with proper speaker identification, punctuation, and DesiVocal-specific fixes\n",
    "4. **Downloads** the optimized `.txt` file ready for TTS\n",
    "\n",
    "Optimized for translating public-domain literary works (Sherlock Holmes, Ramayana, Mahabharata, Dracula, Alice in Wonderland, Pride and Prejudice, etc.) into natural, human-like TTS audio."
]))

# ==============================================================================
# Cell 1: Step 1 Markdown
# ==============================================================================
cells.append(make_cell("markdown", [
    "## Step 1: Install and Setup Ollama\n",
    "Run this cell to install Ollama and start the server in the background."
]))

# ==============================================================================
# Cell 2: Install Ollama
# ==============================================================================
cells.append(make_cell("code", [
    "# Install required packages\n",
    "!pip install -q ollama requests ipywidgets\n",
    "\n",
    "# Install and start Ollama server\n",
    "import subprocess\n",
    "import time\n",
    "import os\n",
    "import sys\n",
    "\n",
    "print(\"Installing Ollama...\")\n",
    "\n",
    "# Install zstd first (required for Ollama extraction)\n",
    "!apt-get update -qq && apt-get install -y -qq zstd > /dev/null 2>&1\n",
    "\n",
    "# Download and install Ollama\n",
    "!curl -fsSL https://ollama.com/install.sh | sh\n",
    "\n",
    "print(\"\\nStarting Ollama server in background...\")\n",
    "\n",
    "# Start Ollama server in background\n",
    "os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'\n",
    "subprocess.Popen(['/usr/local/bin/ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
    "\n",
    "# Wait for server to start\n",
    "time.sleep(5)\n",
    "\n",
    "# Verify server is running\n",
    "try:\n",
    "    import ollama\n",
    "    ollama.list()\n",
    "    print(\"[OK] Ollama server is running and ready!\")\n",
    "except Exception as e:\n",
    "    print(f\"[WARN] Ollama server may not be ready yet. Error: {e}\")\n",
    "    print(\"   Please wait a few seconds and try running the next cell.\")"
]))

# ==============================================================================
# Cell 3: Step 2 Markdown
# ==============================================================================
cells.append(make_cell("markdown", [
    "## Step 2: Select and Download Thinking Model\n",
    "Choose a reasoning-capable model. These models show their thinking process before producing output.\n",
    "\n",
    "**Recommended:**\n",
    "- `deepseek-r1:14b` - Best reasoning quality for the size\n",
    "- `qwen3:14b` - Strong multilingual reasoning\n",
    "- `magistral:24b` - Mistral's reasoning model (needs more VRAM)\n",
    "- `gpt-oss:20b` - OpenAI-style thinking format"
]))

# ==============================================================================
# Cell 4: Model Selection
# ==============================================================================
cells.append(make_cell("code", [
    "import ipywidgets as widgets\n",
    "from IPython.display import display\n",
    "import ollama\n",
    "\n",
    "print(\"Thinking Model Selection\")\n",
    "print(\"=\" * 40)\n",
    "\n",
    "# Thinking model options\n",
    "THINKING_MODELS = {\n",
    "    \"deepseek-r1:14b (Best Reasoning)\": \"deepseek-r1:14b\",\n",
    "    \"qwen3:14b (Strong Multilingual Thinking)\": \"qwen3:14b\",\n",
    "    \"magistral:24b (Mistral Reasoning - Large)\": \"magistral:24b\",\n",
    "    \"gpt-oss:20b (OpenAI-style Thinking)\": \"gpt-oss:20b\",\n",
    "    \"qwen3:8b (Lighter Thinking Model)\": \"qwen3:8b\",\n",
    "    \"deepseek-r1:7b (Lightweight Reasoning)\": \"deepseek-r1:7b\",\n",
    "}\n",
    "\n",
    "model_dropdown = widgets.Dropdown(\n",
    "    options=list(THINKING_MODELS.keys()),\n",
    "    value=\"deepseek-r1:14b (Best Reasoning)\",\n",
    "    description='Model:',\n",
    "    style={'description_width': 'initial'},\n",
    "    layout=widgets.Layout(width='450px')\n",
    ")\n",
    "\n",
    "display(model_dropdown)\n",
    "print(\"\\nSelect a model and run the next cell to download it.\")\n",
    "print(\"NOTE: 14b models need ~10GB VRAM, 24b models need ~16GB VRAM.\")"
]))

# ==============================================================================
# Cell 5: Pull Model
# ==============================================================================
cells.append(make_cell("code", [
    "# Pull the selected thinking model\n",
    "selected_model_name = THINKING_MODELS[model_dropdown.value]\n",
    "print(f\"Downloading model: {selected_model_name}...\")\n",
    "print(\"This may take several minutes for large models.\")\n",
    "\n",
    "try:\n",
    "    current_digest = ''\n",
    "    for progress in ollama.pull(selected_model_name, stream=True):\n",
    "        digest = progress.get('digest', '')\n",
    "        if digest != current_digest and current_digest:\n",
    "             print()\n",
    "        current_digest = digest\n",
    "\n",
    "        status = progress.get('status', '')\n",
    "        if 'completed' in progress and 'total' in progress:\n",
    "             completed = progress['completed']\n",
    "             total = progress['total']\n",
    "             pct = (completed / total * 100) if total > 0 else 0\n",
    "             print(f\"\\r   {status}: {pct:.1f}%\", end='', flush=True)\n",
    "        else:\n",
    "             print(f\"\\r   {status}\", end='', flush=True)\n",
    "\n",
    "    print(f\"\\n\\n[OK] Model '{selected_model_name}' ready to use!\")\n",
    "except Exception as e:\n",
    "    print(f\"\\n[ERROR] Error pulling model: {e}\")"
]))

# ==============================================================================
# Cell 6: Step 3 Markdown
# ==============================================================================
cells.append(make_cell("markdown", [
    "## Step 3: TTS Optimizer Engine (with Thinking Display)\n",
    "This defines the optimizer class that streams the model's thinking process in real-time."
]))

# ==============================================================================
# Cell 7: TTSThinkingOptimizer Class
# ==============================================================================

# Build the prompt as a Python string
optimizer_code = r'''import requests
import json
import sys
import re
import time
from IPython.display import display, HTML, clear_output

# ══════════════════════════════════════════════════════════════
# KNOWN THINKING MODEL PATTERNS
# ══════════════════════════════════════════════════════════════
# Different thinking models use different tag patterns for their reasoning:
#   deepseek-r1  : <think> ... </think>
#   qwen3        : <think> ... </think>
#   magistral    : [Thinking] ... [/Thinking] or just outputs reasoning first
#   gpt-oss      : <|begin_of_thought|> ... <|end_of_thought|>
# We handle all of these patterns.

THINK_START_PATTERNS = ['<think>', '<|begin_of_thought|>', '[Thinking]']
THINK_END_PATTERNS = ['</think>', '<|end_of_thought|>', '[/Thinking]']


class TTSThinkingOptimizer:
    """Optimizes text for DesiVocal.com TTS using thinking models with visible reasoning."""

    def __init__(self, model_name="deepseek-r1:14b", chunk_size=2000, timeout=6000):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = model_name
        self.chunk_size = chunk_size
        self.timeout = timeout
        print(f"Initialized TTS Thinking Optimizer")
        print(f"   Model: {self.model}")
        print(f"   Chunk size: {self.chunk_size} chars")
        print(f"   Timeout: {self.timeout}s per chunk")

    def chunk_text(self, text: str) -> list:
        """Split text into chunks at paragraph/sentence boundaries."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        # First try to split by paragraphs (double newline)
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If any chunk is still too large, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                sentences = re.split(r'([।॥.!?]\s+)', chunk)
                sub_chunk = ""
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    separator = sentences[i+1] if i+1 < len(sentences) else ""
                    if len(sub_chunk) + len(sentence) + len(separator) > self.chunk_size and sub_chunk:
                        final_chunks.append(sub_chunk.strip())
                        sub_chunk = sentence + separator
                    else:
                        sub_chunk += sentence + separator
                if sub_chunk.strip():
                    final_chunks.append(sub_chunk.strip())

        # Fallback: if still no chunks, force-split
        if not final_chunks:
            final_chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        print(f"\nText split into {len(final_chunks)} chunks")
        for idx, chunk in enumerate(final_chunks, 1):
            print(f"   Chunk {idx}: {len(chunk)} characters")

        return final_chunks

    def get_optimization_prompt(self, text: str) -> str:
        """Build the TTS optimization prompt for thinking models."""
        prompt = f"""You are an expert text formatter for DesiVocal.com TTS system. You have strong reasoning abilities - use them to understand context deeply before formatting.

REASONING-FIRST APPROACH

YOUR SPECIAL CAPABILITY:
Unlike basic models, you can THINK THROUGH THE TEXT before formatting.
Use this to identify speakers accurately, resolve pronouns, and create optimal TTS output.

PROCESS:
1. ANALYZE: Read entire text, identify all characters and their relationships
2. MAP: Create character-to-punctuation assignments
3. TRACK: Follow conversation flows and pronoun references
4. FORMAT: Apply formatting with deep contextual understanding
5. VERIFY: Check consistency and speaker identification accuracy

MISSION: Format for Non-SSML TTS with Single Voice

THE CHALLENGE:
DesiVocal.com uses ONE VOICE for everything. No SSML support.
Listeners cannot distinguish speakers unless we explicitly mark them.

YOUR SOLUTION:
- Use DIFFERENT punctuation marks for DIFFERENT characters
- Identify speakers intelligently from context clues
- Make it crystal clear who is speaking at all times

ABSOLUTE RULE: WORD PRESERVATION

Every input word MUST appear in output (except attribution words like "ne kaha" which become speaker tags).

ALLOWED:
- Format numbers, dates, punctuation
- ADD speaker tags
- ADD different dialogue punctuation per character

FORBIDDEN:
- Adding explanations, metadata, annotations
- Translating between languages
- Dropping words
- Adding your own content

SPEAKER IDENTIFICATION - USE YOUR REASONING

THINK THROUGH THE TEXT:

Step 1: Character Discovery
Read through and identify all named characters:
- Direct names: "Holmes ne kaha", "Watson ne poocha"
- Roles/titles: raja, doctor, maharaj
- Pronouns that refer to them: usne, maine

Step 2: Pronoun Resolution
When you see "usne kaha" (he/she said):
- Look back 1-3 sentences
- Identify who was last mentioned or contextually relevant
- Track the conversation flow

Step 3: Conversation Tracking
In dialogue between two people:
- They typically alternate
- Unless context indicates otherwise (interruption, multiple quotes from same person)

Step 4: Narrator Handling
- "main" (I) in narration = narrator speaking (often Watson in Sherlock stories, Sanjeev in memoirs)
- Keep as "main:" or "kathavachak:" based on context
- Do not confuse with "main" inside dialogue quotes

LITERARY CONTEXT AWARENESS:
When processing well-known works, leverage your knowledge:
- Sherlock Holmes stories: Holmes=protagonist, Watson=narrator/sidekick, clients=visitors
- Ramayana/Mahabharata: Identify gods, heroes, sages by their attributes and relationships
- Gothic novels (Dracula, Frankenstein): Track epistolary/journal narration styles
- Classic novels: Maintain the period feel while making dialogue clear
- For ANY book: Identify the narrative structure first (first-person, third-person, epistolary, dramatic)

DIALOGUE WRAPPING - DIFFERENT MARKS FOR DIFFERENT VOICES

PUNCTUATION ASSIGNMENT:

TIER 1 - Main Protagonist: 'single quotes'
  Usually: Holmes, Ram, Elizabeth, Alice, the central hero/heroine

TIER 2 - Companion/Narrator: "double quotes"
  Usually: Watson, Sita, Darcy, narrator, secondary protagonist

TIER 3 - Authority/Antagonist: *asterisks*
  Usually: raja, villain, Ravana, Dracula, authority figures, clients

TIER 4 - Additional Characters: <<guillemets>>
  Others: minor characters, suspects, servants, messengers

CONSISTENCY IS CRUCIAL:
Once assigned, each character keeps their punctuation throughout the ENTIRE text.

FORMAT:
CharacterName: [mark]dialogue text[mark]

REMOVE ATTRIBUTION PHRASES:
"Holmes ne kaha, 'X'" becomes "Holmes: 'X'"

TECHNICAL FORMATTING RULES

1. ROMAN NUMERALS to REGULAR NUMBERS
   I to 1, II to 2, III to 3, IV to 4, V to 5, etc.
   Chapter I to Chapter 1
   Adhyay II to Adhyay 2

2. NUMBERS: Remove all commas
   50,000 to 50000
   1,50,000 to 150000

3. DATES: Use month names
   15/03/2024 to 15 March 2024

4. YEARS: Add "san" prefix (no space)
   1988 to san1988
   "main 1995 mein paida hua" to "main san1995 mein paida hua"

5. TIME: Write in words
   3:30 PM to saadhe teen or teen bajkar tees minute
   10:00 AM to ten baje

6. THE "10" BUG (CRITICAL)
   DesiVocal does not speak "10" properly!
   ALWAYS convert to "ten":
   - 10 books to ten kitaabein
   - Chapter 10 to Chapter ten
   - 10:00 to ten baje
   - 8-10 glasses to 8 se ten gilaas

7. RANGES: Use "se" (no spaces)
   5-8 to 5se8
   10-15 to tense15

8. PERCENTAGES
   50% to 50 percent
   12.5% to 12.5 percent

9. ABBREVIATIONS TO EXPAND
   Dr. to Doctor (or Daktar in Hindi context)
   Rs. to rupaye
   km to kilometer

10. ACRONYMS: Remove periods
    U.S.A. to USA
    N.A.S.A. to NASA

11. EMAILS and URLS: Convert dots
    @ to "at the rate"
    . (in email/URL) to "dot"
    hr@company.com to hr at the rate company dot com

12. COMPOUND WORDS: Remove hyphens
    cross-check to cross check
    post-mortem to post mortem

13. SYMBOLS: Expand
    degree F to degree Fahrenheit
    degree C to degree Celsius
    x to guna (multiply)

PROSODY AND PACING (For Non-Dialogue)

Use punctuation creatively for pacing:

, to Short pause
| to Medium pause (context/topic shift)
. to Long pause (sentence end)
,, to Extended pause
... to Suspense/trailing
!! to Strong excitement
?? to Confusion/disbelief

Context shifts need separation:
"usne khana khaya. | fir so gaya. | subah jaldi utha."

GENRE-SPECIFIC FORMATTING GUIDANCE

FOR DETECTIVE/MYSTERY STORIES (Holmes, Poirot, etc.):
- Mark deductions and observations with measured pacing: commas, pipes
- Use suspense markers (...) before revelations
- Distinguish client narration (*asterisks*) from detective analysis ('single quotes')

FOR EPIC/MYTHOLOGICAL TEXTS (Ramayana, Mahabharata, etc.):
- Preserve the gravitas of divine speech with measured pacing
- Use pipes (|) between shloka-like passages for breathing room
- Mark blessings, curses, and proclamations distinctly
- Character hierarchy: Gods/heroes 'single', sages "double", demons *asterisks*, others <<guillemets>>

FOR GOTHIC/HORROR (Dracula, Frankenstein, etc.):
- Use suspense markers (...) liberally for atmospheric tension
- Differentiate journal entries, letters, and dialogue clearly
- Mark whispered or fearful speech with appropriate pacing

FOR ROMANCE/SOCIAL NOVELS (Pride and Prejudice, etc.):
- Preserve wit and irony through careful pacing
- Distinguish between narration and inner thoughts
- Use commas and pauses to convey social nuance

FOR CHILDREN'S LITERATURE (Alice in Wonderland, etc.):
- Keep pacing lively and bouncy
- Use exclamation for wonder and surprise
- Make character voices very distinct through punctuation

FOR ANY UNKNOWN BOOK:
- Read a few paragraphs to understand the genre and tone
- Identify narrative style (first person, third person, epistolary)
- Detect the emotional register (serious, humorous, dramatic, whimsical)
- Apply formatting that preserves THAT specific tone

COMPREHENSIVE EXAMPLES

Example 1: Sherlock Holmes (Detective - English to Hindi translated text)

INPUT:
adhyay I
yah 15 march, 1988 ki baat hai. Holmes ne kaha, "main 10 baje aaunga." Watson ne poocha, "kyun?" "kyunki yah zaroori hai," Holmes ne kaha.

OUTPUT:
adhyay 1.

yah 15 march, san1988 ki baat hai.

Holmes: 'main ten baje aaunga.'

Watson: "kyun?"

Holmes: 'kyunki yah zaroori hai.'

---

Example 2: Ramayana/Epic style (multiple characters, divine speech)

INPUT:
Ram ne Sita se kaha, "main vanvaas ja raha hun." Sita ne kaha, "main bhi chalungi." Lakshman ne kaha, "main bhi saath chalunga, bhaiya." Dasharath ne vilaap kiya, "mere Ram, mat jao."

OUTPUT:
Ram: 'main vanvaas ja raha hun.'

Sita: "main bhi chalungi."

Lakshman: <<main bhi saath chalunga, bhaiya.>>

Dasharath: *mere Ram,, mat jao.*

---

Example 3: Gothic/Horror style (atmospheric, journal narration)

INPUT:
maine apni diary mein likha. "aaj raat kuch ajeeb hua... castle ki deewaron se awaaz aa rahi thi." Count ne kaha, "dariye mat, aap surakshit hain." lekin uski aankhon mein kuch aur hi tha.

OUTPUT:
maine apni diary mein likha.

main: "aaj raat kuch ajeeb hua... | castle ki deewaron se awaaz aa rahi thi."

Count: *dariye mat,, aap surakshit hain.*

lekin uski aankhon mein kuch aur hi tha...

---

Example 4: Complex pronoun resolution

INPUT:
Holmes kamre mein khada tha. Watson darwaze par aaya. usne poocha, "kya hua?" Holmes ne jawaab diya. "kuch nahi," usne kaha. "tumhe yakeen hai?" Watson ne dobara poocha.

OUTPUT:
Holmes kamre mein khada tha. | Watson darwaze par aaya.

Watson: "kya hua?"

Holmes ne jawaab diya.

Holmes: 'kuch nahi.'

Watson: "tumhe yakeen hai?"

---

Example 5: Romance/Social novel style

INPUT:
"yah ek saarvbhaumik satya hai," usne kaha, "ki ek dhanvaan aadmi ko patni ki zaroorat hoti hai." Mrs. Bennet ne kaha, "oh Mr. Bennet! Kya aapne suna? Netherfield aakhirkaar bik gaya!"

OUTPUT:
kathavachak: "yah ek saarvbhaumik satya hai... | ki ek dhanvaan aadmi ko patni ki zaroorat hoti hai."

Mrs. Bennet: *oh Mr. Bennet!! | kya aapne suna? | Netherfield aakhirkaar bik gaya!!*

FINAL CHECKLIST

Before outputting, verify:

- All Roman numerals converted (I to 1, II to 2, etc.)
- All instances of "10" converted to "ten"
- All years have "san" prefix (1988 to san1988)
- All numbers without commas (50,000 to 50000)
- All speakers identified from context (no lazy vakta1, vakta2)
- Each character has consistent punctuation throughout
- All attribution words removed (ne kaha, ne poocha)
- Email/URL dots converted to "dot"
- Abbreviations expanded (Dr. to Daktar/Doctor)
- No metadata, annotations, or extra content added
- Word count preserved
- Genre-appropriate pacing applied

OUTPUT INSTRUCTIONS

Return ONLY the formatted text.

NO explanations.
NO reasoning shown in the output (keep it in your thinking).
NO bullet points.
NO "CHECK:" sections.
NO metadata.

Just clean, perfectly formatted text ready for TTS.

INPUT TEXT:
{text}"""
        return prompt

    def optimize_chunk_streaming(self, chunk: str, chunk_num: int = 1, total_chunks: int = 1, retry_count: int = 3) -> str:
        """
        Optimize a single chunk with streaming output showing thinking process.
        """
        prompt = self.get_optimization_prompt(chunk)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": -1
            }
        }

        for attempt in range(retry_count):
            try:
                response = requests.post(
                    self.ollama_url,
                    json=payload,
                    stream=True,
                    timeout=self.timeout
                )
                response.raise_for_status()

                full_response = ""
                thinking_content = ""
                output_content = ""
                in_thinking = False
                thinking_started = False
                thinking_ended = False

                print(f"\n{'='*60}")
                print(f"  CHUNK {chunk_num}/{total_chunks} ({len(chunk)} chars)")
                print(f"{'='*60}")

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        full_response += token

                        # Detect thinking start
                        for pattern in THINK_START_PATTERNS:
                            if pattern in full_response and not thinking_started:
                                thinking_started = True
                                in_thinking = True
                                print(f"\n--- MODEL REASONING ---")
                                # Remove the tag from display
                                remaining = full_response.split(pattern, 1)[-1]
                                if remaining:
                                    thinking_content += remaining
                                    sys.stdout.write(remaining)
                                    sys.stdout.flush()
                                full_response = ""  # Reset to avoid re-matching
                                break

                        # Detect thinking end
                        if in_thinking:
                            for pattern in THINK_END_PATTERNS:
                                if pattern in token:
                                    in_thinking = False
                                    thinking_ended = True
                                    # Get any content before the end tag
                                    before_tag = token.split(pattern)[0]
                                    if before_tag:
                                        thinking_content += before_tag
                                        sys.stdout.write(before_tag)
                                        sys.stdout.flush()
                                    print(f"\n--- END REASONING ---\n")
                                    print(f"--- FORMATTED OUTPUT ---")
                                    # Get content after end tag
                                    after_tag = token.split(pattern, 1)[-1]
                                    if after_tag.strip():
                                        output_content += after_tag
                                        sys.stdout.write(after_tag)
                                        sys.stdout.flush()
                                    break
                            else:
                                if in_thinking:
                                    thinking_content += token
                                    sys.stdout.write(token)
                                    sys.stdout.flush()
                        elif thinking_ended:
                            # We are past thinking, collecting output
                            output_content += token
                            sys.stdout.write(token)
                            sys.stdout.flush()
                        elif not thinking_started:
                            # Model might not use thinking tags, treat as direct output
                            output_content += token
                            sys.stdout.write(token)
                            sys.stdout.flush()

                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

                print(f"\n{'='*60}")

                # If we never entered thinking mode, the full response is the output
                if not thinking_started:
                    output_content = full_response

                # Clean the output
                cleaned = self._clean_output(output_content)
                print(f"[OK] Chunk {chunk_num}/{total_chunks} complete! ({len(cleaned)} chars output)")

                return cleaned

            except requests.exceptions.Timeout:
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"\n[WARN] Timeout on attempt {attempt + 1}/{retry_count}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"\n[ERROR] Failed after {retry_count} attempts due to timeout")
                    raise
            except Exception as e:
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"\n[WARN] Error on attempt {attempt + 1}/{retry_count}: {e}")
                    print(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"\n[ERROR] Failed after {retry_count} attempts: {e}")
                    raise

        return None

    def optimize(self, text: str) -> str:
        """Optimize text with automatic chunking, streaming thinking display."""
        chunks = self.chunk_text(text)

        if len(chunks) == 1:
            print(f"\nProcessing single chunk ({len(text)} chars)...")
            return self.optimize_chunk_streaming(chunks[0], 1, 1)

        print(f"\nProcessing {len(chunks)} chunks with visible reasoning...")
        optimized_chunks = []

        for idx, chunk in enumerate(chunks, 1):
            try:
                optimized = self.optimize_chunk_streaming(chunk, idx, len(chunks))
                if optimized:
                    optimized_chunks.append(optimized)
                else:
                    print(f"[WARN] Chunk {idx}/{len(chunks)} failed - using original")
                    optimized_chunks.append(chunk)
            except Exception as e:
                print(f"[ERROR] Error processing chunk {idx}: {e}")
                print("   Using original chunk text")
                optimized_chunks.append(chunk)

        final_text = "\n\n".join(optimized_chunks)
        print(f"\n[OK] All chunks processed! Total output: {len(final_text)} characters")
        return final_text

    def _clean_output(self, text: str) -> str:
        """Clean the model output to remove formatting artifacts."""
        # Remove markdown artifacts
        text = text.replace("```", "").replace("**", "")

        # Remove any remaining thinking tags
        for pattern in THINK_START_PATTERNS + THINK_END_PATTERNS:
            text = text.replace(pattern, "")

        # Remove lines that look like metadata/headers
        lines = []
        for line in text.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('OUTPUT'):
                # Skip lines that are just dashes or equals
                if not re.match(r'^[-=]{3,}$', stripped):
                    lines.append(line.rstrip())

        return '\n'.join(lines).strip()


print("[OK] TTSThinkingOptimizer class loaded with streaming reasoning support!")'''

cells.append(make_cell("code", [optimizer_code]))

# ==============================================================================
# Cell 8: Step 4 Markdown
# ==============================================================================
cells.append(make_cell("markdown", [
    "## Step 4: Upload Text and Configure\n",
    "Upload your `.txt` file and set the chunk size. The model's thinking process will be displayed as it works."
]))

# ==============================================================================
# Cell 9: Upload and Configure
# ==============================================================================
cells.append(make_cell("code", [
    "from google.colab import files\n",
    "import ipywidgets as widgets\n",
    "from IPython.display import display\n",
    "\n",
    "print(\"Upload your text file (.txt):\")\n",
    "uploaded = files.upload()\n",
    "\n",
    "if uploaded:\n",
    "    uploaded_filename = list(uploaded.keys())[0]\n",
    "    file_size = len(uploaded[uploaded_filename])\n",
    "    print(f\"[OK] Uploaded: {uploaded_filename} ({file_size:,} bytes)\")\n",
    "else:\n",
    "    print(\"[WARN] No file uploaded yet.\")\n",
    "\n",
    "# Chunk size selector\n",
    "chunk_size_input = widgets.IntSlider(\n",
    "    value=2000,\n",
    "    min=500,\n",
    "    max=5000,\n",
    "    step=100,\n",
    "    description='Chunk Size:',\n",
    "    style={'description_width': 'initial'},\n",
    "    layout=widgets.Layout(width='400px')\n",
    ")\n",
    "\n",
    "print(\"\\nConfiguration:\")\n",
    "display(chunk_size_input)\n",
    "print(\"\\nChunk size guide: smaller = more API calls but less timeout risk\")\n",
    "print(\"   Recommended: 1500-2500 for 14b models, 1000-1500 for 7b models\")"
]))

# ==============================================================================
# Cell 10: Step 5 Markdown
# ==============================================================================
cells.append(make_cell("markdown", [
    "## Step 5: Run Optimization (with Visible Thinking)\n",
    "The model will show its reasoning process as it analyzes and formats each chunk.\n",
    "You will see:\n",
    "- `--- MODEL REASONING ---` : The model thinking through speaker identification, context analysis\n",
    "- `--- FORMATTED OUTPUT ---` : The actual formatted text for TTS"
]))

# ==============================================================================
# Cell 11: Run Optimization
# ==============================================================================
cells.append(make_cell("code", [
    "# Run Optimization with Thinking Display\n",
    "if not uploaded:\n",
    "    print(\"[WARN] Please upload a file in the previous step first!\")\n",
    "else:\n",
    "    try:\n",
    "        text_content = uploaded[uploaded_filename].decode(\"utf-8\")\n",
    "        print(f\"Read {len(text_content):,} characters from file.\")\n",
    "\n",
    "        # Initialize optimizer with selected model\n",
    "        try:\n",
    "            model_to_use = selected_model_name\n",
    "        except NameError:\n",
    "            model_to_use = \"deepseek-r1:14b\"  # Fallback\n",
    "            print(\"[WARN] Using default model: deepseek-r1:14b\")\n",
    "\n",
    "        optimizer = TTSThinkingOptimizer(\n",
    "            model_name=model_to_use,\n",
    "            chunk_size=chunk_size_input.value,\n",
    "            timeout=6000  # 100 minutes per chunk for thinking models\n",
    "        )\n",
    "\n",
    "        print(f\"\\nStarting optimization with {model_to_use}...\")\n",
    "        print(\"The model's reasoning process will be displayed below.\")\n",
    "        print(\"=\" * 60)\n",
    "\n",
    "        start_time = time.time()\n",
    "        optimized_text = optimizer.optimize(text_content)\n",
    "        end_time = time.time()\n",
    "\n",
    "        processing_time = end_time - start_time\n",
    "\n",
    "        if optimized_text:\n",
    "            print(\"\\n\" + \"=\" * 60)\n",
    "            print(\"OPTIMIZATION COMPLETE!\")\n",
    "            print(\"=\" * 60)\n",
    "            print(f\"Processing time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)\")\n",
    "            print(f\"Input length: {len(text_content):,} chars\")\n",
    "            print(f\"Output length: {len(optimized_text):,} chars\")\n",
    "            print(f\"Size change: {((len(optimized_text) - len(text_content)) / len(text_content) * 100):+.1f}%\")\n",
    "\n",
    "            print(\"\\nPreview (First 800 characters):\")\n",
    "            print(\"=\" * 60)\n",
    "            print(optimized_text[:800])\n",
    "            if len(optimized_text) > 800:\n",
    "                print(\"\\n... (truncated)\")\n",
    "            print(\"=\" * 60)\n",
    "\n",
    "            # Save to file\n",
    "            output_filename = f\"tts_optimized_{uploaded_filename}\"\n",
    "            with open(output_filename, 'w', encoding='utf-8') as f:\n",
    "                f.write(optimized_text)\n",
    "\n",
    "            print(f\"\\nSaved to: {output_filename}\")\n",
    "            print(\"Downloading file...\")\n",
    "\n",
    "            # Trigger download\n",
    "            files.download(output_filename)\n",
    "            print(\"\\n[OK] Done! Check your downloads folder.\")\n",
    "\n",
    "        else:\n",
    "            print(\"\\n[ERROR] Optimization failed. Please check the errors above.\")\n",
    "\n",
    "    except Exception as e:\n",
    "        print(f\"\\n[ERROR] Error reading or processing file: {e}\")\n",
    "        import traceback\n",
    "        print(\"\\nFull error details:\")\n",
    "        print(traceback.format_exc())"
]))

# ==============================================================================
# Cell 12: Troubleshooting Markdown
# ==============================================================================
cells.append(make_cell("markdown", [
    "## Troubleshooting\n",
    "\n",
    "**If you get timeouts:**\n",
    "1. Reduce chunk size to 1000-1500 characters\n",
    "2. Use a smaller model: `deepseek-r1:7b` or `qwen3:8b`\n",
    "3. Check Ollama server: `!ollama ps`\n",
    "4. Restart Ollama: Go back to Step 1 and re-run\n",
    "\n",
    "**Thinking models are slower** than regular models because they reason through the text first. This is expected and produces better results, especially for:\n",
    "- Complex dialogue with many speakers\n",
    "- Pronoun resolution in long passages\n",
    "- Genre-appropriate formatting\n",
    "\n",
    "**For very large files (100k+ chars):**\n",
    "- Use chunk size of 1000 characters\n",
    "- Consider splitting the file manually\n",
    "- Processing will take longer but will be more reliable"
]))

# ==============================================================================
# Save the notebook
# ==============================================================================
output_path = "TTS_Thinking_Optimizer.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"[OK] Created {output_path}")
print(f"    Cells: {len(cells)}")
print(f"    Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
print(f"    Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
