#!/usr/bin/env python3
"""
Context-Aware Book Summarization Tool
Enhanced version with length control and improved prompting
Supports Ollama & HuggingFace
"""

import ollama
import os, sys, time, json, argparse, warnings, re
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------- COLOR SUPPORT ----------------
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    C = True
except ImportError:
    C = False
    class Fore: RED = GREEN = YELLOW = CYAN = MAGENTA = RESET = ""
    class Style: BRIGHT = RESET_ALL = ""

# ---------------- HF SUPPORT ----------------
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ---------------- LENGTH SPECIFICATIONS ----------------
LENGTH_SPECS = {
    "SHORT": {
        "chunk_target": "2-3 sentences",
        "final_ratio": "5-8%",
        "description": "Brief overview hitting only the most critical points"
    },
    "MEDIUM": {
        "chunk_target": "4-6 sentences", 
        "final_ratio": "10-15%",
        "description": "Balanced summary covering main ideas and key details"
    },
    "LONG": {
        "chunk_target": "8-12 sentences",
        "final_ratio": "20-25%", 
        "description": "Comprehensive summary preserving nuance and context"
    }
}

# ---------------- ENHANCED TIERS ----------------
SUMMARY_PROMPTS = {
    "BASIC": {
        "system": """You are a professional book summarizer focused on accuracy and clarity.

CORE PRINCIPLES:
- Preserve factual accuracy above all else
- Capture main ideas, events, and arguments in order
- Use clear, direct language
- NO interpretation, NO opinions, NO embellishment
- Maintain the author's voice and perspective""",
        
        "chunk_user": """TEXT TO SUMMARIZE:
\"\"\"
{chunk}
\"\"\"

CONTEXT FROM PREVIOUS SECTIONS:
{context}

TARGET LENGTH: {length_target}

Create a factual summary that:
1. Captures the key information in this section
2. Connects naturally with what came before
3. Preserves important names, events, and details
4. Uses approximately {length_target}
5. Maintains chronological order

Summary:""",

        "final_user": """You are creating a final cohesive summary from chunk summaries.

TARGET: {final_ratio} of original length
APPROACH: Create ONE flowing narrative (not a list of sections)

CHUNK SUMMARIES:
{summaries}

INSTRUCTIONS:
1. Combine all information into one seamless narrative
2. Remove ALL repetitions and redundancies
3. Maintain chronological/logical flow throughout
4. Preserve all key facts, names, events, and arguments
5. Connect ideas smoothly with transitions
6. Write as if summarizing the complete text directly
7. Target length: {final_ratio} of the original

Create the final summary:"""
    },

    "INTERMEDIATE": {
        "system": """You are an expert analytical summarizer who captures both content and context.

GOALS:
- Capture ideas, arguments, and their relationships
- Preserve narrative/logical flow and transitions
- Identify themes and patterns
- Connect new content with previous context
- Balance detail with conciseness
- Maintain the author's argumentative structure""",
        
        "chunk_user": """TEXT TO SUMMARIZE:
\"\"\"
{chunk}
\"\"\"

PREVIOUS CONTEXT:
{context}

TARGET LENGTH: {length_target}

Create a thematic summary that:
1. Identifies main ideas and how they connect
2. Preserves the logical flow and argumentation
3. Notes significant transitions or shifts
4. Maintains connection with previous context
5. Uses approximately {length_target}
6. Captures both explicit and implicit themes

Summary:""",

        "final_user": """Create a cohesive analytical summary from these chunk summaries.

TARGET: {final_ratio} of original length
FOCUS: Themes, arguments, and narrative flow

CHUNK SUMMARIES:
{summaries}

INSTRUCTIONS:
1. Synthesize into one flowing analytical narrative
2. Highlight thematic connections and patterns
3. Preserve the author's argumentative arc
4. Remove redundancies while keeping nuance
5. Show how ideas develop and connect
6. Write as a unified, coherent analysis
7. Target approximately {final_ratio} of original length

Create the cohesive summary:"""
    },

    "ADVANCED": {
        "system": """You are a senior literary analyst creating publication-quality summaries.

EXPERTISE:
- Capture intent, subtext, and rhetorical structure
- Explain WHY ideas matter, not just WHAT they are
- Identify authorial choices and their effects
- Preserve logical progression and development
- Synthesize rather than merely condense
- Think like a literary editor preparing reader guides""",
        
        "chunk_user": """TEXT TO ANALYZE:
\"\"\"
{chunk}
\"\"\"

NARRATIVE CONTEXT:
{context}

TARGET LENGTH: {length_target}

Create a sophisticated summary that:
1. Captures both surface content and deeper significance
2. Explains the function of this section in the larger work
3. Notes rhetorical choices and structural elements
4. Preserves the author's logic and progression
5. Identifies subtext and implications
6. Uses approximately {length_target}
7. Connects meaningfully with prior context

Analysis:""",

        "final_user": """Synthesize a sophisticated, publication-quality summary from these analytical summaries.

TARGET: {final_ratio} of original length
STANDARD: Literary analysis quality

CHUNK SUMMARIES:
{summaries}

INSTRUCTIONS:
1. Create ONE seamless narrative synthesis
2. Preserve the work's intellectual architecture
3. Show how ideas develop and interconnect
4. Capture both explicit content and deeper significance
5. Eliminate redundancy while preserving nuance
6. Write with the polish of a published analysis
7. Target approximately {final_ratio} of original length
8. Think like you're writing for a literary journal

Create the final synthesis:"""
    }
}

# ---------------- UTILITIES ----------------
def detect_device():
    """Auto-detect available device (CUDA, ROCm, or CPU)."""
    try:
        if HF_AVAILABLE and torch.cuda.is_available():
            # Check if it's ROCm (AMD) or CUDA (NVIDIA)
            if hasattr(torch.version, 'hip') and torch.version.hip:
                print("🔍 ROCm (AMD GPU) detected")
                return "cuda"  # ROCm uses same interface as CUDA
            else:
                print("🔍 CUDA (NVIDIA GPU) detected")
                return "cuda"
    except:
        pass
    print("🔍 No GPU detected, using CPU")
    return "cpu"

def chunk_text(text, chunk_words=400, overlap=80):
    """Chunk text with overlap for context preservation"""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks

def clean_output(text):
    """Remove thinking tags and code blocks from output"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove common prefixes
    text = re.sub(r'^(Summary:|Analysis:|Here is.*?:|Here\'s.*?:)\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

def estimate_word_count(text):
    """Estimate word count from text"""
    return len(text.split())

# ---------------- PROGRESS ----------------
class Progress:
    def __init__(self, path):
        self.path = path
        self.data = self.load()

    def load(self):
        if os.path.exists(self.path):
            return json.load(open(self.path))
        return {"done": [], "context": "", "chunk_summaries": []}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

# ---------------- MODEL PROVIDER ----------------
class Provider:
    def __init__(self, provider, model, device):
        self.provider = provider
        self.model = model
        # Auto-detect AMD GPU if not specified
        if device == "cpu":
            self.device = detect_device()
        else:
            self.device = device
        self.pipeline = None

    def load(self):
        if self.provider == "ollama":
            try:
                ollama.show(self.model)
                return True
            except:
                print(f"{Fore.YELLOW}Pulling model {self.model}...{Style.RESET_ALL}")
                os.system(f"ollama pull {self.model}")
                return True

        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace not installed")

        print(f"{Fore.CYAN}Loading HuggingFace model...{Style.RESET_ALL}")
        
        # Use appropriate dtype based on device
        if self.device in ["cuda"]:
            torch_dtype = torch.float16
            device_map = "auto"
        else:
            torch_dtype = torch.float32
            device_map = None
        
        tok = AutoTokenizer.from_pretrained(self.model)
        mdl = AutoModelForCausalLM.from_pretrained(
            self.model,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
        self.pipeline = pipeline("text-generation", model=mdl, tokenizer=tok)
        
        # Show device type for better user feedback
        if self.device == "cuda":
            device_type = "ROCm" if (hasattr(torch.version, 'hip') and torch.version.hip) else "CUDA"
            print(f"{Fore.GREEN}✅ Model loaded on {device_type}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}✅ Model loaded on CPU{Style.RESET_ALL}")
        
        return True

    def generate(self, system, user, max_tokens=1500):
        if self.provider == "ollama":
            out = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                options={
                    "temperature": 0.2,  # Lower for more consistent summaries
                    "num_ctx": 8192,
                    "num_predict": max_tokens
                }
            )
            return out["message"]["content"]

        prompt = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:"
        r = self.pipeline(prompt, max_new_tokens=max_tokens, temperature=0.2)
        return r[0]["generated_text"].split("Assistant:")[-1]

# ---------------- MAIN ----------------
def main():
    p = argparse.ArgumentParser(
        description="Context-aware book summarization with length control"
    )
    p.add_argument("input", help="Input text file to summarize")
    p.add_argument("-o", "--output", default="summary.txt", 
                   help="Output file for summary (default: summary.txt)")
    p.add_argument("-t", "--tier", choices=SUMMARY_PROMPTS.keys(), default="INTERMEDIATE",
                   help="Summary quality tier (default: INTERMEDIATE)")
    p.add_argument("-l", "--length", choices=LENGTH_SPECS.keys(), default="MEDIUM",
                   help="Summary length: SHORT/MEDIUM/LONG (default: MEDIUM)")
    p.add_argument("-ol", "--ollama", action="store_true",
                   help="Use Ollama provider")
    p.add_argument("-hf", "--huggingface", action="store_true",
                   help="Use HuggingFace provider")
    p.add_argument("-m", "--model", required=True,
                   help="Model name (e.g., llama3.1, mistral)")
    p.add_argument("--chunk-words", type=int, default=500,
                   help="Words per chunk (default: 500)")
    p.add_argument("--overlap", type=int, default=100,
                   help="Overlap between chunks (default: 100)")
    p.add_argument("--device", choices=['cpu', 'cuda', 'rocm', 'auto'], default='auto',
                   help="Device to use (auto-detects CUDA/ROCm if available)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from previous progress")
    args = p.parse_args()

    # Handle device argument
    if args.device == "auto":
        device = "cpu"  # Will be auto-detected in Provider
    elif args.device == "rocm":
        device = "cuda"  # ROCm uses CUDA interface
    else:
        device = args.device

    # Make output path absolute
    args.output = os.path.abspath(args.output)

    # Validate provider selection
    if not args.ollama and not args.huggingface:
        print(f"{Fore.RED}Error: Must specify either --ollama or --huggingface{Style.RESET_ALL}")
        sys.exit(1)

    provider_type = "ollama" if args.ollama else "huggingface"

    print(f"{Fore.CYAN}{Style.BRIGHT}=== Book Summarization Tool ==={Style.RESET_ALL}")
    print(f"Provider: {provider_type}")
    print(f"Model: {args.model}")
    print(f"Tier: {args.tier}")
    print(f"Length: {args.length} ({LENGTH_SPECS[args.length]['description']})")
    print(f"Device: {device}")
    print()

    # Initialize provider
    prov = Provider(provider_type, args.model, device)
    prov.load()

    # Load and chunk text
    text = Path(args.input).read_text(encoding="utf-8")
    original_words = estimate_word_count(text)
    chunks = chunk_text(text, args.chunk_words, args.overlap)
    
    print(f"{Fore.GREEN}Loaded text: {original_words} words, {len(chunks)} chunks{Style.RESET_ALL}")
    print()

    # Progress tracking
    progress = Progress(args.output + ".progress.json")
    context = progress.data.get("context", "")
    chunk_summaries = progress.data.get("chunk_summaries", [])

    # Get prompts and length specs
    prompts = SUMMARY_PROMPTS[args.tier]
    length_spec = LENGTH_SPECS[args.length]

    # Process chunks
    for i, chunk in enumerate(chunks, 1):
        if str(i) in progress.data["done"]:
            print(f"{Fore.BLUE}Skipping chunk {i}/{len(chunks)} (already processed){Style.RESET_ALL}")
            continue

        print(f"{Fore.YELLOW}{Style.BRIGHT}► Processing chunk {i}/{len(chunks)}{Style.RESET_ALL}")

        chunk_prompt = prompts["chunk_user"].format(
            chunk=chunk,
            context=context[-2000:],  # Last 2000 chars of context
            length_target=length_spec["chunk_target"]
        )

        result = prov.generate(prompts["system"], chunk_prompt, max_tokens=800)
        result = clean_output(result)
        
        chunk_summaries.append(result)
        context += "\n\n" + result
        
        progress.data["done"].append(str(i))
        progress.data["context"] = context
        progress.data["chunk_summaries"] = chunk_summaries
        progress.save()
        
        print(f"{Fore.GREEN}✓ Chunk {i} summarized ({estimate_word_count(result)} words){Style.RESET_ALL}")

    # Generate final cohesive summary
    if chunk_summaries:
        print()
        print(f"{Fore.CYAN}{Style.BRIGHT}► Generating final cohesive summary...{Style.RESET_ALL}")
        
        final_prompt = prompts["final_user"].format(
            summaries="\n\n---\n\n".join(chunk_summaries),
            final_ratio=length_spec["final_ratio"]
        )
        
        # Use higher token limit for final summary
        final_summary = prov.generate(
            prompts["system"],
            final_prompt,
            max_tokens=2500
        )
        final_summary = clean_output(final_summary)
        
        # Calculate compression ratio
        final_words = estimate_word_count(final_summary)
        compression = (final_words / original_words) * 100
        
        # Write final summary
        with open(args.output, "w", encoding="utf-8") as out:
            out.write(final_summary)
        
        print()
        print(f"{Fore.GREEN}{Style.BRIGHT}✓ Summary complete!{Style.RESET_ALL}")
        print(f"Original: {original_words} words")
        print(f"Summary: {final_words} words ({compression:.1f}% of original)")
        print(f"Output: {args.output}")
        
        # Clean up progress file
        if os.path.exists(progress.path):
            os.remove(progress.path)
            
    else:
        print(f"{Fore.YELLOW}No new chunks to process.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
