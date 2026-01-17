"""
Summary Engine for ReadLyte MVP
Handles text summarization using Ollama or HuggingFace models
"""

import re
import warnings
warnings.filterwarnings("ignore")

# Try to import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Try to import HuggingFace
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# Length specifications
LENGTH_SPECS = {
    "SHORT": {
        "target": "2-3 sentences",
        "ratio": "5-10%",
        "description": "Brief overview of key points"
    },
    "MEDIUM": {
        "target": "4-6 sentences",
        "ratio": "15-20%",
        "description": "Balanced summary covering main ideas"
    },
    "LONG": {
        "target": "8-12 sentences",
        "ratio": "25-35%",
        "description": "Comprehensive summary preserving nuance"
    }
}

# Summary prompts by tier
SUMMARY_PROMPTS = {
    "BASIC": {
        "system": """You are a professional summarizer focused on accuracy and clarity.

PRINCIPLES:
- Preserve factual accuracy
- Capture main ideas in order
- Use clear, direct language
- NO interpretation or opinions
- Maintain author's perspective""",
        
        "user": """Summarize this text:

"{text}"

TARGET LENGTH: {length_target}

Create a factual summary that:
1. Captures key information
2. Preserves important names and events
3. Uses approximately {length_target}
4. Maintains chronological order

Summary:"""
    },
    
    "INTERMEDIATE": {
        "system": """You are an expert analytical summarizer.

GOALS:
- Capture ideas and their relationships
- Preserve narrative/logical flow
- Identify themes and patterns
- Balance detail with conciseness
- Maintain argumentative structure""",
        
        "user": """Summarize this text analytically:

"{text}"

TARGET LENGTH: {length_target}

Create a thematic summary that:
1. Identifies main ideas and connections
2. Preserves logical flow
3. Notes significant transitions
4. Uses approximately {length_target}
5. Captures explicit and implicit themes

Summary:"""
    },
    
    "ADVANCED": {
        "system": """You are a senior literary analyst creating publication-quality summaries.

EXPERTISE:
- Capture intent, subtext, and rhetorical structure
- Explain WHY ideas matter, not just WHAT they are
- Identify authorial choices and their effects
- Synthesize rather than merely condense
- Think like a literary editor""",
        
        "user": """Create a sophisticated summary:

"{text}"

TARGET LENGTH: {length_target}

Create an analytical summary that:
1. Captures surface content and deeper significance
2. Explains the function in the larger work
3. Notes rhetorical choices
4. Preserves logic and progression
5. Identifies subtext and implications
6. Uses approximately {length_target}

Analysis:"""
    }
}

# Recommended models
MODEL_RECOMMENDATIONS = {
    "FAST": ["llama3.2:3b", "qwen2.5:3b", "phi3:3.8b"],
    "QUALITY": ["llama3.1:8b", "qwen2.5:7b", "mistral:7b"]
}


def clean_summary(text: str) -> str:
    """Clean up summary output."""
    # Remove thinking tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove markdown code blocks
    text = re.sub(r'```\w*\n?', '', text)
    # Remove common prefixes
    text = re.sub(r'^(Summary:|Analysis:|Here is.*?:|Here\'s.*?:)\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


def summarize_text_ollama(text: str, model: str, tier: str, 
                          length_type: str, temperature: float = 0.2) -> str:
    """Summarize using Ollama."""
    if not OLLAMA_AVAILABLE:
        raise ImportError("Ollama not installed. Run: pip install ollama")
    
    prompts = SUMMARY_PROMPTS.get(tier, SUMMARY_PROMPTS["BASIC"])
    length_spec = LENGTH_SPECS.get(length_type, LENGTH_SPECS["MEDIUM"])
    
    user_prompt = prompts["user"].format(
        text=text,
        length_target=length_spec["target"]
    )
    
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": user_prompt}
        ],
        options={
            "temperature": temperature,
            "num_ctx": 8192,
            "num_predict": 1500
        }
    )
    
    return clean_summary(response["message"]["content"])


def summarize_text_huggingface(text: str, model: str, tier: str,
                               length_type: str, temperature: float = 0.2,
                               device: str = "cpu") -> str:
    """Summarize using HuggingFace model."""
    if not HF_AVAILABLE:
        raise ImportError("transformers not installed. Run: pip install transformers torch")
    
    prompts = SUMMARY_PROMPTS.get(tier, SUMMARY_PROMPTS["BASIC"])
    length_spec = LENGTH_SPECS.get(length_type, LENGTH_SPECS["MEDIUM"])
    
    user_prompt = prompts["user"].format(
        text=text,
        length_target=length_spec["target"]
    )
    full_prompt = f"System: {prompts['system']}\n\nUser: {user_prompt}\n\nAssistant:"
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    pipe = pipeline(
        "text-generation",
        model=model_obj,
        tokenizer=tokenizer,
        max_new_tokens=1500,
        temperature=temperature,
        do_sample=True
    )
    
    response = pipe(full_prompt)
    generated = response[0]['generated_text']
    summary = generated.split("Assistant:")[-1].strip()
    
    return clean_summary(summary)


def summarize_text(text: str, model: str, tier: str = "INTERMEDIATE",
                   length_type: str = "MEDIUM", provider: str = "ollama",
                   temperature: float = 0.2) -> str:
    """
    Main summarization function.
    
    Args:
        text: Text to summarize
        model: Model name (e.g., "llama3.1:8b")
        tier: Quality tier - BASIC, INTERMEDIATE, ADVANCED
        length_type: Summary length - SHORT, MEDIUM, LONG
        provider: "ollama" or "huggingface"
        temperature: Generation temperature (0.1-1.0)
    
    Returns:
        Summary text
    """
    if not text or not text.strip():
        return ""
    
    if provider == "ollama":
        return summarize_text_ollama(text, model, tier, length_type, temperature)
    elif provider == "huggingface":
        return summarize_text_huggingface(text, model, tier, length_type, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def chunk_for_summary(text: str, chunk_words: int = 500, overlap: int = 100) -> list:
    """Split text into chunks for summarization with overlap."""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def summarize_long_text(text: str, model: str, tier: str = "INTERMEDIATE",
                        length_type: str = "MEDIUM", provider: str = "ollama",
                        chunk_words: int = 500) -> str:
    """
    Summarize long text by chunking and combining.
    
    For texts longer than chunk_words, splits into chunks,
    summarizes each, then combines into final summary.
    """
    word_count = len(text.split())
    
    # If text is short enough, summarize directly
    if word_count <= chunk_words:
        return summarize_text(text, model, tier, length_type, provider)
    
    # Chunk the text
    chunks = chunk_for_summary(text, chunk_words)
    
    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarize_text(chunk, model, tier, length_type, provider)
        chunk_summaries.append(summary)
    
    # Combine summaries
    combined = "\n\n".join(chunk_summaries)
    
    # If combined is still too long, summarize again
    if len(combined.split()) > chunk_words:
        return summarize_text(combined, model, tier, length_type, provider)
    
    return combined


def get_length_info() -> dict:
    """Get length specification info for UI display."""
    return LENGTH_SPECS


def get_available_models(provider: str = "ollama") -> list:
    """Get list of available summarization models."""
    if provider == "ollama" and OLLAMA_AVAILABLE:
        try:
            models = ollama.list()
            return [m['name'] for m in models.get('models', [])]
        except:
            return MODEL_RECOMMENDATIONS["FAST"] + MODEL_RECOMMENDATIONS["QUALITY"]
    return MODEL_RECOMMENDATIONS["FAST"] + MODEL_RECOMMENDATIONS["QUALITY"]


if __name__ == "__main__":
    # Test summarization
    test_text = """
    The Industrial Revolution, which began in Britain in the late 18th century, 
    fundamentally transformed human society. It marked the transition from 
    hand production methods to machines, new chemical manufacturing and iron 
    production processes, improved water power, the development of machine tools, 
    and the rise of the factory system. The textile industry was the first to use 
    modern production methods, and textiles became the dominant industry in terms 
    of employment, value of output, and capital invested.
    """
    
    print("🧪 Testing summary engine...")
    print(f"📝 Input: {len(test_text.split())} words")
    
    if OLLAMA_AVAILABLE:
        try:
            for length in ["SHORT", "MEDIUM", "LONG"]:
                result = summarize_text(test_text, "llama3.2:3b", "BASIC", length, "ollama")
                print(f"\n{length}: {result}")
                print(f"  ({len(result.split())} words)")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("⚠️ Ollama not available")
