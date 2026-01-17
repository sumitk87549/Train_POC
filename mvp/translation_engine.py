"""
Translation Engine for ReadLyte MVP
Handles Hindi translation using Ollama or HuggingFace models
"""

import re
import time
from typing import Optional, Generator
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


# Translation prompts by tier
TRANSLATION_PROMPTS = {
    "BASIC": {
        "system": """You are a Hindi literary translator. Translate English to Hindi faithfully.

RULES:
1. TRANSLATE EVERYTHING - every word, sentence, paragraph
2. NO SUMMARIZATION - faithful translation only
3. Proper transliteration: London → लंदन, Watson → वॉटसन
4. Maintain paragraph structure
5. Natural Hindi that flows like original prose""",
        
        "user": """Translate this English text to Hindi completely:

"{text}"

Hindi translation:"""
    },
    
    "INTERMEDIATE": {
        "system": """You are an expert Hindi literary translator. Create natural Hindi translations.

COMMANDMENTS:
1. TRANSLATE EVERYTHING - every word, comma, nuance
2. ZERO SUMMARIZATION - this is translation, not reduction
3. LENGTH PRESERVATION - Hindi ≈ same length as English
4. ALL DIALOGUE translated with character voice
5. ALL DESCRIPTION preserved completely

GUIDELINES:
- Transform idioms naturally to Hindi
- Preserve character voices
- Keep narrative pacing
- London → लंदन, Doctor → डॉक्टर""",
        
        "user": """COMPLETE TRANSLATION REQUIRED. NO SUMMARIZATION.

Translate ENTIRE passage to Hindi. Every sentence. Every detail.

English:
"{text}"

Complete Hindi translation:"""
    },
    
    "ADVANCED": {
        "system": """You are a master literary translator creating publication-quality Hindi translations.

SUPREME MANDATE: COMPLETE, FAITHFUL, BEAUTIFUL TRANSLATION

CRITICAL REQUIREMENTS:
1. ABSOLUTE COMPLETENESS - Translate EVERY word, sentence, paragraph
2. ZERO SUMMARIZATION - The gravest translator's sin
3. LENGTH PRESERVATION - Hindi ≈ 0.9-1.2x English
4. ALL DIALOGUE with character voice preserved
5. ALL DESCRIPTIONS with full detail
6. TECHNICAL PRECISION in transliteration

QUALITY BENCHMARKS:
- COMPLETE: 100% of original
- ACCURATE: Faithful to source
- NATURAL: Feels like original Hindi prose
- PUBLISHABLE: Professional quality""",
        
        "user": """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL TRANSLATION TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIREMENTS:
✓ Translate EVERY sentence
✓ Translate EVERY detail
✓ Maintain EVERY paragraph
✓ Include ALL dialogue
✓ Preserve ALL descriptions
✓ Keep similar length (0.9-1.2x)

❌ FORBIDDEN: NO summarization, NO condensing

English:
"{text}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complete Hindi translation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    }
}

# Model recommendations
MODEL_TIERS = {
    "FAST": ["qwen2.5:3b", "phi3.5:3.8b", "llama3.2:3b"],
    "BALANCED": ["qwen2.5:7b", "deepseek-r1:7b", "llama3.1:8b"],
    "QUALITY": ["qwen2.5:14b", "deepseek-r1:14b", "mistral:7b"]
}


def clean_translation(text: str) -> str:
    """Clean up translation output."""
    # Remove thinking tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove markdown code blocks
    text = re.sub(r'```\w*\n?', '', text)
    # Remove translation markers
    text = re.sub(r'(Translation:|Hindi Translation:|Here\'s the translation:)', '', text, flags=re.IGNORECASE)
    # Clean whitespace
    lines = [line.strip() for line in text.split('\n')]
    text = '\n\n'.join(line for line in lines if line)
    return text.strip()


def translate_text_ollama(text: str, model: str, tier: str, 
                          temperature: float = 0.3, stream: bool = False) -> str:
    """Translate using Ollama."""
    if not OLLAMA_AVAILABLE:
        raise ImportError("Ollama not installed. Run: pip install ollama")
    
    prompts = TRANSLATION_PROMPTS.get(tier, TRANSLATION_PROMPTS["BASIC"])
    user_prompt = prompts["user"].format(text=text)
    
    if stream:
        # Streaming mode - returns generator
        return _translate_ollama_stream(prompts["system"], user_prompt, model, temperature)
    else:
        # Non-streaming
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": temperature,
                "num_ctx": 8192
            }
        )
        return clean_translation(response["message"]["content"])


def _translate_ollama_stream(system: str, user: str, model: str, 
                              temperature: float) -> Generator[dict, None, None]:
    """
    Stream translation from Ollama.
    
    Yields dict with:
        - type: 'thinking' | 'translation' | 'status'
        - content: text content
        - full_text: accumulated text so far (for translation)
    """
    full_text = ""
    thinking_text = ""
    translation_text = ""
    in_thinking = False
    
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        options={
            "temperature": temperature,
            "num_ctx": 8192
        },
        stream=True
    )
    
    for chunk in stream:
        if 'message' in chunk and 'content' in chunk['message']:
            token = chunk['message']['content']
            full_text += token
            
            # Detect thinking tags
            if '<think>' in token:
                in_thinking = True
                token = token.replace('<think>', '')
                yield {'type': 'status', 'content': '🧠 Model is thinking...'}
            
            if '</think>' in token:
                in_thinking = False
                token = token.replace('</think>', '')
                yield {'type': 'status', 'content': '✅ Thinking complete, generating translation...'}
            
            # Route to appropriate output
            if in_thinking:
                thinking_text += token
                yield {'type': 'thinking', 'content': token, 'full_text': thinking_text}
            else:
                translation_text += token
                yield {'type': 'translation', 'content': token, 'full_text': translation_text}
    
    # Return final cleaned text
    yield {'type': 'complete', 'content': clean_translation(translation_text), 'full_text': full_text}


def translate_text_huggingface(text: str, model: str, tier: str,
                               temperature: float = 0.3, device: str = "cpu") -> str:
    """Translate using HuggingFace model."""
    if not HF_AVAILABLE:
        raise ImportError("transformers not installed. Run: pip install transformers torch")
    
    prompts = TRANSLATION_PROMPTS.get(tier, TRANSLATION_PROMPTS["BASIC"])
    user_prompt = prompts["user"].format(text=text)
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
        max_new_tokens=4096,
        temperature=temperature,
        do_sample=True
    )
    
    response = pipe(full_prompt)
    generated = response[0]['generated_text']
    translation = generated.split("Assistant:")[-1].strip()
    
    return clean_translation(translation)


def translate_text(text: str, model: str, tier: str = "BASIC",
                   provider: str = "ollama", language: str = "hindi",
                   temperature: float = 0.3, stream: bool = False) -> str:
    """
    Main translation function.
    
    Args:
        text: Text to translate
        model: Model name (e.g., "qwen2.5:3b")
        tier: Quality tier - BASIC, INTERMEDIATE, ADVANCED
        provider: "ollama" or "huggingface"
        language: Target language (default: hindi)
        temperature: Generation temperature (0.1-1.0)
        stream: Whether to stream output (Ollama only)
    
    Returns:
        Translated text in Hindi
    """
    if not text or not text.strip():
        return ""
    
    # Currently only Hindi is implemented
    if language != "hindi":
        raise ValueError(f"Language '{language}' not yet supported. Only 'hindi' is available.")
    
    if provider == "ollama":
        return translate_text_ollama(text, model, tier, temperature, stream)
    elif provider == "huggingface":
        return translate_text_huggingface(text, model, tier, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def chunk_text(text: str, chunk_words: int = 350) -> list:
    """Split text into translatable chunks."""
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = []
    current_count = 0
    
    for para in paragraphs:
        para_words = len(para.split())
        
        if para_words > chunk_words:
            # Save current chunk
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_count = 0
            
            # Split long paragraph
            words = para.split()
            for i in range(0, len(words), chunk_words):
                chunk_text = ' '.join(words[i:i + chunk_words])
                chunks.append(chunk_text)
        else:
            if current_count + para_words > chunk_words and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_count = para_words
            else:
                current_chunk.append(para)
                current_count += para_words
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def get_available_models(provider: str = "ollama") -> list:
    """Get list of available translation models."""
    if provider == "ollama" and OLLAMA_AVAILABLE:
        try:
            models = ollama.list()
            return [m['name'] for m in models.get('models', [])]
        except:
            return MODEL_TIERS["FAST"] + MODEL_TIERS["BALANCED"] + MODEL_TIERS["QUALITY"]
    return MODEL_TIERS["FAST"] + MODEL_TIERS["BALANCED"] + MODEL_TIERS["QUALITY"]


if __name__ == "__main__":
    # Test translation
    test_text = "The quick brown fox jumps over the lazy dog. It was a beautiful morning."
    
    print("🧪 Testing translation engine...")
    print(f"📝 Input: {test_text}")
    
    if OLLAMA_AVAILABLE:
        try:
            result = translate_text(test_text, "qwen2.5:3b", "BASIC", "ollama")
            print(f"✅ Hindi: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("⚠️ Ollama not available")
