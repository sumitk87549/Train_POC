#!/usr/bin/env python3
"""
TTS Text Optimizer for desivocal.com
Optimizes translated text with proper punctuation for natural voice generation
Supports: Ollama (local) and HuggingFace Inference API
"""

import requests
import json
import sys
import os
from pathlib import Path
from typing import Optional

class TTSOptimizer:
    """Optimizes text for desivocal.com TTS generation"""
    
    def __init__(self, backend="ollama", model_name=None, hf_token=None):
        """
        Initialize TTS Optimizer
        
        Args:
            backend: "ollama" or "huggingface"
            model_name: Model to use (if None, uses default)
            hf_token: HuggingFace API token (required for HF backend)
        """
        self.backend = backend
        
        if backend == "ollama":
            self.ollama_url = "http://localhost:11434/api/generate"
            # Best free Ollama models for this task
            self.model = model_name or "qwen2.5:14b"  # Default
            print(f"🤖 Using Ollama model: {self.model}")
            
        elif backend == "huggingface":
            if not hf_token:
                hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("HuggingFace token required! Set HF_TOKEN env var or pass hf_token parameter")
            
            self.hf_token = hf_token
            # Best free HF models for this task
            self.model = model_name or "Qwen/Qwen2.5-14B-Instruct"
            self.hf_url = f"https://api-inference.huggingface.co/models/{self.model}"
            print(f"🤖 Using HuggingFace model: {self.model}")
        
        else:
            raise ValueError("Backend must be 'ollama' or 'huggingface'")
    
    def get_optimization_prompt(self, text: str, language: str = "Hindi") -> str:
        """
        Creates the TTS optimization prompt based on desivocal.com best practices
        
        Args:
            text: Translated text to optimize
            language: Target language (Hindi/Tamil/Bengali etc.)
        """
        
        prompt = f"""You are a TTS punctuation expert optimizing text for natural voice generation on desivocal.com (Indian language TTS platform).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Add PROPER PUNCTUATION to make this {language} text sound NATURAL when read by TTS voice-over system.

DESIVOCAL.COM BEST PRACTICES (MANDATORY):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✓ AMPLE PUNCTUATION for natural pauses and modulation:
   - Use periods (.) to break sentences (max 15-20 words per sentence)
   - Use commas (,) for natural breathing pauses (every 8-12 words)
   - Use question marks (?) for questions
   - Use exclamation marks (!) for emphasis/excitement

2. ✓ MULTIPLE PUNCTUATIONS for emotion/expression:
   - ??? for strong doubt/confusion/repeated questions
   - !!! for excitement/shock/strong emotion
   - ... for hesitation/suspense/trailing off
   - Use these ONLY when appropriate for natural speech

3. ✓ ABBREVIATIONS with dots:
   - AI → A.I.
   - ISO → I.S.O.
   - USA → U.S.A.
   - PhD → Ph.D.

4. ✓ DIALOGUE formatting:
   - Questions should end with ? or ???
   - Excited speech with ! or !!!
   - Keep natural conversation flow

5. ✓ SENTENCE BREAKING (CRITICAL):
   - Break long sentences into shorter ones
   - Each sentence should be 10-20 words maximum
   - Add periods to create natural pause points

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES OF GOOD TTS OPTIMIZATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ BAD (No punctuation, long sentences):
"एक बुद्धिमान राजा था उसका काफी बड़ा साम्राज्य था उसके राज्य में प्रजा हर तरह से खुशहाल थी"

✓ GOOD (Proper punctuation, natural breaks):
"एक बुद्धिमान राजा था। उसका काफी बड़ा साम्राज्य था। उसके राज्य में प्रजा हर तरह से खुशहाल थी।"

❌ BAD (Single question mark):
"क्या आप वॉइस ओवर उत्पन्न करने में सक्षम हैं"

✓ GOOD (Multiple punctuation for emphasis):
"क्या आप वॉइस ओवर उत्पन्न करने में सक्षम हैं??? सर्वश्रेष्ठ भारतीय वॉयस ओवर यहाँ हैं!"

❌ BAD (No emotion markers):
"अगर यह काम आज के आज नहीं हुआ तो तुम्हारी फैमिली कल का सूरज नहीं देख पाएगी"

✓ GOOD (Emotional punctuation):
"अगर यह काम आज के आज नहीं हुआ तो... तुम और तुम्हारी पूरी फैमिली कल का सूरज नहीं देख पाओगे!!!"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES (DO NOT VIOLATE):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✗ DO NOT change ANY words
✗ DO NOT add or remove content
✗ DO NOT translate anything
✗ DO NOT use SSML tags like <break>, <emphasis>, <speak>
✗ DO NOT use special tags like [laugh], <smile>, etc.

✓ ONLY add punctuation marks: . , ? ! ??? !!! ...
✓ ONLY break long sentences with periods
✓ ONLY add dots in abbreviations
✓ Keep 100% of original words intact

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT TEXT TO OPTIMIZE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT (TTS-OPTIMIZED VERSION):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY the optimized text with proper punctuation. No explanations, no extra text, ONLY the optimized version."""

        return prompt
    
    def optimize_with_ollama(self, text: str, language: str = "Hindi") -> str:
        """Optimize text using Ollama local model"""
        
        prompt = self.get_optimization_prompt(text, language)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,  # Low temperature for consistent punctuation
                "top_p": 0.9,
                "num_predict": -1  # No limit on response length
            }
        }
        
        print(f"📤 Sending to Ollama ({self.model})...")
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            optimized_text = result.get("response", "").strip()
            
            # Clean up any markdown or extra formatting
            optimized_text = self._clean_output(optimized_text)
            
            print("✅ Optimization complete!")
            return optimized_text
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error calling Ollama: {e}")
            print("💡 Make sure Ollama is running: ollama serve")
            sys.exit(1)
    
    def optimize_with_huggingface(self, text: str, language: str = "Hindi") -> str:
        """Optimize text using HuggingFace Inference API"""
        
        prompt = self.get_optimization_prompt(text, language)
        
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2000,
                "temperature": 0.3,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        
        print(f"📤 Sending to HuggingFace ({self.model})...")
        
        try:
            response = requests.post(self.hf_url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                optimized_text = result[0].get("generated_text", "").strip()
            elif isinstance(result, dict):
                optimized_text = result.get("generated_text", "").strip()
            else:
                optimized_text = str(result).strip()
            
            # Clean up any markdown or extra formatting
            optimized_text = self._clean_output(optimized_text)
            
            print("✅ Optimization complete!")
            return optimized_text
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error calling HuggingFace API: {e}")
            if response.status_code == 401:
                print("💡 Check your HuggingFace token")
            elif response.status_code == 503:
                print("💡 Model is loading, try again in a moment")
            sys.exit(1)
    
    def _clean_output(self, text: str) -> str:
        """Clean up model output to get only the optimized text"""
        
        # Remove common markdown artifacts
        text = text.replace("```", "")
        text = text.replace("**", "")
        
        # Remove any explanatory text before/after
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, headers, or explanatory text
            if line and not line.startswith('#') and not line.startswith('OUTPUT'):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def optimize(self, text: str, language: str = "Hindi") -> str:
        """
        Main optimization method - routes to appropriate backend
        
        Args:
            text: Text to optimize
            language: Language name (Hindi/Tamil/Bengali etc.)
        
        Returns:
            Optimized text ready for desivocal.com TTS
        """
        
        if self.backend == "ollama":
            return self.optimize_with_ollama(text, language)
        else:
            return self.optimize_with_huggingface(text, language)


def main():
    """Main execution function with examples"""
    
    print("=" * 60)
    print("🎙️  TTS TEXT OPTIMIZER FOR DESIVOCAL.COM")
    print("=" * 60)
    print()
    
    # Example usage with Ollama
    print("📝 Example: Optimizing Hindi text for TTS\n")
    
    # Sample translated text (without proper punctuation)
    sample_text = """शेरलॉक होम्स के लिए वह हमेशा वो औरत थी मैंने उन्हें कभी किसी और नाम से नहीं पुकारते सुना उनकी नज़र में वह सभी औरतों से अलग थी वह उनके लिए बहुत खास थी और वह उन्हें कभी नहीं भूले"""
    
    print("INPUT TEXT (without punctuation):")
    print("-" * 60)
    print(sample_text)
    print("-" * 60)
    print()
    
    # Initialize optimizer (Ollama by default)
    try:
        # Try Ollama first
        optimizer = TTSOptimizer(backend="ollama", model_name="qwen2.5:14b")
        
        # Optimize text
        optimized = optimizer.optimize(sample_text, language="Hindi")
        
        print("\nOUTPUT TEXT (TTS-optimized):")
        print("=" * 60)
        print(optimized)
        print("=" * 60)
        print()
        
        # Save to file
        output_file = "tts_optimized_output.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(optimized)
        
        print(f"✅ Optimized text saved to: {output_file}")
        print(f"📋 Ready to paste into desivocal.com!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure Ollama is running: ollama serve")
        print("   2. Pull the model: ollama pull qwen2.5:14b")
        print("   3. Or try HuggingFace backend instead")


def batch_process(input_file: str, output_file: str, backend: str = "ollama", 
                  model: str = None, language: str = "Hindi", hf_token: str = None):
    """
    Batch process a file with multiple chapters/paragraphs
    
    Args:
        input_file: Path to input text file
        output_file: Path to save optimized text
        backend: "ollama" or "huggingface"
        model: Model name (optional)
        language: Language name
        hf_token: HuggingFace token (if using HF backend)
    """
    
    print(f"📂 Processing file: {input_file}")
    
    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize optimizer
    optimizer = TTSOptimizer(backend=backend, model_name=model, hf_token=hf_token)
    
    # Optimize
    optimized = optimizer.optimize(text, language=language)
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(optimized)
    
    print(f"✅ Optimized text saved to: {output_file}")
    print(f"📊 Original length: {len(text)} chars")
    print(f"📊 Optimized length: {len(optimized)} chars")


if __name__ == "__main__":
    # Run example
    main()
    
    # Uncomment to batch process files:
    # batch_process("chapter1_translated.txt", "chapter1_tts_ready.txt", 
    #               backend="ollama", language="Hindi")