"""
DesiVocal TTS Optimizer - Complete Implementation
Optimized for Gemma3:27b with full DesiVocal TTS quirks handling
"""

import requests
import json
import time
import re


class DesiVocalTTSOptimizer:
    """
    Optimizes text for DesiVocal.com TTS with specific handling for:
    - Numbers (no commas)
    - Dates (written format)
    - Abbreviations (expanded)
    - Special characters (written out)
    - And 50+ other DesiVocal-specific quirks
    """
    
    def __init__(self, model_name="gemma3:27b", chunk_size=2000, timeout=600):
        """
        Initialize optimizer
        
        Args:
            model_name: Ollama model (recommend: gemma3:27b)
            chunk_size: Max characters per chunk (1500-2500 recommended)
            timeout: Timeout per chunk in seconds
        """
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = model_name
        self.chunk_size = chunk_size
        self.timeout = timeout
        
        print(f"🤖 DesiVocal TTS Optimizer initialized")
        print(f"   Model: {self.model}")
        print(f"   Chunk size: {self.chunk_size} chars")
        print(f"   Timeout: {self.timeout}s per chunk")
    
    def get_optimization_prompt(self, text: str, language: str = "Hindi") -> str:
        """
        Get the optimized prompt for DesiVocal TTS
        This prompt includes all discovered quirks and rules
        """
        prompt = f"""You are an expert TTS text optimizer for DesiVocal.com voice generation system.

Your ONLY task: Transform text to work perfectly with DesiVocal TTS by fixing numbers, dates, symbols, and abbreviations while preserving 100% of the original words.

═══════════════════════════════════════════════════════════════
🎯 CORE PRINCIPLE: PRESERVE ALL WORDS, FIX ONLY FORMAT
═══════════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 CRITICAL TRANSFORMATIONS (DesiVocal-Specific):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NUMBERS - NEVER USE COMMAS:
   ❌ 50,000 रुपये → Sounds like "50 rupaye" 
   ✅ 50000 रुपये → Correct
   ✅ पचास हजार रुपये → Best
   
   ❌ 1,50,000 → Wrong
   ✅ 150000 → Correct
   ✅ एक लाख पचास हजार → Best

2. DATES - ALWAYS WRITE IN WORDS:
   ❌ 15/03/2024 → Sounds like "fifteen slash oh three slash..."
   ❌ 15-03-2024 → Same issue
   ✅ 15 मार्च 2024 → Good
   ✅ पंद्रह मार्च 2024 → Best

3. TIME - WRITE IN WORDS:
   ❌ 3:30 → Sounds like "teen tees"
   ✅ साढ़ेतीन बजे → Best
   ✅ तीन बजकर तीस मिनट → Also good

4. NUMBER 10 - SPECIAL CASE:
   ❌ 10 → Not spoken at all!
   ❌ दस → Also not working!
   ✅ ten → Works (English)
   Note: This is a known bug in DesiVocal

5. ZEROS IN DATES:
   ❌ 07/09/2024 → Sounds like "seven nine two thousand..."
   ✅ 7 सितंबर 2024 → Correct
   ✅ सात सितंबर 2024 → Best

6. RANGES - USE "से" NOT HYPHEN:
   ❌ 8-10 गिलास → Wrong pronunciation
   ✅ 8से10 गिलास → Good
   ✅ आठ से दस गिलास → Best
   
   ❌ 3-5 घंटे → Wrong
   ✅ 3से5 घंटे → Good
   ✅ तीन से पांच घंटे → Best

7. PERCENTAGES:
   ❌ 50% → Often skipped
   ✅ 50 percent → Good
   ✅ 50 प्रतिशत → Better
   ✅ पचास प्रतिशत → Best

8. ALPHANUMERIC CODES - ADD SEPARATOR:
   ❌ 1A → Sounds like "eka"
   ✅ 1 A → Good (space)
   ✅ 1.A → Also good
   ✅ एक A → Best

9. QUESTION CODES:
   ❌ Q2 → Sounds like "kyu doo"
   ✅ Q दो → Good
   ✅ Q-दो → Also good
   ✅ प्रश्न दो → Best

10. ABBREVIATIONS - EXPAND:
    ❌ डॉ. → Sounds like "daawww"
    ✅ डाक्टर → Correct
    ✅ डॉक्टर → Best
    
    ❌ रु. → Sounds like "ruu"
    ✅ रुपये → Correct

11. ACRONYMS - NO PERIODS:
    ❌ N.A.S.A. → Read completely wrong
    ✅ NASA → Works
    ✅ नासा → Best (if known in Hindi)
    
    ❌ A.I. → Wrong
    ✅ AI → Works
    ✅ कृत्रिम बुद्धिमत्ता → Best

12. PLURAL ACRONYMS - AVOID 's' or 'ज़':
    ❌ PCs → Sounds like "PCS"
    ❌ SOPs → Sounds like "sops" 
    ✅ PC → Singular form works
    ✅ SOP → Singular form works

13. SYMBOLS - WRITE OUT:
    ❌ @ → Sounds like "har" (hr@company = "har company")
    ✅ at the rate → Correct
    ✅ at → Acceptable
    
    ❌ & → Often skipped (R&D = "RD")
    ✅ and → Correct (R and D)
    ✅ एंड → Hindi equivalent
    
    ❌ / → Sounds like "slash"
    ✅ Remove or use appropriate word
    
    ❌ - (hyphen in words) → Creates unnecessary pause
    ✅ Use space instead (क्रॉस-चेक → क्रॉस चेक)

14. SPECIAL CHARACTERS:
    ❌ <> → Not read
    ✅ Remove brackets, keep content
    
    ❌ °F → Sounds like "f"
    ✅ डिग्री फ़ारेनहाइट → Full form
    
    ❌ 6.5" → Sounds like "cheeh pannch"
    ✅ 6.5 इंच → Correct

15. TEMPERATURE:
    ❌ 102°F → Sounds like "102f"
    ✅ 102 डिग्री फ़ारेनहाइट → Good
    ✅ एक सौ दो डिग्री फ़ारेनहाइट → Best

16. MULTIPLICATION:
    ❌ 3x → Sounds like "teen exx"
    ✅ 3 गुना → Good
    ✅ तीन गुना → Best

17. CURRENCY:
    ❌ रु. 5,000 → Sounds like "ruu five"
    ✅ 5000 रुपये → Good
    ✅ पांच हजार रुपये → Best

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ALLOWED ADDITIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PUNCTUATION (use wisely):
   • . (period) - for sentence endings
   • , (comma) - for short pauses (NOT in numbers!)
   • ? (question mark) - for questions
   • ! (exclamation) - use sparingly
   • ... (ellipsis) - for hesitation
   
2. SENTENCE BREAKING:
   • Break sentences that are 30+ words long
   • Aim for 15-20 words per sentence
   • Use natural breaking points

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ STRICT PROHIBITIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DO NOT change, add, or remove any content words
2. DO NOT translate anything
3. DO NOT use commas in numbers (1,234 → 1234)
4. DO NOT use slashes in dates (15/03 → 15 मार्च)
5. DO NOT use hyphens in ranges (3-5 → 3से5)
6. DO NOT use abbreviations (डॉ. → डाक्टर)
7. DO NOT add periods to acronyms (NASA not N.A.S.A.)
8. DO NOT use special symbols (@, &, /, etc.)
9. DO NOT use SSML tags
10. DO NOT add explanations or comments

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 INPUT TEXT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎤 OUTPUT (DesiVocal-Optimized):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY the optimized text. No explanations, no preamble, no markdown."""
        
        return prompt
    
    def chunk_text(self, text: str) -> list:
        """
        Split text into chunks at sentence boundaries
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by common sentence endings
        sentences = re.split(r'([।॥.!?।]\\s+)', text)
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            separator = sentences[i+1] if i+1 < len(sentences) else ""
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + len(separator) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + separator
            else:
                current_chunk += sentence + separator
        
        # Add the last chunk if not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If we still have no chunks, split by character count
        if not chunks:
            chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        print(f"\\n📊 Split text into {len(chunks)} chunks")
        for idx, chunk in enumerate(chunks, 1):
            print(f"   Chunk {idx}: {len(chunk)} characters")
        
        return chunks
    
    def optimize_chunk(self, chunk: str, language: str = "Hindi", retry_count: int = 3) -> str:
        """
        Optimize a single chunk with retry logic
        """
        prompt = self.get_optimization_prompt(chunk, language)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower for more consistent output
                "top_p": 0.9,
                "num_predict": -1
            }
        }
        
        for attempt in range(retry_count):
            try:
                response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                optimized_text = result.get("response", "").strip()
                optimized_text = self._clean_output(optimized_text)
                return optimized_text
            except requests.exceptions.Timeout:
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"\\n⚠️ Timeout on attempt {attempt + 1}/{retry_count}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"\\n❌ Failed after {retry_count} attempts due to timeout")
                    raise
            except Exception as e:
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"\\n⚠️ Error on attempt {attempt + 1}/{retry_count}: {e}")
                    print(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"\\n❌ Failed after {retry_count} attempts: {e}")
                    raise
        
        return None
    
    def optimize(self, text: str, language: str = "Hindi") -> str:
        """
        Optimize text with automatic chunking
        """
        chunks = self.chunk_text(text)
        
        if len(chunks) == 1:
            print(f"\\n📤 Processing single chunk ({len(text)} chars)...")
            return self.optimize_chunk(chunks[0], language)
        
        # Process multiple chunks
        print(f"\\n🔄 Processing {len(chunks)} chunks...")
        optimized_chunks = []
        
        for idx, chunk in enumerate(chunks, 1):
            print(f"\\n📤 Processing chunk {idx}/{len(chunks)} ({len(chunk)} chars)...")
            try:
                optimized = self.optimize_chunk(chunk, language)
                if optimized:
                    optimized_chunks.append(optimized)
                    print(f"✅ Chunk {idx}/{len(chunks)} complete!")
                else:
                    print(f"❌ Chunk {idx}/{len(chunks)} failed - using original")
                    optimized_chunks.append(chunk)
            except Exception as e:
                print(f"❌ Error processing chunk {idx}: {e}")
                print("   Using original chunk text")
                optimized_chunks.append(chunk)
        
        # Combine all chunks
        final_text = " ".join(optimized_chunks)
        print(f"\\n✅ All chunks processed! Total output: {len(final_text)} characters")
        return final_text
    
    def _clean_output(self, text: str) -> str:
        """Clean the model output"""
        text = text.replace("```", "").replace("**", "")
        lines = [line.strip() for line in text.split('\\n') 
                 if line.strip() and not line.strip().startswith('#') 
                 and not line.strip().startswith('OUTPUT')]
        return '\\n'.join(lines).strip()


# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = DesiVocalTTSOptimizer(
        model_name="gemma3:27b",
        chunk_size=2000,
        timeout=600
    )
    
    # Test cases
    test_texts = [
        "बैठक 15/03/2024 को दोपहर 3:30 बजे है और budget 50,000 रुपये है।",
        "डॉ. शर्मा ने N.A.S.A. के साथ A.I. project शुरू किया।",
        "कृपया hr@company.com पर 8-10 दिनों में RSVP करें।",
        "Q2 में 1,50,000 रुपये का revenue था और growth 25% थी।"
    ]
    
    print("\\n" + "="*60)
    print("TESTING DESIVOCAL TTS OPTIMIZER")
    print("="*60)
    
    for i, test in enumerate(test_texts, 1):
        print(f"\\n--- Test {i} ---")
        print(f"Input:  {test}")
        optimized = optimizer.optimize(test)
        print(f"Output: {optimized}")
        print()