#!/usr/bin/env python3
"""
Improved Multilingual Book Narrator - Fixed for Hindi
Key improvements:
1. Stricter prompts to prevent hallucination
2. Better validation
3. Post-processing to remove repetitions
4. Support for Indian-specific models
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import re
from collections import OrderedDict

# Try to import dependencies
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class ImprovedNarratorPrompts:
    """Strictly controlled prompts with better constraints."""
    
    SYSTEM_PROMPT_HINDI = """आप एक पेशेवर ऑडियोबुक कथावाचक हैं। आपका काम केवल दिए गए टेक्स्ट को प्राकृतिक आवाज़ में सुनाना है।

महत्वपूर्ण नियम:
1. मूल टेक्स्ट में जो लिखा है वही बोलें - कुछ भी नया न जोड़ें
2. कुछ भी न छोड़ें - हर शब्द महत्वपूर्ण है
3. व्याख्या न करें, सारांश न दें - बस वही पढ़ें जो लिखा है
4. केवल ये मार्कर जोड़ें: [PAUSE-SHORT], [PAUSE-MEDIUM], [PAUSE-LONG]
5. टोन मार्कर (अंग्रेजी में): [TONE: serious/thoughtful/curious/calm]
6. लंबे वाक्यों को प्राकृतिक ठहराव से तोड़ें
7. कोई अतिरिक्त विवरण, संदर्भ या स्पष्टीकरण न जोड़ें

उदाहरण:
गलत: "होम्स एक रहस्यमय व्यक्ति है जिसमें बौद्धिक प्रतिभा है। यह उसकी विरोधाभासी प्रकृति को दर्शाता है..."
सही: "होम्स को एक रहस्यमय व्यक्ति के रूप में परिचित किया जाता है, [PAUSE-SHORT] जिसमें बौद्धिक प्रतिभा एवं अपने कार्यों के प्रति सूक्ष्म दृष्टिकोण दोनों ही हैं।"

आप केवल आवाज़ हैं। मूल शब्दों को बदलें नहीं।"""

    SYSTEM_PROMPT_ENGLISH = """You are a professional audiobook narrator. Your job is ONLY to read the text aloud naturally.

CRITICAL RULES:
1. Speak EXACTLY what's written - add NOTHING new
2. Skip NOTHING - every word matters
3. DO NOT interpret, summarize, or explain - just read what's written
4. ONLY add these markers: [PAUSE-SHORT], [PAUSE-MEDIUM], [PAUSE-LONG]
5. Tone markers (in English): [TONE: serious/thoughtful/curious/calm]
6. Break long sentences with natural pauses
7. NO additional details, context, or clarifications

Example:
Wrong: "Holmes is a mysterious person with intellectual talent. This shows his contradictory nature..."
Right: "Holmes is introduced as a mysterious person, [PAUSE-SHORT] with both intellectual talent and a meticulous approach to his work."

You are a VOICE only. Do not change the original words."""

    NARRATION_TEMPLATE_HINDI = """नीचे दिया गया टेक्स्ट बिलकुल वैसे ही सुनाएं जैसे लिखा है। कुछ भी नया न जोड़ें।

मूल टेक्स्ट:
\"\"\"
{text}
\"\"\"

निर्देश:
- ऊपर के शब्दों को बिलकुल वैसे ही बोलें
- केवल [PAUSE-SHORT], [PAUSE-MEDIUM], [PAUSE-LONG] जोड़ें
- कोई व्याख्या, सारांश या अतिरिक्त विवरण न दें
- मूल वाक्यों को बदलें नहीं

कथन (मूल शब्दों में):"""

    NARRATION_TEMPLATE_ENGLISH = """Read the text below EXACTLY as written. Add NOTHING new.

ORIGINAL TEXT:
\"\"\"
{text}
\"\"\"

INSTRUCTIONS:
- Speak the exact words above
- ONLY add [PAUSE-SHORT], [PAUSE-MEDIUM], [PAUSE-LONG]
- NO interpretation, summary, or additional details
- DO NOT change the original sentences

NARRATION (using original words):"""

    @staticmethod
    def detect_language(text):
        """Detect if text is primarily Hindi or English."""
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = hindi_chars + english_chars
        if total_chars == 0:
            return "english"
        hindi_ratio = hindi_chars / total_chars
        return "hindi" if hindi_ratio > 0.3 else "english"


class RepetitionRemover:
    """Remove repetitive content from narration."""
    
    @staticmethod
    def remove_repetitions(text):
        """Remove repeated sentences and phrases."""
        sentences = re.split(r'(?<=[.!?।])\s+', text)
        seen = OrderedDict()
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # Create a normalized key (first 50 chars)
            key = ' '.join(sent.split()[:10]).lower()
            
            if key not in seen:
                seen[key] = sent
        
        return ' '.join(seen.values())
    
    @staticmethod
    def remove_meta_commentary(text, original):
        """Remove sentences that aren't in the original."""
        # Find sentences that discuss the text rather than narrate it
        meta_patterns = [
            r'यह.*?(दर्शाता|रेखांकित|स्थापित|विस्तारित).*?है',
            r'यह अध्याय.*?(उजागर|बनाता|स्पष्ट).*?है',
            r'This.*?(shows|demonstrates|establishes|highlights)',
            r'This chapter.*?(reveals|creates|clarifies)'
        ]
        
        sentences = re.split(r'(?<=[.!?।])\s+', text)
        filtered = []
        
        for sent in sentences:
            is_meta = False
            for pattern in meta_patterns:
                if re.search(pattern, sent, re.IGNORECASE):
                    is_meta = True
                    break
            
            if not is_meta:
                filtered.append(sent)
        
        return ' '.join(filtered)


class ImprovedLLMNarrator:
    """Improved LLM narrator with better validation."""
    
    def __init__(self, provider="ollama", model_name=None, device="cpu", language="auto"):
        self.provider = provider
        self.model_name = model_name or self._get_default_model()
        self.device = device
        self.language = language
        self.model = None
        self.tokenizer = None
        self.prompts = ImprovedNarratorPrompts()
        self.repetition_remover = RepetitionRemover()
        
        print(f"🎭 Initializing {provider} narrator...")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {device}")
        print(f"   Language: {language}")
        
        self._load_model()
    
    def _get_default_model(self):
        """Get best default model based on provider."""
        if self.provider == "ollama":
            # Try to use better models if available
            return "gemma2:9b"  # Better for Hindi than qwen
        else:
            return "ai4bharat/Airavata"  # Indian-specific model
    
    def _load_model(self):
        """Load the LLM model."""
        if self.provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama not installed. Install: pip install ollama")
            try:
                ollama.list()
                print("✅ Ollama connection successful")
            except Exception as e:
                raise RuntimeError(f"Cannot connect to Ollama: {e}")
        
        elif self.provider == "huggingface":
            if not HF_AVAILABLE:
                raise ImportError("Transformers not installed.")
            
            print(f"📥 Loading HuggingFace model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            
            print("✅ HuggingFace model loaded")
    
    def generate(self, prompt, system_prompt, max_tokens=2048, temperature=0.2):
        """Generate with lower temperature for more faithful reproduction."""
        if self.provider == "ollama":
            return self._generate_ollama(prompt, system_prompt, max_tokens, temperature)
        else:
            return self._generate_huggingface(prompt, system_prompt, max_tokens, temperature)
    
    def _generate_ollama(self, prompt, system_prompt, max_tokens, temperature):
        """Generate using Ollama with strict parameters."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": temperature,  # Lower = more faithful
                    "num_predict": max_tokens,
                    "top_p": 0.85,  # Lower = less creative
                    "repeat_penalty": 1.3,  # Higher = less repetition
                    "top_k": 40,  # Lower = more focused
                }
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ Ollama generation error: {e}")
            return None
    
    def _generate_huggingface(self, prompt, system_prompt, max_tokens, temperature):
        """Generate using HuggingFace with strict parameters."""
        try:
            # Format depends on model
            if "Airavata" in self.model_name or "sarvam" in self.model_name:
                formatted_prompt = f"### System:\n{system_prompt}\n\n### User:\n{prompt}\n\n### Assistant:\n"
            else:
                formatted_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.85,
                    top_k=40,
                    repetition_penalty=1.3
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()
            elif "### Assistant:" in response:
                response = response.split("### Assistant:")[-1].strip()
            
            return response
        except Exception as e:
            print(f"⚠️ HuggingFace generation error: {e}")
            return None
    
    def validate_and_clean(self, original, narration):
        """Validate and clean the narration."""
        if not narration:
            return None, "Empty narration"
        
        # Remove meta-commentary
        cleaned = self.repetition_remover.remove_meta_commentary(narration, original)
        
        # Remove repetitions
        cleaned = self.repetition_remover.remove_repetitions(cleaned)
        
        # Check if too much was added
        original_words = set(original.lower().split())
        clean_narration = re.sub(r'\[(?:TONE|PAUSE|PRONOUNCE|EMPHASIS):[^\]]*\]', '', cleaned)
        clean_narration = re.sub(r'\[PAUSE-(?:SHORT|MEDIUM|LONG)\]', '', clean_narration)
        narration_words = set(clean_narration.lower().split())
        
        # Calculate how many new words were added
        new_words = narration_words - original_words
        
        # More lenient for Hindi due to grammatical variations
        lang = self.prompts.detect_language(original)
        threshold = 0.6 if lang == "hindi" else 0.4
        
        if len(new_words) > len(original_words) * threshold:
            return None, f"Too many new words added ({len(new_words)} new vs {len(original_words)} original)"
        
        return cleaned, "Valid"
    
    def narrate_text(self, text, max_retries=3):
        """Convert text to narration with strict validation."""
        detected_lang = self.prompts.detect_language(text) if self.language == "auto" else self.language
        
        system_prompt = (self.prompts.SYSTEM_PROMPT_HINDI if detected_lang == "hindi" 
                        else self.prompts.SYSTEM_PROMPT_ENGLISH)
        template = (self.prompts.NARRATION_TEMPLATE_HINDI if detected_lang == "hindi" 
                   else self.prompts.NARRATION_TEMPLATE_ENGLISH)
        
        prompt = template.format(text=text)
        
        for attempt in range(max_retries):
            # Reduce temperature with each retry
            temp = 0.2 - (attempt * 0.05)
            
            narration = self.generate(prompt, system_prompt, max_tokens=3072, temperature=temp)
            
            if not narration:
                continue
            
            cleaned, reason = self.validate_and_clean(text, narration)
            
            if cleaned:
                return cleaned, True, detected_lang
            else:
                print(f"      ⚠️ Attempt {attempt + 1} failed: {reason}")
                if attempt < max_retries - 1:
                    print(f"      🔄 Retrying with temperature {temp - 0.05:.2f}...")
        
        # Fallback
        print(f"      ⚠️ All attempts failed, using minimal narration")
        return self._minimal_narration(text), False, detected_lang
    
    def _minimal_narration(self, text):
        """Minimal fallback - just add pauses."""
        sentences = re.split(r'([.!?।]+\s+)', text)
        result = []
        
        for i, sent in enumerate(sentences):
            if not sent.strip():
                continue
            
            result.append(sent)
            
            # Add pause after sentences
            if sent.strip() in '.!?।':
                if i < len(sentences) - 1:
                    result.append(" [PAUSE-SHORT] ")
        
        return ''.join(result)


class TextPreprocessor:
    """Preprocess text with better chapter detection."""
    
    def __init__(self):
        self.chapter_pattern = re.compile(
            r'^(={3,}\s*)?(Chapter|CHAPTER|अध्याय|CHAPTER)\s+(\d+|[IVXivx]+|[०-९]+):?\s*(.*)(\s*={3,})?$',
            re.MULTILINE
        )
    
    def split_into_chapters(self, text):
        """Split text into chapters."""
        chapters = []
        matches = list(self.chapter_pattern.finditer(text))
        
        if not matches:
            return [{
                "number": 1,
                "title": "Complete Text",
                "content": text
            }]
        
        for i, match in enumerate(matches):
            chapter_num = match.group(3)
            chapter_title = match.group(4).strip() or f"Chapter {chapter_num}"
            
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            content = text[start_pos:end_pos].strip()
            
            chapters.append({
                "number": i + 1,
                "title": chapter_title,
                "content": content
            })
        
        return chapters
    
    def split_into_sentences(self, text):
        """Split into sentences (Hindi + English)."""
        sentences = re.split(r'(?<=[.!?।])\s+(?=[A-ZА-Я"\u0900-\u097F])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, sentences, chunk_size=8, overlap=1):
        """Create smaller overlapping chunks."""
        chunks = []
        i = 0
        
        while i < len(sentences):
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = ' '.join(chunk_sentences)
            
            chunks.append({
                'text': chunk_text,
                'start_idx': i,
                'end_idx': i + len(chunk_sentences)
            })
            
            i += max(1, chunk_size - overlap)
        
        return chunks


class ImprovedTranscriptionGenerator:
    """Improved transcription generator."""
    
    def __init__(self, provider="ollama", model_name=None, output_dir=".", 
                 device="cpu", language="auto"):
        self.narrator = ImprovedLLMNarrator(provider, model_name, device, language)
        self.preprocessor = TextPreprocessor()
        self.output_dir = Path(output_dir) / "transcriptions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_from_file(self, input_file, chunk_size=8):
        """Generate transcription from file."""
        print("=" * 70)
        print("📚 IMPROVED MULTILINGUAL NARRATOR (Hindi + English)")
        print("=" * 70)
        
        print(f"\n📖 Reading: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        primary_lang = ImprovedNarratorPrompts.detect_language(text)
        print(f"🌍 Detected language: {primary_lang.upper()}")
        
        chapters = self.preprocessor.split_into_chapters(text)
        print(f"✅ Found {len(chapters)} chapters")
        
        transcription_data = {
            "metadata": {
                "source_file": str(input_file),
                "generated_at": datetime.now().isoformat(),
                "primary_language": primary_lang,
                "total_chapters": len(chapters),
                "narrator_model": self.narrator.model_name,
                "chunk_size": chunk_size
            },
            "chapters": []
        }
        
        total_start = time.time()
        successful = 0
        total_chunks = 0
        
        for ch_idx, chapter in enumerate(chapters, 1):
            print(f"\n{'=' * 70}")
            print(f"📖 Chapter {ch_idx}/{len(chapters)}: {chapter['title']}")
            print(f"{'=' * 70}")
            
            sentences = self.preprocessor.split_into_sentences(chapter['content'])
            chunks = self.preprocessor.create_chunks(sentences, chunk_size=chunk_size, overlap=1)
            
            print(f"📦 Processing {len(chunks)} chunks...")
            total_chunks += len(chunks)
            
            narrated_chunks = []
            
            for c_idx, chunk in enumerate(chunks, 1):
                print(f"   🎙️ Chunk {c_idx}/{len(chunks)}... ", end="", flush=True)
                
                start_time = time.time()
                narration, is_valid, lang = self.narrator.narrate_text(chunk['text'])
                elapsed = time.time() - start_time
                
                if is_valid:
                    successful += 1
                    print(f"✅ [{lang}] ({elapsed:.1f}s)")
                else:
                    print(f"⚠️ Fallback [{lang}] ({elapsed:.1f}s)")
                
                narrated_chunks.append({
                    "chunk_number": c_idx,
                    "original_text": chunk['text'],
                    "narration": narration,
                    "language": lang,
                    "is_valid": is_valid
                })
            
            transcription_data["chapters"].append({
                "chapter_number": ch_idx,
                "title": chapter['title'],
                "chunks": narrated_chunks
            })
        
        total_time = time.time() - total_start
        
        # Save files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.output_dir / f"transcription_{timestamp}.json"
        txt_file = self.output_dir / f"transcription_{timestamp}.txt"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, ensure_ascii=False, indent=2)
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            for chapter in transcription_data["chapters"]:
                f.write(f"\n{'='*70}\n")
                f.write(f"CHAPTER {chapter['chapter_number']}: {chapter['title']}\n")
                f.write(f"{'='*70}\n\n")
                
                for chunk in chapter['chunks']:
                    f.write(f"{chunk['narration']}\n\n")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"🎉 TRANSCRIPTION COMPLETE!")
        print(f"{'='*70}")
        print(f"⏱️ Total time: {total_time/60:.2f} minutes")
        print(f"🌍 Language: {primary_lang.upper()}")
        print(f"📚 Chapters: {len(chapters)}")
        print(f"📦 Total chunks: {total_chunks}")
        print(f"✅ Successful: {successful}/{total_chunks} ({100*successful/total_chunks:.1f}%)")
        print(f"💾 JSON: {json_file}")
        print(f"📄 TXT: {txt_file}")
        print(f"{'='*70}")
        
        return str(txt_file), str(json_file)


def main():
    parser = argparse.ArgumentParser(
        description='Improved Multilingual Narrator for Hindi + English',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended Models:
  Ollama:
    - gemma2:9b (best for Hindi)
    - aya:8b (multilingual specialist)
    - qwen2.5:14b (better than 7b)
    - llama3.1:8b (good instruction following)
  
  HuggingFace:
    - ai4bharat/Airavata (Indian languages)
    - sarvamai/sarvam-2b-v0.5 (Indian LLM)
    - CohereForAI/aya-23-8B (multilingual)

Examples:
  python transcribe_improved.py -f hindi_book.txt -m gemma2:9b
  python transcribe_improved.py -f book.txt -p huggingface -m ai4bharat/Airavata
        """
    )
    
    parser.add_argument('-f', '--file', required=True, help='Input text file')
    parser.add_argument('-p', '--provider', choices=['ollama', 'huggingface'],
                        default='ollama', help='LLM provider')
    parser.add_argument('-m', '--model', help='Model name')
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--language', default='auto', choices=['auto', 'hindi', 'english'])
    parser.add_argument('--chunk-size', type=int, default=8,
                        help='Sentences per chunk (smaller = better quality)')
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)
    
    try:
        generator = ImprovedTranscriptionGenerator(
            provider=args.provider,
            model_name=args.model,
            output_dir=args.output,
            device=args.device,
            language=args.language
        )
        
        txt_file, json_file = generator.generate_from_file(
            args.file,
            chunk_size=args.chunk_size
        )
        
        print(f"\n✅ Ready for TTS: {txt_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()