#!/usr/bin/env python3
"""
Human-like Book Narrator - Transcription Generator
Converts book text into natural, conversational narration with explanations
Supports both Ollama and HuggingFace LLMs
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import re

# Try to import dependencies
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class NarratorPrompts:
    """Professionally crafted prompts for natural narration."""
    
    SYSTEM_PROMPT = """You are a professional audiobook narrator with years of experience. Your narration style is:
- Natural and conversational, like a friend explaining a story
- Warm, engaging, and expressive
- Clear pronunciation guidance for complex words
- Contextual explanations for difficult concepts (only when necessary)
- Appropriate pacing with natural pauses
- Emotional variations matching the content
- Cultural sensitivity for Hindi/English mix

When narrating:
1. Convert written text to spoken narrative
2. Add natural speech patterns (um, well, you see, etc. - but sparingly)
3. Break complex sentences into digestible chunks
4. Add brief explanations for technical/difficult terms in [EXPLAIN: ...] markers
5. Mark emotional tone with [TONE: happy/sad/serious/excited/thoughtful]
6. Indicate pauses with [PAUSE-SHORT], [PAUSE-MEDIUM], [PAUSE-LONG]
7. For difficult words, add pronunciation: word [PRONOUNCE: pronunciation]
8. Keep the narrative flowing naturally - don't over-explain

Remember: You're narrating to someone listening, not reading. Make it sound human."""

    NARRATION_TEMPLATE = """Convert this text into natural audiobook narration:

TEXT:
{text}

CONTEXT: {context}

Provide the narration with appropriate markers for:
- [TONE: emotion] for emotional context
- [PAUSE-SHORT/MEDIUM/LONG] for natural pauses
- [EXPLAIN: brief explanation] for complex terms (max 1-2 sentences)
- [PRONOUNCE: pronunciation] for difficult words
- [EMPHASIS: word/phrase] for important points

Keep it conversational and engaging. The listener should feel like a human narrator is reading to them.

NARRATION:"""

    CHAPTER_OPENING = """Create an engaging opening for this chapter that sets the mood:

CHAPTER TITLE: {title}
CHAPTER SUMMARY: {summary}

Create a 2-3 sentence opening that:
- Welcomes the listener
- Sets the emotional tone
- Creates anticipation
- Feels natural and warm

OPENING:"""

    SECTION_TRANSITION = """Create a smooth transition between these sections:

PREVIOUS SECTION: {prev_section}
NEXT SECTION: {next_section}

Create a 1-2 sentence transition that flows naturally.

TRANSITION:"""


class LLMNarrator:
    """LLM-powered narrator for human-like transcription."""
    
    def __init__(self, provider="ollama", model_name=None, device="cpu"):
        self.provider = provider
        self.model_name = model_name or self._get_default_model()
        self.device = device
        self.model = None
        self.tokenizer = None
        self.prompts = NarratorPrompts()
        
        print(f"🎭 Initializing {provider} narrator...")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {device}")
        
        self._load_model()
    
    def _get_default_model(self):
        """Get default model based on provider."""
        if self.provider == "ollama":
            return "qwen2.5:7b"  # Excellent for creative tasks
        else:  # huggingface
            return "mistralai/Mistral-7B-Instruct-v0.2"
    
    def _load_model(self):
        """Load the LLM model."""
        if self.provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama not installed. Install: pip install ollama")
            
            # Test connection
            try:
                ollama.list()
                print("✅ Ollama connection successful")
            except Exception as e:
                raise RuntimeError(f"Cannot connect to Ollama: {e}")
        
        elif self.provider == "huggingface":
            if not HF_AVAILABLE:
                raise ImportError("Transformers not installed. Install: pip install transformers torch")
            
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
    
    def generate(self, prompt, max_tokens=2048, temperature=0.7):
        """Generate text using the LLM."""
        if self.provider == "ollama":
            return self._generate_ollama(prompt, max_tokens, temperature)
        else:
            return self._generate_huggingface(prompt, max_tokens, temperature)
    
    def _generate_ollama(self, prompt, max_tokens, temperature):
        """Generate using Ollama."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ Ollama generation error: {e}")
            return None
    
    def _generate_huggingface(self, prompt, max_tokens, temperature):
        """Generate using HuggingFace."""
        try:
            # Format prompt for instruction model
            formatted_prompt = f"<s>[INST] {self.prompts.SYSTEM_PROMPT}\n\n{prompt} [/INST]"
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the response part
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()
            
            return response
        except Exception as e:
            print(f"⚠️ HuggingFace generation error: {e}")
            return None
    
    def narrate_text(self, text, context=""):
        """Convert text to natural narration."""
        prompt = self.prompts.NARRATION_TEMPLATE.format(
            text=text,
            context=context
        )
        return self.generate(prompt, max_tokens=2048, temperature=0.7)
    
    def create_chapter_opening(self, title, summary=""):
        """Create engaging chapter opening."""
        prompt = self.prompts.CHAPTER_OPENING.format(
            title=title,
            summary=summary
        )
        return self.generate(prompt, max_tokens=256, temperature=0.8)
    
    def create_transition(self, prev_section, next_section):
        """Create smooth section transition."""
        prompt = self.prompts.SECTION_TRANSITION.format(
            prev_section=prev_section,
            next_section=next_section
        )
        return self.generate(prompt, max_tokens=128, temperature=0.7)


class TextPreprocessor:
    """Preprocess and structure book text for narration."""
    
    def __init__(self):
        self.chapter_pattern = re.compile(r'^(Chapter|CHAPTER|अध्याय)\s+(\d+|[IVXivx]+):?\s*(.*)$', re.MULTILINE)
        self.section_pattern = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)
    
    def detect_language(self, text):
        """Detect if text is Hindi, English, or mixed."""
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u0900-\u097F]', text))
        
        if total_chars == 0:
            return "unknown"
        
        hindi_ratio = hindi_chars / total_chars
        
        if hindi_ratio > 0.7:
            return "hindi"
        elif hindi_ratio > 0.2:
            return "mixed"
        else:
            return "english"
    
    def split_into_chapters(self, text):
        """Split text into chapters."""
        chapters = []
        matches = list(self.chapter_pattern.finditer(text))
        
        if not matches:
            # No chapters found, treat entire text as one chapter
            return [{
                "number": 1,
                "title": "Complete Text",
                "content": text
            }]
        
        for i, match in enumerate(matches):
            chapter_num = match.group(2)
            chapter_title = match.group(3).strip() or f"Chapter {chapter_num}"
            
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            content = text[start_pos:end_pos].strip()
            
            chapters.append({
                "number": i + 1,
                "title": chapter_title,
                "content": content
            })
        
        return chapters
    
    def split_into_paragraphs(self, text, max_length=800):
        """Split text into manageable paragraphs."""
        # Split by double newlines or devanagari full stop
        paragraphs = re.split(r'\n\s*\n|।\s*।', text)
        
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is too long, split by sentences
            if len(para) > max_length:
                sentences = re.split(r'[.!?।]\s+', para)
                current_chunk = []
                current_length = 0
                
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    
                    if current_length + len(sent) > max_length and current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sent]
                        current_length = len(sent)
                    else:
                        current_chunk.append(sent)
                        current_length += len(sent)
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
            else:
                chunks.append(para)
        
        return chunks


class TranscriptionGenerator:
    """Main transcription generator."""
    
    def __init__(self, provider="ollama", model_name=None, output_dir=".", device="cpu"):
        self.narrator = LLMNarrator(provider, model_name, device)
        self.preprocessor = TextPreprocessor()
        self.output_dir = Path(output_dir) / "transcriptions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_from_file(self, input_file, quality="high"):
        """Generate transcription from input file."""
        print("=" * 70)
        print("📚 HUMAN-LIKE AUDIOBOOK NARRATOR")
        print("=" * 70)
        
        # Read input
        print(f"\n📖 Reading: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Detect language
        language = self.preprocessor.detect_language(text)
        print(f"🌍 Detected language: {language}")
        
        # Split into chapters
        print(f"\n📑 Processing structure...")
        chapters = self.preprocessor.split_into_chapters(text)
        print(f"✅ Found {len(chapters)} chapters")
        
        # Generate transcription
        transcription_data = {
            "metadata": {
                "source_file": str(input_file),
                "generated_at": datetime.now().isoformat(),
                "language": language,
                "total_chapters": len(chapters),
                "quality": quality,
                "narrator_model": self.narrator.model_name
            },
            "chapters": []
        }
        
        total_start = time.time()
        
        for ch_idx, chapter in enumerate(chapters, 1):
            print(f"\n{'=' * 70}")
            print(f"📖 Chapter {ch_idx}/{len(chapters)}: {chapter['title']}")
            print(f"{'=' * 70}")
            
            chapter_start = time.time()
            
            # Generate chapter opening
            print("🎬 Creating chapter opening...")
            opening = self.narrator.create_chapter_opening(
                chapter['title'],
                chapter['content'][:200]
            )
            
            # Split content into paragraphs
            paragraphs = self.preprocessor.split_into_paragraphs(chapter['content'])
            print(f"📝 Processing {len(paragraphs)} sections...")
            
            narrated_sections = []
            
            for p_idx, para in enumerate(paragraphs, 1):
                print(f"   🎙️ Section {p_idx}/{len(paragraphs)}... ", end="", flush=True)
                
                # Build context from previous section
                context = ""
                if p_idx > 1:
                    context = f"Previous section: {paragraphs[p_idx-2][:150]}..."
                
                # Generate narration
                section_start = time.time()
                narration = self.narrator.narrate_text(para, context)
                section_time = time.time() - section_start
                
                if narration:
                    narrated_sections.append({
                        "section_number": p_idx,
                        "original_text": para,
                        "narration": narration,
                        "generation_time": section_time
                    })
                    print(f"✅ ({section_time:.1f}s)")
                else:
                    print(f"⚠️ Failed")
                    # Fallback to original text
                    narrated_sections.append({
                        "section_number": p_idx,
                        "original_text": para,
                        "narration": para,
                        "generation_time": 0
                    })
            
            chapter_time = time.time() - chapter_start
            
            transcription_data["chapters"].append({
                "chapter_number": ch_idx,
                "title": chapter['title'],
                "opening": opening,
                "sections": narrated_sections,
                "generation_time": chapter_time
            })
            
            print(f"⏱️ Chapter completed in {chapter_time/60:.2f} minutes")
        
        total_time = time.time() - total_start
        transcription_data["metadata"]["total_generation_time"] = total_time
        
        # Save transcription
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.output_dir / f"transcription_{timestamp}.json"
        txt_file = self.output_dir / f"transcription_{timestamp}.txt"
        
        # Save JSON (complete data)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, ensure_ascii=False, indent=2)
        
        # Save TXT (narration only, ready for TTS)
        with open(txt_file, 'w', encoding='utf-8') as f:
            for chapter in transcription_data["chapters"]:
                f.write(f"\n{'='*70}\n")
                f.write(f"CHAPTER {chapter['chapter_number']}: {chapter['title']}\n")
                f.write(f"{'='*70}\n\n")
                
                if chapter['opening']:
                    f.write(f"{chapter['opening']}\n\n")
                
                for section in chapter['sections']:
                    f.write(f"{section['narration']}\n\n")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"🎉 TRANSCRIPTION COMPLETE!")
        print(f"{'='*70}")
        print(f"⏱️ Total time: {total_time/60:.2f} minutes")
        print(f"📚 Chapters processed: {len(chapters)}")
        print(f"💾 JSON output: {json_file}")
        print(f"📄 TXT output (TTS-ready): {txt_file}")
        print(f"{'='*70}")
        
        return str(txt_file), str(json_file)


def main():
    parser = argparse.ArgumentParser(
        description='Human-like Audiobook Narrator - Transcription Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Ollama (recommended for quality)
  python narrator.py -f book.txt -p ollama -m qwen2.5:7b
  python narrator.py -f book.txt -p ollama -m llama3.2:3b
  
  # Using HuggingFace
  python narrator.py -f book.txt -p huggingface -m mistralai/Mistral-7B-Instruct-v0.2
  
  # High quality with GPU
  python narrator.py -f book.txt -p ollama -m qwen2.5:14b --device cuda
  
  # Quick mode with smaller model
  python narrator.py -f book.txt -p ollama -m llama3.2:1b --quality medium

Recommended Ollama Models:
  - qwen2.5:7b - Best quality/speed balance (recommended)
  - llama3.2:3b - Fast, good quality
  - mistral:7b - Excellent for creative narration
  - llama3.2:1b - Very fast, decent quality
        """
    )
    
    parser.add_argument('-f', '--file', required=True, help='Input text file')
    parser.add_argument('-p', '--provider', choices=['ollama', 'huggingface'],
                        default='ollama', help='LLM provider')
    parser.add_argument('-m', '--model', help='Model name')
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--quality', choices=['medium', 'high'], default='high',
                        help='Narration quality level')
    
    args = parser.parse_args()
    
    # Check file exists
    if not Path(args.file).exists():
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)
    
    # Check provider availability
    if args.provider == "ollama" and not OLLAMA_AVAILABLE:
        print("❌ Error: Ollama not installed")
        print("   Install: pip install ollama")
        print("   Also ensure Ollama service is running: ollama serve")
        sys.exit(1)
    
    if args.provider == "huggingface" and not HF_AVAILABLE:
        print("❌ Error: Transformers not installed")
        print("   Install: pip install transformers torch")
        sys.exit(1)
    
    try:
        generator = TranscriptionGenerator(
            provider=args.provider,
            model_name=args.model,
            output_dir=args.output,
            device=args.device
        )
        
        txt_file, json_file = generator.generate_from_file(args.file, args.quality)
        
        print(f"\n✅ Transcription ready for TTS!")
        print(f"   Use this file with listen.py: {txt_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Generation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()