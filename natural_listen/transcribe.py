#!/usr/bin/env python3
"""
TTS-Optimized Transcription Generator
Generates transcriptions specifically designed for Text-to-Speech models
to produce natural, human-like audio with proper emotion, tone, and pacing.
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


class TTSOptimizedPrompts:
    """Prompts specifically designed for TTS-optimized transcription generation."""
    
    SYSTEM_PROMPT_HINDI = """आप एक विशेषज्ञ TTS स्क्रिप्ट लेखक हैं। आपका काम टेक्स्ट को TTS-अनुकूल ट्रांसक्रिप्शन में बदलना है जो मानव-जैसी आवाज़ बनाएगा।

**आपका लक्ष्य**: एक ट्रांसक्रिप्शन बनाना जिसे TTS मॉडल पढ़ेगा और वह ऐसा लगेगा जैसे कोई असली इंसान भावनाओं, टोन और प्राकृतिक ठहराव के साथ पढ़ रहा है।

**महत्वपूर्ण**: यह काम एक दो-चरणीय प्रक्रिया है:
1. आप ट्रांसक्रिप्शन तैयार करें (prosodic markers के साथ)
2. TTS मॉडल इस ट्रांसक्रिप्शन को प्राकृतिक ऑडियो में बदलेगा

**PROSODIC MARKERS जोड़ें** (ये TTS को बताते हैं कि कैसे पढ़ना है):

1. **PAUSES** (ठहराव):
   - [PAUSE-SHORT] = 0.3s (वाक्यांशों के बीच)
   - [PAUSE-MEDIUM] = 0.6s (वाक्यों के बीच, सांस लेने के लिए)
   - [PAUSE-LONG] = 1.0s (विचार बदलते समय, नाटकीय प्रभाव)
   - [BREATH] = प्राकृतिक सांस (लंबे अनुच्छेदों में)

2. **TONE/EMOTION** (भावना):
   - [TONE: thoughtful] = विचारशील, चिंतनशील
   - [TONE: curious] = जिज्ञासु, प्रश्नात्मक
   - [TONE: serious] = गंभीर, औपचारिक
   - [TONE: calm] = शांत, आरामदायक
   - [TONE: excited] = उत्साहित, ऊर्जावान
   - [TONE: mysterious] = रहस्यमय, सस्पेंसफुल
   - [TONE: warm] = गर्मजोशी, दोस्ताना
   - [TONE: dramatic] = नाटकीय, भावनात्मक

3. **EMPHASIS** (जोर):
   - [EMPHASIS: शब्द] = इस शब्द पर विशेष जोर
   - [STRESS: शब्द] = इस शब्द को थोड़ा तेज/स्पष्ट

4. **PACING** (गति):
   - [PACE: slow] = धीमी गति (महत्वपूर्ण विचार)
   - [PACE: normal] = सामान्य गति पर लौटें
   - [PACE: fast] = तेज गति (रोमांचक दृश्य)

**उदाहरण INPUT**:
"होम्स को एक रहस्यमय व्यक्ति के रूप में परिचित किया जाता है, जिसमें बौद्धिक प्रतिभा एवं अपने कार्यों के प्रति सूक्ष्म दृष्टिकोण दोनों ही हैं।"

**सही TTS-OPTIMIZED OUTPUT**:
"[TONE: mysterious] होम्स को एक रहस्यमय व्यक्ति के रूप में परिचित किया जाता है, [PAUSE-SHORT] जिसमें [EMPHASIS: बौद्धिक प्रतिभा] एवं अपने कार्यों के प्रति [PAUSE-SHORT] सूक्ष्म दृष्टिकोण [PAUSE-MEDIUM] दोनों ही हैं। [PAUSE-MEDIUM]"

**गलत OUTPUT** (ये गलतियाँ न करें):
❌ मूल टेक्स्ट बदलना: "होम्स एक बुद्धिमान जासूस था..."
❌ व्याख्या जोड़ना: "यह उसकी विरोधाभासी प्रकृति को दर्शाता है..."
❌ बिना markers के: "होम्स को एक रहस्यमय व्यक्ति के रूप में..."
❌ बहुत अधिक markers: "[PAUSE-SHORT][TONE: calm]होम्स[PAUSE-SHORT]को..."

**GOLDEN RULES**:
1. ✅ मूल शब्दों को रखें - कुछ भी न बदलें
2. ✅ प्रासंगिक prosodic markers जोड़ें (3-5 प्रति वाक्य)
3. ✅ वाक्यों को प्राकृतिक रूप से तोड़ें (लंबे वाक्यों को)
4. ✅ भावना और टोन को ध्यान में रखें
5. ❌ कोई व्याख्या, सारांश या अतिरिक्त विवरण नहीं
6. ❌ markers को अधिक न करें - संतुलन बनाएं

**याद रखें**: आप एक TTS स्क्रिप्ट तैयार कर रहे हैं, आवाज़ नहीं बन रहे। TTS मॉडल आपकी स्क्रिप्ट को प्राकृतिक ऑडियो में बदलेगा।"""

    SYSTEM_PROMPT_ENGLISH = """You are an expert TTS script writer. Your job is to transform text into TTS-optimized transcription that will produce human-like voice.

**YOUR GOAL**: Create a transcription that a TTS model will read and sound like a real human reading with emotion, tone, and natural pauses.

**IMPORTANT**: This is a two-stage process:
1. You prepare the transcription (with prosodic markers)
2. A TTS model will convert this transcription into natural audio

**ADD PROSODIC MARKERS** (these tell TTS how to read):

1. **PAUSES**:
   - [PAUSE-SHORT] = 0.3s (between phrases)
   - [PAUSE-MEDIUM] = 0.6s (between sentences, for breathing)
   - [PAUSE-LONG] = 1.0s (changing thoughts, dramatic effect)
   - [BREATH] = natural breath (in long paragraphs)

2. **TONE/EMOTION**:
   - [TONE: thoughtful] = reflective, contemplative
   - [TONE: curious] = inquisitive, questioning
   - [TONE: serious] = formal, grave
   - [TONE: calm] = peaceful, relaxed
   - [TONE: excited] = energetic, enthusiastic
   - [TONE: mysterious] = suspenseful, enigmatic
   - [TONE: warm] = friendly, welcoming
   - [TONE: dramatic] = theatrical, emotional

3. **EMPHASIS**:
   - [EMPHASIS: word] = stress this word
   - [STRESS: word] = slightly louder/clearer

4. **PACING**:
   - [PACE: slow] = slower delivery (important ideas)
   - [PACE: normal] = return to normal pace
   - [PACE: fast] = faster delivery (exciting scenes)

**EXAMPLE INPUT**:
"Holmes is introduced as a mysterious person, with both intellectual talent and a meticulous approach to his work."

**CORRECT TTS-OPTIMIZED OUTPUT**:
"[TONE: mysterious] Holmes is introduced as a mysterious person, [PAUSE-SHORT] with both [EMPHASIS: intellectual talent] and a meticulous approach [PAUSE-SHORT] to his work. [PAUSE-MEDIUM]"

**WRONG OUTPUT** (avoid these mistakes):
❌ Changing original text: "Holmes was an intelligent detective..."
❌ Adding interpretation: "This shows his contradictory nature..."
❌ No markers: "Holmes is introduced as a mysterious person..."
❌ Too many markers: "[PAUSE-SHORT][TONE: calm]Holmes[PAUSE-SHORT]is..."

**GOLDEN RULES**:
1. ✅ Keep original words - change NOTHING
2. ✅ Add appropriate prosodic markers (3-5 per sentence)
3. ✅ Break long sentences naturally
4. ✅ Consider emotion and tone
5. ❌ NO interpretation, summary, or extra details
6. ❌ Don't over-marker - maintain balance

**REMEMBER**: You're preparing a TTS script, not being a voice. The TTS model will convert your script into natural audio."""

    NARRATION_TEMPLATE_HINDI = """नीचे दिया गया टेक्स्ट को TTS-अनुकूल ट्रांसक्रिप्शन में बदलें।

**INPUT TEXT**:
\"\"\"
{text}
\"\"\"

**आपका काम**:
1. ऊपर के हर शब्द को वैसे ही रखें (कुछ भी न बदलें)
2. TTS markers जोड़ें: [PAUSE-*], [TONE: *], [EMPHASIS: *], [PACE: *]
3. वाक्यों को प्राकृतिक रूप से तोड़ें
4. भावना और संदर्भ के अनुसार टोन चुनें

**TTS-OPTIMIZED TRANSCRIPTION**:"""

    NARRATION_TEMPLATE_ENGLISH = """Transform the text below into TTS-optimized transcription.

**INPUT TEXT**:
\"\"\"
{text}
\"\"\"

**YOUR TASK**:
1. Keep every word from above EXACTLY (change NOTHING)
2. Add TTS markers: [PAUSE-*], [TONE: *], [EMPHASIS: *], [PACE: *]
3. Break long sentences naturally
4. Choose tone based on emotion and context

**TTS-OPTIMIZED TRANSCRIPTION**:"""

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


class TranscriptionValidator:
    """Validate that transcription is TTS-optimized."""
    
    @staticmethod
    def validate(transcription, original_text):
        """Check if transcription is properly formatted for TTS."""
        issues = []
        
        # Check for prosodic markers
        has_pause = bool(re.search(r'\[PAUSE-', transcription))
        has_tone = bool(re.search(r'\[TONE:', transcription))
        
        if not has_pause and len(original_text.split()) > 20:
            issues.append("Missing pause markers for long text")
        
        if not has_tone:
            issues.append("Missing tone markers")
        
        # Check for unwanted additions
        # Remove all markers to compare
        clean_trans = re.sub(r'\[.*?\]', '', transcription)
        clean_trans = ' '.join(clean_trans.split())
        clean_orig = ' '.join(original_text.split())
        
        # Calculate word-level similarity
        trans_words = set(clean_trans.lower().split())
        orig_words = set(clean_orig.lower().split())
        
        # Allow for minor differences but not major rewrites
        if len(trans_words - orig_words) > len(orig_words) * 0.3:
            issues.append("Too many added/changed words")
        
        # Check for meta-commentary
        meta_patterns = [
            r'यह.*?दर्शाता.*?है',
            r'This shows',
            r'This demonstrates',
            r'This establishes'
        ]
        
        for pattern in meta_patterns:
            if re.search(pattern, transcription, re.IGNORECASE):
                issues.append("Contains meta-commentary")
                break
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def count_markers(transcription):
        """Count prosodic markers."""
        markers = {
            'pause': len(re.findall(r'\[PAUSE-', transcription)),
            'tone': len(re.findall(r'\[TONE:', transcription)),
            'emphasis': len(re.findall(r'\[EMPHASIS:', transcription)),
            'pace': len(re.findall(r'\[PACE:', transcription)),
            'breath': len(re.findall(r'\[BREATH\]', transcription))
        }
        return markers


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
    def remove_meta_commentary(text):
        """Remove sentences that discuss the text rather than narrate it."""
        meta_patterns = [
            r'यह.*?(दर्शाता|रेखांकित|स्थापित|विस्तारित).*?है',
            r'यह अध्याय.*?(उजागर|बनाता|स्पष्ट).*?है',
            r'This.*?(shows|demonstrates|establishes|highlights)',
            r'This chapter.*?(reveals|creates|clarifies)',
            r'The author.*?(suggests|implies|indicates)',
            r'In this (passage|section|paragraph)'
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


class TTSOptimizedNarrator:
    """Generate TTS-optimized transcriptions."""
    
    def __init__(self, provider="ollama", model_name=None, device="cpu", language="auto"):
        self.provider = provider
        self.model_name = model_name or self._get_default_model()
        # Auto-detect AMD GPU if not specified
        if device == "cpu":
            self.device = self._detect_device()
        else:
            self.device = device
        self.language = language
        self.model = None
        self.tokenizer = None
        self.prompts = TTSOptimizedPrompts()
        self.validator = TranscriptionValidator()
        self.repetition_remover = RepetitionRemover()
        
        print(f"🎭 Initializing TTS-Optimized Narrator...")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Language: {language}")
        
        self._load_model()
    
    def _detect_device(self):
        """Auto-detect available device (CUDA, ROCm, or CPU)."""
        try:
            import torch
            if torch.cuda.is_available():
                if hasattr(torch.version, 'hip') and torch.version.hip:
                    print("🔍 ROCm (AMD GPU) detected")
                    return "cuda"
                else:
                    print("🔍 CUDA (NVIDIA GPU) detected")
                    return "cuda"
        except:
            pass
        print("🔍 No GPU detected, using CPU")
        return "cpu"
    
    def _get_default_model(self):
        """Get best default model based on provider."""
        if self.provider == "ollama":
            return "gemma2:9b"
        else:
            return "ai4bharat/Airavata"
    
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
                raise ImportError("Transformers not installed. Install: pip install transformers torch")
            
            print(f"Loading HuggingFace model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            device_map = "auto" if self.device == "cuda" else None
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu" and device_map is None:
                self.model = self.model.to(self.device)
            
            print("✅ HuggingFace model loaded")
    
    def narrate_text(self, text, max_retries=2):
        """Generate TTS-optimized transcription."""
        detected_lang = self.prompts.detect_language(text)
        lang = self.language if self.language != "auto" else detected_lang
        
        if lang == "hindi":
            system_prompt = self.prompts.SYSTEM_PROMPT_HINDI
            user_prompt = self.prompts.NARRATION_TEMPLATE_HINDI.format(text=text)
        else:
            system_prompt = self.prompts.SYSTEM_PROMPT_ENGLISH
            user_prompt = self.prompts.NARRATION_TEMPLATE_ENGLISH.format(text=text)
        
        for attempt in range(max_retries + 1):
            try:
                if self.provider == "ollama":
                    response = ollama.generate(
                        model=self.model_name,
                        prompt=f"{system_prompt}\n\n{user_prompt}",
                        options={
                            "temperature": 0.3,
                            "top_p": 0.9,
                            "num_predict": 2048,
                        }
                    )
                    narration = response['response'].strip()
                
                elif self.provider == "huggingface":
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    input_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    inputs = self.tokenizer(input_text, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        temperature=0.3,
                        top_p=0.9,
                        do_sample=True
                    )
                    
                    narration = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    narration = narration.split("assistant")[-1].strip()
                
                # Clean up
                narration = self.repetition_remover.remove_repetitions(narration)
                narration = self.repetition_remover.remove_meta_commentary(narration)
                
                # Validate
                is_valid, issues = self.validator.validate(narration, text)
                
                if is_valid or attempt == max_retries:
                    markers = self.validator.count_markers(narration)
                    return narration, is_valid, lang, markers
                
            except Exception as e:
                if attempt == max_retries:
                    print(f"\n⚠️ Error generating transcription: {e}")
                    return text, False, lang, {}
        
        return text, False, lang, {}


class TextPreprocessor:
    """Preprocess text for TTS generation."""
    
    def split_into_chapters(self, text):
        """Split text into chapters."""
        chapter_pattern = r'(?:^|\n)(?:Chapter|CHAPTER|अध्याय)\s+(\d+|[IVX]+)(?:\s*[-:.]\s*(.+?))?(?=\n|$)'
        
        matches = list(re.finditer(chapter_pattern, text, re.MULTILINE | re.IGNORECASE))
        
        if not matches:
            return [{
                'number': 1,
                'title': 'Full Text',
                'content': text.strip()
            }]
        
        chapters = []
        
        for i, match in enumerate(matches):
            chapter_num = match.group(1)
            chapter_title = match.group(2) or ""
            
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            content = text[start_pos:end_pos].strip()
            
            chapters.append({
                'number': chapter_num,
                'title': chapter_title.strip() or f"Chapter {chapter_num}",
                'content': content
            })
        
        return chapters
    
    def split_into_sentences(self, text):
        """Split into sentences (Hindi + English)."""
        sentences = re.split(r'(?<=[.!?।])\s+(?=[A-ZА-Я"\u0900-\u097F])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, sentences, chunk_size=6, overlap=1):
        """Create smaller overlapping chunks for better TTS quality."""
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


class TTSTranscriptionGenerator:
    """Main class for generating TTS-optimized transcriptions."""
    
    def __init__(self, provider="ollama", model_name=None, output_dir=".", 
                 device="cpu", language="auto"):
        self.narrator = TTSOptimizedNarrator(provider, model_name, device, language)
        self.preprocessor = TextPreprocessor()
        self.output_dir = Path(output_dir) / "tts_transcriptions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_from_file(self, input_file, chunk_size=6):
        """Generate TTS-optimized transcription from file."""
        print("=" * 80)
        print("🎙️ TTS-OPTIMIZED TRANSCRIPTION GENERATOR")
        print("=" * 80)
        
        print(f"\n📖 Reading: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        primary_lang = TTSOptimizedPrompts.detect_language(text)
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
                "chunk_size": chunk_size,
                "optimization": "TTS-ready with prosodic markers"
            },
            "chapters": []
        }
        
        total_start = time.time()
        successful = 0
        total_chunks = 0
        total_markers = {'pause': 0, 'tone': 0, 'emphasis': 0, 'pace': 0, 'breath': 0}
        
        for ch_idx, chapter in enumerate(chapters, 1):
            print(f"\n{'=' * 80}")
            print(f"📖 Chapter {ch_idx}/{len(chapters)}: {chapter['title']}")
            print(f"{'=' * 80}")
            
            sentences = self.preprocessor.split_into_sentences(chapter['content'])
            chunks = self.preprocessor.create_chunks(sentences, chunk_size=chunk_size, overlap=1)
            
            print(f"📦 Processing {len(chunks)} chunks...")
            total_chunks += len(chunks)
            
            narrated_chunks = []
            
            for c_idx, chunk in enumerate(chunks, 1):
                print(f"   🎙️ Chunk {c_idx}/{len(chunks)}... ", end="", flush=True)
                
                start_time = time.time()
                narration, is_valid, lang, markers = self.narrator.narrate_text(chunk['text'])
                elapsed = time.time() - start_time
                
                # Update marker counts
                for key in total_markers:
                    total_markers[key] += markers.get(key, 0)
                
                if is_valid:
                    successful += 1
                    marker_str = f"P:{markers.get('pause',0)} T:{markers.get('tone',0)} E:{markers.get('emphasis',0)}"
                    print(f"✅ [{lang}] {marker_str} ({elapsed:.1f}s)")
                else:
                    print(f"⚠️ Fallback [{lang}] ({elapsed:.1f}s)")
                
                narrated_chunks.append({
                    "chunk_number": c_idx,
                    "original_text": chunk['text'],
                    "tts_transcription": narration,
                    "language": lang,
                    "is_valid": is_valid,
                    "markers": markers
                })
            
            transcription_data["chapters"].append({
                "chapter_number": ch_idx,
                "title": chapter['title'],
                "chunks": narrated_chunks
            })
        
        total_time = time.time() - total_start
        
        # Save files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.output_dir / f"tts_transcription_{timestamp}.json"
        txt_file = self.output_dir / f"tts_transcription_{timestamp}.txt"
        
        # Save detailed JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, ensure_ascii=False, indent=2)
        
        # Save clean TTS-ready text
        with open(txt_file, 'w', encoding='utf-8') as f:
            for chapter in transcription_data["chapters"]:                
                for chunk in chapter['chunks']:
                    f.write(f"{chunk['tts_transcription']}\n\n")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"🎉 TTS TRANSCRIPTION COMPLETE!")
        print(f"{'='*80}")
        print(f"⏱️ Total time: {total_time/60:.2f} minutes")
        print(f"🌍 Language: {primary_lang.upper()}")
        print(f"📚 Chapters: {len(chapters)}")
        print(f"📦 Total chunks: {total_chunks}")
        print(f"✅ Successful: {successful}/{total_chunks} ({100*successful/total_chunks:.1f}%)")
        print(f"\n🎭 Prosodic Markers Added:")
        print(f"   Pauses: {total_markers['pause']}")
        print(f"   Tones: {total_markers['tone']}")
        print(f"   Emphasis: {total_markers['emphasis']}")
        print(f"   Pace: {total_markers['pace']}")
        print(f"   Breaths: {total_markers['breath']}")
        print(f"   Total: {sum(total_markers.values())}")
        print(f"\n💾 JSON: {json_file}")
        print(f"📄 TXT (TTS-ready): {txt_file}")
        print(f"{'='*80}")
        print("\n✨ This transcription is optimized for TTS models!")
        print("   Feed it to your TTS model for natural, human-like audio.")
        
        return str(txt_file), str(json_file)


def main():
    parser = argparse.ArgumentParser(
        description='TTS-Optimized Transcription Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool generates transcriptions specifically designed for Text-to-Speech models.
The output includes prosodic markers (pauses, tone, emphasis) that help TTS models
produce natural, human-like audio with proper emotion and pacing.

Recommended Models:
  Ollama:
    - gemma2:9b (best for Hindi)
    - aya:8b (multilingual specialist)
    - qwen2.5:14b (excellent instruction following)
    - llama3.1:8b (good for English)
  
  HuggingFace:
    - ai4bharat/Airavata (Indian languages)
    - sarvamai/sarvam-2b-v0.5 (Indian LLM)
    - CohereForAI/aya-23-8B (multilingual)

Examples:
  python transcribe.py -f book.txt -p ollama -m gemma2:9b --language hindi
  python transcribe.py -f book.txt -p ollama -m qwen2.5:14b --device cuda
  python transcribe.py -f book.txt -p huggingface -m ai4bharat/Airavata --language hindi
        """
    )
    
    parser.add_argument('-f', '--file', required=True, help='Input text file')
    parser.add_argument('-p', '--provider', choices=['ollama', 'huggingface'],
                        default='ollama', help='LLM provider')
    parser.add_argument('-m', '--model', help='Model name')
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'rocm', 'auto'],
                        help='Device to use (auto-detects CUDA/ROCm if available)')
    parser.add_argument('--language', default='auto', choices=['auto', 'hindi', 'english'])
    parser.add_argument('--chunk-size', type=int, default=6,
                        help='Sentences per chunk (smaller = better TTS quality, default: 6)')
    
    args = parser.parse_args()
    
    # Handle device argument
    if args.device == "auto":
        device = "cpu"  # Will be auto-detected in TTSOptimizedNarrator
    elif args.device == "rocm":
        device = "cuda"  # ROCm uses CUDA interface
    else:
        device = args.device
    
    if not Path(args.file).exists():
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)
    
    try:
        generator = TTSTranscriptionGenerator(
            provider=args.provider,
            model_name=args.model,
            output_dir=args.output,
            device=device,
            language=args.language
        )
        
        txt_file, json_file = generator.generate_from_file(
            args.file,
            chunk_size=args.chunk_size
        )
        
        print(f"\n✅ TTS-ready transcription: {txt_file}")
        print(f"📊 Detailed data: {json_file}")
        
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