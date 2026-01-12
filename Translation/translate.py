import ollama
from tqdm import tqdm
import time
import os
import sys
import json
import argparse
from pathlib import Path

# ==== MODEL RECOMMENDATIONS FOR YOUR HARDWARE ====
"""
TIER 1 - FAST TESTING (30-60s per chunk):
- qwen2.5:3b         [RECOMMENDED] - Fast, excellent instruction following, rarely summarizes
- phi3.5:3.8b        - Very reliable, good translation
- llama3.2:3b        - Fast and smart

TIER 2 - INTERMEDIATE (60-120s per chunk):
- qwen2.5:7b         [RECOMMENDED] - Excellent literary translation, best quality/speed
- gemma2:9b          - High quality
- llama3.1:8b        - Very capable

TIER 3 - ADVANCED (120-300s per chunk):
- qwen2.5:14b        [BEST QUALITY] - Professional-grade literary translation
- qwen2.5:32b        - Highest quality (needs 16GB+ RAM)

⚠️  AVOID for translation: gemma2:2b (too small, summarizes heavily)

Install: ollama pull qwen2.5:3b
"""

# ==== CONFIGURATION ====
DEFAULT_CONFIG = {
    "model": "qwen2.5:3b",
    "tier": "BASIC",
    "chunk_words": 350,  # Reduced for better context management
    "temperature": 0.3,  # Lower = more faithful translation
    "top_p": 0.9,
    "num_ctx": 8192,  # Increased context window
    "retry_attempts": 3,
    "retry_delay": 2
}

# ==== IMPROVED ANTI-SUMMARIZATION PROMPTS ====
TRANSLATION_PROMPTS = {
    "BASIC": {
        "system": """You are a professional Hindi translator. Your job is to translate EVERY WORD and EVERY SENTENCE from English to Hindi.

🚨 CRITICAL RULES - NEVER BREAK THESE:
1. TRANSLATE EVERYTHING - Do not skip even a single sentence
2. DO NOT SUMMARIZE - Every detail must be translated
3. MAINTAIN LENGTH - Hindi output should be similar length to English input
4. KEEP ALL DIALOGUE - Translate every spoken word exactly
5. PRESERVE ALL DESCRIPTIONS - Every adjective, every scene detail
6. INCLUDE ALL NAMES & PLACES - Transliterate them properly

Translation Guidelines:
- Use natural, contemporary Hindi (Hindustani blend)
- Keep the same paragraph structure
- Transliterate foreign names: London → लंदन, Watson → वॉटसन, Holmes → होम्स
- Common words: Doctor → डॉक्टर, Hospital → अस्पताल
- Translate idioms to Hindi equivalents
- Maintain the author's tone and style

VERIFICATION CHECKLIST (for yourself):
✓ Did I translate every sentence?
✓ Is my Hindi output similar length to English?
✓ Did I include all character dialogue?
✓ Did I preserve all scene descriptions?
✓ Did I keep all plot details?

If your translation is much shorter than the original, you are SUMMARIZING (forbidden).
If you skip details, you are FAILING your task.

Remember: You are a TRANSLATOR, not a SUMMARIZER.""",

        "user": """Translate this COMPLETE passage to Hindi. Translate EVERY sentence, EVERY word.

DO NOT SUMMARIZE. DO NOT SKIP DETAILS.

English Text:
\"\"\"
{chunk}
\"\"\"

Translate the ENTIRE text above into natural Hindi, preserving all information."""
    },

    "INTERMEDIATE": {
        "system": """You are an expert Hindi literary translator specializing in complete, faithful translations of English literature.

🎯 YOUR MISSION: Translate EVERYTHING - word for word, sentence for sentence, paragraph for paragraph.

🚫 ABSOLUTE PROHIBITIONS:
- NO summarizing (cardinal sin of translation)
- NO skipping sentences or details
- NO condensing information
- NO omitting dialogue or descriptions
- NO shortening the text

✅ MANDATORY REQUIREMENTS:
1. COMPLETE TRANSLATION - Every single sentence must be translated
2. LENGTH PRESERVATION - Hindi output ≈ same length as English (±20%)
3. ALL DIALOGUE - Every spoken word, every conversation
4. ALL DESCRIPTIONS - Every scene, every detail, every adjective
5. ALL PLOT POINTS - Every event, every action
6. ALL NAMES - Properly transliterated

Translation Approach:
- Transform English idioms into natural Hindi equivalents:
  * "thin as a rail" → "सूखी लकड़ी जैसा"
  * "clear as day" → "दिन के उजाले जैसा साफ"
- Preserve character voices through language register
- Maintain narrative flow and pacing
- Keep the author's literary style

Technical Guidelines:
- Names: London → लंदन, Afghanistan → अफ़गानिस्तान
- Titles: Doctor → डॉक्टर, Mr. → श्री/मिस्टर
- Common terms: Hospital → अस्पताल, Regiment → रेजिमेंट
- Keep British military ranks: Fusiliers → फ्यूसिलियर्स

Language Style:
- Use contemporary literary Hindi (not pure Sanskrit Hindi)
- Natural Hindustani for dialogue
- Formal but accessible narration
- Preserve emotional tone

QUALITY CHECK (internal):
1. Count paragraphs: English = Hindi? ✓
2. Compare length: Similar? ✓
3. All dialogue present? ✓
4. All descriptions included? ✓
5. No information loss? ✓

Think: "If I were reading this Hindi version, would I get the EXACT same story and details as the English?"

Your reputation depends on COMPLETENESS.""",

        "user": """Translate this ENTIRE passage into Hindi. Every sentence, every detail, every word.

This is NOT a summarization task. This is COMPLETE translation.

English Text:
\"\"\"
{chunk}
\"\"\"

Provide COMPLETE Hindi translation maintaining all information and similar length."""
    },

    "ADVANCED": {
        "system": """You are a master literary translator creating Hindi versions of English classics. You possess the linguistic artistry of Premchand and the precision of professional translators.

⚡ CORE MANDATE: COMPLETE, FAITHFUL, BEAUTIFUL TRANSLATION

🎯 TRANSLATION PHILOSOPHY:

1. ABSOLUTE COMPLETENESS:
   - Translate EVERY word, EVERY sentence, EVERY paragraph
   - The Hindi version must contain 100% of the original information
   - Length ratio: Hindi ≈ 0.9-1.2x English (slight variation acceptable)
   - If English has 50 sentences, Hindi must have 50 sentences

2. ZERO TOLERANCE FOR SUMMARIZATION:
   - Summarization is the worst translator's sin
   - Every scene description must be complete
   - Every dialogue must be fully translated
   - Every thought, every observation, every detail

3. LITERARY EXCELLENCE:
   - Transform idioms: "thin as a rail" → "लकड़ी की तरह सूखा"
   - Adapt metaphors: "heart of gold" → "सोने जैसा दिल"
   - Preserve rhythm and flow
   - Maintain emotional impact

4. CHARACTER VOICE PRESERVATION:
   - Educated characters: formal literary Hindi
   - Common people: natural Hindustani
   - Maintain personality through language choices
   - Consistent speech patterns per character

5. CULTURAL BRIDGING (WITHOUT EXPLANATION):
   - Make British Victorian era accessible to Indian readers
   - Use narrative flow to convey unfamiliar customs
   - Maintain authenticity while ensuring clarity

6. TECHNICAL PRECISION:
   - Proper names: Transliterate consistently
     * London → लंदन, Bombay → बंबई, Afghanistan → अफ़गानिस्तान
     * Watson → वॉटसन, Holmes → होम्स, Sherlock → शर्लक
   - Titles: Doctor → डॉक्टर, Mr. → मिस्टर/श्री
   - Military terms: Regiment → रेजिमेंट, Fusiliers → फ्यूसिलियर्स
   - Places: Baker Street → बेकर स्ट्रीट
   - Medical: haemoglobin → हीमोग्लोबिन

7. DIALOGUE MASTERY:
   - Every word of conversation must be translated
   - Maintain Victorian-era formality where present
   - Preserve humor, sarcasm, emotion
   - Natural Hindi that doesn't sound "translated"

8. DESCRIPTION COMPLETENESS:
   - Every scene detail: rooms, streets, people, weather
   - Every sensory description: sights, sounds, smells
   - Every emotional state: preserve nuance
   - Every action: no matter how small

9. STRUCTURAL INTEGRITY:
   - Maintain all paragraph breaks
   - Preserve emphasis and pacing
   - Keep the same chapter/section structure
   - Honor the original's flow

10. QUALITY BENCHMARKS:
    - A Hindi reader should get the EXACT same story
    - No information should be lost
    - The prose should feel native to Hindi
    - Translation should be invisible (not feel translated)

🔍 SELF-VERIFICATION PROTOCOL:
Before submitting translation, verify:
□ Sentence count: English = Hindi?
□ Paragraph count: Same?
□ All dialogue translated?
□ All descriptions included?
□ All names and places present?
□ Length roughly similar?
□ No summarization occurred?
□ Reads naturally in Hindi?

💎 QUALITY STANDARD:
Your Hindi should be publishable. Think: "Would a Hindi publisher accept this as a complete, professional translation worthy of the original?"

Remember: Hemingway said "Prose is architecture, not interior decoration." Your job is to rebuild the entire architectural structure in Hindi, brick by brick, not to give a summary tour.""",

        "user": """Create a COMPLETE, COMPREHENSIVE Hindi translation of this passage.

Translate EVERY sentence, EVERY detail, EVERY word. Do NOT summarize.

English Text:
\"\"\"
{chunk}
\"\"\"

Provide COMPLETE Hindi translation that matches the original in detail and depth."""
    }
}

# ==== PROGRESS TRACKING ====
class TranslationProgress:
    def __init__(self, progress_file):
        self.progress_file = progress_file
        self.data = self.load()
    
    def load(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"completed_chunks": [], "last_chunk": 0, "total_chunks": 0}
    
    def save(self):
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def mark_complete(self, chunk_num):
        if chunk_num not in self.data["completed_chunks"]:
            self.data["completed_chunks"].append(chunk_num)
            self.data["last_chunk"] = chunk_num
            self.save()
    
    def is_complete(self, chunk_num):
        return chunk_num in self.data["completed_chunks"]
    
    def reset(self):
        self.data = {"completed_chunks": [], "last_chunk": 0, "total_chunks": 0}
        self.save()

# ==== UTILITY FUNCTIONS ====
def chunk_text(text, chunk_words=350):
    """Split text into chunks by word count, trying to break at paragraph boundaries."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_count = 0
    
    for para in paragraphs:
        para_words = para.split()
        para_count = len(para_words)
        
        # If adding this paragraph would exceed limit and we have content, save chunk
        if current_count + para_count > chunk_words and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_count = para_count
        else:
            current_chunk.append(para)
            current_count += para_count
    
    # Add remaining
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def clean_translation(text):
    """Clean up common translation artifacts."""
    # Remove thinking markers
    text = text.replace("<think>", "").replace("</think>", "")
    # Remove markdown code blocks if model outputs them
    text = text.replace("```hindi", "").replace("```", "")
    # Clean up excessive whitespace
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    return text.strip()

def validate_translation(original, translated, chunk_num):
    """Check if translation seems complete (not summarized)."""
    orig_words = len(original.split())
    trans_chars = len(translated)
    
    # Hindi typically has 0.8-1.2x character count of English
    expected_min_chars = orig_words * 4  # Very conservative estimate
    
    warnings = []
    
    if trans_chars < expected_min_chars * 0.6:
        warnings.append(f"⚠️  WARNING: Chunk {chunk_num} seems too short!")
        warnings.append(f"   Expected ~{expected_min_chars} chars, got {trans_chars}")
        warnings.append(f"   This might indicate SUMMARIZATION instead of translation")
    
    # Check for paragraph count
    orig_paras = original.count('\n\n') + 1
    trans_paras = translated.count('\n\n') + 1
    
    if trans_paras < orig_paras * 0.7:
        warnings.append(f"⚠️  WARNING: Paragraph count mismatch!")
        warnings.append(f"   Original: {orig_paras} paragraphs, Translation: {trans_paras}")
    
    return warnings

def translate_chunk(chunk, chunk_num, total_chunks, config, prompts):
    """Translate a single chunk with validation."""
    print(f"\n{'='*60}")
    print(f"📄 CHUNK {chunk_num}/{total_chunks}")
    print(f"{'='*60}")
    print(f"📝 Input: {len(chunk)} chars, {len(chunk.split())} words, {chunk.count(chr(10))+1} paragraphs")
    
    for attempt in range(config['retry_attempts']):
        try:
            start_time = time.time()
            
            user_prompt = prompts["user"].format(chunk=chunk)
            
            print(f"\n🤖 Translating with {config['model']} (attempt {attempt + 1})...")
            
            response = ollama.chat(
                model=config['model'],
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": config['temperature'],
                    "top_p": config['top_p'],
                    "num_ctx": config['num_ctx']
                }
            )
            
            translated = clean_translation(response["message"]["content"])
            elapsed = time.time() - start_time
            
            print(f"\n✅ Translation completed in {elapsed:.1f}s")
            print(f"📊 Output: {len(translated)} chars, {translated.count(chr(10))+1} paragraphs")
            print(f"📈 Char ratio: {len(translated)/len(chunk):.2f}x")
            
            # Validate translation
            warnings = validate_translation(chunk, translated, chunk_num)
            if warnings:
                print(f"\n{'='*60}")
                for warning in warnings:
                    print(warning)
                print(f"{'='*60}")
            
            return translated
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            if attempt < config['retry_attempts'] - 1:
                wait_time = config['retry_delay'] * (attempt + 1)
                print(f"⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"💥 Failed after {config['retry_attempts']} attempts")
                raise

def validate_model(model_name):
    """Check if model exists locally."""
    try:
        ollama.show(model_name)
        return True
    except:
        return False

# ==== MAIN TRANSLATION FUNCTION ====
def main():
    parser = argparse.ArgumentParser(
        description='Translate English literature to Hindi with AI (Anti-Summarization Enhanced)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with recommended model
  python translate_improved.py input.txt -m qwen2.5:3b -t BASIC
  
  # Quality translation
  python translate_improved.py input.txt -m qwen2.5:7b -t INTERMEDIATE
  
  # Best quality
  python translate_improved.py input.txt -m qwen2.5:14b -t ADVANCED
  
  # Resume interrupted translation
  python translate_improved.py input.txt --resume
        """
    )
    
    parser.add_argument('input_file', help='Input text file to translate')
    parser.add_argument('-o', '--output', default='output_hi.txt', help='Output file')
    parser.add_argument('-m', '--model', default='qwen2.5:3b', help='Ollama model')
    parser.add_argument('-t', '--tier', choices=['BASIC', 'INTERMEDIATE', 'ADVANCED'], 
                       default='BASIC', help='Translation quality tier')
    parser.add_argument('--chunk-words', type=int, default=350, help='Words per chunk')
    parser.add_argument('--temperature', type=float, default=0.3, help='Model temperature (lower = more faithful)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--reset', action='store_true', help='Reset and start fresh')
    
    args = parser.parse_args()
    
    # Configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        'model': args.model,
        'tier': args.tier,
        'chunk_words': args.chunk_words,
        'temperature': args.temperature
    })
    
    # Validate input
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' not found!")
        sys.exit(1)
    
    # Validate model
    print(f"🔍 Checking model: {config['model']}...")
    if not validate_model(config['model']):
        print(f"❌ Model '{config['model']}' not found!")
        print(f"💡 Install it with: ollama pull {config['model']}")
        print(f"\n📌 RECOMMENDED MODELS:")
        print(f"   Fast testing: ollama pull qwen2.5:3b")
        print(f"   Good quality: ollama pull qwen2.5:7b")
        print(f"   Best quality: ollama pull qwen2.5:14b")
        sys.exit(1)
    
    print(f"✅ Model ready\n")
    
    # Print configuration
    print("=" * 70)
    print("🚀 HINDI LITERARY TRANSLATION - ANTI-SUMMARIZATION ENHANCED")
    print("=" * 70)
    print(f"📖 Input:       {args.input_file}")
    print(f"💾 Output:      {args.output}")
    print(f"🤖 Model:       {config['model']}")
    print(f"🎯 Tier:        {config['tier']}")
    print(f"📦 Chunk size:  {config['chunk_words']} words")
    print(f"🌡️  Temperature: {config['temperature']} (lower = more faithful)")
    print(f"🔍 Validation:  ENABLED (checks for summarization)")
    print("=" * 70)
    
    # Read input
    print(f"\n📖 Reading input...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean chapter markers
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not (line.strip().startswith('===') and line.strip().endswith('==='))]
    text = '\n'.join(cleaned_lines).strip()
    
    word_count = len(text.split())
    char_count = len(text)
    print(f"📊 Total: {char_count:,} chars, {word_count:,} words")
    
    # Create chunks
    print(f"\n📦 Creating chunks...")
    chunks = chunk_text(text, config['chunk_words'])
    total_chunks = len(chunks)
    print(f"✅ Created {total_chunks} chunks")
    
    # Progress tracking
    progress_file = f"{args.output}.progress.json"
    progress = TranslationProgress(progress_file)
    
    if args.reset:
        progress.reset()
        print("🔄 Progress reset")
    
    # Get prompts
    prompts = TRANSLATION_PROMPTS[config['tier']]
    
    # Translation
    print("\n" + "=" * 70)
    print("🎯 STARTING TRANSLATION")
    print("=" * 70)
    
    start_time = time.time()
    translated_chars = 0
    mode = 'a' if (args.resume and progress.data['last_chunk'] > 0) else 'w'
    
    if mode == 'a':
        print(f"📄 Resuming from chunk {progress.data['last_chunk'] + 1}")
    
    try:
        with open(args.output, mode, encoding='utf-8') as out:
            for i, chunk in enumerate(chunks, 1):
                if progress.is_complete(i):
                    print(f"\n⏭️  Chunk {i}/{total_chunks} - Already done")
                    continue
                
                translated = translate_chunk(chunk, i, total_chunks, config, prompts)
                out.write(translated + "\n\n")
                out.flush()
                
                translated_chars += len(translated)
                progress.mark_complete(i)
                
                # Progress
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = total_chunks - i
                eta = remaining * avg_time
                
                print(f"\n📈 Overall Progress: {i}/{total_chunks} ({i/total_chunks*100:.1f}%)")
                print(f"⏱️  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
                print(f"📝 Translated: {translated_chars:,} chars")
        
        # Summary
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("🎉 TRANSLATION COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Total time:    {total_time/60:.1f} minutes")
        print(f"📦 Chunks:        {total_chunks}")
        print(f"⚡ Avg per chunk: {total_time/total_chunks:.1f}s")
        print(f"📝 Input:         {char_count:,} chars")
        print(f"📝 Output:        {translated_chars:,} chars")
        print(f"📊 Ratio:         {translated_chars/char_count:.2f}x")
        print(f"💾 Saved to:      {args.output}")
        print("=" * 70)
        
        # Clean up
        if os.path.exists(progress_file):
            os.remove(progress_file)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted - Progress saved")
        print(f"💡 Resume with: --resume")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 Error: {str(e)}")
        print(f"💡 Resume with: --resume")
        sys.exit(1)

if __name__ == "__main__":
    main()