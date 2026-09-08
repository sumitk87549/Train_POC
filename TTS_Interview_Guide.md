# 🎙️ VoxCPM2 JARVIS TTS Engine (v4) — Complete Interview Preparation Guide

> **Target Role**: AI/ML Engineer / Speech AI / Audio LLM Developer  
> **Language**: Easy & Natural Hinglish (Hindi + English)  
> **Source Code**: [`TTS/VoxCPM2_JARVIS_TTS_v4.ipynb`](file:///home/sumit/Documents/GitHub/Train_POC/TTS/VoxCPM2_JARVIS_TTS_v4.ipynb)  
> *(Note: Code snippets nahi hain, sirf simplified concepts, function names, aur interview explanation guidance hai so aap easily notebook refer kar sako.)*

---

## 1. Project Overview & High-Level Architecture

### Summary (Interview intro line)
> *"Maine ek Production-Grade Hindi & Hinglish Audiobook TTS Pipeline build kiya hai using OpenBMB's VoxCPM2 model (2B Parameters, 48 kHz output). Yeh pipeline raw text ko studio-quality narration audio mein convert karti hai with automatic text normalization, intelligent sentence-boundary chunking, style propagation, acoustic artifact cleanup, aur seamless audio stitching without any clicks or pops."*

### Key Specs & Tech Stack
- **Base Model**: OpenBMB VoxCPM2 (2 Billion Parameters, 30+ Languages supported).
- **Audio Output**: 48 kHz High-Resolution Studio Quality Audio (PCM 16-bit WAV).
- **Target Hardware**: NVIDIA T4 GPU (Google Colab / Cloud environment with >= 8 GB VRAM).
- **Core Python Libraries**: `voxcpm`, `transformers`, `torch`, `soundfile`, `scipy`, `numpy`, `librosa`.

---

## 2. Notebook Workflow & Structural Breakdown

Interview mein interviewer aksar puchta hai: *"Aapka notebook end-to-end kaise run hota hai?"*

Notebook **7 Cells** mein structured hai:
1. **Cell 1 — System Diagnostics & Boot Sequence**: GPU VRAM, CUDA compatibility, storage space checks (`nvidia-smi` query).
2. **Cell 2 — Package Installation & Verification**: Pip installation of `voxcpm`, `transformers`, `accelerate`, `soundfile`, `scipy`, `librosa`.
3. **Cell 3 — Text Upload & Preprocessing Pipeline**: UTF-8 `.txt` file upload + Layer 1 & Layer 2 normalization toggles.
4. **Cell 4 — Reference Audio / Voice Cloning Config**: Optional voice cloning setup supporting 3 modes:
   - **Default TTS**: VoxCPM2 built-in voice.
   - **Voice Clone**: Reference audio clip (10-20 sec clean clip).
   - **Ultimate Clone**: Reference audio + exact transcript prompt for maximum voice fidelity.
5. **Cell 5 — User Configuration & Parameter Locking**: Chunk sizes, inference timesteps, CFG values, silence gap timings, crossfade duration, checkpoint directory setup.
6. **Cell 6 — TTS Generation Engine**: Load model, smart audiobook chunker, audio generation loop with retry & caching, silence trimming, Hanning crossfade stitching.
7. **Cell 7 — Playback & Download**: In-notebook playback, peak normalization report (-1 dBFS), file export.

---

## 3. Core Technical Concepts & Engineering Decisions

### A. Text Preprocessing & Normalization Pipeline (Cell 3)
Raw text ko bina clean kiye TTS model mein bhejne se hallucination, weird pauses, aur phonetic errors aate hain. Maine **2-Layer Normalization Pipeline** build kiya:

1. **Layer 1 (`_clean_text`) — Always-On Structural Cleanup**:
   - **Line-Ending Unification**: `\r\n` ko `\n` mein convert karta hai.
   - **Decorative Line Removal**: Lines like `=====`, `-----` carry no spoken meaning. Unhe strip karke blank lines banaya jata hai jo natural pauses bante hain.
   - **Unicode NFC Composition**: Devanagari characters mein vowel marks (matras) base letters ke saath properly compose hone chahiye (`unicodedata.normalize("NFC")`).
   - **Zero-Width Character Cleanup**: Invisible control characters like `\u200b` (zero-width space), `\ufeff` (BOM), soft-hyphens ko remove karta hai.
   - **Whitespace Collapsing**: Multiple spaces/tabs ko single space mein replace karta hai, trailing line spaces strip karta hai.

2. **Layer 2 (`_normalize_text`) — Typographic Normalization**:
   - **Curly Quotes to ASCII**: Smart single/double quotes (`‘’“”`) ko straight ASCII (`'"`) mein change karta hai.
   - **Dash & Ellipsis Normalization**: En-dash (`–`) ko hyphen (`-`) mein, ellipsis character (`…`) ko three dots (`...`) mein convert karta hai.
   - **Punctuation Spacing**: Stray spaces before punctuation (` hello .`) ko fix karta hai. Sentence-ending marks ke baad single space ensure karta hai, but closing quotes/brackets ke cases ko carefully handle karta hai.

---

### B. Smart Audiobook Chunker (`smart_chunk_audiobook` in Cell 6)

#### Interview Question: *"Aap pooray chapter ko ek saath TTS model mein kyu nahi bhejte?"*
> **Answer**: VoxCPM2 ki safe context limit 480 characters tak hai. Agar 480 chars se bada text bhejenge to GPU VRAM spike hoga, model hallucinate karega, aur speech rate unnatural (choppy ya fast) ho jayegi. Isliye **Text Chunking** zaroori hai.

#### Chunker Architecture:
1. **Chapter Heading Detection (`_CHAPTER_RE`)**:
   - Pattern match karta hai like `Chapter 1`, `अध्याय 2`, `Part IV`, `## Heading`.
   - Chapter headings ko dedicated 2.0 second silence pause assign kiya jata hai.
2. **Hierarchical Splitting**:
   - Pehle double-newlines (`\n\n`) se Paragraphs split hote hain.
   - Phir sentence boundary regex (`_SENT_RE`) se sentences split hote hain.
   - Agar sentence tab bhi target chunk size (`CHUNK_SIZE` = 380 chars) se bada ho, to clause boundary regex (`_CLAUSE_RE` — commas, semicolons, em-dashes) par split hota hai.
   - Last resort: `_hard_split_words` word-boundary split.
3. **Minimum Chunk Enforcement (`MIN_CHUNK_SIZE` = 80 chars)**:
   - Sub-80 character short sentences ko previous neighbor chunk ke saath merge kar diya jata hai taaki short isolated fragments natural prosody spoil na karein.
4. **Style Cue Extraction & Propagation (`_extract_style_cue`)**:
   - Text paragraphs ke shuru mein `(warm engaging narrator, Hinglish flow)` jaisa parenthetical style cue ho sakta hai.
   - Jab paragraph multiple chunks mein split hota hai, yeh extractor style cue ko extract karta hai aur **har child chunk ke aage re-attach** kar deta hai. Isse har inference call mein identical voice guidance rehti hai.

---

## 4. The 4 Legendary Bug-Fix Stories (🔥 Interview Masterclass)

Interview mein jab aap ye specific bugs aur unke fixes discuss karoge, to interviewer ko 100% confidence ho jayega ki aapne real-world acoustic problems solve kiye hain.

---

### 🐛 Bug-001 Fix: Dialogue Terminal Punctuation (`_ensure_terminal_punct`)
- **Problem**: Hindi dialogue chunks aksar `।"` (Devanagari danda + closing quote) par end hote hain. Purana code sirf `text[-1]` check karta tha. Since last character `"` tha (jo terminal punctuation set mein nahi hai), code uske baad ek period add kar deta tha -> `।".`.
- **Acoustic Impact**: `।".` sequence VoxCPM2 ke phoneme tokenizer ko confuse kar deta tha, jis se narration ke end mein ek sharp **high-pitch screeching noise spike** generate hota tha.
- **Fix**: `_ensure_terminal_punct` trailing closing characters (`"`, `'`, `)`, `]`, `}`, `»`) ke peeche dekhta hai (walk back). Agar real last character pehle se terminal mark (`।`, `.`, `!`, `?`) hai, to text untouched rehta hai. Agar terminal mark missing hai, to closing quote ke **pehle** Devanagari danda (`।`) ya period (`.`) insert karta hai.

---

### 🐛 Bug-005 Fix: Short Paragraph Merging (`_merge_short_paragraphs`)
- **Problem**: Text mein short single-line paragraphs like `"कहाँ?"` (7 chars) ya `"Exactly।"` (10 chars) rehte hain. Standard chunking inside paragraph kaam karti thi, isliye ye short paragraphs isolate hokar sub-15-char orphan chunks ban jaate the.
- **Acoustic Impact**: Model sub-15-char inputs par speech cadence miss kar deta tha, resulting in distorted, metallic, or truncated audio clips.
- **Fix**: Paragraph list level par `_merge_short_paragraphs` chalaya jo short paragraphs ko aage waale paragraph ke saath merge kar deta hai (forward merge) pehle hi stage par. Heading paragraphs ko merge hone se protect kiya gaya hai.

---

### 🐛 Bug-004 Fix: Model-Padding Silence Trimming (`_trim_silence`)
- **Problem**: VoxCPM2 generation ke aage-peeche naturally 50 ms se 200 ms ki model-padding silence wrap karke deta hai.
- **Acoustic Impact**: Jab hum 15 ms crossfade apply karte the, to crossfade speech audio par nahi balki model-padding silence par apply hota tha! Pure speech boundary unfaded reh jaati thi, jis se audio stitching points par **pop/click noise** aata tha.
- **Fix**: `_trim_silence` function output numpy array par per-frame RMS energy compute karta hai (8 ms frames at -42 dBFS threshold) aur leading/trailing padding silence ko trim kar deta hai with 6 ms safety margin. Crossfade hamesha real speech boundaries par lagta hai.

---

### 🐛 Bug-002 & Bug-003 Fix: Smooth Audio Stitching & Click Elimination (`stitch_audio`)
- **Problem**: 
  1. *Linear Crossfade Ramp*: Linear ramp mein onset par slope discontinuity hoti hai (derivative step change), jo 48 kHz sample rate par micro-click sound produce karti hai.
  2. *Chunk Head Discontinuity*: VoxCPM2 ka output waveform non-zero amplitude se start ho sakta hai. Jab true silence (`np.zeros`) se non-zero waveform join hota hai, to step-discontinuity hoti hai -> **broadband click sound**.
- **Fix**:
  1. Linear ramp ko replace karke **Raised-Cosine (Hanning half-window)** apply kiya (`np.hanning`). Hanning window ke dono ends par zero derivative hota hai -> smooth entry and exit without slope click.
  2. Output chunk tail par Fade-OUT **aur** chunk head par **Fade-IN** dono apply kiye (`seg_arr[:fade_n] *= fade_in`), jo initial step-discontinuity click ko poori tarah eliminate kar deta hai.

---

## 5. Audio Post-Processing & Silence Gap Mapping

### Silence Gap Hierarchy (`SILENCE_MAP`):
Audiobooks mein pacing natural lagne ke liye different structural boundaries par alag gap durations add kiye gaye hain:
- **Chapter Break**: 2.00 seconds (`SILENCE_CHAPTER`)
- **Paragraph Break**: 0.55 seconds (`SILENCE_PARAGRAPH`)
- **Sentence Break**: 0.20 seconds (`SILENCE_SENTENCE`)
- **Clause Break**: 0.08 seconds (`SILENCE_CLAUSE`)

### Peak Normalization (-1 dBFS):
Audio stitching ke baad total output array ki maximum peak amplitude check hoti hai. Peak ko `-1 dBFS` (`target = 10 ** (-1.0 / 20)`) par scale kiya jata hai.
- **Why -1 dBFS?** 0 dBFS par audio digital clipping and DAC inter-sample peaks causes distortion on speakers. -1 dBFS industry standard headroom provide karta hai for audiobook distribution (ACX / Audible standards).

---

## 6. Fault Tolerance, Caching & VRAM Resilience

### A. Per-Chunk Disk Checkpointing (`RESUME_CHUNKS`)
- Target directory: `CHUNKS_DIR = "/content/chunks"`
- Har generated audio chunk numpy array format (`.npy`) mein save hota hai as `chunk_0001.npy`.
- Agar Google Colab GPU timeout ya disconnect ho jaye, next run par pehle se completed chunks disk se instant load ho jaate hain. System waheen se resume karta hai jahan crash hua tha.

### B. 3-Attempt Retry Loop with Memory Cleanup
- Har chunk generation `for attempt in range(3):` block mein wrapped hai.
- Agar `torch.cuda.OutOfMemoryError` aati hai:
  1. `gc.collect()` Python garbage collector run hota hai.
  2. `torch.cuda.empty_cache()` VRAM clear karta hai.
  3. `time.sleep(4)` wait karke model retry karta hai.

---

## 7. How to Answer Core Interview Questions (Cheat Sheet)

### Q1: "Aapne TTS Pipeline mein audio clicking aur popping sounds ko kaise eliminate kiya?"
> **Answer**:  
> *"Audio click do jagah aate hain: speech-to-silence transition aur non-zero waveform onset. Maine 3-step solution implement kiya:*  
> *1. **Silence Trimming (`_trim_silence`)**: Pehle VoxCPM2 ki 50-200ms model-padding silence ko RMS energy analysis (-42 dBFS) se trim kiya taaki crossfade exact speech edge par lage.*  
> *2. **Hanning Window Fade-Out**: Linear ramp ki jagah Raised-Cosine (Hanning) window use kiya kyunki Linear ramp mein slope discontinuity aati hai jo click karti hai, jabki Hanning window ki derivative boundary par zero hoti hai.*  
> *3. **Chunk Head Fade-IN**: Non-zero start amplitude ko zero-silence se bridge karne ke liye 15ms Fade-IN lagaya."*

### Q2: "VoxCPM2 model ke saath Text Preprocessing kyu zaroori thi?"
> **Answer**:  
> *"TTS models raw characters, unicode composite marks, aur strange punctuation se heavily affect hote hain. Maine 2-layer pipeline banayi:*  
> *Layer 1 Unicode NFC normalization and zero-width char cleanup karti hai.*  
> *Layer 2 smart quotes aur dash ko normalize karti hai. Specially, maine `_ensure_terminal_punct` function banaya jo dialogue trailing quotes `।"` ke piche dekhta hai aur correct danda `।` insert karta hai before closing quote, prevent kisibhi high-pitch phoneme noise spikes ko."*

### Q3: "Audiobook length text ko handle karne ke liye chunking strategy kya thi?"
> **Answer**:  
> *"Maine `smart_chunk_audiobook` algorithm design kiya:*  
> *1. Sentence aur clause boundaries par split karta hai with a max limit of 380 chars (VoxCPM2 safe VRAM window).*  
> *2. Short chunks (<80 chars) ko neighbour chunks ke saath merge karta hai (`MIN_CHUNK_SIZE`) for smooth natural prosody.*  
> *3. Heading detection karta hai for 2.0s chapter breaks.*  
> *4. Paragraph level style cues `(warm engaging narrator)` ko extract karke sabhi child chunks par propagate karta hai for consistent voice continuity."*

---

## Summary Checklist for Interview
- [x] Model Name & Params: **VoxCPM2 (2B Params, 48 kHz)**
- [x] Target Chunk Size: **380 chars (max 480), Min chunk: 80 chars**
- [x] Text Cleanup: **Unicode NFC, Layer 1 & 2 Normalization**
- [x] Fix 1: **Terminal Punctuation Lookahead (`।"` fix)**
- [x] Fix 2: **Short Paragraph Forward Merge**
- [x] Fix 3: **RMS Silence Trimming (-42 dBFS)**
- [x] Fix 4: **Hanning Window Crossfade + Head Fade-IN**
- [x] Audio Standard: **Peak Normalized to -1 dBFS, PCM_16 WAV**
- [x] Crash Recovery: **Per-chunk `.npy` cache (`CHUNKS_DIR`)**
