# 📚 Multi-Format Book Summarization Studio — Complete Interview Preparation Guide

> **Target Role**: AI/ML Engineer / GenAI Developer / Prompt Engineer  
> **Language**: Easy & Natural Hinglish (Hindi + English)  
> **Source Code**: [`summarize/summary_generator_colab.ipynb`](file:///home/sumit/Documents/GitHub/Train_POC/summarize/summary_generator_colab.ipynb)  
> *(Note: Code snippets nahi hain, sirf simplified concepts, function names, aur interview explanation guidance hai so aap easily notebook refer kar sako.)*

---

## 1. Project Overview & High-Level Architecture

### Summary (Interview intro line)
> *"Maine ek Commercial-Grade Multi-Format Book Summarization Engine build kiya hai using Ollama served open-weights LLMs (Gemma 3 27B, Qwen 3 14B, Llama 3.1 8B). Yeh system long books ko Map-Reduce Two-Pass architecture se process karke 9 publication-ready formats mein summarize karta hai — ranging from factual overviews to Instagram captions, YouTube narration scripts, Twitter threads, aur SEO blog posts."*

### Key Specs & Tech Stack
- **Supported Models**: 
  - `gemma3:27b` (Best creative writing & prose quality)
  - `gemma3:12b` (Balanced quality for T4 GPU)
  - `qwen3:14b` (Strong reasoning & multilingual/Hindi support)
  - `llama3.1:8b` (Fast, 128K context window)
- **Model Server**: Ollama background server running on Colab T4 GPU.
- **Python Stack**: `ollama`, `ipywidgets`, `colorama`, `google.colab.files`.
- **Core Design Pattern**: **Map-Reduce (Two-Pass) Hierarchical Summarization** with sentence-aware chunking and word overlap.

---

## 2. Notebook Workflow & Execution Flow

Interview mein interviewer puchta hai: *"Aapka summarization studio step-by-step kaise execution manage karta hai?"*

Notebook **6 Clear Steps** mein divided hai:
1. **Step 1 — Dependency & Ollama Setup**: Installing `ollama`, `ipywidgets`, system `zstd`, launching Ollama server in background (`OLLAMA_HOST = 127.0.0.1:11434`).
2. **Step 2 — Model Selection & Pull**: Dropdown widget se model select karke `ollama.pull()` streaming download execute karna.
3. **Step 3 — Summarization Engine Definition**: `SUMMARY_TYPES` dict (9 specifications), `PROMPTS` dict, sentence overlap chunker (`chunk_text_with_overlap`), `clean_output` function, and `generate_summary` driver function load karna.
4. **Step 4 — Text Upload**: `.txt` book upload karke word count preview dikhana.
5. **Step 5 — Format Choice & Execution**: Output type, chunk words (e.g. 500 words), overlap words (e.g. 100 words) select karke Map-Reduce generation run karna.
6. **Step 6 — Result Export**: Timestamped summary text file formatted headers ke saath local storage mein download karna.

---

## 3. Core Architectural Pattern: Map-Reduce (Two-Pass Summarization)

#### Interview Question: *"Aap pooray book ko single LLM prompt mein kyu nahi bhejte? 128K context window wale models to available hain!"*

> **Answer (Interview Masterclass)**:  
> *"Single-pass summarization on large context (100k+ words) mein do major problems aate hain:*  
> *1. **Lost in the Middle Effect**: LLMs long context ke start aur end par jyada focus karte hain, middle chapters ke key plot points bhool jaate hain.*  
> *2. **Superficiality**: Single-pass prompt output space limit (e.g., 2000 tokens) ke karan details compression mein major narrative beats drop kar deta hai.*  
> *Isliye maine **Hierarchical Map-Reduce (Two-Pass) Architecture** design ki."*

---

### Map-Reduce Flow Breakdown:

```
[Full Book Text]
       │
       ▼
[Chunker with Overlap] ➔ Chunk 1 (400w), Chunk 2 (400w), Chunk 3 (400w) ...
       │
  PASS 1 (MAP) ────────► Summarize Chunk 1 ➔ Summary 1
                         Summarize Chunk 2 ➔ Summary 2
                         Summarize Chunk 3 ➔ Summary 3
       │
  PASS 2 (REDUCE) ─────► Concatenate (Summary 1 + 2 + 3 ...) ➔ Final Master Synthesis Prompt
       │
       ▼
[Publication-Ready Format (Instagram / YouTube / Thread / Factual Summary)]
```

1. **Pass 1: Map (Chunk-Level Extraction)**:
   - Text ko ~400-word sentence-boundary chunks mein split karke har chunk se format-specific highlights extract kiye jaate hain.
   - Har chunk processing ke waqt prior running context bhejha jata hai.
2. **Pass 2: Reduce (Final Synthesis)**:
   - Saare chunk-level summaries ko merge karke single Master Synthesis Prompt bhejha jata hai.
   - Final prompt redundancy remove karta hai, story arc build karta hai, aur exact target word count / format structure enforce karta hai.

---

## 4. Sentence-Aware Chunker with Word Overlap (`chunk_text_with_overlap`)

- **Sentence-Boundary Regex**: `re.split(r'(?<=[।॥.!?])\s+', text)`  
  *(Hindi Devanagari danda `।`, double danda `॥`, and English punctuation `.!?'* support karta hai).*
- **Target Chunk Size**: ~400-500 words.
- **Word Overlap (`overlap_prefix`)**: ~80-100 words.
- **Why Word Overlap is Critical**:
  - Boundary sentences par plot events adhoore split ho jaate hain.
  - Previous chunk ke last 80 words ko `overlap_prefix` ki tarah next chunk mein include karne se model boundary plot context loss nahi karta.

---

## 5. The 9 Specialized Output Formats & Temperature Strategy

Interviewer aksar puchta hai: *"Aap different content formats ke liye LLM ke hyper-parameters kaise adjust karte hain?"*

Maine 9 formats ko **Factual vs Creative** categories mein divide karke temperature tune kiya:

| Format Code | Label | Target Words | Temperature | Output Purpose & Style |
|---|---|---|---|---|
| `SHORT` | Short Overview | 150-300 | **0.2** | Factual, concise 2-3 sentence overview |
| `MEDIUM` | Medium Summary | 400-700 | **0.2** | Balanced overview covering key ideas |
| `LONG` | Long Detailed | 800-1500 | **0.2** | Detailed literature analysis preserving subtext |
| `INSTAGRAM` | Instagram Caption | 150-200 | **0.7** | Hook + 2-3 caption paras + 15-20 hashtags + emojis |
| `YOUTUBE_TTS` | YouTube / TTS Script | 500-800 | **0.5** | Narration script with dramatic pauses (`...`), no markdown |
| `TWITTER_THREAD` | Twitter/X Thread | 400-600 | **0.65** | 8-15 numbered tweets (<280 chars each) with viral hook |
| `BLOG_POST` | Blog Article | 800-1200 | **0.5** | SEO article with H1/H2 headings, intro, verdict |
| `NEWSLETTER` | Email Newsletter | 300-500 | **0.6** | Personal email digest with Subject line, P.S. |
| `PODCAST_SCRIPT` | Podcast Script | 600-1000 | **0.55** | Conversational script with host intro, segment transitions |

---

### Key Temperature Trade-Off (Interview Points):
- **Low Temperature (0.2)**: Factual summaries (`SHORT`, `MEDIUM`, `LONG`) mein hallucination strictly avoid karne ke liye use hota hai. Model original text se strictly stick rehta hai.
- **Higher Temperature (0.5 - 0.7)**: Social media and script formats (`INSTAGRAM`, `TWITTER_THREAD`, `PODCAST_SCRIPT`) mein engaging vocabulary, punchy hooks, and creative storytelling flow ke liye higher temperature zaroori hota hai.

---

## 6. Indian English & Audience Adaptation (`INDIAN_ENGLISH_GUIDELINES`)

Target audience Indian readers and social media users hai, isliye prompts mein explicit guidance inject ki gayi hai:
- **Language Tier**: Everyday clear Indian English (avoiding archaic British literary jargon).
- **Natural Expressions**: Expressions like *"mind-blowing"*, *"too good"*, *"literally amazing"*, *"seriously"* natural context mein allow kiye gaye hain.
- **Engagement Prompts**: Hook lines like *"Want to know what happens next?"* or *"Tag that bookworm friend!"* social formats mein add kiye gaye hain.

---

## 7. Output Cleaning & Hallucination Guard (`clean_output`)

LLM outputs mein 4 types of noise aate hain jinhe `clean_output` function clean karta hai:
1. **Reasoning Tags Removal**: Modern reasoning models (e.g. Qwen 3, DeepSeek) output mein `<think>...</think>` chain-of-thought blocks include kar dete hain. Regex `re.sub(r'<think>.*?</think>', '', text)` se unhe strip karta hai.
2. **Code Blocks Removal**: ` ```...``` ` markdown code blocks remove karta hai.
3. **Meta-Prefix Stripping**: Starting headers like `"Summary:"`, `"Analysis:"`, `"Here is the summary:"` ko remove karta hai.
4. **Chinese Character Filtering**: Open-weights models ke occasional foreign character hallucinations ko filter karta hai: `re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf]+', '', text)`.

---

## 8. How to Answer Core Interview Questions (Cheat Sheet)

### Q1: "Full book summarization ke liye Map-Reduce pattern kyu choose kiya?"
> **Answer**:  
> *"Full book (50k-100k words) ko single LLM call mein bhejne se 'Lost in the Middle' effect aata hai aur fine details compression mein lost ho jaati hain. Maine Map-Reduce two-pass system banaya:*  
> *Pass 1 (Map) 400-word overlapping chunks se key points extract karta hai.*  
> *Pass 2 (Reduce) un saare chunk summaries ko combine karke target format (e.g. Twitter Thread ya Blog Post) ke according master synthesis create karta hai."*

### Q2: "Factual Summaries aur Creative Social Media Captions ke temperature aur prompts mein kya difference rakha?"
> **Answer**:  
> *"Factual summaries (Short, Medium, Long) ke liye temperature **0.2** rakha taaki hallucination zero ho aur factual precision maintain rahe. Social formats (Instagram, YouTube TTS, Twitter Thread) ke liye temperature **0.5 - 0.7** set kiya with custom prompts enforcing hooks, emojis, hashtags, and script pacing."*

### Q3: "Summarization pipeline mein data cleanliness aur edge case handling kaise ki?"
> **Answer**:  
> *"Maine `chunk_text_with_overlap` mein Devanagari danda `।` aur English punctuation handling di with 80-word overlap prefix to prevent plot loss at boundary sentences. Furthermore, `clean_output` function regex se reasoning tags `<think>`, markdown fences, meta-prefixes, aur Chinese character hallucinations ko strip kar deta hai."*

---

## Summary Checklist for Interview
- [x] Supported Models: **Gemma 3 27B/12B, Qwen 3 14B, Llama 3.1 8B via Ollama**
- [x] Design Pattern: **Map-Reduce (Two-Pass) Hierarchical Architecture**
- [x] Chunker Strategy: **Sentence-Boundary Aware + 80-word Overlap Prefix**
- [x] Formats Count: **9 Publication-Ready Formats**
- [x] Temperature Tuning: **0.2 (Factual) vs 0.5-0.7 (Creative Social)**
- [x] Post-Processing: **`<think>` tag removal, prefix stripping, Chinese char cleanup**
