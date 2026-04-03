#!/usr/bin/env python3
"""
Transform the copied JARVIS v10.2 notebook into Hindi(Latin)->Hinglish(Latin) v1.0.
Modifies cells in-place, preserving all working code (chunker, dashboard, validation, etc.)
and only changing what's needed for the new task.
"""
import json, re

NB_PATH = '/home/sumit/Documents/GitHub/Train_POC/Translation/JARVIS_Hindi_to_Hinglish_v1.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ============================================================================
# CELL 0 — Header markdown
# ============================================================================
cells[0]['source'] = [
    "# ◈ STARK INDUSTRIES · HINDI→HINGLISH REFORMULATION ARRAY\n",
    "### `v1.0` · Gemma3:27b · Story State Engine v6.0 · 1-Pass · ToneGuard Pro\n",
    "T4 Session Guard                                                                          \n",
    "```\n",
    "  ╔══════════════════════════════════════════════════════════════════════════╗\n",
    "  ║  Hindi(Latin) → Hinglish(Latin) · 1-Pass Reformulation                 ║\n",
    "  ║  Model: gemma3:27b  ·  1-Pass  ·  T4 (15GB)  ·  PROD Edition          ║\n",
    "  ╚══════════════════════════════════════════════════════════════════════════╝\n",
    "```\n",
    "\n",
    "| Step | Cell | Mission |\n",
    "|------|------|---------|\n",
    "| **①** | 2 | Install dependencies |\n",
    "| **②** | 4–5 | Boot Ollama · Pull gemma3:27b |\n",
    "| **③** | 7 | Upload source `.txt` (Hindi in Latin script) |\n",
    "| **④** | 9–10 | Configure parameters |\n",
    "| **⑤** | 12 | Story State Engine v6.0 |\n",
    "| **⑥** | 14–15 | Load Translation Engine v1.0 |\n",
    "| **⑦** | 17 | **⚡ Execute** |\n",
    "| **⑧** | 19 | Download output |\n",
    "\n",
    "> **v1.0 Features**  \n",
    "> • **1-Pass Reformulation** — Hindi(Latin) → natural Hinglish in one pass  \n",
    "> • **ToneGuard Pro** — No foul words, respect system enforced  \n",
    "> • **16 Genre Few-Shots** — Hindi→Hinglish examples across all scene types  \n",
    "> • **Hinglish Voice Guide** — Researched tone: OTT-India register (Made in Heaven, Little Things style)  \n",
    "> • **Anti-Hallucination Layer** — Zero-fabrication, source-anchoring  \n",
    "> • **Foul Word Filter** — PROD-grade: abe, sale, harami, etc. blocked  \n",
    "> • **Story State v6.0** — Rolling 5-chunk context summary  \n",
    "> • **T4 Session Guard** — Real-time session timer, adaptive chunking, auto-checkpoint  \n",
    "> • **Background I/O** — File writes off main thread, no translation blocking  \n",
]

# ============================================================================
# CELL 2 — Dependencies (update branding)
# ============================================================================
cells[2]['source'] = [
    "!pip install -q ollama ipywidgets\n",
    "import torch\n",
    "from IPython.display import display, HTML\n",
    "print(f'✅ Dependencies ready | CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')\n",
    "display(HTML('''\n",
    "<div style=\"background:#070710;border:2px solid #c0392b;border-radius:8px;\n",
    "            padding:12px 18px;font-family:'Courier New',monospace;margin-top:10px;\">\n",
    "  <div style=\"color:#c0392b;font-size:1.1em;font-weight:bold;letter-spacing:2px;\">[ STARK INDUSTRIES — Hindi→Hinglish v1.0 ]</div>\n",
    "  <div style=\"color:#4CAF50;font-size:0.85em;margin-top:4px;\">ollama · ipywidgets · torch ready</div>\n",
    "  <div style=\"color:#FFD700;font-size:0.78em;margin-top:3px;\">ToneGuard Pro · T4 Session Guard · Anti-Hallucination · Foul Word Filter</div>\n",
    "</div>\n",
    "'''))\n",
]

# ============================================================================
# CELL 3 — Boot markdown (update time estimate)
# ============================================================================
cells[3]['source'] = [
    "## 🦙 Step 2a — Boot Ollama Server\n",
    "Run Cell 4, then Cell 5.\n",
    "\n",
    "> ⚠️ T4 = 15 GB VRAM. gemma3:27b Q4_K_M needs ~17 GB — Ollama uses CPU offload for extra layers. Expect ~30–60s/chunk (1-pass). For 40K words (~91 chunks): **1–1.5 hours**.\n",
]

# ============================================================================
# CELL 4 — Ollama boot (update branding)
# ============================================================================
src4 = ''.join(cells[4]['source'])
src4 = src4.replace('gemma3:27b | v10.0 | T4 Session Guard active',
                     'gemma3:27b | v1.0 Hindi→Hinglish | T4 Session Guard active')
cells[4]['source'] = [src4]

# ============================================================================
# CELL 6 — Upload markdown
# ============================================================================
cells[6]['source'] = [
    "## 📤 Step 3 — Upload Source File\n",
    "Upload your `.txt` book file — **Hindi text in Roman/Latin script**.\n",
]

# ============================================================================
# CELL 7 — File upload (adapt for Hindi Latin source)
# ============================================================================
cells[7]['source'] = [
    "from IPython.display import display, HTML\n",
    "from google.colab import files\n",
    "import re\n",
    "\n",
    "display(HTML('<div style=\"background:linear-gradient(135deg,#0a0a0f,#1a0505);border:2px solid #c0392b;'\n",
    "            'border-radius:8px;padding:14px 18px;font-family:Courier New,monospace;margin-bottom:8px;\">'\n",
    "            '<div style=\"color:#c0392b;font-size:1.15em;font-weight:bold;letter-spacing:2px;\">◈ FILE UPLINK</div>'\n",
    "            '<div style=\"color:#FFD700;font-size:0.85em;margin-top:4px;\">Select your .txt file (Hindi in Roman/Latin script).</div></div>'))\n",
    "\n",
    "uploaded = files.upload()\n",
    "UPLOADED_FILE = list(uploaded.keys())[0]\n",
    "\n",
    "with open(UPLOADED_FILE, 'r', encoding='utf-8') as f:\n",
    "    _raw = f.read()\n",
    "\n",
    "# Strip any Devanagari that may have leaked into the source\n",
    "_cleaned = re.sub(r'[\\u0900-\\u097F]+', '', _raw).strip()\n",
    "if _cleaned != _raw:\n",
    "    with open(UPLOADED_FILE, 'w', encoding='utf-8') as f:\n",
    "        f.write(_cleaned)\n",
    "    print('[OK] Devanagari characters stripped from source.')\n",
    "\n",
    "_words = len(_cleaned.split())\n",
    "print(f'\\n✅ File ready: {UPLOADED_FILE}')\n",
    "print(f'   {_words:,} words | {len(_cleaned):,} chars')\n",
    "print(f'   Preview: {_cleaned[:200].strip()!r}...')\n",
]

# ============================================================================
# CELL 8 — Config markdown
# ============================================================================
cells[8]['source'] = [
    "## ⚙️ Step 4 — Configure Parameters\n",
    "\n",
    "| Parameter | Default | Notes |\n",
    "|-----------|---------|-------|\n",
    "| Chunk size | **350 words** | Optimized for T4 — good quality/speed balance |\n",
    "| Overlap | **80 words** | Context continuity between chunks |\n",
    "| num_ctx | **8192** | Model context window |\n",
    "| Session | **270 min** | Colab T4 timeout guard |\n",
]

# ============================================================================
# CELL 9 — Config widgets (remove tier dropdown, 1-pass only)
# ============================================================================
cells[9]['source'] = [
    "import ipywidgets as widgets\n",
    "from IPython.display import display, HTML\n",
    "\n",
    "chunk_slider = widgets.IntSlider(\n",
    "    value=350, min=150, max=550, step=25,\n",
    "    description='Chunk size (words):', style={'description_width':'initial'},\n",
    "    layout=widgets.Layout(width='520px')\n",
    ")\n",
    "overlap_slider = widgets.IntSlider(\n",
    "    value=80, min=0, max=150, step=10,\n",
    "    description='Overlap (words):', style={'description_width':'initial'},\n",
    "    layout=widgets.Layout(width='520px')\n",
    ")\n",
    "num_ctx_slider = widgets.IntSlider(\n",
    "    value=8192, min=4096, max=12288, step=1024,\n",
    "    description='num_ctx (tokens):', style={'description_width':'initial'},\n",
    "    layout=widgets.Layout(width='520px')\n",
    ")\n",
    "session_slider = widgets.IntSlider(\n",
    "    value=270, min=60, max=360, step=10,\n",
    "    description='Session budget (min):', style={'description_width':'initial'},\n",
    "    layout=widgets.Layout(width='520px')\n",
    ")\n",
    "display(HTML('<div style=\"background:#0a0a0f;border:2px solid #c0392b;border-radius:8px;'\n",
    "            'padding:12px 18px;font-family:Courier New,monospace;margin-bottom:10px;\">'\n",
    "            '<div style=\"color:#c0392b;font-weight:bold;letter-spacing:2px;\">◈ MISSION PARAMETERS — Hindi→Hinglish v1.0</div>'\n",
    "            '<div style=\"color:#888;font-size:0.78em;margin-top:4px;\">1-Pass Reformulation | Session Guard | ToneGuard Pro | Anti-Hallucination | Foul Word Filter</div>'\n",
    "            '</div>'))\n",
    "display(chunk_slider, overlap_slider, num_ctx_slider, session_slider)\n",
    "print('\\n💡 Adjust then run Cell 10 to lock config.')\n",
]

# ============================================================================
# CELL 10 — Lock config (remove tier, force 1-pass)
# ============================================================================
cells[10]['source'] = [
    "import os\n",
    "MODEL          = 'gemma3:27b'\n",
    "CHUNK_SIZE     = chunk_slider.value\n",
    "OVERLAP_WORDS  = overlap_slider.value\n",
    "TRANSLATION_TIER = 'ADVANCED'  # Always 1-pass reformulation for Hindi→Hinglish\n",
    "NUM_CTX        = num_ctx_slider.value\n",
    "SESSION_BUDGET = session_slider.value * 60\n",
    "OUTPUT_DIR     = './translation_output'\n",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
    "\n",
    "from IPython.display import display, HTML\n",
    "display(HTML(f'''\n",
    "<div style=\"background:#0a0f0a;border:2px solid #4CAF50;border-radius:8px;\n",
    "            padding:12px 18px;font-family:Courier New,monospace;\">\n",
    "  <div style=\"color:#4CAF50;font-weight:bold;\">[CONFIG LOCKED]</div>\n",
    "  <div style=\"color:#e8e8e8;font-size:0.88em;margin-top:6px;\">\n",
    "    Model: {MODEL} | Chunk: {CHUNK_SIZE}w | Overlap: {OVERLAP_WORDS}w<br>\n",
    "    num_ctx: {NUM_CTX} | Session: {SESSION_BUDGET//60} min | Mode: 1-Pass Hindi→Hinglish Reformulation<br>\n",
    "    Output: {OUTPUT_DIR}\n",
    "  </div>\n",
    "</div>\n",
    "'''))\n",
]

# ============================================================================
# CELL 11 — Step 5 markdown
# ============================================================================
cells[11]['source'] = [
    "## 🧠 Step 5 — Story State Engine v6.0\n",
    "\n",
    "**v6.0 Features:**\n",
    "- ✅ Rolling 5-chunk event summary\n",
    "- ✅ Character tracking with address/speech-style\n",
    "- ✅ Genre detection adapted for Hindi(Latin) keywords\n",
    "- ✅ Established vocab (Hindi-formal → Hinglish-casual mappings)\n",
    "- ✅ Setting extraction\n",
    "- ✅ Resume from last checkpoint\n",
    "- ✅ Context compression (~1200 chars cap)\n",
]

print("Cells 0-11 transformed successfully.")

# Save intermediate
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Saved to {NB_PATH}")
