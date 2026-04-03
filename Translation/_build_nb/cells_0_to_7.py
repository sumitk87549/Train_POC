# Cells 0-7: Header, Dependencies, Ollama Boot, File Upload

CELLS = []

# Cell 0 - Markdown header
CELLS.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
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
        "> • **Anti-Hallucination Layer** — Zero-fabrication, length ratio, loop detection  \n",
        "> • **Foul Word Filter** — PROD-grade commercial output  \n",
        "> • **Story State v6.0** — Rolling context, character tracking, vocab  \n",
        "> • **T4 Session Guard** — Auto-checkpoint, adaptive chunking  \n",
        "> • **Background I/O** — File writes off main thread  \n"
    ]
})

# Cell 1 - Markdown
CELLS.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## ⚡ Step 1 — Install Dependencies\n", "Run once per Colab session."]
})

# Cell 2 - Code: Install deps
CELLS.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "!pip install -q ollama ipywidgets\n",
        "import torch\n",
        "from IPython.display import display, HTML\n",
        "print(f'✅ Dependencies ready | CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')\n",
        "display(HTML('''\n",
        "<div style=\"background:#070710;border:2px solid #c0392b;border-radius:8px;\n",
        "            padding:12px 18px;font-family:'Courier New',monospace;margin-top:10px;\">\n",
        "  <div style=\"color:#c0392b;font-size:1.1em;font-weight:bold;letter-spacing:2px;\">[ STARK INDUSTRIES — JARVIS Hindi→Hinglish v1.0 ]</div>\n",
        "  <div style=\"color:#4CAF50;font-size:0.85em;margin-top:4px;\">ollama · ipywidgets · torch ready</div>\n",
        "  <div style=\"color:#FFD700;font-size:0.78em;margin-top:3px;\">ToneGuard Pro · T4 Session Guard · Anti-Hallucination · Foul Word Filter</div>\n",
        "</div>\n",
        "'''))\n"
    ],
    "outputs": [],
    "execution_count": None
})

# Cell 3 - Markdown
CELLS.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🦙 Step 2a — Boot Ollama Server\n",
        "Run Cell 4, then Cell 5.\n",
        "\n",
        "> ⚠️ T4 = 15 GB VRAM. gemma3:27b Q4_K_M needs ~17 GB — Ollama uses CPU offload for extra layers. Expect ~30–60s/chunk (1-pass). For 40K words (~91 chunks): **1–1.5 hours**.\n"
    ]
})

# Cell 4 - Code: Ollama boot
CELLS.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "import subprocess, time, os\n",
        "from IPython.display import display, HTML\n",
        "\n",
        "display(HTML('<div style=\"background:linear-gradient(135deg,#0a0a0f,#1a0505);border:2px solid #c0392b;'\n",
        "            'border-radius:8px;padding:14px 18px;font-family:Courier New,monospace;\">'\n",
        "            '<div style=\"color:#c0392b;font-size:1.2em;font-weight:bold;letter-spacing:3px;\">◈ OLLAMA BOOT SEQUENCE</div>'\n",
        "            '<div style=\"color:#FFD700;font-size:0.82em;margin-top:4px;\">gemma3:27b | v1.0 | T4 Session Guard active</div></div>'))\n",
        "\n",
        "!apt-get update -qq && apt-get install -y -qq zstd > /dev/null 2>&1\n",
        "!curl -fsSL https://ollama.com/install.sh | sh\n",
        "\n",
        "print('\\n🚀 Starting Ollama server...')\n",
        "os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'\n",
        "os.environ['OLLAMA_GPU_OVERHEAD'] = '512000000'\n",
        "subprocess.Popen(['/usr/local/bin/ollama', 'serve'],\n",
        "                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
        "time.sleep(8)\n",
        "\n",
        "try:\n",
        "    import ollama; ollama.list()\n",
        "    display(HTML('<div style=\"background:#0a0f0a;border:2px solid #4CAF50;border-radius:6px;'\n",
        "                'padding:10px 18px;font-family:Courier New,monospace;margin-top:8px;\">'\n",
        "                '<span style=\"color:#4CAF50;font-weight:bold;\">[OK] ARC REACTOR STABLE — Ollama server operational.</span><br>'\n",
        "                '<span style=\"color:#888;font-size:0.82em;\">Run Cell 5 to pull gemma3:27b (~17 GB — takes 10-25 min).</span></div>'))\n",
        "except Exception as e:\n",
        "    print(f'⚠️ Server may still be starting: {e} — wait 5s and re-run.')\n"
    ],
    "outputs": [],
    "execution_count": None
})

# Cell 5 - Code: Pull model
CELLS.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "import ollama\n",
        "\n",
        "MODEL_NAME = 'gemma3:27b'\n",
        "print(f'📥 Pulling {MODEL_NAME} (Q4_K_M ~17 GB) — go make chai ☕')\n",
        "\n",
        "try:\n",
        "    current_digest = ''\n",
        "    for progress in ollama.pull(MODEL_NAME, stream=True):\n",
        "        digest = progress.get('digest', '')\n",
        "        if digest != current_digest and current_digest: print()\n",
        "        current_digest = digest\n",
        "        status = progress.get('status', '')\n",
        "        if 'completed' in progress and 'total' in progress:\n",
        "            pct = (progress['completed'] / progress['total'] * 100) if progress['total'] else 0\n",
        "            bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))\n",
        "            print(f'\\r   [{bar}] {pct:.1f}%', end='', flush=True)\n",
        "        else:\n",
        "            print(f'\\r   {status}', end='', flush=True)\n",
        "    print(f'\\n\\n✅ {MODEL_NAME} ready!')\n",
        "    for m in ollama.list().get('models', []):\n",
        "        print(f\"   • {m.get('name')} ({m.get('size',0)/(1024**3):.2f} GB)\")\n",
        "except Exception as e:\n",
        "    print(f'\\n❌ Pull failed: {e}\\n   Make sure Ollama server is running (Cell 4).')\n"
    ],
    "outputs": [],
    "execution_count": None
})

# Cell 6 - Markdown
CELLS.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📤 Step 3 — Upload Source File\n",
        "Upload your `.txt` book file — **Hindi text in Roman/Latin script**.\n"
    ]
})

# Cell 7 - Code: File upload
CELLS.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
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
        "print(f'   Preview: {_cleaned[:200].strip()!r}...')\n"
    ],
    "outputs": [],
    "execution_count": None
})
