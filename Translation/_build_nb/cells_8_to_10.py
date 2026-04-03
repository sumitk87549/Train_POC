# Cells 8-10: Config parameters

CELLS_8_10 = []

# Cell 8 - Markdown
CELLS_8_10.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## ⚙️ Step 4 — Configure Parameters\n",
        "\n",
        "| Parameter | Default | Notes |\n",
        "|-----------|---------|-------|\n",
        "| Chunk size | **350 words** | Optimized for T4 — good quality/speed balance |\n",
        "| Overlap | **80 words** | Context continuity between chunks |\n",
        "| num_ctx | **8192** | Model context window |\n",
        "| Session | **270 min** | Colab T4 timeout guard |\n"
    ]
})

# Cell 9 - Code: Parameter widgets
CELLS_8_10.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
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
        "            '<div style=\"color:#888;font-size:0.78em;margin-top:4px;\">1-Pass Reformulation | Session Guard | ToneGuard Pro | Anti-Hallucination</div>'\n",
        "            '</div>'))\n",
        "display(chunk_slider, overlap_slider, num_ctx_slider, session_slider)\n",
        "print('\\n💡 Adjust then run Cell 10 to lock config.')\n"
    ],
    "outputs": [],
    "execution_count": None
})

# Cell 10 - Code: Lock config
CELLS_8_10.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "import os\n",
        "MODEL          = 'gemma3:27b'\n",
        "CHUNK_SIZE     = chunk_slider.value\n",
        "OVERLAP_WORDS  = overlap_slider.value\n",
        "NUM_CTX        = num_ctx_slider.value\n",
        "SESSION_BUDGET = session_slider.value * 60  # convert to seconds\n",
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
        "    num_ctx: {NUM_CTX} | Session: {SESSION_BUDGET//60} min | Mode: 1-Pass Reformulation<br>\n",
        "    Output: {OUTPUT_DIR}\n",
        "  </div>\n",
        "</div>\n",
        "'''))\n"
    ],
    "outputs": [],
    "execution_count": None
})
