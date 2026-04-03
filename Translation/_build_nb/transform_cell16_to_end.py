#!/usr/bin/env python3
"""
Transform Step 4: Cells 16-22 — Book Profile, Run execution, Download, Appendix
"""
import json

NB_PATH = '/home/sumit/Documents/GitHub/Train_POC/Translation/JARVIS_Hindi_to_Hinglish_v1.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ============================================================================
# CELL 16 — Markdown
# ============================================================================
cells[16]['source'] = [
    "## 📚 Step 7a — Book Profile\n",
    "Set up the identity for your Hindi(Latin) book. This helps the engine maintain consistency.\n"
]

# ============================================================================
# CELL 17 — Book Profile and Execution Code
# ============================================================================
cells[17]['source'] = [
    "BOOK_PROFILE = {\n",
    "    'title': 'Sample Hindi Book (Romanized)',\n",
    "    'genre': 'literary fiction', # Option: thriller, romance, coming-of-age, sci-fi\n",
    "\n",
    "    # Characters you want the engine to remember (optional)\n",
    "    'characters': {\n",
    "        'Rajesh':  {'role': 'Protagonist', 'address': 'tum'},\n",
    "        'Pitaji':  {'role': 'Father', 'address': 'aap'},\n",
    "    },\n",
    "\n",
    "    # Vocab: Force formal Hindi words to become casual Hinglish\n",
    "    'extra_vocab': {\n",
    "        'aavashyakta': 'zaroorat',\n",
    "        'parivartan': 'badlaav',\n",
    "        'sahayata': 'madad',\n",
    "        'prabhat': 'subah'\n",
    "    },\n",
    "\n",
    "    # Tone Rules for the whole book\n",
    "    'tone_rules': {\n",
    "        'default': 'Modern Indian OTT register (Made in Heaven style)',\n",
    "        'avoid': ['tu', 'tujhe', 'abe', 'sale', 'saala', 'gali', 'shuddh hindi'],\n",
    "        'extra': 'Keep sentences smooth and conversational. Respect elders with aap.'\n",
    "    }\n",
    "}\n",
    "\n",
    "# 1. Reset state for the new book\n",
    "reset_for_new_book(title=BOOK_PROFILE['title'], genre=BOOK_PROFILE['genre'])\n",
    "\n",
    "# 2. Load characters\n",
    "for name, info in BOOK_PROFILE.get('characters', {}).items():\n",
    "    update_character(name, role=info.get('role'), address=info.get('address'))\n",
    "\n",
    "# 3. Set vocab overrides\n",
    "for eng, hin in BOOK_PROFILE.get('extra_vocab', {}).items():\n",
    "    add_vocab_entry(eng, hin)\n",
    "\n",
    "print('\\n')\n",
    "print_state_summary()\n",
    "print('\\n🚀 LAUNCHING REFORMULATION PIPELINE...')\n",
    "\n",
    "# 4. RUN!\n",
    "generator = OllamaTranslationGenerator(\n",
    "    model_name     = MODEL,\n",
    "    target_lang    = 'Hinglish_Latin',\n",
    "    output_dir     = OUTPUT_DIR,\n",
    "    tier           = TRANSLATION_TIER,\n",
    "    chunk_size     = CHUNK_SIZE,\n",
    "    overlap_words  = OVERLAP_WORDS,\n",
    "    num_ctx        = NUM_CTX,\n",
    "    session_budget = SESSION_BUDGET,\n",
    "    tone_rules     = BOOK_PROFILE.get('tone_rules')\n",
    ")\n",
    "\n",
    "# Optional: Set `resume_from_chunk=X` if Colab crashed and you need to restart from chunk 45\n",
    "out_txt, state_json = generator.translate_file(UPLOADED_FILE, resume_from_chunk=0)\n"
]

# ============================================================================
# CELL 18-20 — Download section
# ============================================================================
cells[18]['source'] = [
    "## 💾 Step 8 — Download Output\n",
    "\n",
    "The output is saved in `./translation_output/`.\n"
]

cells[20]['source'] = [
    "## (Optional) Backup to Google Drive\n",
    "Run this to save the final Output File directly to your Drive to avoid losing it when Colab disconnects.\n"
]

# ============================================================================
# CELL 22 — Appendix (Hinglish Voice Reference)
# ============================================================================
cell22_src = ''.join(cells[22]['source'])
cell22_src = cell22_src.replace(
    "# 📚 Appendix: The JARVIS Hinglish Voice\n\n*(For reference only — already built into the engine's prompts)*",
    "# 📚 Appendix: Hindi→Hinglish Voice Reference\n\n*(For reference only — already built into the engine's prompts)*"
)
cell22_src = cell22_src.replace(
    "How to translate English classics to Hinglish without ruining them",
    "How to reformulate Hindi(Latin) to Hinglish without losing the depth"
)
cells[22]['source'] = [cell22_src]

# Save notebook
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Cells 16-22 transformed successfully.")
print(f"Saved to {NB_PATH}")
