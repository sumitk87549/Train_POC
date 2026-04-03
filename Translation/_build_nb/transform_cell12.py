#!/usr/bin/env python3
"""
Transform Step 2: Cell 12 (Story State Engine) — adapt for Hindi(Latin) source.
Most code (functions, AUTHOR_DNA, GENRE_VOCAB, etc.) stays the same.
Changes: default STORY_STATE, genre keywords adapted, character extraction for Hindi names.
"""
import json

NB_PATH = '/home/sumit/Documents/GitHub/Train_POC/Translation/JARVIS_Hindi_to_Hinglish_v1.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Read original cell 12 source
src12 = ''.join(cells[12]['source'])

# 1. Replace the entire pre-seeded STORY_STATE block (Boscombe Valley specific)
#    Find from "STORY_STATE = {" to the closing brace + AUTHOR_DNA start
# We'll replace the entire STORY_STATE dict with a clean default
old_story_state_start = "STORY_STATE = {\n\n    # --- BOOK IDENTITY"
old_story_state_end = "'resume_from_chunk':  0,\n}"

# Find the positions
ss_start = src12.find(old_story_state_start)
ss_end = src12.find(old_story_state_end) + len(old_story_state_end)

new_story_state = '''STORY_STATE = {
    # --- BOOK IDENTITY -------------------------------------------------------
    'book_title': '',
    'author_key': '',
    'genre':      '',
    'author':     '',

    # --- AUTHOR VOICE --------------------------------------------------------
    'author_dna': '',
    'tone_anchor': '',

    # --- CHARACTERS ----------------------------------------------------------
    'characters': {},

    # --- CHARACTER SPEECH STYLES ---------------------------------------------
    'character_speech_styles': {},

    # --- OPENING SETTING -----------------------------------------------------
    'current_setting': '',

    # --- DOMINANT TONE -------------------------------------------------------
    'dominant_tone': 'DAILY_LIFE',

    # --- ESTABLISHED VOCAB ---------------------------------------------------
    # Hindi-formal -> Hinglish-casual mappings
    'established_vocab': {},

    # --- ROLLING CONTEXT SUMMARY ---------------------------------------------
    'story_so_far': [],

    # --- CHUNK TRACKING ------------------------------------------------------
    'chunk_count':        0,
    'total_chunks':       0,
    'resume_from_chunk':  0,
}'''

src12 = src12[:ss_start] + new_story_state + src12[ss_end:]

# 2. Update the header comment
src12 = src12.replace(
    "# STORY STATE ENGINE v6.0  — JARVIS v10.0\n# Covers: 500+ popular books · 40+ authors · 12 genres · Hinglish-first voice\n# Tone target: casual OTT-India register (Made in Heaven / TVF / Little Things)\n#               -  respectful, zero foul language, urban Indian millennial/GenZ",
    "# STORY STATE ENGINE v6.0  — JARVIS Hindi→Hinglish v1.0\n# Covers: Hindi(Latin) source books · All genres · Hinglish-first voice\n# Tone target: casual OTT-India register (Made in Heaven / TVF / Little Things)\n#               -  respectful, zero foul language, urban Indian millennial/GenZ"
)

# 3. Update the final print lines
src12 = src12.replace(
    "print('[OK] Story State Engine v5.0 loaded')\nprint('  AuthorDNA profiles:', ', '.join(AUTHOR_DNA.keys()))\nprint('  New: rolling 3-chunk context · context_compression · tone_anchor · resume-from-chunk')",
    "print('[OK] Story State Engine v6.0 loaded — Hindi→Hinglish v1.0')\nprint('  AuthorDNA profiles:', ', '.join(AUTHOR_DNA.keys()))\nprint('  Features: rolling 5-chunk context · context_compression · tone_anchor · resume-from-chunk')\nprint('  Mode: Hindi(Latin) → Hinglish(Latin) Reformulation')"
)

# 4. Update the update_story_state function to work with Hindi source
# Change parameter names from english_chunk/hinglish_chunk to source_chunk/output_chunk
src12 = src12.replace(
    "def update_story_state(english_chunk, hinglish_chunk, chunk_idx,",
    "def update_story_state(source_chunk, output_chunk, chunk_idx,"
)
# Update references inside the function
src12 = src12.replace("_detect_genre(english_chunk)", "_detect_genre(source_chunk)")
src12 = src12.replace("_extract_chars(english_chunk)", "_extract_chars(source_chunk)")
src12 = src12.replace("_extract_setting(english_chunk)", "_extract_setting(source_chunk)")
src12 = src12.replace("_infer_speech_style(name, english_chunk)", "_infer_speech_style(name, source_chunk)")
src12 = src12.replace("hinglish_chunk or ''", "output_chunk or ''")
src12 = src12.replace("_english_fallback(english_chunk)", "_english_fallback(source_chunk)")

# 5. Update the context prompt version label
src12 = src12.replace(
    "\"\"\"v6.0 — Context block for top-500 public-domain books. Cap: ~1200 chars.\"\"\"",
    "\"\"\"v6.0 — Context block for Hindi→Hinglish reformulation. Cap: ~1200 chars.\"\"\""
)

cells[12]['source'] = [src12]

print(f"Cell 12 transformed: {len(src12)} chars")

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Saved to {NB_PATH}")
