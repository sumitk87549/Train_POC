import json
import re

nb_path = "HindiTranslator_fixed.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Change 1: Replace SYSTEM_PROMPT
new_system_prompt = '''SYSTEM_PROMPT = """You are an expert literary translator. Translate the given English text into Hindi written in Latin/Roman script (romanized Hindi).

═══════════════════════════════════════════════
CORE PHILOSOPHY — MEANING OVER WORDS
═══════════════════════════════════════════════
Translate the MEANING and FEELING of the original — not word by word.
Ask yourself: "How would a Hindi speaker naturally say this?"
A fluent Hindi speaker never says "Main ahsankrit rahunga agar main nahin tha" —
they say "Agar main shukriya na ada karta, toh yeh meri buri baat hoti."

═══════════════════════════════════════════════
DIALOGUE REGISTER — VERY IMPORTANT
═══════════════════════════════════════════════
Match closeness of the relationship in pronouns:
- Husband/wife, close friends, siblings → use "tum" / "tumhara" / "tumhe"
- Strangers, elders, formal settings     → use "aap" / "aapka" / "aapko"
- Superior addressing inferior (master→servant, boss→junior) → use "tu" / "tera"

═══════════════════════════════════════════════
VOCABULARY RULES
═══════════════════════════════════════════════
- Use simple, everyday spoken Hindi — like how urban Indians talk in real life
- FORBIDDEN words (too archaic/Sanskrit): pratham, vimarsh, sanchalan, anugraha,
  atyadhikta, suhint, ahsankrit, prakar, sambandh (use "baare mein"), kshetra (use "ilaqa")
- FORBIDDEN: Invented words. If you don't know the Hindi equivalent, use the English word as-is.
- Common English words to KEEP as-is: phone, car, office, station, platform, platform,
  game-keeper, farm, pool, ticket, cab, telegram, train
- Quantities: "quarter mile" → "paav mile" (NOT "charter mile" — never invent words)
- Regions/districts: "country district" → "gramin ilaqa" (NOT "gaon" which means village)
- Farm/estate: use "farm" or "khet-baadi" (NOT just "khet" which means a small field)
- "Singularity" in detective context → "asamaanya baat" or "koi ajeeb cheez"
- "Clue" → "suraag" (NEVER "suhint" — that is not a Hindi word)

═══════════════════════════════════════════════
PARAGRAPH AND NARRATIVE RULES
═══════════════════════════════════════════════
- PRESERVE paragraph breaks exactly as in the original
- Long narrative paragraphs must stay as one flowing paragraph — do NOT split them into many tiny sentences
- Holmes/narrator exposition paragraphs should flow like a connected story, not a bullet list
- Dialogue lines: each speaker line stays on its own line, exactly as in original

═══════════════════════════════════════════════
ANTI-HALLUCINATION RULES — CRITICAL
═══════════════════════════════════════════════
- NEVER repeat a phrase or sentence you already wrote in this chunk
- If you find yourself writing the same words twice, STOP and move to the next sentence
- Do NOT add any content that is not in the original English text
- Do NOT add commentary, footnotes, translator notes, or explanations
- If a sentence is unclear, translate your best understanding — do NOT skip it

═══════════════════════════════════════════════
SCRIPT RULE
═══════════════════════════════════════════════
- Output ONLY Latin/Roman script (a-z, A-Z). ZERO Devanagari characters.
- Good: "Woh ghar gaya"   Bad: "वह घर गया"

═══════════════════════════════════════════════
FEW-SHOT EXAMPLES — STUDY THESE CAREFULLY
═══════════════════════════════════════════════

--- EXAMPLE 1: Dialogue Register (husband/wife = tum) ---
English:
  "What do you say, dear?" said my wife, looking across at me. "Will you go?"
  "I really don't know what to say. I have a fairly long list at present."

Hindi (Latin):
  "Tum kya sochte ho, jaan?" meri patni ne meri taraf dekh kar kaha. "Kya tum jaoge?"
  "Sachmuch mujhe samajh nahi aa raha. Mere paas abhi kaafi kaam hai."

--- EXAMPLE 2: Meaning-based, not word-for-word ---
English:
  "I should be ungrateful if I were not, seeing what I gained through one of them."

Hindi (Latin):
  "Agar main shukriya na ada karta toh yeh meri burai hoti — un cases mein se ek ne mujhe itna kuch diya hai."

--- EXAMPLE 3: Singularity / clue vocabulary ---
English:
  "Singularity is almost invariably a clue. The more featureless and commonplace
  a crime is, the more difficult it is to bring it home."

Hindi (Latin):
  "Koi bhi asamaanya ya ajeeb cheez hamesha ek suraag hoti hai. Jitna sadharan aur aam ek jurm ho, utna hi usse sabit karna mushkil hota hai."

--- EXAMPLE 4: Preserving narrative paragraph flow ---
English:
  They appear to have avoided the society of the neighbouring English families
  and to have led retired lives, though both the McCarthys were fond of sport
  and were frequently seen at the race-meetings of the neighbourhood.

Hindi (Latin):
  Lagta hai ki unhone aas-paas ke Angrezi parivaron se door rehne ki koshish ki aur ek seedha-saadha zindagi jeete the — lekin dono McCarthys khel-kood ke shaukeen the aur aas-paas ki race-meetings mein unhe aksar dekha jaata tha.

--- EXAMPLE 5: Correct geography/measurement words ---
English:
  "From Hatherley Farmhouse to the Boscombe Pool is a quarter of a mile"
  "Boscombe Valley is a country district not very far from Ross"

Hindi (Latin):
  "Hatherley Farmhouse se Boscombe Pool tak paav mile ki doori hai"
  "Boscombe Valley, Ross se zyada door nahi ek gramin ilaqa hai"

--- EXAMPLE 6: Formal Holmes speech stays formal with aap ---
English:
  "It is really very good of you to come, Watson," said he.
  "Local aid is always either worthless or else biassed."

Hindi (Latin):
  "Watson, aapka aana bahut achha laga," usne kaha.
  "Yahan ki sthaniya madad hamesha ya to bekar hoti hai, ya ek-tarafaa."

--- EXAMPLE 7: Servant/master register uses tu/tera ---
English:
  He had told the man that he must hurry, as he had an appointment.

Hindi (Latin):
  Usne naukar se kaha tha ki jaldi kar, kyunki use ek zaroori kaam pe jaana hai.

═══════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════
Output ONLY the translated text. No labels like "Translation:" or "Hindi:".
No notes. No explanations. Just the translation, preserving all paragraph breaks."""'''

# Change 2
new_repetition_score = '''def repetition_score(text):
    """
    Detects if the model has gone into a repetition loop.
    Returns a score 0.0-1.0 where >0.4 means dangerous repetition.
    Strategy: split into sentences, check what fraction are near-duplicates.
    """
    if not text or len(text) < 80:
        return 0.0
    # Split on sentence-ending punctuation
    sentences = [s.strip() for s in re.split(r'[।.!?]\s+', text) if len(s.strip()) > 15]
    if len(sentences) < 3:
        return 0.0

    # Build bigram sets per sentence for fuzzy comparison
    def bigrams(s):
        words = s.lower().split()
        return set(zip(words, words[1:])) if len(words) > 1 else set()

    seen = []
    repeat_count = 0
    for sent in sentences:
        bg = bigrams(sent)
        if not bg:
            continue
        for prev_bg in seen:
            if prev_bg and bg:
                overlap = len(bg & prev_bg) / max(len(bg | prev_bg), 1)
                if overlap > 0.55:   # 55% bigram overlap = near-duplicate sentence
                    repeat_count += 1
                    break
        seen.append(bg)

    return repeat_count / max(len(sentences), 1)

'''

# Change 3
new_repetition_check = '''        # -- Check for repetition loop (model stuck repeating phrases) -----------
        rep_score = repetition_score(raw)
        if rep_score > 0.40:
            return None, t_end - t_start, f'REPETITION_LOOP: {rep_score:.0%} of sentences are near-duplicates'

'''

# Change 4
new_options = '''            options={
                'temperature'   : TEMPERATURE,
                'num_ctx'       : NUM_CTX,
                'num_predict'   : 2048,
                'top_k'         : 20,          # was 30 — tighter vocab = fewer invented words
                'top_p'         : 0.85,        # was 0.90 — more focused
                'repeat_penalty': 1.35,        # was 1.15 — much stronger anti-loop
                'repeat_last_n' : 128,         # look back 128 tokens for repeats
            }'''

# Change 5:
new_sliders = """chunk_slider = widgets.IntSlider(
    value=250, min=100, max=500, step=50,        # default 250 (was 400) — smaller = fewer loops
    description='Chunk size (words):',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='500px')
)
temp_slider = widgets.FloatSlider(
    value=0.15, min=0.05, max=0.6, step=0.05,   # default 0.15 (was 0.3) — more deterministic
    description='Temperature:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='500px')
)
ctx_slider = widgets.IntSlider(
    value=6144, min=4096, max=12288, step=1024,  # default 6144 (was 8192) — matches smaller chunks
    description='num_ctx (tokens):',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='500px')
)"""

# Change 6: 
new_preamble_stripper = """        # -- Remove preamble labels (model sometimes prefixes output) -------------
        raw = re.sub(
            r'^(?:Translation|Hindi(?: \\(Latin(?: script)?\\))?|Romanized Hindi|'
            r'Output|Roman[- ]alphabet Hindi translation|'
            r'Hindi \\(Latin\\) translation|Translated text)'
            r'\\s*:?\\s*',
            '', raw, flags=re.IGNORECASE
        ).strip()

        # Also strip if model echoed the source English back at the top
        # (sometimes model outputs "English: ... \\n Hindi: ..." format)
        if re.search(r'^English\\s*:', raw, re.IGNORECASE):
            parts = re.split(r'Hindi\\s*(?:\\(Latin[^)]*\\))?\\s*:', raw, flags=re.IGNORECASE)
            if len(parts) > 1:
                raw = parts[-1].strip()"""

# Change 7: 
new_style_guide = """### ✅ Good Examples — Meaning-Based Translation
| English | Hindi (Latin) | Why |
|---------|---------------|-----|
| What do you say, dear? (wife→husband) | Tum kya sochte ho, jaan? | tum = intimate |
| It is good of you to come, Watson | Watson, aapka aana bahut achha laga | aap = formal |
| Singularity is a clue | Koi ajeeb cheez hamesha ek suraag hoti hai | meaning, not literal |
| A quarter of a mile | Paav mile | paav = quarter |
| Country district | Gramin ilaqa | not "gaon" (village) |
| Farm / Farmhouse | Farm / Farmhouse | keep English, don't over-translate |
| He led a retired life | Woh ek seedha-saadha zindagi jeeta tha | natural Hindi idiom |
| Local aid is biassed | Yahan ki madad hamesha ek-tarafaa hoti hai | ek-tarafaa = biased |

### ❌ What we avoid
- Devanagari script (हिंदी) — output is Latin only  
- Archaic words: *pratham, vimarsh, atyadhikta, suhint, ahsankrit*
- Invented words: **never** write a word you're not sure exists in Hindi
- Word-for-word literal translation that sounds robotic
- Repetition of phrases — if you wrote it once, never write it again
- Splitting one flowing paragraph into many choppy sentences
- Using "aap" for husband/wife or close friends (use "tum")
- Using "gaon" for a rural area/district (use "gramin ilaqa" or "dehaati علاقہ")

---
### ⚙️ Model Notes"""

for i, cell in enumerate(nb["cells"]):
    if not cell.get("source"): continue
    
    src = "".join(cell["source"])
    original_src = src
    
    if "SYSTEM_PROMPT =" in src: 
        # Change 1
        src = re.sub(r'SYSTEM_PROMPT = """.*?"""', new_system_prompt, src, flags=re.DOTALL)
        
        # Change 2
        src = src.replace("# -- Prompts", new_repetition_score + "# -- Prompts")
        
        # Change 3
        # find the alpha_density check block and put new check after it
        alpha_block = "GARBAGE: alpha density only {a_density:.0%}'"
        if alpha_block in src:
            src = repr(src)
            src = src.replace("GARBAGE: alpha density only {a_density:.0%}'\\n\\n", 
                              "GARBAGE: alpha density only {a_density:.0%}'\\n\\n" + new_repetition_check.replace('\\n', '\\\\n').replace('\\', '\\\\') ) # wait using ast or just string replace nicely
            src = eval(src)
            # Actually easier to just replace string naturally without repr
            src = src.replace("GARBAGE: alpha density only {a_density:.0%}'\n\n", 
                              "GARBAGE: alpha density only {a_density:.0%}'\n\n" + new_repetition_check)
        
        # Change 4
        src = re.sub(r'options=\{.*?\}', new_options, src, flags=re.DOTALL)
        
        # Change 6
        pre_strip_start = src.find("# -- Remove preamble labels")
        pre_strip_end = src.find("if len(raw) < 20:", pre_strip_start)
        if pre_strip_start != -1 and pre_strip_end != -1:
            src = src[:pre_strip_start] + new_preamble_stripper + "\n\n        " + src[pre_strip_end:]
            
    # Change 5
    if "chunk_slider =" in src:
        src = re.sub(r'chunk_slider = widgets\.IntSlider\(.*?display\(chunk_slider, temp_slider, ctx_slider\)', 
                     new_sliders + "\n\ndisplay(chunk_slider, temp_slider, ctx_slider)", 
                     src, flags=re.DOTALL)
                     
    # Change 7
    if "### ✅ Good Examples" in src and cell.get("cell_type") == "markdown":
        src = re.sub(r'### ✅ Good Examples.*?---\n### ⚙️ Model Notes', new_style_guide, src, flags=re.DOTALL)

    # Re-assign back to cell format (split by lines keeping newlines)
    # Jupyter notebooks typically store strings line by line, but a single string is also valid. 
    # For safety, let's split it nicely as it was:
    
    if src != original_src:
        lines = [line + '\\n' for line in src.split('\\n')]
        if len(lines) > 0:
            lines[-1] = lines[-1][:-2] # strip last \n
        cell["source"] = src.splitlines(True)


with open("HindiTranslator_fixed_updated.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print("Changes applied. Output saved to HindiTranslator_fixed_updated.ipynb")
