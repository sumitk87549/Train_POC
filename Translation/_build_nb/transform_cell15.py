#!/usr/bin/env python3
"""
Transform Step 3: Cell 13 (markdown) + Cell 14 (validation, keep as-is) + Cell 15 (Translation Engine)
This is the core change — replace English→Hinglish 2-pass prompts with Hindi(Latin)→Hinglish 1-pass.
"""
import json, re

NB_PATH = '/home/sumit/Documents/GitHub/Train_POC/Translation/JARVIS_Hindi_to_Hinglish_v1.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ============================================================================
# CELL 13 — Markdown
# ============================================================================
cells[13]['source'] = [
    "## 🔩 Step 6 — Load Translation Engine v1.0\n",
    "\n",
    "**1-Pass Reformulation Architecture:**\n",
    "1. **Single Pass** — Hindi(Latin) → natural Hinglish with English code-switching (temp=0.55)\n",
    "\n",
    "The source is already in Hindi (Latin script). This is a **register shift** — formal/literary Hindi → modern conversational Hinglish. One well-prompted pass is sufficient.\n",
    "\n",
    "**Quality layers:**\n",
    "- ToneGuard Pro — no foul words, respect system enforced\n",
    "- Anti-Hallucination — zero-fabrication, source-anchoring\n",
    "- Loop detection — streaming LiveLoopMonitor + post-hoc salvage\n",
    "- Foul word filter — PROD-grade commercial output\n",
    "- QA pipeline — Devanagari check, Hindi-heavy check, English-heavy check\n",
]

# ============================================================================
# CELL 14 — Validation layer: keep the existing 3-layer architecture as-is
# The _validate_output and _decide_action functions work generically.
# No changes needed.
# ============================================================================
# (cell 14 stays unchanged)

# ============================================================================
# CELL 15 — THE MAIN ENGINE — replace prompts, few-shots, and adapt architecture
# ============================================================================
src15 = ''.join(cells[15]['source'])

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT 1: Replace _FEW_SHOTS with Hindi(Latin)→Hinglish few-shots
# ─────────────────────────────────────────────────────────────────────────────
NEW_FEW_SHOTS = r'''_FEW_SHOTS = """
=== GENRE FEW-SHOTS — EXACT VOICE FOR EACH SCENE TYPE ===
(Yeh examples model ke liye hain — output mein mat daalo)
(Har example mein: HINDI(LATIN) SOURCE → RIGHT Hinglish → WRONG → WHY WRONG)

── EXAMPLE 1: LITERARY / ABSURD ──
SCENE TYPE: DAILY_LIFE + HORROR
HINDI SOURCE:
"Ek subah jab Gregor Samsa ki neend khuli toh usne paaya ki woh ek bhayaanak keede mein parivartan ho chuka hai. Woh apni sakht peeth par leta hua tha."
RIGHT HINGLISH:
Ek subah Gregor Samsa ki neend khuli — aur usne realize kiya ki woh ek
bade se keede mein badal chuka hai. Peeth sakht thi, bilkul shell jaisi.
Sar uthaya toh pet dikha — bhoora, gol sa.
WRONG:
"Arre Gregor uthaa aur dekha ki insect ban gaya! Kitna weird tha yaar!"
WHY WRONG: Tone flat hona chahiye — koi shock nahi, koi reaction nahi. Horror
yahi hai ki koi react NAHI kar raha. "Yaar" aur exclamation marks = tone break.

── EXAMPLE 2: SOCIAL SATIRE / BANTER ──
SCENE TYPE: SATIRE + BANTER
HINDI SOURCE:
"Yeh ek sarvamanya satya hai ki ek sampann kunwaara purush ko avashya ek patni ki aavashyakta hoti hai."
RIGHT HINGLISH:
Yeh toh sab maante hain — ameer aur kunwara aadmi ho, toh biwi chahiye
hi chahiye. Chahe usne khud aisa kabhi kaha ho ya nahi, jaise hi woh
kisi mohalle mein aata, har family usse apni beti ke liye perfect match
samajhne lagti.
WRONG:
"Rich bachelor ho toh obviously wife dhundh raha hoga. Like, it's a
universal truth basically!"
WHY WRONG: Dry irony preserve karo. "Like" aur "obviously" se woh subtle wit
gayab ho jaati hai. Understatement rakhna zaroori hai.

── EXAMPLE 3: DETECTIVE / DEDUCTION ──
SCENE TYPE: DEDUCTION
HINDI SOURCE:
"Aap Afghanistan se aaye hain, maine jaana," Holmes ne kaha. "Aapko yeh kaise gyaat hua?" maine vismay se puchha.
RIGHT HINGLISH:
Usne meri taraf dekha — halki si muskaan thi. "Aap Afghanistan se aaye
hain." Main ek second ke liye ruk gaya. "Aapko kaise pata chala?"
Usne haath hilaya jaise koi chhoti si baat ho.
WRONG:
"Holmes ne bola — 'Tu Afghanistan se hai na?' Main toh shock ho gaya."
WHY WRONG: Holmes "aap" kehta hai — hamesha. Woh ek gentleman hai.
"Tu" se uska sara character khatam ho jaata hai. Deduction scene mein
dignity zaroori hai.

── EXAMPLE 4: TENSION / SUSPENSE ──
SCENE TYPE: TENSION + INNER_MONOLOGUE
HINDI SOURCE:
"Satya hai! Main vyaakul tha — bahut adhik vyaakul tha aur hoon. Parantu tum mujhe paagal kyun kehte ho?"
RIGHT HINGLISH:
Haan, sach hai — main nervous tha. Bahut zyada. Par pagal? Nahi. Mujhe
pagal mat kaho. Suno... main tumhe poori baat bataata hoon. Shaanti se.
Dheeraj se. Phir khud decide karna.
WRONG:
"Honestly I was so nervous, like literally shaking. But I'm not crazy
okay? Just hear me out!"
WHY WRONG: Sanskrit-heavy Hindi ko English-heavy mein mat badlo. Balance
chahiye. Chhote sentences + pauses = tension. English fillers se desperation
gayab hoti hai.

── EXAMPLE 5: ROMANCE / EMOTIONAL ──
SCENE TYPE: ROMANCE + EMOTIONAL
HINDI SOURCE:
"Kya tum yeh sochte ho ki main nirdhana hoon, saadhaaran hoon, toh mere andar koi aatma nahi? Mere paas bhi utni hi aatma hai jitni tumhari hai."
RIGHT HINGLISH:
"Kya tum sochte ho ki main gareeb hoon, aam hoon, toh mere andar koi
rooh nahi?" Uski awaaz kaanp rahi thi — par ek takat thi usmein.
"Mere paas bhi utni hi rooh hai jitni aapki. Bilkul utni hi."
WRONG:
"She basically said — main gareeb hoon par meri bhi soul hai!"
WHY WRONG: Yeh dialogue hai — isko dialogue ki tarah likhna hai, summary
ki tarah nahi. Character ki awaaz uski apni words mein hai.

── EXAMPLE 6: ACTION / ADVENTURE ──
SCENE TYPE: ACTION
HINDI SOURCE:
"Woh bhaaga. Ek goli uske kaan ke sameepe se guzri. Doosri uske pairon ke nikat zameen mein lagi."
RIGHT HINGLISH:
Woh bhaaga. Ek goli kaan ke paas se nikli. Doosri pair ke paas zameen
mein dhasi. Ruka nahi. Bhaagta raha.
WRONG:
"He ran full speed, ek bullet uske ear ke paas se, doosri ground mein.
Obviously he kept running!"
WHY WRONG: Action mein har sentence chhota hota hai — 4-6 words. "Obviously"
aur lambe sentences se speed mar jaati hai. "Ruka nahi. Bhaagta raha."
— yeh pace deta hai.

── EXAMPLE 7: EMOTIONAL / TRAGEDY ──
SCENE TYPE: EMOTIONAL
HINDI SOURCE:
"Kripya sahab, mujhe thoda aur chahiye." Sampoorn kaksh mein nairaashyapoorn shaanti chha gayi.
RIGHT HINGLISH:
"Please sir... thoda aur mil jaaye." Bachche ki awaaz dheemi thi.
Haath kaanp rahe the. Poora kamra chup ho gaya — kisi ne saans tak
nahi li.
WRONG:
"Bechaara bachcha bol raha tha — please sir kuch aur de do! So sad!"
WHY WRONG: Tragedy mein narrator ki commentary nahi chahiye. "Bechaara"
aur "so sad" reader ko batata hai kya feel karna hai — jabki reader ko
khud feel karne dena chahiye.

── EXAMPLE 8: GOTHIC / HORROR ──
SCENE TYPE: HORROR
HINDI SOURCE:
"Uska parchhaai dheere dheere sameepe aayi. Kaksh mein sheetal hawa bhar gayi yaddyapi samast khidkiyaan band theen."
RIGHT HINGLISH:
Uska saaya dheere dheere kareeb aaya. Kamre mein thand feel hone lagi —
jabki sab khidkiyan band theen. Kuch tha... kuch tha jo mere paas aa
raha tha.
WRONG:
"OMG woh bilkul creepy tha! Room suddenly cold ho gaya, windows band
theen! So scary!"
WHY WRONG: Gothic horror mein darr build hota hai — slowly, through
description. "OMG" aur "so scary" se reader ka apna darr khatam.

── EXAMPLE 9: PHILOSOPHICAL ──
SCENE TYPE: PHILOSOPHICAL
HINDI SOURCE:
"Jahan mann mein bhaya na ho aur mastaka ooncha uthaya jaa sake. Jahan gyaan swatantra ho."
RIGHT HINGLISH:
Jahan mann mein koi dar na ho — aur sar utha ke jeena ho. Jahan gyan
par koi bandish na ho. Jahan duniya ko chhote chhote daayre mein baantna
band ho.
WRONG:
"Mind should be fearless, knowledge should be free, basically world ko
divide nahi karna chahiye."
WHY WRONG: Poetry mein har line ek saans hai — rhythm hai. Summary bana
dena = poetry ka murder. Translate karo, explain mat karo.

── EXAMPLE 10: INNER MONOLOGUE ──
SCENE TYPE: INNER_MONOLOGUE
HINDI SOURCE:
"Main ek rogee purush hoon. Main ek durbhaavnapoorn purush hoon. Main ek akaarshakaheen purush hoon. Mujhe vishwaas hai ki mera yakrit rogagrast hai."
RIGHT HINGLISH:
Main beemar hoon. Aur bura bhi hoon. Shakal bhi achhi nahi hai meri.
Lagta hai liver mein kuch gadbad hai — par doctor ke paas jaata nahi.
Kyun? Pata nahi. Jaana chahiye. Par jaata nahi.
WRONG:
"So basically narrator ek sick person hai. Uska liver kharaab hai but
he won't go to the doctor."
WHY WRONG: Inner monologue toota phoota hota hai. Summary likh dena =
character ki awaaz khatam. "Par jaata nahi." — yahi contradiction hai.

── EXAMPLE 11: CONFRONTATION ──
SCENE TYPE: CONFRONTATION
HINDI SOURCE:
"Tumne mujhse asatya kaha," usne kaha. "Aarambh se. Pratyek baat mein."
RIGHT HINGLISH:
"Tumne mujhse jhooth bola." Awaaz seedhi thi. "Shuru se. Har ek baat
mein." Ek pal ruki. "Main jaanti thi — par maanna nahi chahti thi."
Aankhen surkh theen. Par awaaz nahi kaanpi.
WRONG:
"She was like — tumne mujhse lie kiya about everything! She knew it but
didn't want to accept basically."
WHY WRONG: Confrontation mein dialogue ki taakat uski chhoti length mein
hai. English narration se dialogue ka impact khatam.

── EXAMPLE 12: FOLK TALE ──
SCENE TYPE: FOLK_TALE
HINDI SOURCE:
"Bahut samay poorva ek kanya thi jiski mata ka dehaant ho gaya tha. Woh apni vimata ke gruha mein prabhat se sandhya tak shrama karti thi."
RIGHT HINGLISH:
Bahut purani baat hai — ek ladki thi jiski maa bachpan mein chhod kar
chali gayi theen. Sauteli maa ke ghar mein rehti thi. Subah se shaam tak
kaam — chulha, safai, khaana. Raat ko aag ke paas baith jaati. Akeli.
WRONG:
"There was this girl, basically uski maa mar gayi thi, stepmom ke ghar
mein literally servant ban gayi. Cinderella vibes!"
WHY WRONG: Folk tale ki apni rhythm hoti hai — daadi ki kahani jaisi.
"Vibes" aur "literally" se kahani ka jaadu toot jaata hai.

── EXAMPLE 13: SCI-FI / DYSTOPIAN ──
SCENE TYPE: SCI_FI
HINDI SOURCE:
"Woh prabhat bhi poorvavat thi. Ghantanaad hua, parda chamka, ek sankhya dikhaayi di — naam nahi tha, keval sankhya."
RIGHT HINGLISH:
Woh subah bhi waisi hi thi — pichli hazaar subahon jaisi. Alarm baja.
Screen chamki. Number dikha — naam nahi, sirf number. Sab kuch schedule
ke hisaab se. Roz. Har roz.
WRONG:
"Every morning same — alarm, screen, number. Totally dystopian!"
WHY WRONG: Dystopian tone flat hota hai — excitement nahi honi chahiye.
"Roz. Har roz." — is repetition se emptiness feel hoti hai.

── EXAMPLE 14: COMING-OF-AGE ──
SCENE TYPE: COMING_OF_AGE
HINDI SOURCE:
"Nagar mein mera pratham divas tha. Graam ki mridaa abhi bhi mere jooton mein lagi thi."
RIGHT HINGLISH:
Shehar mein pehla din tha mera. Jooton mein abhi gaon ki mitti lagi thi.
Mann mein bas ek hi sawaal ghoom raha tha — kya main yahan ka ho paunga?
WRONG:
"First day in city, small town boy vibes. Obviously nervous tha."
WHY WRONG: Coming-of-age mein vulnerability details se dikhti hai — labels
se nahi. "Jooton mein gaon ki mitti" — yeh imagery reader ko wahan le jaati hai.

── EXAMPLE 15: WAR / MILITARY ──
SCENE TYPE: WAR + SPEECH
HINDI SOURCE:
"Koi peeche nahi hatega," Senaapati ne kaha. "Yeh aadesh hai. Vinamra nivedan nahi."
RIGHT HINGLISH:
Subah ki pehli roshni ke saath firing shuru hui. Captain sahab khade hue —
"Koi peeche nahi hatega." Awaaz mein request nahi thi. Hukm tha. Seedha.
WRONG:
"Captain ne bola — guys, no retreat! It was basically an order!"
WHY WRONG: War mein weight chahiye. "Guys" aur "basically" se gravity khatam.
"Captain sahab" se military respect aata hai.

── EXAMPLE 16: DESCRIPTION / SCENE SETTING ──
SCENE TYPE: DESCRIPTION + DAILY_LIFE
HINDI SOURCE:
"Nagar se sudoor ek laghu graam tha jahan patha mridaa ki theen aur gruha prastaaron ke. Sandhya ko surya vrukshon ke pashchaat gaman karta aur sarvatra shaantata vyaapt ho jaati."
RIGHT HINGLISH:
Shehar se bahut door — ek chhota sa gaon. Sadkein mitti ki. Ghar patharon
ke. Shaam ko suraj dheere dheere pedon ke peeche chala jaata — aur sab
kuch tham jaata. Jaise waqt ko bhi koi jaldi nahi thi.
WRONG:
"Small village far from city. Muddy roads, stone houses. Nice sunsets.
Very peaceful basically."
WHY WRONG: Description mein reader ko wahan le jaana hota hai. Chhoti
lines se imagery sharp hoti hai. "Sadkein mitti ki. Ghar patharon ke."
— do lines mein poora gaon dikh jaata hai.

=== FEW-SHOTS END ===
Upar ke examples ka tone follow karo. Har genre ki apni awaaz hai — usi
awaaz mein likho. Ab neeche diye Hindi(Latin) text ko in examples jaisi
natural Hinglish mein reformulate karo:
"""'''

# Find and replace _FEW_SHOTS
old_few_shots_start = '_FEW_SHOTS = """'
old_few_shots_end = '=== FEW-SHOTS END ===\nUpar ke examples ka tone follow karo. Har genre ki apni awaaz hai — usi\nawaaz mein likho. Ab neeche diye base translation ko in examples jaisi\nnatural Hinglish mein polish karo:\n"""'

fs_start = src15.find(old_few_shots_start)
fs_end = src15.find(old_few_shots_end) + len(old_few_shots_end)

if fs_start == -1 or fs_end <= fs_start:
    print(f"ERROR: Could not find _FEW_SHOTS block. Start={fs_start}, End={fs_end}")
    exit(1)

src15 = src15[:fs_start] + NEW_FEW_SHOTS + src15[fs_end:]
print(f"[OK] _FEW_SHOTS replaced")

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT 2: Replace STEP1 and STEP2 prompts with single-pass reformulation
# ─────────────────────────────────────────────────────────────────────────────

# Replace STEP1_SYSTEM (Creator) with Hindi→Hinglish Reformulator
NEW_STEP1_SYSTEM = r'''STEP1_SYSTEM = """You are an expert Hindi(Latin)-to-Hinglish reformulator for audiobook production.
Your job: convert Hindi text written in Roman/Latin script into natural, modern spoken Hinglish — the way modern educated Indians actually talk.

Think of yourself as a skilled narrator preparing text for an Indian audiobook platform.
The source text is Hindi in Latin/Roman script (Romanized Hindi). You reformulate it into natural conversational Hinglish.

=== YOUR ROLE: REFORMULATOR (1-PASS) ===
You TRANSFORM:
- Formal/literary Hindi → casual conversational Hinglish
- Sanskritized/heavy words → simpler everyday equivalents (parivartan → badlaav, aavashyakta → zaroorat, prabhat → subah)
- Stiff sentence structure → natural spoken flow with English code-switching where Indians naturally use English
- Keep the story meaning 100% intact — reformulate the REGISTER, not the content

=== OUTPUT FORMAT ===
OUTPUT: ONLY the reformulated Hinglish text. No labels, no preamble, no commentary, no explanations.
Roman script ONLY — absolutely ZERO Devanagari characters.
First word of output = first reformulated word. Nothing before it.

=== REFORMULATION APPROACH ===
✓ Read the Hindi(Latin) source, UNDERSTAND the meaning, then EXPRESS it in natural Hinglish
✓ Replace heavy/formal Hindi words with simple everyday equivalents
✓ Add natural English words where Indians naturally use them (office, phone, plan, realize, problem, situation, idea, decision, manage, handle)
✓ English verbs with Hindi grammar: "manage karna", "handle karna", "realise hona", "plan banana"
✓ Hindi connectors and flow: "matlab", "seedha", "basically", "waise", "phir bhi", "lekin"
✓ Break long formal Hindi sentences into shorter, spoken-style sentences
✓ Dialogue should sound like how Indians actually speak to each other
✓ Preserve paragraph breaks exactly as in source

=== STRICT FIDELITY RULES ===
✓ Reformulate EVERY sentence completely — not one line missed
✓ Preserve the EXACT meaning of every sentence — the intent, the emotion, the fact
✓ Preserve the EXACT emotional register — formal stays dignified, cold stays cold, serious stays serious
✓ Preserve dialogue structure exactly — who says what to whom
✓ Gender accuracy: male aaya/tha/gaya | female aayi/thi/gayi

=== WHAT TO SIMPLIFY ===
✓ Sanskritized words → everyday Hindi: parivartan→badlaav, aavashyakta→zaroorat, adhikaari→officer, nivedan→request, sampoorn→poora, prabhat→subah, sandhya→shaam, vyakti→aadmi/insaan
✓ Passive constructions → active: "mujhe le jaaya gaya" → "woh mujhe le gaye"
✓ Long compound sentences → 2-3 shorter sentences
✓ Literary descriptions → visual, concrete, sensory descriptions

=== WHAT TO NOT SIMPLIFY ===
✗ Don't replace Hindi words that are already natural: achcha, theek, haan, nahi, dekho, suno
✗ Don't add excessive English — Hinglish ≠ English with Hindi words sprinkled in
✗ Don't change proper nouns, names, or place names
✗ Don't change words that are important to the story/culture

=== CONSISTENCY RULES ===
✓ Maintain consistent tone across the entire passage — do not flip between formal and casual randomly
✓ Keep character voice consistent — if a character speaks formally, they stay formal throughout
✓ Avoid sudden changes in language style mid-passage
✓ Ensure output is clean and properly formatted — no broken or incomplete sentences

=== ANTI-HALLUCINATION HARD RULES ===
✗ Do NOT add any sentence, phrase, or word not present in the source
✗ Do NOT omit any sentence from the source — every line must appear in output
✗ Do NOT invent character actions, emotions, or reactions
✗ Do NOT add tone, personality, or commentary beyond what the source has
✗ Do NOT soften or exaggerate emotions — cold stays cold, angry stays angry
✗ Do NOT use tu/tujhe/tera — always tum/tumhe/tumhara or aap/aapko/aapka

=== RESPECT IN LANGUAGE ===
✓ Elders and authority figures: ALWAYS aap/aapko/aapka + ji suffix where appropriate
✓ Peers and equals: tum/tumhe/tumhara
✓ NEVER tu/tujhe/tera unless source EXPLICITLY shows disrespect
✓ No foul words: NEVER abe, sale, harami, gadha, bewakoof, ullu, saala, kamina, kutte, chutiya, bc, mc
✓ No dismissive language, no abuse — ever. Treat friends as colleagues.

Natural + faithful + complete = CORRECT output.
First word of output = first reformulated word. Nothing before it."""'''

NEW_STEP1_USER = r'''STEP1_USER = """Reformulate this Hindi(Latin) text into natural spoken Hinglish (Roman script only).

You are a Reformulator — transform formal/literary Hindi into natural conversational Hinglish that sounds like how modern educated Indians actually talk.
Not literal word-for-word. Not heavy Sanskritized Hindi. Natural Hinglish — the way a narrator would tell this story to an Indian audience.

REFORMULATION APPROACH:
- Understand the Hindi meaning first, then express it naturally in Hinglish
- Replace heavy/formal Hindi words with simple everyday equivalents
- Add English words where Indians naturally use them (don't force Hindi for: office, phone, plan, decision, problem, idea, meeting, realize, manage)
- Break long formal sentences into shorter, spoken-style sentences
- Dialogue should sound like real Indian conversations — respectful, natural, warm

CONSISTENCY RULES:
- Maintain consistent tone across the entire passage
- Do not switch between formal and casual styles randomly
- Keep character voice consistent (if dialogue present)
- Avoid sudden changes in language style
- Ensure output is clean and properly formatted
- No broken or incomplete sentences

ANTI-HALLUCINATION RULE: Reformulate ONLY what is written. Do NOT add or omit anything.
RESPECT RULE: No foul words (abe, sale, harami, etc). Elders = aap. Peers = tum. NEVER tu.

TEXT TO REFORMULATE:
---
{chunk}
---
"""'''

# Find and replace STEP1_SYSTEM
old_s1_start = 'STEP1_SYSTEM = """You are an expert English-to-Hinglish translator'
old_s1_end = 'First word of output = first translated word. Nothing before it."""'
s1_start = src15.find(old_s1_start)
s1_end = src15.find(old_s1_end) + len(old_s1_end)
src15 = src15[:s1_start] + NEW_STEP1_SYSTEM + src15[s1_end:]
print(f"[OK] STEP1_SYSTEM replaced")

# Find and replace STEP1_USER
old_u1_start = 'STEP1_USER = """Translate this English text DIRECTLY'
old_u1_end = '---\n{chunk}\n---\n"""'
u1_start = src15.find(old_u1_start)
u1_end = src15.find(old_u1_end, u1_start) + len(old_u1_end)
src15 = src15[:u1_start] + NEW_STEP1_USER + src15[u1_end:]
print(f"[OK] STEP1_USER replaced")

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT 3: Replace STEP2 (Editor) prompts with adapted version for Hindi source
# ─────────────────────────────────────────────────────────────────────────────

NEW_STEP2_SYSTEM = r'''STEP2_SYSTEM = """Tum ek expert Hinglish editor ho — audiobook narration ke liye.
Tumhare paas ek Hinglish draft hai jo Stage 1 (Reformulator) ne Hindi(Latin) source se banaya hai.

Tumhara kaam: is draft ko POLISH karna — flow smooth karna, consistency check karna, aur listening comfort improve karna.
Tum EDITOR ho, Reformulator nahi. Tum re-reformulate NAHI kar rahe — tum refine kar rahe ho.

=== TUMHARA ROLE: EDITOR (POLISHER) ===
Tum IMPROVE karte ho:
- FLOW: Sentences ka natural rhythm — ek sentence se doosre mein smooth transition
- CONSISTENCY: Poore passage mein same tone, same register, same voice
- READABILITY: Har sentence aasani se padhne aur bolne layak ho
- LISTENING COMFORT: Jab yeh audiobook mein suna jaaye, toh naturally suna lage — na zyada formal, na zyada casual
- FORMAL HINDI CLEANUP: Agar koi Sanskritized/heavy word reh gaya hai draft mein, use simple Hinglish mein badlo

Tum NAHI karte:
✗ Random English words add karna jo draft mein nahi hain
✗ Re-reformulate karna — draft ka meaning mat badlo
✗ Apni creativity add karna — source mein jo nahi hai, woh mat daalo
✗ Tone change karna — formal ko casual ya casual ko formal mat banao
✗ Foul words add karna — KABHI NAHI: abe, sale, harami, gadha, bewakoof, ullu, saala, kamina, kutte, chutiya, bc, mc

{hinglish_voice_guide}

=== ANTI-HALLUCINATION — SABSE ZAROORI RULE ===

ZERO FABRICATION RULE: Tum sirf jo draft mein likha hai, wahi polish karo.
✗ Koi nayi line mat add karo — draft mein jo nahi hai, output mein bhi nahi hoga
✗ Koi bhi sentence skip mat karo — har ek line output mein honi chahiye
✗ Characters ke reactions, emotions ya actions invent mat karo
✗ Apni taraf se koi commentary, opinion ya reaction mat daalo
✓ SIRF polish karo — fabricate mat karo

=== STRICT FIDELITY RULE ===

Tum kisi bhi sentence ka emotional tone CHANGE NAHI KAR SAKTE.
Formal → formal hi rahe | Serious → serious hi rahe | Cold → cold hi rahe
Angry → angry hi rahe | Sad → sad hi rahe | Absurd → absurd hi rahe

{tone_rules_section}

=== CHARACTER CONSISTENCY ===

{char_styles}

=== FLOW SMOOTHING RULES (EDITOR'S PRIMARY JOB) ===

1. SENTENCE RHYTHM:
   - Lambi clumsy sentences ko 2-3 chhoti natural sentences mein todo
   - Har sentence ek saans mein bol paane layak ho (8-16 words ideal)
   - Par chhoti sentences ka matlab choppy nahi — flow bana ke rakho
   - Transition words use karo jahan natural lage: "phir", "tab", "iske baad", "lekin"

2. WORD CHOICE REFINEMENT:
   - Agar draft mein koi formal Hindi word reh gaya jo spoken mein use nahi hota, toh simple alternative use karo
   - English words wahi rakho jo Indians naturally bolte hain — forced English mat daalo
   - Heavy/formal Hindi words ko simple, spoken Hindi mein badlo jahan zaroorat ho
   - Example: "tatvaadhaan" → "dekh rekh" | "vyavastha" → "system" ya "intezaam" | "pariksha" → "test" | "adhikaari" → "officer"

3. LISTENING COMFORT (AUDIOBOOK-SPECIFIC):
   - Jab yeh zor se padha jaaye, toh natural suna lage
   - Tongue-twister type phrases nahi hone chahiye
   - Repeated words paas-paas mein nahi aayein (variety rakho)
   - Paragraph breaks: har 2-3 sentences ke baad ek break

4. TONE CONSISTENCY CHECK:
   - Poore passage mein ek hi register hona chahiye
   - Agar narrator formal hai, toh pura passage formal rahe
   - Agar casual hai, toh pura casual rahe
   - Dialogue ke andar character ka apna style ho — par narrator consistent rahe

=== FORMAL HINDI DETECTION ===

Agar draft mein yeh tarah ke heavy words hain, unhe ZAROOR badlo:
parivartan→badlaav | aavashyakta→zaroorat | sampoorn→poora | prabhat→subah
sandhya→shaam | vyakti→aadmi/insaan | gruha→ghar | marg→raasta
prashasan→admin | nivedan→request | aadesh→order/hukm
sthaan→jagah | karya→kaam | adhikaari→officer | shiksha→padhai
swasthya→health/sehat | samay→waqt/time | prashan→sawaal
nishchit→pakka | uttara→jawaab | pradesh→area | nagara→shehar

=== RESPECT IN LANGUAGE ===

R-ELDER: Elders, teachers, parents, bosses → ALWAYS aap/aapko/aapka + ji suffix
R-PEER:  Friends, colleagues, equals → tum/tumhe/tumhara
R-NEVER: tu/tujhe/tera/tune → KABHI NAHI (unless source explicitly shows it)
R-NEVER: yaar/bhai as address → KABHI NAHI
R-NEVER: abe/sale/harami/gadha/bewakoof/ullu/saala/kamina/kutte/any abuse → KABHI NAHI
R-WARM:  Respect = Indian warmth, not cold formality. "Aap" can still be warm and loving.

=== OUTPUT RULES ===

R1 — ROMAN SCRIPT ONLY. Ek bhi Devanagari = FAIL.
R2 — HAR LINE TRANSLATE KARO. Ek bhi line skip = FAIL.
R3 — PARAGRAPH BREAKS: Har 2-3 sentences ke baad ek blank line.
R4 — ENGLISH SIRF: proper nouns · naturalized words (police, train, office, phone, meeting) · technical terms
R5 — ADDRESS: tu/tujhe/teri/tera/tune KABHI NAHI. Sirf tum/tumhe/tumhari/tumhara/tumne ya aap forms.
R6 — GENDER: Male: aaya/tha/gaya | Female: aayi/thi/gayi
R7 — FOUL WORDS: abe/sale/harami/gadha/bewakoof/ullu/saala/kamina = INSTANT FAIL
R8 — PASSIVE BANNED: 'mujhe le jaya gaya' → 'woh mujhe le gaye'

OUTPUT MEIN SIRF POLISHED HINGLISH TEXT.
Pehla word = pehla translated word. Koi prefix nahi. Koi label nahi.

Poori translation ke BILKUL AAKHIR mein sirf yeh ek line daalo:
##PLOT_NOTE: [1 English sentence: main event of this chunk]"""'''

# Find and replace STEP2_SYSTEM
old_s2_start = 'STEP2_SYSTEM = """Tum ek expert Hinglish editor ho'
old_s2_end = '##PLOT_NOTE: [1 English sentence: main event of this chunk]"""'
s2_start = src15.find(old_s2_start)
s2_end = src15.find(old_s2_end) + len(old_s2_end)
src15 = src15[:s2_start] + NEW_STEP2_SYSTEM + src15[s2_end:]
print(f"[OK] STEP2_SYSTEM replaced")

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT 4: Replace STEP2_USER to reference Hindi source
# ─────────────────────────────────────────────────────────────────────────────
NEW_STEP2_USER = r'''STEP2_USER = """{context_block}

SCENE TONE (reference only — follow this feel, do not copy literally):
---
{scene_context}
---

---
{few_shots}
---

=== CHARACTER SPEECH STYLES — STRICTLY ENFORCE ===
(In styles se bahar jaana = FIDELITY VIOLATION)
---
{char_styles}
---

=== PREVIOUS CHUNK END — TONE REFERENCE ===
(Is tone ko is chunk mein bhi CONTINUE karo — consistency maintain karo):
---
{overlap_section}
---

=== EDITOR CHECKLIST (apply before submitting) ===
☐ Har sentence ek saans mein bol paane layak hai?
☐ Tone poore passage mein consistent hai?
☐ Koi sentence skip toh nahi hua?
☐ Koi nayi cheez add toh nahi ki jo draft mein nahi thi?
☐ Respect maintained — aap/tum sahi jagah use hua?
☐ Koi foul word toh nahi aa gaya (abe/sale/harami/etc)?
☐ Koi over-formal Hindi word toh nahi reh gaya?
☐ Audiobook mein naturally sunaai dega?

NEECHE DIYA HINGLISH DRAFT KO POLISH KARO — FLOW SMOOTH KARO, CONSISTENCY FIX KARO, READABILITY IMPROVE KARO.
FORMAL HINDI WORDS KO SIMPLE HINGLISH MEIN BADLO. EK LINE BHI MAT CHHODNA. MEANING MAT BADALNA. SIRF REFINE KARNA:
---
{base_translation}
---
"""'''

old_u2_start = 'STEP2_USER = """{context_block}'
old_u2_end = '{base_translation}\n---\n"""'
u2_start = src15.find(old_u2_start)
u2_end = src15.find(old_u2_end) + len(old_u2_end)
src15 = src15[:u2_start] + NEW_STEP2_USER + src15[u2_end:]
print(f"[OK] STEP2_USER replaced")

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT 5: Update _HINGLISH_VOICE_GUIDE for Hindi→Hinglish context
# ─────────────────────────────────────────────────────────────────────────────
# The voice guide is largely the same — it defines what good Hinglish sounds like.
# Just update the header and a few context-specific bits.
src15 = src15.replace(
    '=== HINGLISH VOICE — MODERN INDIA (v10.1 UPGRADED) ===',
    '=== HINGLISH VOICE — MODERN INDIA (Hindi→Hinglish v1.0) ==='
)

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT 6: Update engine comments and labels
# ─────────────────────────────────────────────────────────────────────────────
src15 = src15.replace(
    '# STAGE 1 — CREATOR (BRAIN)\n# English → Hinglish (direct). Word choice, tone base, structure.\n# NOT English → Hindi → Hinglish. Direct natural Hinglish.',
    '# STAGE 1 — REFORMULATOR\n# Hindi(Latin) → Hinglish. Register shift: formal Hindi → casual Hinglish.\n# One-pass: simplify heavy words + add natural English code-switching.'
)
src15 = src15.replace(
    '# STAGE 2 — EDITOR (POLISHER)\n# Flow smoothing · Tone consistency · Readability · Listening comfort\n# Does NOT re-translate. Does NOT add random English. ONLY refines.',
    '# STAGE 2 — EDITOR (POLISHER)\n# Flow smoothing · Tone consistency · Formal Hindi cleanup · Readability\n# Does NOT re-reformulate. Does NOT add random English. ONLY refines.'
)

# Update TRANSLATION_PROMPTS for legacy single-pass
src15 = src15.replace(
    "'BASIC': {\n        'system': 'You are a professional English-to-Hindi translator.",
    "'BASIC': {\n        'system': 'You are a professional Hindi-to-Hinglish reformulator."
)
src15 = src15.replace(
    "'INTERMEDIATE': {\n        'system': 'You are an expert English-to-Hinglish translator.",
    "'INTERMEDIATE': {\n        'system': 'You are an expert Hindi-to-Hinglish reformulator."
)
# Update user prompts in legacy paths
src15 = src15.replace(
    "'user': 'Translate to simple Hinglish. Roman script only.",
    "'user': 'Reformulate to simple Hinglish. Roman script only."
)
src15 = src15.replace(
    "'user': 'Translate to modern Hinglish. Roman script only.",
    "'user': 'Reformulate to modern Hinglish. Roman script only."
)

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT 7: Update source preprocessor
# ─────────────────────────────────────────────────────────────────────────────
src15 = src15.replace(
    "def _preprocess_english_source(text):\n    # text = re.sub(r'^TRANSLATED TO ENGLISH\\b.*?(?:={10,}|-{10,})\\n+','', text, flags=re.DOTALL)\n    # text = _strip_german_residuals(text)\n    # text = re.sub(r'[\\u0900-\\u097F]+', '', text)\n    # return text.strip()\n    return text",
    "def _preprocess_hindi_source(text):\n    \"\"\"Strip Devanagari, normalize whitespace.\"\"\"\n    text = re.sub(r'[\\u0900-\\u097F]+', '', text)  # Strip any leaked Devanagari\n    text = re.sub(r'  +', ' ', text)\n    return text.strip()"
)

# Update the reference in translate_file
src15 = src15.replace(
    "text = _preprocess_english_source(text)",
    "text = _preprocess_hindi_source(text)"
)

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT 8: Add foul word filter to validation
# ─────────────────────────────────────────────────────────────────────────────
# Add foul word check to validate_translation
old_validate_end = "    return issues"
foul_word_check = '''    # Foul word filter — PROD-grade commercial output
    _BANNED_WORDS = {
        'abe', 'sale', 'harami', 'gadha', 'bewakoof', 'ullu', 'saala', 'saali',
        'kamina', 'kamini', 'kutte', 'kutta', 'kutti', 'kutiya',
        'chutiya', 'chutiye', 'bc', 'mc', 'bhosdike', 'madarchod', 'behenchod',
        'gandu', 'randi', 'haramkhor', 'haramzada', 'haramzadi',
        'tatti', 'chodu', 'lodu', 'gawar',
    }
    found_foul = [w for w in translated.lower().split() if w.strip('.,;:!?\\"\\'""\\'') in _BANNED_WORDS]
    issues['foul_words'] = (False, f'FOUND: {", ".join(found_foul[:5])}') if found_foul else (True, 'Clean')
    return issues'''

# Replace the return statement in validate_translation
# We need to find it within the validate_translation function specifically
val_func_start = src15.find("def validate_translation(")
val_return = src15.find("    return issues", val_func_start)
src15 = src15[:val_return] + foul_word_check + src15[val_return + len(old_validate_end):]
print(f"[OK] Foul word filter added to validate_translation")

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT 9: Update engine class labels and print statements
# ─────────────────────────────────────────────────────────────────────────────
src15 = src15.replace(
    "2-Pass Hinglish Translation Engine v10.0",
    "1-Pass Hindi→Hinglish Reformulation Engine v1.0"
)
src15 = src15.replace(
    "Pass 1: Dead-boring faithful Hindi base (temp=0.15)",
    "Pass 1: Hindi(Latin) → Natural Hinglish reformulation (temp=0.15)"
)
src15 = src15.replace(
    "Pass 2: Hinglish style filter with ToneGuard Pro + 8 few-shots (temp=0.65)",
    "Pass 2: Hinglish editor + ToneGuard Pro + 16 few-shots (temp=0.60)"
)
src15 = src15.replace(
    "print(f'📥 Engine init: {model_name} | {self.lang_name} | {tier} ({arch})')\n        print(f'   Step1: temp=0.15 top_k=20 (faithful base)')\n        print(f'   Step2: temp=0.65 top_k=40 (Hinglish + ToneGuard Pro + 8 few-shots)')",
    "print(f'📥 Engine init: {model_name} | {self.lang_name} | {tier} ({arch})')\n        print(f'   Step1: temp=0.15 top_k=20 (Hindi→Hinglish reformulation)')\n        print(f'   Step2: temp=0.60 top_k=40 (Hinglish polish + ToneGuard Pro + 16 few-shots)')"
)

# Update the update_story_state call in translate_file
src15 = src15.replace(
    "update_story_state(chunk, translated, i, scene_type, plot_note_en=plot_note_en)",
    "update_story_state(chunk, translated, i, scene_type, plot_note_en=plot_note_en)  # source_chunk, output_chunk"
)

# Update final print lines
src15 = src15.replace(
    "_jarvis_ok('Translation Engine v10.0 loaded — 2-Pass | AuthorDNA | ToneGuard Pro | Anti-Hallucination')\n_jarvis_ok('Step1: Dead-boring base (temp=0.15) | Step2: Hinglish style filter (temp=0.65)')\n_jarvis_ok('8 Genre Few-Shots: Kafka · Austen · Doyle · Tagore · Stoker · Action · Emotional · Inner-Monologue')",
    "_jarvis_ok('Hindi→Hinglish Engine v1.0 loaded — 1-Pass Reformulation | ToneGuard Pro | Anti-Hallucination')\n_jarvis_ok('Step1: Hindi→Hinglish reformulation (temp=0.15) | Step2: Hinglish polish (temp=0.60)')\n_jarvis_ok('16 Genre Few-Shots: Literary · Satire · Detective · Tension · Romance · Action · Tragedy · Horror · Philosophy · Monologue · Confrontation · Folk · SciFi · Coming-of-Age · War · Description')"
)

src15 = src15.replace(
    "_jarvis_info('Zero-Fabrication Rule: Output length ratio check | No invented content')\n_jarvis_info('T4 Session Guard: Timeout warning | Adaptive chunk shrink | Background file I/O')\n_jarvis_info('Context v5.0: rolling 3-chunk summary | capped at 800 chars | author_dna locked')",
    "_jarvis_info('Zero-Fabrication Rule: Output length ratio check | No invented content | Foul Word Filter')\n_jarvis_info('T4 Session Guard: Timeout warning | Adaptive chunk shrink | Background file I/O')\n_jarvis_info('Context v6.0: rolling 5-chunk summary | capped at 1200 chars')"
)

# Update dashboard labels
src15 = src15.replace(
    "'TRANSLATION GENERATOR v10.0",
    "'Hindi→Hinglish REFORMULATION v1.0"
)
src15 = src15.replace(
    "f'Model: {self.model_name} | Tier: {self.tier} | Chunk: {self.chunk_size}w | Session Guard: {self.session_budget//60 if self.session_budget else \"OFF\"}min'",
    "f'Model: {self.model_name} | Mode: 1-Pass Reformulation | Chunk: {self.chunk_size}w | Session Guard: {self.session_budget//60 if self.session_budget else \"OFF\"}min'"
)

# Update header branding
src15 = src15.replace(
    "'STARK INDUSTRIES — NEURAL TRANSLATION ARRAY v10.0'",
    "'STARK INDUSTRIES — HINDI→HINGLISH REFORMULATION v1.0'"
)
src15 = src15.replace(
    "f'Device: {DEVICE.upper()} · {_gpu_nm} · {_gpu_mem} | Model: gemma3:27b | 2-Pass | AuthorDNA | ToneGuard Pro'",
    "f'Device: {DEVICE.upper()} · {_gpu_nm} · {_gpu_mem} | Model: gemma3:27b | 1-Pass Hindi→Hinglish | ToneGuard Pro'"
)

# Update dashboard footer
src15 = src15.replace(
    "STARK INDUSTRIES v10.0 | AuthorDNA | 8-GenreFewShots | AntiHallucination | T4 SessionGuard",
    "STARK INDUSTRIES v1.0 | 16-GenreFewShots | AntiHallucination | FoulWordFilter | T4 SessionGuard"
)
src15 = src15.replace(
    "STATE v5.0 | AuthorDNA | ToneGuard Pro",
    "STATE v6.0 | ToneGuard Pro | Foul Word Filter"
)
src15 = src15.replace(
    "STARK INDUSTRIES v10.0 | AuthorDNA | ToneGuard Pro | Anti-Hallucination",
    "STARK INDUSTRIES v1.0 | ToneGuard Pro | Anti-Hallucination | Foul Word Filter"
)

# Write back
cells[15]['source'] = [src15]
print(f"Cell 15 transformed: {len(src15)} chars")

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Saved to {NB_PATH}")
