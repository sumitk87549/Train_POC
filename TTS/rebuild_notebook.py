import json

path = '/home/sumit/Documents/GitHub/Train_POC/TTS/fish_s2pro_hinglish_audiobook_2_5_L4_LONG QT4.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def cell(text):
    return [line + '\n' for line in text.strip('\n').split('\n')]

# Cell 0: Markdown
nb['cells'][0]['source'] = cell("""
# 🐟 S2-PRO (GGUF) — Long-Form Hinglish Audiobook Pipeline
## T4 GPU · s2.cpp Vulkan Inference · Smart Chunking

**Model**: `rodrigomt/s2-pro-gguf` (Q8_0 Quantized)
**Runtime**: Google Colab · **NVIDIA T4 GPU**
**Features**:
- 🎯 High-speed pure C++ inference via `s2.cpp`
- ⚡ Vulkan GPU acceleration
- 🧠 Fits easily in 15GB VRAM (Q8_0 is ~6.11 GB)
- 📖 Smart sentence-boundary text chunking for long stories
- 🎙️ Reference audio voice cloning across all chunks
- 🌐 Hinglish (Devanagari + Latin script) fully supported

**Workflow**: Cell 1→8 in order. No restarts needed.

| Cell | Purpose |
|------|--------|
| 1 | Hardware audit (Vulkan support check) |
| 2 | Configuration |
| 3 | Upload text file (.txt) |
| 4 | Upload reference audio (optional) |
| 5 | **Build s2.cpp engine** (CMake/Vulkan) |
| 6 | Download GGUF weights |
| 7 | **S2 Synthesis** (chunked CLI pipeline) |
| 8 | Play & download output |
""")

# Cell 1: Environment Audit
nb['cells'][1]['source'] = cell(r"""
# ════════════════════════════════════════════════════════════
# CELL 1 — ENVIRONMENT & HARDWARE AUDIT
# ════════════════════════════════════════════════════════════
import os, sys, subprocess, platform, time

SEP = "═" * 62
print(f"\n{SEP}")
print("  CELL 1 — ENVIRONMENT & HARDWARE AUDIT")
print(f"{SEP}\n")

print("🐍 PYTHON")
print(f"   Version      : {sys.version.split()[0]}")
print(f"   Executable   : {sys.executable}")

print(f"\n💻 CPU & RAM")
try:
    with open('/proc/meminfo') as f:
        mi = {l.split(':')[0]: l.split(':')[1].strip() for l in f}
    total_gb = int(mi['MemTotal'].split()[0])     / 1e6
    avail_gb = int(mi['MemAvailable'].split()[0]) / 1e6
    print(f"   Host RAM     : {total_gb:.1f} GB ({avail_gb:.1f} GB available)")
except Exception as e:
    print(f"   /proc/meminfo: {e}")

print(f"\n⚡ GPU ACCELERATOR (Vulkan Target)")
_has_gpu = False
try:
    smi = subprocess.check_output(
        ['nvidia-smi','--query-gpu=name,memory.total,driver_version',
         '--format=csv,noheader,nounits'],
        stderr=subprocess.DEVNULL).decode().strip()
    parts = [p.strip() for p in smi.split(",")]
    gpu_name = parts[0] if parts else "Unknown"
    gpu_vram = parts[1] if len(parts) > 1 else "?"
    gpu_driver = parts[2] if len(parts) > 2 else "?"
    print(f"   GPU          : {gpu_name}")
    print(f"   VRAM         : {gpu_vram} MB")
    print(f"   Driver       : {gpu_driver}")
    vram_mb = int(gpu_vram)
    _has_gpu = True
    if vram_mb >= 8000:
        print(f"   ✅ Sufficient VRAM for Q8_0 weights (~6.1 GB + context)")
    else:
        print(f"   ⚠️  Low VRAM — might need Q6_K or CPU compilation.")
except Exception:
    print(f"   GPU          : none detected (Will fall back to CPU inference)")

print(f"\n🔧 TOOLS")
for tool in ['git','cmake','ffmpeg','python3','curl']:
    try:
        path = subprocess.check_output(['which',tool],stderr=subprocess.DEVNULL).decode().strip()
        print(f"   ✅ {tool:12s} → {path}")
    except:
        print(f"   ℹ️ {tool:12s} → Will be installed during dependencies.")

print(f"\n{SEP}")
print("  ✅ CELL 1 COMPLETE — Proceed to Cell 2")
print(f"{SEP}")
""")

# Cell 2: Config
nb['cells'][2]['source'] = cell(r"""
# ════════════════════════════════════════════════════════════
# CELL 2 — CONFIGURATION
# ════════════════════════════════════════════════════════════
import os

SEP = "═" * 62
print(f"\n{SEP}")
print("  CELL 2 — CONFIGURATION")
print(f"{SEP}\n")

# ── EDIT THESE ────────────────────────────────────────────────
DO_INSTALL     = True           # False = skip engine build
MODEL_DIR      = '/content/models/s2-pro-gguf'
OUTPUT_DIR     = '/content/inference_outputs'
S2_BINARY      = '/content/s2.cpp/build/s2'

# Model weights size targeting (Q8_0 recommended for Colab T4)
QUANT_LEVEL    = 's2-pro-q8_0.gguf'

# Device: 0 = GPU Vulkan, -1 = CPU mode
VULKAN_DEVICE  = 0

# ── VOICE CLONING ─────────────────────────────────────────────
PROMPT_TEXT    = 'नमस्कार। मैं हूँ आपका storyteller. आज की कहानी शुरू होती है एक ऐसी जगह से — जहाँ राज़ छुपे हैं, जहाँ सच और झूठ के बीच सिर्फ एक पतली सी लकीर है। सुनिए ध्यान से। क्योंकि यह कहानी सिर्फ सुनने की नहीं — महसूस करने की है.'

# ── GENERATION PARAMS ─────────────────────────────────────────
TEMPERATURE    = 0.7
TOP_P          = 0.8
TOP_K          = 50
MAX_TOKENS     = 2048

# ── LONG TEXT CHUNKING ─────────────────────────────────────────
CHUNK_SIZE     = 100            # Words per chunk (reduced slightly for s2.cpp max safety)
CROSSFADE_MS   = 50             # Crossfade overlap between chunks (ms)
# ──────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('/content/uploads', exist_ok=True)

print(f"\n{'─'*40}")
print(f"   DO_INSTALL     : {DO_INSTALL}")
print(f"   MODEL_DIR      : {MODEL_DIR}/")
print(f"   WEIGHTS        : {QUANT_LEVEL}")
print(f"   VULKAN_DEVICE  : {VULKAN_DEVICE}")
print(f"   TEMPERATURE    : {TEMPERATURE}")
print(f"   MAX_TOKENS     : {MAX_TOKENS}")
print(f"   CHUNK_SIZE     : {CHUNK_SIZE} words")
print(f"{'─'*40}")

TEXT_TO_SYNTH   = "नमस्ते! यह FishAudio S2-Pro का परीक्षण है।"
REFERENCE_AUDIO = ''

print(f"\n{SEP}")
print("  ✅ CELL 2 COMPLETE — Proceed to Cell 3")
print(f"{SEP}")
""")

# Note: Cell 3 and 4 were left mostly intact in previous step except we added the os.makedirs fallback.
# Let's ensure Cell 3 has NO SYNTAX ERRORS from the generic string replacements I previously ran
# We will regenerate it fully from its original clean state but add the fallbacks properly.

nb['cells'][3]['source'] = cell(r"""
# ════════════════════════════════════════════════════════════
# CELL 3 — UPLOAD TEXT FILE  (.txt)
# ════════════════════════════════════════════════════════════
import os
from google.colab import files

os.makedirs('/content/uploads', exist_ok=True)
if 'CHUNK_SIZE' not in locals():
    CHUNK_SIZE = 100

SEP = "═" * 62
print(f"\n{SEP}")
print("  CELL 3 — UPLOAD TEXT FILE")
print(f"{SEP}")
print('''
  Accepted : .txt  (UTF-8 encoding)
  Emotion tags supported:
    [excited]  [whisper]  [pause]   [laugh]   [sad]
    [singing]  [shouting] [emphasis][sigh]    [fast pace]
    [chuckle]  [inhale]   [volume up/down]
  Example:
    [excited] नमस्ते भाई! [pause] आज बहुत मज़ा आएगा।

  Long text? No problem! Auto-chunking handles it.
''')

print("📂 File picker opening — select your .txt file...")
try:
    uploaded_text = files.upload()
except Exception:
    uploaded_text = {}

if uploaded_text:
    fname = list(uploaded_text.keys())[0]
    raw   = uploaded_text[fname]

    dest = os.path.join('/content/uploads', fname)
    with open(dest, 'wb') as f:
        f.write(raw)

    try:
        TEXT_TO_SYNTH = raw.decode('utf-8').strip()
    except UnicodeDecodeError:
        TEXT_TO_SYNTH = raw.decode('utf-8', errors='replace').strip()

    word_count = len(TEXT_TO_SYNTH.split())
    char_count = len(TEXT_TO_SYNTH)
    line_count = TEXT_TO_SYNTH.count('\n') + 1

    print(f"\n✅ FILE LOADED")
    print(f"{'─'*50}")
    print(f"   Filename   : {fname}")
    print(f"   Saved to   : {dest}")
    print(f"   Bytes      : {len(raw):,}")
    print(f"   Characters : {char_count:,}")
    print(f"   Words      : {word_count:,}")
    print(f"   Lines      : {line_count:,}")
    print(f"{'─'*50}")

    est_chunks = max(1, word_count // CHUNK_SIZE + (1 if word_count % CHUNK_SIZE else 0))
    est_min_gpu = word_count / 300  # GPU ~300 words/min
    print(f"   📊 Estimated chunks : {est_chunks} (at {CHUNK_SIZE} words/chunk)")
    print(f"   ⏱  Estimated time   : ~{est_min_gpu:.1f} min on L4 GPU")

    print(f"\n📖 PREVIEW (first 500 chars):")
    print(f"{'─'*50}")
    print(TEXT_TO_SYNTH[:500] + ('...' if char_count > 500 else ''))
    print(f"{'─'*50}")
else:
    if 'TEXT_TO_SYNTH' not in locals():
        TEXT_TO_SYNTH = "नमस्ते! यह FishAudio S2-Pro का परीक्षण है। कृपया इसे हिंदी में बोलिए।"
    print("\n⚠️  No file uploaded — using default Hindi text.")
    print(f"   Text: {TEXT_TO_SYNTH}")

print(f"\n   TEXT_TO_SYNTH set → {len(TEXT_TO_SYNTH)} chars / {len(TEXT_TO_SYNTH.split())} words")
print(f"\n{SEP}")
print("  ✅ CELL 3 COMPLETE — Proceed to Cell 4")
print(f"{SEP}")
""")

nb['cells'][4]['source'] = cell(r"""
# ════════════════════════════════════════════════════════════
# CELL 4 — UPLOAD REFERENCE AUDIO  (OPTIONAL — voice cloning)
# ════════════════════════════════════════════════════════════
import os, json
from google.colab import files

os.makedirs('/content/uploads', exist_ok=True)
if 'PROMPT_TEXT' not in locals():
    PROMPT_TEXT = ''

SEP = "═" * 62
print(f"\n{SEP}")
print("  CELL 4 — UPLOAD REFERENCE AUDIO  (OPTIONAL)")
print(f"{SEP}")
print('''
  ┌─────────────────────────────────────────────────────┐
  │  OPTIONAL — skip this if you want a random voice.  │
  │  Click Cancel or don't upload to use random voice. │
  └─────────────────────────────────────────────────────┘

  If uploading audio:
    1. Upload .wav or .mp3 here
    2. Set PROMPT_TEXT in Cell 2 = exact transcript of clip

  Without PROMPT_TEXT, voice cloning quality drops.
''')

REFERENCE_AUDIO      = ''
REFERENCE_AUDIO_NAME = ''

print("📂 File picker — upload reference audio or cancel to skip...")
try:
    uploaded_audio = files.upload()
except Exception:
    uploaded_audio = {}

if uploaded_audio:
    aname = list(uploaded_audio.keys())[0]
    raw   = uploaded_audio[aname]
    ext   = os.path.splitext(aname)[1].lower()

    SUPPORTED = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
    if ext not in SUPPORTED:
        print(f"\n   ⚠️  Extension '{ext}' not in {SUPPORTED}")
        print(f"   Continuing anyway — ffmpeg may still handle it.")

    _uploads_dir = '/content/uploads'
    dest = os.path.join(_uploads_dir, aname)
    with open(dest, 'wb') as f:
        f.write(raw)

    REFERENCE_AUDIO      = dest
    REFERENCE_AUDIO_NAME = aname
    size_kb = len(raw) / 1024

    print(f"\n✅ REFERENCE AUDIO LOADED")
    print(f"{'─'*50}")
    print(f"   Filename    : {aname}")
    print(f"   Saved to    : {dest}")
    print(f"   Size        : {size_kb:.1f} KB  ({len(raw):,} bytes)")
    print(f"   Format      : {ext}")
    print(f"{'─'*50}")

    try:
        import subprocess as _sp
        r = _sp.run(
            ['ffprobe','-v','quiet','-print_format','json','-show_streams', dest],
            capture_output=True, text=True, timeout=10
        )
        probe = json.loads(r.stdout)
        st = probe['streams'][0]
        duration    = float(st.get('duration', 0))
        sample_rate = st.get('sample_rate', '?')
        channels_n  = st.get('channels', '?')
        codec       = st.get('codec_name', '?')
        print(f"   Duration    : {duration:.2f} s")
        print(f"   Sample rate : {sample_rate} Hz")
        print(f"   Channels    : {channels_n}")
        print(f"   Codec       : {codec}")
        if duration < 2:
            print(f"   ⚠️  Very short (<2s) — quality may suffer.")
        elif duration > 30:
            print(f"   ⚠️  Long clip (>30s) — first ~15s will be used.")
        else:
            print(f"   ✅ Good clip length for voice cloning.")
    except Exception as e:
        print(f"   (Could not probe audio: {e})")

    if not PROMPT_TEXT.strip():
        print(f"\n   ⚠️  PROMPT_TEXT is empty in Cell 2!")
        print(f"   Voice cloning quality is much better with a transcript.")
        print(f"   → Go back to Cell 2 and set PROMPT_TEXT to what is spoken.")
    else:
        excerpt = PROMPT_TEXT[:80] + ('...' if len(PROMPT_TEXT) > 80 else '')
        print(f"\n   PROMPT_TEXT : '{excerpt}'")
        print(f"   ✅ Transcript provided — full voice cloning enabled.")

else:
    REFERENCE_AUDIO = ''
    print("\n✅ No reference audio — will use random built-in voice.")
    print("   S2-Pro's default voices are expressive and multilingual.")

print(f"\n{'─'*50}")
print(f"   REFERENCE_AUDIO = '{REFERENCE_AUDIO}'")
mode = f"VOICE CLONE: {REFERENCE_AUDIO_NAME}" if REFERENCE_AUDIO else "RANDOM VOICE (no reference)"
print(f"   Mode            = {mode}")
print(f"{'─'*50}")

print(f"\n{SEP}")
print("  ✅ CELL 4 COMPLETE — Proceed to Cell 5")
print(f"{SEP}")
""")

# Cell 5: Build s2.cpp
nb['cells'][5]['source'] = cell(r"""
# ════════════════════════════════════════════════════════════
# CELL 5 — BUILD ENGINE  (s2.cpp with Vulkan)
# ════════════════════════════════════════════════════════════
import os, sys, subprocess, time

SEP = "═" * 62
print(f"\n{SEP}")
print("  CELL 5 — BUILDING S2.CPP VULKAN ENGINE")
print(f"{SEP}\n")

if 'DO_INSTALL' not in locals():
    DO_INSTALL = True

if not DO_INSTALL:
    print("   ⏭️  DO_INSTALL = False — skipping.\n")
else:
    t0 = time.time()
    
    print("   📦 Installing system dependencies (cmake, vulkan-tools)...")
    subprocess.run(['apt-get', 'update', '-qq'])
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'cmake', 'libvulkan-dev', 'vulkan-tools', 'ffmpeg', 'build-essential', 'git'], capture_output=True)
    
    REPO_DIR = '/content/s2.cpp'
    if not os.path.exists(os.path.join(REPO_DIR, '.git')):
        print("   📥 Cloning rodrigomatta/s2.cpp...")
        subprocess.run(['git', 'clone', '--recurse-submodules', 'https://github.com/rodrigomatta/s2.cpp.git', REPO_DIR])
    else:
        print("   ✅ Repo already cloned.")
        
    print("   🔨 Compiling s2.cpp with Vulkan support (this takes 1-3 mins)...")
    os.chdir(REPO_DIR)
    subprocess.run(['cmake', '-B', 'build', '-DCMAKE_BUILD_TYPE=Release', '-DS2_VULKAN=ON'])
    r = subprocess.run(['cmake', '--build', 'build', '--parallel', str(os.cpu_count())], capture_output=True, text=True)
    
    if r.returncode == 0 and os.path.exists('./build/s2'):
        print(f"   ✅ Engine built successfully! ({time.time() - t0:.1f}s)")
    else:
        print(f"   ❌ Build failed! Check cmake logs:\n{r.stderr[-500:]}")
        raise RuntimeError("Failed to build s2.cpp renderer.")

    print("   📦 Installing huggingface-cli for model downloads...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'])
    print("   ✅ huggingface_hub ready.")

print(f"\n{SEP}")
print("  ✅ CELL 5 COMPLETE — Proceed to Cell 6")
print(f"{SEP}")
""")

# Cell 6: Download Model
nb['cells'][6]['source'] = cell(r"""
# ════════════════════════════════════════════════════════════
# CELL 6 — DOWNLOAD GGUF WEIGHTS
# ════════════════════════════════════════════════════════════
import os, sys, subprocess, time

SEP = "═" * 62
print(f"\n{SEP}")
print("  CELL 6 — DOWNLOADING MODEL WEIGHTS")
print(f"{SEP}")

t0 = time.time()
print(f"   📥 Downloading {QUANT_LEVEL} and tokenizer.json from rodrigomt/s2-pro-gguf...")
cmd = [
    'huggingface-cli', 'download', 'rodrigomt/s2-pro-gguf',
    QUANT_LEVEL, 'tokenizer.json',
    '--local-dir', MODEL_DIR
]
proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.returncode == 0:
    print(f"   ✅ Download complete/verified ({time.time() - t0:.1f}s)")
else:
    print(f"   ❌ Download failed:\n{proc.stderr}")
    raise RuntimeError("Failed to download GGUF model.")

for f in [QUANT_LEVEL, 'tokenizer.json']:
    size = os.path.getsize(os.path.join(MODEL_DIR, f)) / 1e6
    print(f"   📄 {f}: {size:.1f} MB")

print(f"\n{SEP}")
print("  ✅ CELL 6 COMPLETE — Proceed to Cell 7")
print(f"{SEP}")
""")

# Cell 7: Inference
nb['cells'][7]['source'] = cell(r"""
# ════════════════════════════════════════════════════════════
# CELL 7 — S2 SYNTHESIS  (Chunked CLI Pipeline)
# ════════════════════════════════════════════════════════════
import os, sys, time, gc, re, glob, shlex, subprocess
import html as _html
import numpy as np
from IPython.display import display, HTML, clear_output

SEP = "═" * 62
print(f"\n{SEP}")
print("  CELL 7 — S2 SYNTHESIS  (VULKAN GPU · CHUNKED)")
print(f"{SEP}\n")

if not os.path.exists(S2_BINARY):
    raise FileNotFoundError(f"s2 binary not found at {S2_BINARY}. Please run Cell 5.")
if not os.path.exists(os.path.join(MODEL_DIR, QUANT_LEVEL)):
    raise FileNotFoundError(f"{QUANT_LEVEL} not found. Please run Cell 6.")

for f in glob.glob(os.path.join(OUTPUT_DIR, 'chunk_*.wav')):
    os.remove(f)
for f in glob.glob(os.path.join(OUTPUT_DIR, 'output_s2pro.wav')):
    os.remove(f)

def chunk_text(text, max_words=100):
    sentences = re.split(r'(?<=[।.!?\n])\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks, current, current_wc = [], [], 0
    for sent in sentences:
        sent_wc = len(sent.split())
        if sent_wc > max_words:
            if current: chunks.append(' '.join(current)); current, current_wc = [], 0
            words = sent.split()
            for i in range(0, len(words), max_words): chunks.append(' '.join(words[i:i+max_words]))
            continue
        if current_wc + sent_wc > max_words and current:
            chunks.append(' '.join(current)); current = [sent]; current_wc = sent_wc
        else:
            current.append(sent); current_wc += sent_wc
    if current: chunks.append(' '.join(current))
    return chunks if chunks else [text.strip()]

text_clean = TEXT_TO_SYNTH.replace('\r\n', '\n').strip()
total_words = len(text_clean.split())
text_chunks = chunk_text(text_clean, max_words=CHUNK_SIZE)
num_chunks = len(text_chunks)

print(f"📄 Text split into {num_chunks} chunk(s)  ({total_words} total words)")
for ci, ch in enumerate(text_chunks):
    preview = ch[:60].replace('\n',' ') + ('...' if len(ch) > 60 else '')
    print(f"   Chunk {ci+1}: {len(ch.split())} words — {preview}")

t0 = time.time()
chunk_wavs = []

for ci, chunk_text in enumerate(text_chunks):
    chunk_num = ci + 1
    chunk_t0 = time.time()
    chunk_wav_path = os.path.join(OUTPUT_DIR, f'chunk_{chunk_num:04d}.wav')
    
    print(f"\n⏳ Synthesizing chunk {chunk_num}/{num_chunks}...")
    cmd = [
        S2_BINARY,
        '-m', os.path.join(MODEL_DIR, QUANT_LEVEL),
        '-t', os.path.join(MODEL_DIR, 'tokenizer.json'),
        '-v', str(VULKAN_DEVICE),
        '-max-tokens', str(MAX_TOKENS),
        '-temp', str(TEMPERATURE),
        '-top-p', str(TOP_P),
        '-top-k', str(TOP_K),
        '-o', chunk_wav_path,
        '-text', chunk_text.replace('\n', ' ')
    ]
    if REFERENCE_AUDIO and os.path.exists(REFERENCE_AUDIO):
        cmd.extend(['-pa', REFERENCE_AUDIO])
        if PROMPT_TEXT.strip():
            cmd.extend(['-pt', PROMPT_TEXT.replace('\n', ' ')])
            
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    if proc.returncode == 0 and os.path.exists(chunk_wav_path):
        print(f"   ✅ Chunk {chunk_num} complete ({time.time() - chunk_t0:.1f}s)")
        chunk_wavs.append(chunk_wav_path)
    else:
        print(f"   ❌ Chunk {chunk_num} failed!")
        print(proc.stderr)
        raise RuntimeError(f"s2.cpp inference failed on chunk {chunk_num}")

print(f"\n{SEP}")
print(f"  ✅ ALL {num_chunks} CHUNKS SYNTHESIZED")

if num_chunks == 1:
    import shutil
    OUTPUT_WAV_PATH = os.path.join(OUTPUT_DIR, 'output_s2pro.wav')
    shutil.move(chunk_wavs[0], OUTPUT_WAV_PATH)
else:
    import soundfile as sf
    import resampy
    
    all_audio = []
    target_sr = None
    for wpath in chunk_wavs:
        data, sr = sf.read(wpath)
        if data.ndim > 1:
            data = data[:, 0]  # mono
        if target_sr is None: 
            target_sr = sr
        elif sr != target_sr: 
            data = resampy.resample(data, sr, target_sr)
        all_audio.append(data)
        
    crossfade_samples = int(target_sr * CROSSFADE_MS / 1000)
    if crossfade_samples < 1: crossfade_samples = 1
    
    result = all_audio[0]
    for i in range(1, len(all_audio)):
        nxt = all_audio[i]
        if len(result) >= crossfade_samples and len(nxt) >= crossfade_samples:
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in  = np.linspace(0, 1, crossfade_samples)
            overlap = result[-crossfade_samples:] * fade_out + nxt[:crossfade_samples] * fade_in
            result = np.concatenate([result[:-crossfade_samples], overlap, nxt[crossfade_samples:]])
        else:
            result = np.concatenate([result, nxt])

    OUTPUT_WAV_PATH = os.path.join(OUTPUT_DIR, 'output_s2pro.wav')
    sf.write(OUTPUT_WAV_PATH, result, target_sr)

print(f"  🎉 Final concatenated output: {OUTPUT_WAV_PATH}")
print(f"  Total time: {time.time() - t0:.1f}s")
print(f"{SEP}\n")
""")

# Cell 8: Play and Download (Unchanged mainly, let's keep it pristine as it was to avoid issues, but inject it cleanly)
nb['cells'][8]['source'] = cell(r"""
# ════════════════════════════════════════════════════════════
# CELL 8 — PLAY & DOWNLOAD OUTPUT AUDIO
# ════════════════════════════════════════════════════════════
import os, glob
from IPython.display import Audio, display, HTML
from google.colab import files

SEP = "═" * 62
print(f"\n{SEP}")
print("  CELL 8 — PLAY & DOWNLOAD OUTPUT")
print(f"{SEP}\n")

# Find output wav
if 'OUTPUT_WAV_PATH' not in dir() or not OUTPUT_WAV_PATH:
    cands = (sorted(glob.glob('/content/inference_outputs/output_s2pro.wav'))
           + sorted(glob.glob('/content/inference_outputs/*.wav')))
    if cands:
        OUTPUT_WAV_PATH = cands[0]
        print(f"   Found: {OUTPUT_WAV_PATH}")
    else:
        raise FileNotFoundError("No output WAV found. Run Cell 7 first.")

if not os.path.exists(OUTPUT_WAV_PATH):
    raise FileNotFoundError(f"WAV not found: {OUTPUT_WAV_PATH}")

wav_mb = os.path.getsize(OUTPUT_WAV_PATH) / 1e6

try:
    import soundfile as sf
    data, sr = sf.read(OUTPUT_WAV_PATH)
    duration = len(data) / sr
    channels = 'Stereo' if len(data.shape) > 1 else 'Mono'
except Exception as e:
    data, sr, duration, channels = None, 44100, 0, 'Unknown'
    print(f"   ⚠️  Could not read audio metadata: {e}")

print(f"  📄 File     : {OUTPUT_WAV_PATH}")
print(f"  📦 Size     : {wav_mb:.2f} MB")
print(f"  ⏱  Duration : {duration:.2f}s  ({duration/60:.2f} min)")
print(f"  🎵 Rate     : {sr} Hz  |  {channels}\n")

print("\n🔊 Audio player (press ▶ to listen):")
display(HTML('''
<div style="background:#080c18;border:1px solid #00d4ff;border-radius:6px;
   padding:10px 14px;font-family:monospace;color:#00d4ff;margin:6px 0;
   display:inline-block">
  ▶ Press play to listen to your S2-Pro generated audio
</div>'''))
display(Audio(OUTPUT_WAV_PATH, autoplay=False))

print("\n💾 Downloading to your local machine...")
files.download(OUTPUT_WAV_PATH)
""")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook strings rebuilt completely.")
