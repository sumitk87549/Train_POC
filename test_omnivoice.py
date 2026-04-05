import time
import sys
import torch
import torchaudio

print("Importing model...")
start = time.time()
try:
    from omnivoice import OmniVoice
except ImportError:
    print("omnivoice not installed")
    sys.exit(1)
print(f"Import took {time.time() - start:.2f}s")

start = time.time()
print("Loading model on CPU with float32...")
try:
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", 
        device_map="cpu", 
        dtype=torch.float32
    )
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)
print(f"Loading took {time.time() - start:.2f}s")

# Create a small dummy reference audio
dummy_ref = torch.randn(1, 48000) # 2 seconds at 24kHz
torchaudio.save("dummy_ref.wav", dummy_ref, 24000)

print("Starting generation test (short sentence)...")
start = time.time()
try:
    audio = model.generate(
        text="यह एक परीक्षण है। OmniVoice हिंदी में कैसा काम करता है, हमें यह देखना है।",
        ref_audio="dummy_ref.wav",
        ref_text="This is a test reference text for the dummy audio used in cloning.",
    )
except Exception as e:
    print(f"Failed to generate: {e}")
    sys.exit(1)
gen_time = time.time() - start
print(f"Generation took {gen_time:.2f}s")

torchaudio.save("out.wav", audio[0], 24000)
print("Saved to out.wav")
