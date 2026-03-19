import torchaudio

def prepare_reference(input_path, output_path="ref.wav"):
    wav, sr = torchaudio.load(input_path)

    # convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # resample to 24kHz
    wav = torchaudio.transforms.Resample(sr, 24000)(wav)

    torchaudio.save(output_path, wav, 24000)

    return output_path

ref_path = prepare_reference("./Fiction_Hello…_this.mp3")