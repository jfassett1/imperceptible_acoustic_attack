import whisper



sample_raw = whisper.load_audio("/home/jaydenfassett/audioversarial/imperceptible/original_audio.wav")
def raw_to_mel(x,device="cpu"):
    x = whisper.pad_or_trim(x)
    mel = whisper.log_mel_spectrogram(x).to(device).unsqueeze(dim=0)
    return mel

sample_mel = raw_to_mel(sample_raw)