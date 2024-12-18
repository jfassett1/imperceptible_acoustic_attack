import whisper
import matplotlib.pyplot as plt
from scipy.io import wavfile
from whisper import log_mel_spectrogram
import numpy as np
import librosa
import torch
def raw_to_mel(x,device="cpu"):
    x = whisper.pad_or_trim(x)
    mel = whisper.log_mel_spectrogram(x).to(device).unsqueeze(dim=0)
    return mel

def overlay_np(noise, raw_audio):
    # Ensure noise and raw_audio have at least 2 dimensions
    noise = np.atleast_2d(noise)
    raw_audio = np.atleast_2d(raw_audio)

    # Trim or pad noise to match raw_audio's length
    if noise.shape[-1] != raw_audio.shape[-1]:
        noise = noise[:, :raw_audio.shape[-1]]
        noise = np.pad(noise, ((0, 0), (0, raw_audio.shape[-1] - noise.shape[-1])))

    # Broadcast noise to match the batch size of raw_audio
    noise = np.broadcast_to(noise, raw_audio.shape)

    # Add noise to raw_audio
    result = noise + raw_audio

    return result.squeeze(0) if result.shape[0] == 1 else result

def overlay_torch(noise, raw_audio):
    # Ensure noise and raw_audio are tensors with at least 2 dimensions
    noise = torch.atleast_2d(noise)
    raw_audio = torch.atleast_2d(raw_audio)

    # Ensure noise and raw_audio have the same length in the last dimension
    if noise.shape[-1] != raw_audio.shape[-1]:
        if noise.shape[-1] > raw_audio.shape[-1]:
            noise = noise[:, :raw_audio.shape[-1]]
        else:
            pad_length = raw_audio.shape[-1] - noise.shape[-1]
            noise = torch.nn.functional.pad(noise, (0, pad_length))

    # Broadcast noise to match the batch size of raw_audio
    if noise.shape[0] == 1:
        noise = noise.expand_as(raw_audio)

    # Add noise to raw_audio
    result = noise + raw_audio

    # Return the result, squeeze if batch size is 1
    return result.squeeze(0) if result.shape[0] == 1 else result

if __name__ == "__main__":
    aud = wavfile.read("/home/jaydenfassett/audioversarial/imperceptible/original_audio.wav")[1]
    aud = aud[np.newaxis,:]
    aud = np.repeat(aud,16,axis=0)
    noise = np.random.randn(16000)
    # raw_audio = np.random.rand(4, 16000)
    qq = overlay_np(noise,aud).shape
    print(qq)