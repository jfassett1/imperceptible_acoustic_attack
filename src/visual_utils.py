from whisper import log_mel_spectrogram
import matplotlib.pyplot as plt
import numpy as np
import torch
import librosa
from pathlib import Path


def overlay(noise, raw_audio):
    if noise.dim() == 1:
        noise = noise.unsqueeze(0)
    if raw_audio.dim() == 1:
        raw_audio = raw_audio.unsqueeze(0)
    padding = torch.zeros_like(raw_audio[:, :-noise.shape[-1]])
    noise_padded = torch.cat([noise, padding], dim=-1)
    return noise_padded + raw_audio

def audio_to_img(noise, #Learned noise
                 audio_list, #Sample noises to overlay over audio
                 raw_sample, #Raw audio sample to overlay audio over
                 save_dir: Path, #Directory for saving resultant images
                 sampling_rate=16000):
    """
    Takes in audio
    Returns list of paths to images
    """
    paths = []
    audio_list.insert(0,noise)
    for i,audio in enumerate(audio_list):

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio = audio.astype(np.float32)

        hop_length = 160
        mel_spec1 = log_mel_spectrogram(audio).numpy().squeeze()

        plt.figure(figsize=(16, 5))  # Create a figure directly
        librosa.display.specshow(mel_spec1, sr=sampling_rate, hop_length=hop_length, 
                                x_axis='time', y_axis='mel', shading="gouraud", cmap="magma")
        plt.title(f"Unperturbed Audio ({audio.shape[1]/sampling_rate:.2f} seconds)")
        plt.colorbar()
        image_path = save_dir / f"{i}.png"
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        paths.append(save_dir / f"{i}.png")
    # plt.colorbar(format="%+2.0f dB")
    return paths  # Return the current figure


if __name__ == "__main__":
    print("")