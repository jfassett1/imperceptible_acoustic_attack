import whisper
import matplotlib.pyplot as plt
from scipy.io import wavfile
from whisper import log_mel_spectrogram
import numpy as np
import librosa
sample_raw = whisper.load_audio("/home/jaydenfassett/audioversarial/imperceptible/original_audio.wav")
def raw_to_mel(x,device="cpu"):
    x = whisper.pad_or_trim(x)
    mel = whisper.log_mel_spectrogram(x).to(device).unsqueeze(dim=0)
    return mel

def save_photo_overlay(noise, audio, savepath):
    """
    Save an overlay of the noise and audio spectrogram as an image.

    Parameters:
    noise (np.ndarray): The noise array.
    audio (np.ndarray): The audio array.
    savepath (str): Path where the image will be saved.
    """
    # Ensure noise and audio have the same length

    noise = np.concatenate([noise.squeeze(), np.zeros(len(audio) - len(noise))])

    # Combine noise and audio
    combined_audio = noise + audio

    hop_length = 160
    mel_spec1 = log_mel_spectrogram(audio.astype(np.float32)).numpy()
    mel_spec2 = log_mel_spectrogram(combined_audio.astype(np.float32)).numpy()

    # mel_spectrogram1 = librosa.feature.melspectrogram(y=aud1, sr=fs1, n_mels=80, hop_length=hop_length)
    # mel_spectrogram_db1 = librosa.power_to_db(mel_spectrogram1, ref=np.max)

    # mel_spectrogram2 = librosa.feature.melspectrogram(y=aud2, sr=fs1, n_mels=80, hop_length=hop_length)
    # mel_spectrogram_db2 = librosa.power_to_db(mel_spectrogram2, ref=np.max)

    fig,axes = plt.subplots(1,2,figsize = (12,5))

    librosa.display.specshow(mel_spec1, sr=16000, hop_length=hop_length, 
                            x_axis='time', y_axis='mel', ax=axes[0],shading="gouraud",cmap="magma")
    
    librosa.display.specshow(mel_spec2, sr=16000, hop_length=hop_length, 
                            x_axis='time', y_axis='mel', ax=axes[1],shading="gouraud",cmap="magma")
    axes[0].set_title(f"Unperturbed Audio ({len(combined_audio)/160000:.2f} seconds)")
    axes[1].set_title(f"Perturbed Audio ({len(combined_audio)/160000:.2f} seconds)")
    # Save the figure
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close(fig)



sample_mel = raw_to_mel(sample_raw)

def contrast(aud1,aud2):

    fs1, aud1 = aud1
    fs2, aud2 = aud2
    aud1 = aud1.astype(np.float32)
    aud2 = aud2.astype(np.float32)

    hop_length = 160
    mel_spec1 = log_mel_spectrogram(aud1).numpy()
    mel_spec2 = log_mel_spectrogram(aud2).numpy()
    # mel_spectrogram1 = librosa.feature.melspectrogram(y=aud1, sr=fs1, n_mels=80, hop_length=hop_length)
    # mel_spectrogram_db1 = librosa.power_to_db(mel_spectrogram1, ref=np.max)

    # mel_spectrogram2 = librosa.feature.melspectrogram(y=aud2, sr=fs1, n_mels=80, hop_length=hop_length)
    # mel_spectrogram_db2 = librosa.power_to_db(mel_spectrogram2, ref=np.max)

    fig,axes = plt.subplots(1,2,figsize = (12,5))

    librosa.display.specshow(mel_spec1, sr=fs1, hop_length=hop_length, 
                            x_axis='time', y_axis='mel', ax=axes[0],shading="gouraud",cmap="magma")
    
    librosa.display.specshow(mel_spec2, sr=fs2, hop_length=hop_length, 
                            x_axis='time', y_axis='mel', ax=axes[1],shading="gouraud",cmap="magma")
    axes[0].set_title(f"Unperturbed Audio ({len(aud1)/fs1:.2f} seconds)")
    axes[1].set_title(f"Perturbed Audio ({len(aud2)/fs2:.2f} seconds)")
    return fig


if __name__ == "__main__":
    aud = wavfile.read("/home/jaydenfassett/audioversarial/imperceptible/original_audio.wav")

    noise = np.random.randn(16000)