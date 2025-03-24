from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from whisper import load_audio, pad_or_trim
from whisper.audio import HOP_LENGTH, N_FFT, mel_filters

from src.data import AudioDataModule
from src.visual_utils import show_mel_spectrograms

DEVICE = "cuda"
def estimate_mel_T(audio_len, hop_length=HOP_LENGTH, n_fft=N_FFT):
    return 1 + (audio_len - n_fft) // hop_length


def plotnshow(q1, q2):
    plt.figure()
    plt.plot(q1, label="Samp Strength Vals")
    plt.plot(q2, label="Quiet Vals")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/jaydenfassett/audioversarial/imperceptible/quiets.png")


N_MELS = 80  # Whisper default setting
SAMPLING_RATE = 16000


def compute_mel_bin_centers(mel_filter_bank, sr, n_fft):
    """
    Computes frequencies of each mel bin.
    """
    # mel_filter_bank: numpy array or torch tensor of shape (n_mels, n_fft//2 + 1)
    fft_freqs = np.linspace(0, sr / 2, int(n_fft / 2) + 1)
    mel_filter_bank_np = (
        mel_filter_bank.cpu().numpy()
        if torch.is_tensor(mel_filter_bank)
        else mel_filter_bank
    )
    centers = np.array(
        [
            (filter_weights * fft_freqs).sum() / (filter_weights.sum() + 1e-10)
            for filter_weights in mel_filter_bank_np
        ]
    )
    return centers


mel_filter_bank = mel_filters("cpu", N_MELS)
mel_frequencies = torch.from_numpy(
    compute_mel_bin_centers(mel_filter_bank, SAMPLING_RATE, N_FFT)
)


def log_mel_spectrogram_raw(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Modified log_mel_spectrogram function from whisper.audio
    It is un-normalized.
    This is essential for maintaining the same units for use in the mel spectrogram mask.

    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    return log_spec * 10


def mask_conv(mel_col):
    """
    Does a 2nd derivative convolution, and removes all negative values

    """
    frames = mel_col.shape[-1]
    # print(mel_col.shape)
    # exit()
    smoothed = moving_average(mel_col, groups=frames)  # Smooths first
    # exit()
    kernel = torch.tensor([1, -16, 30, -16, 1], dtype=torch.float32, device=DEVICE) / 12.0
    kernel = kernel.view(1, 1, -1)
    if frames > 1:
        kernel = kernel.expand(frames, -1, -1)

    conved = F.conv1d(smoothed, kernel, padding=2, groups=frames).permute((0, 2, 1))
    # print(conved.shape)
    # exit()
    return F.relu(conved)


def moving_average(x, window_size=3, groups=1):
    kernel = torch.ones(window_size, device=DEVICE) / window_size
    kernel = kernel.view(1, 1, window_size)

    if groups > 1:
        kernel = kernel.expand(groups, -1, -1)
    padding = (window_size - 1) // 2
    x = x.permute(0, 2, 1)
    return F.conv1d(x, kernel, padding=padding, groups=groups)


def quiet(f):
    """returns threshold in quiet measured in SPL at frequency f with an offset 12(in Hz)"""
    thresh = -6.5 * torch.exp(-0.6 * torch.pow(0.001 * f - 3.3, 2))
    return thresh


def db_to_whisper_unit(db: torch.Tensor):
    """
    Convert a tensor of threshold dB values into normalized log10 units.
    """
    # Divide by 10 to get log10-power units.
    log10_power = db / 10.0
    return log10_power


def generate_mel_th(samp_mel: torch.tensor, lengths) -> torch.tensor:
    """
    Generates 1d frequency mask given a sample

    """
    assert len(samp_mel.shape) >= 3, "Expecting log mel spectrogram size (N, 80, T)"
    batch_size, frequencies, frames = samp_mel.shape
    lengths = estimate_mel_T(lengths).int()

    samp_mel = samp_mel.clone()
    conv = mask_conv(samp_mel)  # Power values at each frequency

    quiets = quiet(mel_frequencies).to(DEVICE) # Calculate the quiet values at all mel frequencies
    quiets = quiets.expand(batch_size, -1)
    # Normalize quiets
    ref_freq = torch.argmin(
        torch.abs(mel_frequencies - 1000)
    )  # Finding closest bin to 1000hz
    # ref_vals = samp_mel[:, ref_freq, :]  # [B, T]
    # mask = torch.arange(ref_vals.size(1), device=DEVICE).unsqueeze(0) < lengths.unsqueeze(1)  # [B, T]
    # ref_db = (ref_vals * mask).sum(dim=1) / lengths

    # ref_db = (ref_freqs.sum(dim=-1)) / lengths # Get average using the true lengths
    # print(samp_mel[:,ref_freq,:])
    ref_db = torch.zeros_like(lengths, dtype=float, device=DEVICE)

    for i, (samp, rel_length) in enumerate(
        zip(samp_mel[:, ref_freq, :], lengths)
    ):  # Find the average value at 1000hz bin. Exclude the padded values
        ref_db[i] = samp[:rel_length].mean()

    # print(lengths.shape)
    # print("fake",ref_db)
    diff = quiets[:, ref_freq] - ref_db

    quiets = quiets - diff.view(-1, 1)  # Aligning quiets with sample
    quiets = quiets.unsqueeze(-1).expand(
        batch_size, frequencies, frames
    )  # Expand across time dim

    conv[samp_mel < quiets] = 0
    samp_mel[conv == 0] = (
        torch.inf
    )  # Set all zero vals to infinite (will always be louder than perturbation)
    return samp_mel


if __name__ == "__main__":
    output = "/home/jaydenfassett/audioversarial/imperceptible/src/masking/vis2.png"

    # print(neighborhood_size(mel_frequencies))
    qr = AudioDataModule("tedlium:", batch_size=15)
    samp = pad_or_trim(qr.sample[0])
    dl = next(iter(qr.random_all_dataloader()))
    tests = dl[0].to(DEVICE)
    lengths = dl[-1].to(DEVICE)
    samp_mel = log_mel_spectrogram_raw(tests)
    threshold = generate_mel_th(samp_mel, lengths)

    # Vis code
    q = []
    # print(dl[2][0])
    for i in range(5):
        # print(tests[i].shape)
        mel_len = int(estimate_mel_T(lengths[i]))
        samp = log_mel_spectrogram_raw(pad_or_trim(tests[i]))
        # print(mel_len,samp.shape)
        samp = samp[:, :mel_len]
        q.append(samp)

    # print(samp_mel_len)
    # print(samp_mel.mean(dim=-1) - threshold)
    # print(threshold)
    fig = show_mel_spectrograms(q)
    # fig = show_mel_spectrograms([samp_mel.squeeze(),samp_mel.squeeze(0),conv.squeeze(0)],titles=["Audio Spectrogram","Mean Strengths Visualization","Local Maxima strengths"])
    # fig.savefig(output)
