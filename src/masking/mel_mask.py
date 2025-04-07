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


def plotnshow(q1, q2,q3):
    plt.figure()
    plt.plot(q1, label="Samp Strength Vals")
    plt.plot(q2, label="Updated threshold")
    plt.plot(q3,label="ATH Threshold")
    # plt.plot(np.random.randn(80),label="Random noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/jaydenfassett/audioversarial/imperceptible/quiets.png")

# def plotnshow(q1, q2, q3):
#     plt.figure()
#     plt.plot(q1, label="Samp Strength Vals")
#     plt.plot(q2, label="ATH Threshold")

#     # Highlight the area between q1 and q2
#     plt.fill_between(
#         np.arange(len(q1)), 
#         q1, 
#         q2, 
#         where=(q1 > q2), 
#         interpolate=True, 
#         alpha=0.3, 
#         color='red', 
#         label="Above Threshold"
#     )
#     # plt.fill_between(
#     #     np.arange(len(q1)), 
#     #     q1, 
#     #     q2, 
#     #     where=(q1 <= q2), 
#     #     interpolate=True, 
#     #     alpha=0.3, 
#     #     color='blue', 
#     #     label="Below Threshold"
#     # )

#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("/home/jaydenfassett/audioversarial/imperceptible/quiets.png")
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
    """
    compute the absolute threshold of hearing (ATH) in dB SPL for frequencies f in Hz.
    """
    f_kHz = f * 0.001
    term1 = 3.64 * torch.pow(f_kHz, -0.8)
    term2 = -6.5 * torch.exp(-0.6 * torch.pow(f_kHz - 3.3, 2))
    term3 = 0.001 * torch.pow(f_kHz, 4)
    return term1 + term2 + term3

def Bark(f):
    """Returns the bark-scale value for input frequency f (in Hz)"""
    return 13 * torch.atan(0.00076 * f) + 3.5 * torch.atan((f / 7500.0) ** 2)


def db_to_whisper_unit(db: torch.Tensor):
    """
    Convert a tensor of threshold dB values into normalized log10 units.
    """
    # Divide by 10 to get log10-power units.
    log10_power = db / 10.0
    return log10_power

def bin_indices(x: torch.Tensor):
    """Returns the first index where each unique value in sorted 1D tensor `x` appears."""
    change = torch.cat([torch.tensor([True], device=x.device), x[1:] != x[:-1]])
    return torch.nonzero(change, as_tuple=False).flatten()

def generate_mel_th(samp_mel: torch.tensor, lengths) -> torch.tensor:
    """
    Generates a 1D frequency mask given a sample using a two-slope spread function.
    The approach is similar to the two_slops function but adjusted so that the threshold
    is only raised (never lowered).

    Assumptions:
      - samp_mel has shape (N, 80, T) (or similar)
      - mel_frequencies, DEVICE, estimate_mel_T, mask_conv, quiet, Bark, and bin_indices
        are defined elsewhere.
    """
    assert len(samp_mel.shape) >= 3, "Expecting log mel spectrogram size (N, 80, T)"
    batch_size, frequencies, frames = samp_mel.shape

    lengths = estimate_mel_T(lengths).int()

    samp_mel = samp_mel.clone()
    # conv = mask_conv(samp_mel)  # Power values at each frequency

    mel_max = samp_mel.max()

    samp_mel = samp_mel +  (56.6 - mel_max)

    quiets = quiet(mel_frequencies).to(DEVICE)  # Quiet threshold for each mel frequency
    quiets = quiets.view(1, -1, 1).expand(batch_size, -1, frames)
    # quiets = quiets - 56.6  # empirically chosen offset

    # Compute bark scale values from the mel frequencies (assumed monotonic)
    barks = Bark(mel_frequencies).to(DEVICE)

    # Use the bark values to form bins (each bin corresponding roughly to a critical band)
    bark_bins = barks.floor().to(DEVICE, dtype=torch.int64)
    bin_idx = bin_indices(bark_bins)

    # Start with the quiet threshold and update with masker-based thresholds
    threshold = quiets.clone()

    # Process each bark bin (frequency chunk)
    for i in range(len(bin_idx) - 1):
        start, stop = bin_idx[i], bin_idx[i + 1]
        chunk = samp_mel[:, start:stop, :]

        max_val, max_idx = chunk.max(dim=1, keepdim=True)  # shape: (batch_size, 1, frames)

        # Get the bark values corresponding to the current chunk frequencies.
        bark_chunk = barks[start:stop].view(1, -1, 1).expand(batch_size, -1, frames)
        # Obtain the bark value of the masker by gathering using max_idx (relative index in the chunk)
        bark_masker = bark_chunk.gather(dim=1, index=max_idx)

        # Compute the difference (dz) in the bark scale between each frequency in the chunk and the masker.
        # This will be negative for frequencies below the masker and positive for frequencies above.
        dz = bark_chunk - bark_masker

        # Initialize the spread function (sf) tensor.
        sf = torch.zeros_like(dz)

        # Create a tensor of indices for the chunk frequency bins to identify positions relative to the masker.
        freq_indices = torch.arange(chunk.shape[1], device=chunk.device).view(1, -1, 1).expand_as(chunk)

        # Define masks: for frequencies below the masker index and above it.
        mask_lower = freq_indices < max_idx
        mask_upper = freq_indices > max_idx

        # For frequencies below the masker: use a positive slope with the absolute difference.
        G = 5
        sf[mask_lower] = G * torch.abs(dz[mask_lower])
        
        # For frequencies above the masker: compute a level-dependent slope.
        # The original two_slops used: (-27 + 0.37 * max(masker_level-40,0)) * dz,
        upper_slope = G - 0.37 * torch.clamp(max_val - 40, min=0)
        upper_slope_expanded = upper_slope.expand_as(dz)
        sf[mask_upper] = upper_slope_expanded[mask_upper] * dz[mask_upper]

        spread = max_val + sf

        threshold[:, start:stop, :] = torch.maximum(threshold[:, start:stop, :], spread)

    # plotnshow(samp_mel.cpu().numpy()[0,:,3],threshold[0,:,3].cpu().numpy(),quiets[0,:,3].cpu().numpy())
    # exit()
    # exit()
    return threshold

if __name__ == "__main__":

    output = "/home/jaydenfassett/audioversarial/imperceptible/src/masking/vis2.png"
    from whisper import log_mel_spectrogram
    # print(neighborhood_size(mel_frequencies))
    qr = AudioDataModule("librispeech:clean-100", batch_size=2)
    samp = pad_or_trim(qr.sample[0])
    dl = next(iter(qr.random_all_dataloader()))
    tests = dl[0].to(DEVICE)
    lengths = dl[-1].to(DEVICE)
    # tests -= 0.0709
    samp_mel = log_mel_spectrogram_raw(tests)
    threshold = generate_mel_th(samp_mel, lengths)
    random_mel = log_mel_spectrogram_raw(torch.randn(1,480000).clamp_(max=0.02,min=-0.02))



    sample_idx = 0
    frame_idx = 5 
    plt.plot(threshold[sample_idx, :, frame_idx].cpu().numpy(), label="Masking Threshold")
    # plt.plot(random_mel[sample_idx, :, frame_idx].cpu().numpy(), '--', label="Actual Audio Threshold")

    # plt.plot(quiets[sample_idx, :, frame_idx].cpu().numpy(), '--', label="Quiet Threshold")
    plt.xlabel("Mel/Frequency Bin")
    plt.ylabel("dB")
    plt.title("Threshold Curve at Frame 100")
    plt.grid()
    plt.legend()
    plt.savefig("/home/jaydenfassett/audioversarial/imperceptible/quiets.png")
    plt.show()


    exit()
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
    fig.savefig(output)
