from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from whisper import load_audio, pad_or_trim
from whisper.audio import HOP_LENGTH, N_FFT, mel_filters

from src.data import AudioDataModule
# from src.visual_utils import show_mel_spectrograms

DEVICE = "cuda"
def estimate_mel_T(audio_len, hop_length=HOP_LENGTH, n_fft=N_FFT):
    return 1 + (audio_len - n_fft) // hop_length


def plotnshow(q1, q2=None, q3=None, local_max=None):
    plt.figure()
    x = np.arange(len(q1))
    plt.plot(x, q1, label="Original Sample Frequency Vals")
    if q2 is not None:
        plt.plot(q2, color="red", label="Threshold")
    if local_max is not None:
        # If local_max is a boolean mask, get indices; otherwise assume it's a list of indices.
        if local_max.dtype == bool:
            indices = np.nonzero(local_max)[0]
        else:
            indices = np.array(local_max)
        plt.scatter(indices, np.array(q1)[indices], color="green", marker="x", s=100, label="Local Maxima")
    if q3 is not None:
        plt.plot(x, q3, label="ATH Threshold")
    plt.legend()
    plt.xlabel("Mel Frequency Bins")
    plt.ylabel("Audio in DB SPL")
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
    It is not in db and is not normalized.
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
    # log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # return log_spec * 10
    return mel_spec


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
    return F.conv1d(x, kernel, padding=padding, groups=groups).permute((0, 2, 1))


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




# Display results
# for i, group in enumerate(neighborhoods):
#     group_vals = [values[j] for j in group]
#     print(f"Value {i} ({values[i]:.4f}) -> Indices: {group}")

def bin_indices(x: torch.Tensor):
    """Returns the first index where each unique value in sorted 1D tensor `x` appears."""
    change = torch.cat([torch.tensor([True], device=x.device), x[1:] != x[:-1]])
    return torch.nonzero(change, as_tuple=False).flatten()
def two_slops(chunk, bark_values, max_val, max_idx, G=5):
    """
    Computes the spread function given a chunk of the spectrum and its associated bark values.


    """
    batch_size, num_freqs, frames = chunk.shape

    # Expand bark_values to match the chunk's dimensions.
    bark_chunk = bark_values.view(1, -1, 1).expand(batch_size, -1, frames)
    bark_masker = bark_chunk.gather(dim=1, index=max_idx)

    # Compute difference on bark scale.
    dz = bark_chunk - bark_masker
    sf = torch.zeros_like(dz)

    # Create frequency indices tensor.
    freq_indices = torch.arange(num_freqs, device=chunk.device).view(1, -1, 1).expand_as(chunk)

    # Masks for frequencies lower and higher than the masker.
    mask_lower = freq_indices < max_idx
    mask_upper = freq_indices > max_idx

    # For frequencies below the masker index: use a positive slope with the absolute difference.
    sf[mask_lower] = G * torch.abs(dz[mask_lower])
    
    # For frequencies above: compute level-dependent slope.
    upper_slope = G - 0.37 * torch.clamp(max_val - 40, min=0)
    upper_slope_expanded = upper_slope.expand_as(dz)
    sf[mask_upper] = upper_slope_expanded[mask_upper] * dz[mask_upper]

    spread = max_val + sf
    return spread
def generate_mel_th(samp_mel: torch.tensor, lengths, method = "groups") -> torch.tensor:
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
    smoothed = moving_average(samp_mel, groups=frames)
    samp_mel = torch.clamp(smoothed, min=1e-10).log10() * 10 #convert to DB
    mel_max = samp_mel.max()

    # OFFSET = 56.6
    OFFSET = 60
    samp_mel = samp_mel +  (OFFSET - mel_max)

    quiets = quiet(mel_frequencies).to(DEVICE)  # Quiet threshold for each mel frequency
    quiets = quiets.view(1, -1, 1).expand(batch_size, -1, frames)

    max_tensor = torch.zeros_like((samp_mel), dtype=torch.bool, device=samp_mel.device)


    prev_diff = samp_mel[:, 1:-1, :] - samp_mel[:, :-2, :]
    next_diff = samp_mel[:, 2:, :]   - samp_mel[:, 1:-1, :]
    max_tensor[:, 1:-1, :] = (prev_diff > 0) & (next_diff < 0)

    # quiets = quiets - 56.6  # empirically chosen offset

    # Compute bark scale values from the mel frequencies (assumed monotonic)
    # barks = barks.to(DEVICE)

    # Use the bark values to form bins (each bin corresponding roughly to a critical band)
    bark_bins = barks.floor().to(DEVICE, dtype=torch.int64)
    bin_idx = bin_indices(bark_bins)
    
    # exit()
    # Start with the quiet threshold and update with masker-based thresholds
    threshold = quiets.clone()
    if method == "bins":
        # Process each bark bin (frequency chunk)
        for i in range(len(bin_idx) - 1):
            start, stop = bin_idx[i], bin_idx[i + 1]
            chunk = samp_mel[:, start:stop, :]


            max_val, max_idx = chunk.max(dim=1, keepdim=True)  # shape: (batch_size, 1, frames)

            # Get the bark values corresponding to the current chunk frequencies.
            bark_chunk = barks[start:stop].view(1, -1, 1).expand(batch_size, -1, frames).to(DEVICE)
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
            G = 27
            sf[mask_lower] = G * torch.abs(dz[mask_lower])
            
            # For frequencies above the masker: compute a level-dependent slope.
            # The original two_slops used: (-27 + 0.37 * max(masker_level-40,0)) * dz,
            upper_slope = G - 0.37 * torch.clamp(max_val - 40, min=0)
            upper_slope_expanded = upper_slope.expand_as(dz)
            sf[mask_upper] = upper_slope_expanded[mask_upper] * dz[mask_upper]

            spread = max_val + sf

            threshold[:, start:stop, :] = torch.maximum(threshold[:, start:stop, :], spread)
    else:
        # PART 1. BUILD THRESHOLD MATRIX BASED ON THREE CONDITIONS
        # Build new max vector
        thresh_mask = torch.zeros_like(samp_mel)
        for i, (group, idx) in enumerate(zip(bark_groups, group_indices)):
            f1, f2 = group[0], group[-1]
            chunk = samp_mel[:, f1:f2 + 1, :]

            assert len(chunk.shape) > 2
            local_max = max_tensor[:, i, :].unsqueeze(1)

            # CONDITION 1. It must be the max within 0.5 Bark
            max_val, max_idx = chunk.max(dim=1, keepdim=True)

            # Check if our target frequency is the masker of its group
            pruned = (max_idx == idx)

            # CONDITION 2. Must be a local maximum in its area (apply the local max mask)
            pruned = pruned * local_max  # Removing non-local-max values

            thresh_mask[:, i, :] = pruned.squeeze(1)

        max_only = samp_mel * thresh_mask
        max_only.masked_fill_(max_only == 0, -torch.inf)

        # CONDITION 3. Masker must be audible (louder than the ATH threshold)
        thresh_mask = max_only > quiets
        thresh_vals = torch.where(thresh_mask, samp_mel, -torch.inf)



        # Compute bark tensor and an additional offset matrix delta_m for later adjustment.
        bark_tensor = barks.view(1, -1, 1).expand(batch_size, -1, frames).to(DEVICE)
        delta_m = -6.025 - 0.275 * bark_tensor

        # Transpose candidate masker levels to shape (batch, T, F)
        candidate_levels = thresh_vals.transpose(1, 2)  # shape: (B, T, F)

        # Precompute the F×F delta matrix (difference between each target and candidate masker frequency).
        # barks is assumed to be a 1D tensor of shape (F,)
        barks_target = barks.view(frequencies, 1)  # shape: (F, 1)
        barks_candidate = barks.view(1, frequencies)  # shape: (1, F)
        delta = (barks_target - barks_candidate).to(DEVICE)  # shape: (F, F)

        # Set the base G constant
        G = -27

        # Instead of expanding candidate_levels to (B, T, F, F), iterate over the target frequencies.
        spread_list = []
        for target_idx in range(frequencies):
            # Get the precomputed delta values for this target frequency (shape: (F,))
            current_delta = delta[target_idx]  # shape: (F,)
            # Reshape for broadcasting over (B, T, F)
            current_delta_exp = current_delta.view(1, 1, frequencies)

            # Compute the level-dependent slope for candidate maskers. This is computed per (B, T, F).
            slope = G - 0.37 * torch.clamp(candidate_levels - 40, min=0)

            # Compute candidate masker contributions:
            # - For target frequencies below the masker (current_delta < 0):
            #       contribution = candidate_level + G * |current_delta|
            # - For target frequencies above the masker (current_delta >= 0):
            #       contribution = candidate_level + slope * current_delta
            contrib = torch.where(
                current_delta_exp < 0,
                candidate_levels + G * torch.abs(current_delta_exp),
                candidate_levels + slope * current_delta_exp
            )
            # For this target frequency, take the maximum contribution over candidate maskers (last dimension).
            contrib_max, _ = contrib.max(dim=-1)  # shape: (B, T)
            # Append with an extra frequency dimension, so that later we can concatenate along the frequency axis.
            spread_list.append(contrib_max.unsqueeze(-1))  # shape: (B, T, 1)

        # Concatenate contributions along the frequency axis to get (B, T, F),
        # then transpose to (B, F, T) and add the delta_m offset.
        spread_concat = torch.cat(spread_list, dim=-1)  # shape: (B, T, F)
        spread_full = spread_concat.transpose(1, 2) + delta_m

        # Instead of taking the element-wise maximum in dB,
        # convert to the linear scale, sum, and then convert back to dB.
        spread_lin = torch.pow(10, spread_full / 10.0)   # Convert spread threshold to linear
        quiets_lin = torch.pow(10, quiets / 10.0)          # Convert quiet threshold to linear
        # Sum the quiet threshold and the masker contributions in the linear domain.
        threshold_lin = spread_lin + quiets_lin
        # Convert the summed value back to dB.
        threshold = 10 * torch.log10(threshold_lin)

        plotnshow(
            samp_mel[1, :, 5].cpu().numpy(),
            threshold[1, :, 5].cpu().numpy(),
            quiets[1, :, 5].cpu().numpy(),
            local_max=max_tensor[1, :, 5].cpu().numpy()
        )

        return threshold

# All pre-computed values
# |--------------------------------------------------------------
barks = Bark(mel_frequencies)
# x_cpu = barks.cpu()
values = barks.tolist()
bark_groups = []
group_indices = []  # List of the local indices for each value

for i, val_i in enumerate(values):
    group = [j for j, val_j in enumerate(values) if abs(val_j - val_i) <= 0.5]
    bark_groups.append(group)
    
    # Find the index of 'i' in its group
    local_index = group.index(i)
    group_indices.append(local_index)

total = sum(len(group) for group in bark_groups)

if __name__ == "__main__":
    import os
    output = "/home/jaydenfassett/audioversarial/imperceptible/src/masking/vis2.png"
    from whisper import log_mel_spectrogram
    from time import time
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # print(neighborhood_size(mel_frequencies))
    qr = AudioDataModule("librispeech:clean-100", batch_size=128)
    samp = pad_or_trim(qr.sample[0])
    dl = next(iter(qr.random_all_dataloader()))
    tests = dl[0].to(DEVICE)
    lengths = dl[-1].to(DEVICE)
    # tests -= 0.0709
    samp_mel = log_mel_spectrogram_raw(tests)
    start1 = time()
    generate_mel_th(samp_mel, lengths,method="bins")
    end1 = time()
    threshold = generate_mel_th(samp_mel, lengths,method="2")
    end2 = time()

    print(f"Bin Method: {(end1 - start1):.2f}")
    print(f"Group Method: {(end2 - end1):.2f}")
    # random_mel = log_mel_spectrogram_raw(torch.randn(1,480000).clamp_(max=0.02,min=-0.02))
    exit()


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
    # fig = show_mel_spectrograms(q)
    # fig = show_mel_spectrograms([samp_mel.squeeze(),samp_mel.squeeze(0),conv.squeeze(0)],titles=["Audio Spectrogram","Mean Strengths Visualization","Local Maxima strengths"])
    # fig.savefig(output)
