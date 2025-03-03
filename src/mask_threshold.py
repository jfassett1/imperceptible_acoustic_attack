import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy import signal
import scipy
import librosa
import torch



def compute_PSD_matrix(audio, window_size):
    """
	First, perform STFT.
	Then, compute the PSD.
	Last, normalize PSD.
    """

    win = np.sqrt(8.0/3.) * librosa.core.stft(audio, center=False)
    z = abs(win / window_size) #Normalizes STFT. Dividing by window_size basically reverses any frequencies being artificially amplified by the overlapping windows
    psd_max = np.max(z*z)
    psd = 10 * np.log10(z * z + 0.0000000000000000001)
    PSD = 96 - np.max(psd) + psd # Normalization process, hinges on related work.
    return PSD, psd_max   
def compute_PSD_matrix_batch(audio, window_size, hop_length=None, transpose=False):
    """
    First, perform STFT.
    Then, compute the PSD.
    Last, normalize PSD.
    """
    if hop_length is None:
        hop_length = window_size // 4

    # Ensure audio is 2D: (BATCH_SIZE, n_samples)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    win = torch.hann_window(window_size, device=audio.device) * torch.sqrt(torch.tensor(8.0 / 3.0, device=audio.device))

    # Compute STFT
    stft = torch.stft(audio, n_fft=window_size, hop_length=hop_length, 
                      window=win, center=False, return_complex=True)
    # Normalize STFT
    z = stft.abs() / window_size  
    psd_linear = z ** 2
    # Clamp to avoid exploding vals
    psd = 10 * torch.log10(torch.clamp(psd_linear, min=1e-10))
    # Normalization process (same as original)
    psd_max = psd.amax(dim=(-1, -2), keepdim=True)
    PSD = 96 - psd_max + psd  

    psd_max = psd_linear.amax(dim=(-1, -2))
    if PSD.shape[0] == 1:
        psd_max = psd_max.item()

    if transpose:
        PSD = PSD.transpose(1, 2)
    return PSD, psd_max
def Bark(f):
    """returns the bark-scale value for input frequency f (in Hz)"""
    return 13*np.arctan(0.00076*f) + 3.5*np.arctan(pow(f/7500.0, 2))

def quiet(f):
     """returns threshold in quiet measured in SPL at frequency f with an offset 12(in Hz)"""
     thresh = 3.64*pow(f*0.001,-0.8) - 6.5*np.exp(-0.6*pow(0.001*f-3.3,2)) + 0.001*pow(0.001*f,4) - 12
     return thresh

def two_slops(bark_psd, delta_TM, bark_maskee):
    """
	returns the masking threshold for each masker using two slopes as the spread function 
    """
    Ts = []
    for tone_mask in range(bark_psd.shape[0]):
        bark_masker = bark_psd[tone_mask, 0]
        dz = bark_maskee - bark_masker
        zero_index = np.argmax(dz > 0)
        sf = np.zeros(len(dz))
        sf[:zero_index] = 27 * dz[:zero_index]
        sf[zero_index:] = (-27 + 0.37 * max(bark_psd[tone_mask, 1] - 40, 0)) * dz[zero_index:] 
        T = bark_psd[tone_mask, 1] + delta_TM[tone_mask] + sf
        Ts.append(T)
    return Ts
    
def compute_th(PSD, barks, ATH, freqs):
    """ returns the global masking threshold
    """
    # Identification of tonal maskers
    # find the index of maskers that are the local maxima
    length = len(PSD)
    masker_index = signal.argrelextrema(PSD, np.greater)[0]
    
    
    # delete the boundary of maskers for smoothing
    if 0 in masker_index:
        masker_index = np.delete(0)
    if length - 1 in masker_index:
        masker_index = np.delete(length - 1)
    num_local_max = len(masker_index)

    # treat all the maskers as tonal (conservative way)
    # smooth the PSD 
    p_k = pow(10, PSD[masker_index]/10.)    
    p_k_prev = pow(10, PSD[masker_index - 1]/10.)
    p_k_post = pow(10, PSD[masker_index + 1]/10.)
    P_TM = 10 * np.log10(p_k_prev + p_k + p_k_post)
        # bark_psd: the first column bark, the second column: P_TM, the third column: the index of points
    _BARK = 0
    _PSD = 1
    _INDEX = 2
    bark_psd = np.zeros([num_local_max, 3])
    bark_psd[:, _BARK] = barks[masker_index]
    bark_psd[:, _PSD] = P_TM
    bark_psd[:, _INDEX] = masker_index
    

    # delete the masker that doesn't have the highest PSD within 0.5 Bark around its frequency
    # for s in 
    # print(bark_psd.shape)
    """
    for i in range(num_local_max):
        next = i + 1
        if next >= bark_psd.shape[0]:
            break
        
        while bark_psd[next, _BARK] - bark_psd[i, _BARK]  < 0.5:
            # masker must be higher than quiet threshold
            if quiet(freqs[int(bark_psd[i, _INDEX])]) > bark_psd[i, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=0)
            if next == bark_psd.shape[0]:
                break
                
            if bark_psd[i, _PSD] < bark_psd[next, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=0)
            else:
                bark_psd = np.delete(bark_psd, (next), axis=0)
            if next == bark_psd.shape[0]:
                break   
    """

    # bmax = np.vectorize(bark_max)

    bark_psd = bark_max_vectorized(barks,masker_index,P_TM,freqs)
    # compute the individual masking threshold
    delta_TM = 1 * (-6.025  -0.275 * bark_psd[:, 0])
    Ts = two_slops(bark_psd, delta_TM, barks) 
    Ts = np.array(Ts)
    
    # compute the global masking threshold
    theta_x = np.sum(pow(10, Ts/10.), axis=0) + pow(10, ATH/10.) 
    # print(theta_x.max())
    return theta_x
# def bark_max(barks,masker_index,P_TM,freqs,num_local_max):
#     for i in range(num_local_max):
#         next = i + 1
#         if next >= bark_psd.shape[0]:
#             break
        
#         while bark_psd[next, _BARK] - bark_psd[i, _BARK]  < 0.5:
#             # masker must be higher than quiet threshold
#             if quiet(freqs[int(bark_psd[i, _INDEX])]) > bark_psd[i, _PSD]:
#                 bark_psd = np.delete(bark_psd, (i), axis=0)
#             if next == bark_psd.shape[0]:
#                 break
                
#             if bark_psd[i, _PSD] < bark_psd[next, _PSD]:
#                 bark_psd = np.delete(bark_psd, (i), axis=0)
#             else:
#                 bark_psd = np.delete(bark_psd, (next), axis=0)
#             if next == bark_psd.shape[0]:
#                 break   
#     return bark_psd
def bark_max_vectorized(barks, masker_index, P_TM, freqs):
    """
    A more 'vectorized' approach to select peaks (maskers) 
    that are >= quiet threshold and are local maxima 
    within a 0.5-Bark neighborhood.
    
    Parameters
    ----------
    barks : 1D array of Bark values for *all* frequencies.
    masker_index : 1D array (int) of indices in barks/freqs 
                   that correspond to local maxima.
    P_TM : 1D array of PSD (or power) values at those maskers.
    freqs : 1D array of frequencies (same indexing as barks).
    
    Returns
    -------
    bark_psd_kept : 2D array of shape (M,3), where:
        - column 0 = bark (of the chosen maskers)
        - column 1 = PSD (power) of the chosen maskers
        - column 2 = original index into the `freqs` array
    """
    # 1) Build an array of [bark, PSD, index].
    bark_psd = np.column_stack((
        barks[masker_index],  # Bark values for each local max
        P_TM,                 # PSD at those maxima
        masker_index          # Original indices into barks/freqs
    ))

    # 2) Discard any that fail the quiet threshold test:
    #    quiet_threshold at the frequency of each local max
    q = quiet(freqs[masker_index])       # shape matches P_TM
    mask = (P_TM >= q)                   # keep only where PSD >= quiet
    bark_psd = bark_psd[mask]

    # If everything got removed, just return empty.
    if bark_psd.size == 0:
        return bark_psd

    # 3) Sort by Bark (column 0).
    #    That way we can find consecutive maskers that are < 0.5 Bark apart.
    sort_idx = np.argsort(bark_psd[:, 0])
    bark_psd = bark_psd[sort_idx]

    # 4) Identify clusters of maskers where consecutive Bark-distance < 0.5
    #    We do this by taking differences in sorted Bark values:
    sorted_barks = bark_psd[:, 0]
    diffs = np.diff(sorted_barks)              # shape (N-1, )
    cluster_breaks = (diffs >= 0.5).astype(int)
    # cluster_breaks[i] = 1 means there's a "break" between item i and i+1.

    # We can then build an integer "cluster_id" via cumulative sums:
    # e.g., cluster_id = [0,0,0,1,1,2,2,2, ...] etc.
    cluster_id = np.cumsum(np.insert(cluster_breaks, 0, 0))

    # 5) For each cluster, keep only the masker with the highest PSD.
    #    We can do this in a simple Python loop.  There are ways to do it 
    #    more purely with NumPy, but a small loop is usually quite fast.
    bark_psd_kept = []
    for c in np.unique(cluster_id):
        # Indices belonging to cluster c:
        cluster_mask = (cluster_id == c)
        # The row with the maximum PSD in that cluster:
        cluster_rows = np.where(cluster_mask)[0]
        i_best = cluster_rows[np.argmax(bark_psd[cluster_rows, 1])]  # col=1 is PSD
        bark_psd_kept.append(bark_psd[i_best])

    # Convert list-of-rows back to NumPy array, shape (M,3)
    bark_psd_kept = np.array(bark_psd_kept, dtype=float)

    # 6) Return final array of shape (M,3).
    #    col 0 = Bark, col 1 = PSD, col 2 = original index into barks/freqs
    return bark_psd_kept
    
def generate_th(audio, fs, window_size=2048):
    """
	returns the masking threshold theta_xs and the max psd of the audio
    """
    PSD, psd_max= compute_PSD_matrix(audio , window_size)  
    freqs = librosa.core.fft_frequencies(sr=fs, n_fft=window_size)
    barks = Bark(freqs)
    

    # compute the quiet threshold 
    ATH = np.zeros(len(barks)) - 1e-9
    bark_ind = np.argmax(barks > 1)
    ATH[bark_ind:] = quiet(freqs[bark_ind:])

    # compute the global masking threshold theta_xs 
    theta_xs = []
    # compute the global masking threshold in each window
    for i in range(PSD.shape[1]):
        theta_xs.append(compute_th(PSD[:,i], barks, ATH, freqs))
    theta_xs = np.array(theta_xs)
    return theta_xs, psd_max, PSD

def generate_th_batch(audio, fs, window_size=2048):
    """
	returns the masking threshold theta_xs and the max psd of the audio
    """
    torch_psd = compute_PSD_matrix_batch(torch.from_numpy(audio) , window_size)  #Getting PSD matrix. It 
    # print("torch_psd",torch_psd[0].shape)
    # PSD, psd_max = torch_psd
    PSD, psd_max = [x.numpy() if isinstance(x, torch.Tensor) else x for x in torch_psd]

    freqs = librosa.core.fft_frequencies(sr=fs, n_fft=window_size)
    barks = Bark(freqs)

    # compute the quiet threshold 
    ATH = np.zeros(len(barks)) - np.inf
    bark_ind = np.argmax(barks > 1)
    ATH[bark_ind:] = quiet(freqs[bark_ind:])

    # compute the global masking threshold theta_xs 
    theta_xs = []
    # compute the global masking threshold in each window
    PSD = PSD[np.newaxis, ...] if len(PSD.shape) < 3 else PSD #Add extra dim if needed
    # print(PSD.shape)
    for samp in range(PSD.shape[0]):
        theta_i = []
        for i in range(PSD.shape[2]):
            theta_i.append(compute_th(PSD[samp,:,i], barks, ATH, freqs))
        theta_xs.append(theta_i)
    theta_xs = np.array(theta_xs)
    return theta_xs, psd_max, PSD

if __name__ == "__main__":

    import whisper
    x = whisper.load_audio("/home/jaydenfassett/audioversarial/imperceptible/original_audio.wav")
    np.random.seed(0)
    qq = np.random.randn(16000)
    # print(compute_PSD_matrix(qq,2048)[0])
    #Checking equality between batched function, and normal function

    print("Testing PSD matrix:")

    from time import perf_counter

    # psd1 = compute_PSD_matrix(qq[0,:],2048)[0]
    # psd2 = compute_PSD_matrix_batch(torch.tensor(qq),2048)[0]

    q1 = generate_th_batch(qq,16000)[0]

    print(q1.max(),q1.min())
    # psd1 = torch.tensor(psd1)
    # psddiff = torch.abs(psd1 - psd2).sum().sum()
    # print(psd1.shape)
    # print(psd2.shape)
    # print("tensors equal? ",torch.equal(psd1,psd2))
    # print("tensors close? ", torch.allclose(psd1,psd2,atol=1e-6))
    # print(f"difference: { psddiff}")
    # print(compute_PSD_matrix(qq,2048)[0].shape)

    # print("\n")
    # print()
    # qq = torch.from_numpy(qq)
    # x = x.reshape(1,-1)
    # start = perf_counter()
    # print(x.shape)


    # theta_xs = generate_th_batch(qq,16000)[0]

    # end = perf_counter() - start
    # print("time",end)
    # q1 = []
    # start2 = perf_counter()
    # for q in range(qq.shape[0]):
    #     rr = generate_th(qq[q,:],16000)[0]
    #     q1.append(rr)
    
    # end2 = perf_counter() - start2
    
    # print("Batched: ",end)
    # print("Individual: ",end2)
    # print(end / end2)

    # theta_xs = torch.from_numpy(theta_xs)
    # theta_xs = theta_xs.mean(dim=1)
    # print(theta_xs.shape)
    # print("theta min:", theta_xs.min().item(), "thetao max:", theta_xs.max().item())
    # # print(r.shape)
    # import matplotlib.pyplot as plt
    # print(theta_xs[0,:].shape)
    # plt.plot(np.log10(theta_xs[0,:]))
    # plt.savefig("/home/jaydenfassett/audioversarial/imperceptible/src/nuts.png")
    # exit()
    # q1 = []
    # for q in range(qq.shape[0]):
    #     rr = generate_th(qq[q,:],16000)[0]
    #     q1.append(rr)
    
    # q1 = np.array(q1)
    # print(q1.shape)
    # q2 = generate_th_batch(qq,fs=16000)[0]
    # print(q2.shape)

    # print(compute_PSD_matrix_batch(torch.from_numpy(qq),2048,transpose=True)[0].shape)
    # print(compute_PSD_matrix_batch(torch.from_numpy(qq),2048)[1].shape)
    # print(compute_PSD_matrix(qq,2048)[1].shape)

    # y = np.array_equal(q1,q2)
    # y2 = np.allclose(q1, q2, atol=1e-8)

    # y3 = ((q1 - q2) ** 2).mean().mean()
    # print(f"Equality: {y}")
    # print(f"Approximately Equal: {y2}")
    # print(f"Difference: {y3}")