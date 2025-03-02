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
def compute_PSD_matrix_batch(audio, window_size, hop_length=None):
    if hop_length is None:
        hop_length = window_size // 4

    # Ensure audio is 2D: (BATCH_SIZE, n_samples)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Create a Hann window scaled by sqrt(8/3)
    win = torch.hann_window(window_size, device=audio.device) * (8.0/3.)**0.5

    stft = torch.stft(audio, n_fft=window_size, hop_length=hop_length, 
                      window=win, center=False, return_complex=True)
    stft = stft / window_size

    # Compute magnitude and power spectrum
    z = stft.abs()
    psd_linear = z ** 2
    psd = 10 * torch.log10(psd_linear + 1e-18)
    max_psd_db = psd.amax(dim=(-1, -2))
    PSD = 96 - max_psd_db.unsqueeze(-1).unsqueeze(-1) + psd
    psd_max = psd_linear.amax(dim=(-1, -2))
    
    #If single sample
    if PSD.shape[0] == 1:
        PSD = PSD.squeeze(0)
        psd_max = psd_max.item()
    # print(type(PSD),type(psd_max))
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
    for i in range(num_local_max):
        next = i + 1
        if next >= bark_psd.shape[1]:
            break
            
        while bark_psd[next, _BARK] - bark_psd[i, _BARK]  < 0.5:
            # masker must be higher than quiet threshold
            if quiet(freqs[int(bark_psd[i, _INDEX])]) > bark_psd[i, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=_BARK)
            if next == bark_psd.shape[_BARK]:
                break
                
            if bark_psd[i, _PSD] < bark_psd[next, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=_BARK)
            else:
                bark_psd = np.delete(bark_psd, (next), axis=_BARK)
            if next == bark_psd.shape[_BARK]:
                break        
    
    # compute the individual masking threshold
    delta_TM = 1 * (-6.025  -0.275 * bark_psd[:, 0])
    Ts = two_slops(bark_psd, delta_TM, barks) 
    Ts = np.array(Ts)
    
    # compute the global masking threshold
    theta_x = np.sum(pow(10, Ts/10.), axis=0) + pow(10, ATH/10.) 
 
    return theta_x


def generate_th(audio, fs, window_size=2048):
    """
	returns the masking threshold theta_xs and the max psd of the audio
    """
    PSD, psd_max= compute_PSD_matrix(audio , window_size)  
    freqs = librosa.core.fft_frequencies(sr=fs, n_fft=window_size)
    barks = Bark(freqs)
    

    # compute the quiet threshold 
    ATH = np.zeros(len(barks)) - np.inf
    bark_ind = np.argmax(barks > 1)
    ATH[bark_ind:] = quiet(freqs[bark_ind:])

    # compute the global masking threshold theta_xs 
    theta_xs = []
    # compute the global masking threshold in each window
    for i in range(PSD.shape[1]):
        theta_xs.append(compute_th(PSD[:,i], barks, ATH, freqs))
    theta_xs = np.array(theta_xs)
    return theta_xs, psd_max

def generate_th_batch(audio, fs, window_size=2048):
    """
	returns the masking threshold theta_xs and the max psd of the audio
    """
    torch_psd = compute_PSD_matrix_torch(torch.from_numpy(audio) , window_size)  #Getting PSD matrix. It 
    PSD, psd_max = (x.numpy() for x in torch_psd)
    freqs = librosa.core.fft_frequencies(sr=fs, n_fft=window_size)
    barks = Bark(freqs)

    # compute the quiet threshold 
    ATH = np.zeros(len(barks)) - np.inf
    bark_ind = np.argmax(barks > 1)
    ATH[bark_ind:] = quiet(freqs[bark_ind:])

    # compute the global masking threshold theta_xs 
    theta_xs = []
    # compute the global masking threshold in each window
    for samp in range(PSD.shape[0]):
        theta_i = []
        for i in range(PSD.shape[2]):
            theta_i.append(compute_th(PSD[samp,:,i], barks, ATH, freqs))
        theta_xs.append(theta_i)
    theta_xs = np.array(theta_xs)
    return theta_xs, psd_max

if __name__ == "__main__":



    qq = np.random.randn(2,24000)
    # print(compute_PSD_matrix(qq,2048)[0])
    #Checking equality between batched function, and normal function

    print("Testing PSD matrix:")

    # psd1 = compute_PSD_matrix(qq[0,:],2048)[0]
    # psd2 = compute_PSD_matrix_torch(torch.tensor(qq[0,:]),2048)[0]
    # psd1 = torch.tensor(psd1)
    # psddiff = torch.abs(psd1 - psd2).sum().sum()
    # print(psd1.shape)
    # print(psd2.shape)
    # print("tensors equal? ",torch.equal(psd1,psd2))
    # print("tensors close? ", torch.allclose(psd1,psd2,atol=1e-6))
    # print(f"difference: { psddiff}")
    # print(compute_PSD_matrix(qq,2048)[0].shape)

    # exit()
    q1 = []
    for q in range(qq.shape[0]):
        rr = generate_th(qq[q,:],16000)[0]
        q1.append(rr)
    
    q1 = np.array(q1)
    print(q1.shape)
    q2 = generate_th_batch(qq,fs=16000)[0]
    print(q2.shape)

    y = np.array_equal(q1,q2)
    y2 = np.allclose(q1, q2, atol=1e-8)

    y3 = ((q1 - q2) ** 2).mean().mean()
    print(f"Equality: {y}")
    print(f"Approximately Equal: {y2}")
    print(f"Difference: {y3}")