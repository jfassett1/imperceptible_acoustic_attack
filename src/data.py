import torchaudio
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pathlib import Path
from whisper import pad_or_trim
from tqdm import tqdm
import numpy as np
from time import time
import types
root_dir = Path(__file__).parent.parent
data_dir = Path("/media/drive1/jaydenfassett/audio_data")
# data_dir = root_dir / "data"

PAD_PLACEHOLDER = 1.337e-10

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name:str ="librispeech:dev-clean", attack_len = 1,batch_size=32, num_workers=0):
        super().__init__()
        self.dataset_name, self.split = dataset_name.split(":")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.iqr = None
        self.q3 = None
        self.q1 = None
        self.max = None
        self.min = None
        self.attack_len = attack_len
        self.setup()
        #Search for a sample > 4 seconds
        while True:
            samp_id = np.random.randint(0,len(self.dataset))
            if self.dataset[samp_id][0].shape[-1] >= 16000 * 5:
                self.sample = self.dataset[samp_id]
                break

    def prepare_data(self):
        # Download the dataset if it is not already available TODO: Full conditionals
        match self.dataset_name.lower():
            case "librispeech":
                self.dataset = torchaudio.datasets.LIBRISPEECH(data_dir, self.split, download=True)

            case "tedlium": 
                self.dataset = torchaudio.datasets.TEDLIUM(data_dir, release="release3", subset=self.split, audio_ext=".flac",download=True)
                self.dataset._load_audio = types.MethodType(_load_audio_patched, self.dataset) #Monkey Patching original load_data function, because it does not default to the correct backend.

            case "commonvoice":
                self.dataset = torchaudio.datasets.COMMONVOICE(data_dir, version="cv-corpus-13.0-2023-03-09", download=True)

            case "vctk":
                self.dataset = torchaudio.datasets.VCTK(data_dir, download=True)
            case _:
                raise ValueError(f"Dataset {self.dataset_name} is not supported.")
    
    def setup(self, stage=None):
        self.prepare_data()
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
    def _normalize_3std(self,audio):
        pass
    def get_IQR(self,
                 N=10,
                 sec_audio = 5): # Length of audio
        """
        Calculates IQR of dataset by random sampling
        
        
        
        """
        start = time()
        dl = self.padded_dataloader()
        # print(f"Dataset length: {len(dl)}")
        if (len(dl)) < N:
            N = len(dl)
        pbar = tqdm(enumerate(dl),total=N-1)

        size_test = int(16000 * self.attack_len) # NOTE: 
        size_flat = self.batch_size * size_test
        waveform_values = np.zeros(N * size_flat,dtype=np.float32) #TODO: FIX DTYPE
        size_gb = 0
        valid = 0

        print("Calculating Data Statistics...")
        for i, (waveform, sample_rate, transcript) in pbar: # Waveform is (N, 480000)
            
            # print(type(waveform))

            waveform = waveform[:, :size_test]
            # print(waveform.shape)
            if waveform.size(0) < self.batch_size:  # Skip batch if's not the right size
                continue  

            #Tracking memory usage #NOTE Vestigial artifact of me using lists previously
            batch_size_gb = waveform.element_size() * waveform.numel() / 1e9
            size_gb += batch_size_gb

            start_idx = i * size_flat
            end_idx = start_idx + size_flat
            waveform_values[start_idx:end_idx] = waveform.cpu().numpy().flatten()
            # Update progress bar with memory usage

            pbar.set_description(f"Memory Usage: {size_gb:.3f} GB")
            valid += 1
            if valid == N:
                break
        # Removing zeroes as to not include padding
        print(f"Removing {(waveform_values  == PAD_PLACEHOLDER).sum()} padded values from calculation")
        waveform_values = waveform_values[waveform_values != PAD_PLACEHOLDER]

        # Compute global IQR
        q1, q3 = np.percentile(waveform_values, [25, 75])
        iqr = q3 - q1
        # print(iqr)
        print(f"Completed in {(time() - start):.6f} seconds")
        self.iqr = iqr
        self.q3 = q3
        self.q1 = q1
        return iqr, q3, q1

    def collate_fn(self, batch,test=False): #Collate not the problem
        # Collate function to handle variable-length audio sequences

        audio, sampling_rate,transcript,utt,speak,chap= zip(*batch)
        #No need to trim
        if self.dataset_name == "librispeech":
            resized_audio = [pad_or_trim_patch(aud[:,16000:]) if test else pad_or_trim(aud[:,16000:]) for aud in audio] #Trimming first sec for librispeech
        else:
            resized_audio = [pad_or_trim_patch(aud) if test else pad_or_trim(aud) for aud in audio]

        audio_signals = torch.cat(resized_audio,dim=0)
        # labels = torch.tensor(labels)
        return audio_signals, sampling_rate, transcript

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)
    def all_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)        
    def random_all_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                    num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)
    def padded_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                    num_workers=self.num_workers, shuffle=True, collate_fn=lambda x: self.collate_fn(x,test=True))
    def _normalize(self,data):
        """
        Normalizing audio based on IQR.
        """

        if None in (self.iqr,self.q3,self.q1):
            print(self.get_IQR(5))
        q = 1.5
        lower,upper = (self.q1 - q*self.iqr),(self.q3 + q*self.iqr)
        print("Upper/Lower:",upper,lower)
        # data_x = torch.clamp(data,min=lower,max=upper)
        return data,lower,upper


N_SAMPLES = 480000
import torch.nn.functional as F
def pad_or_trim_patch(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes],value=PAD_PLACEHOLDER)
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)
    return array

def _load_audio_patched (self, path: str, start_time: float, end_time: float, sample_rate: int = 16000):

    start_time = int(float(start_time) * sample_rate)
    end_time = int(float(end_time) * sample_rate)
    kwargs = {"frame_offset": start_time, "num_frames": end_time - start_time, "backend":"soundfile"}
    return torchaudio.load(path, **kwargs)

if __name__ == "__main__":
    # ""

    # dataset = torchaudio.datasets.LIBRISPEECH(data_dir, "dev-clean", download=True)

    # num_zeroes = 0
    # total_len = 0
    # for sample in tqdm(dataset):
    #     aud = sample[0].squeeze()

    #     zero_mask = (aud == 0.0)
    #     # print(aud.shape)
    #     # exit()
    #     num_zeroes += zero_mask.sum()
    #     total_len += len(aud)
    
    # print(num_zeroes)
    # print(total_len)
    # print(num_zeroes/total_len)

    qr = AudioDataModule(dataset_name="librispeech:train-clean-100",attack_len=2,batch_size=128,num_workers=0)
    qr.get_IQR(N=10)
    dl = qr.padded_dataloader()
    samp = next(iter(dl))[0]

    normalized,lower,upper = qr._normalize(samp)
    # lower,upper = lower*0.02,upper*0.02
    # print(normalized.shape)
    normalized = normalized.flatten()
    print("Num of pads removed", (normalized == PAD_PLACEHOLDER).sum()/ len(normalized))
    filtered = normalized[normalized != PAD_PLACEHOLDER]
    print("Percentage of vals on bound",((filtered == lower) | (filtered == upper)).sum() / len(filtered ))

    filtered = filtered.numpy()

    # iq = qr.get_IQR(N=10)
    # print(iq)
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm
    plt.figure()
    plt.hist(filtered, bins=200, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean( filtered), np.std( filtered))
    plt.plot(x, p, 'k', linewidth=2)
    plt.title("Histogram and Normal Curve")
    plt.show()
    plt.savefig("/home/jaydenfassett/audioversarial/imperceptible/test.png")
    # print(filtered[:2_000_000])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # First subplot: Filtered data with constraint

    length = 5*16000
    axes[0].plot(filtered[:length])
    axes[0].set_title("Filtered with Constraint")
    axes[0].axvline(x=16000, color="g", linestyle="--")
    axes[0].axhline(y=lower, color='r', linestyle='--')
    axes[0].axhline(y=upper, color='r', linestyle='--')
    axes[0].set_ylim(-1, 1)
    axes[0].set_xlim(0, length)
    # Second subplot: First 16000 samples
    axes[1].plot(filtered[:16000* qr.attack_len])
    axes[1].set_title("First second")
    axes[1].axvline(x=16000, color="g", linestyle="--")
    axes[1].axhline(y=lower, color='r', linestyle='--')
    axes[1].axhline(y=upper, color='r', linestyle='--')
    axes[1].set_ylim(-1, 1)
    axes[1].set_xlim(0, 16000)

    # Adjust layout and save the figure
    # plt.tight_layout()
    plt.savefig("/home/jaydenfassett/audioversarial/imperceptible/test2.png")
    plt.show()