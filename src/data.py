import torchaudio
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pathlib import Path
from whisper import pad_or_trim
from tqdm import tqdm
import numpy as np
root_dir = Path(__file__).parent.parent
data_dir = Path("/media/drive1/jaydenfassett/audio_data")
# data_dir = root_dir / "data"



class AudioDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name="librispeech", batch_size=32, num_workers=0):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.max = None
        self.min = None
        self.setup()

    def prepare_data(self):
        # Download the dataset if it is not already available TODO: Full conditionals
        # if self.dataset_name == "librispeech":
        self.dataset = torchaudio.datasets.LIBRISPEECH(data_dir,
                            # 'dev-clean',
                            # 'train-clean-100',
                            self.dataset_name,
                            # 'train-other-500',
                                    download = True)
    
    def setup(self, stage=None):
        self.prepare_data()
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
    def _normalize_3std(self,audio):
        pass
    def get_IQR(self,
                 N=5,
                 sec_audio = 5): # Length of audio
        """
        Calculates IQR of dataset by random sampling
        
        
        
        """
        dl = self.random_all_dataloader()
        pbar = tqdm(enumerate(dl),total=N-1)

        size_test = 16000 * 5 # NOTE: Only first n seconds to avoid zeroes from pad_or_clip
        size_flat = self.batch_size * size_test
        waveform_values = np.zeros(N * size_flat)
        size_gb = 0
        valid = 0
        print("Calculating Data Statistics...")
        for i, (waveform, sample_rate, transcript) in pbar:
            waveform = waveform[:, :size_test]
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
                print(valid)
                break
        # Compute global IQR
        q1, q3 = np.percentile(waveform_values, [25, 75])
        iqr = q3 - q1
        print("Complete")
        return iqr

    def collate_fn(self, batch):
        # Collate function to handle variable-length audio sequences

        audio, sampling_rate,transcript,utt,speak,chap= zip(*batch)

        #No need to trim 
        resized_audio = [pad_or_trim(aud) for aud in audio]
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
                    num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)  

if __name__ == "__main__":

    q = AudioDataModule(dataset_name="train-clean-100")

    p = q.get_IQR(N=100)
    print(p)
    # dl = q.train_dataloader()
    
    # for i in range(1):
    #     l = iter(dl)
    #     print(next(l)[0].shape)
    # print(librispeech)