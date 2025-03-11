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


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name:str ="librispeech:dev-clean", batch_size=32, num_workers=0):
        super().__init__()
        self.dataset_name, self.split = dataset_name.split(":")
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.max = None
        self.min = None
        self.setup()

    def prepare_data(self):
        # Download the dataset if it is not already available TODO: Full conditionals
        match self.dataset_name.lower():
            case "librispeech":
                self.dataset = torchaudio.datasets.LIBRISPEECH(data_dir, self.split, download=True)

            case "tedlium": 
                self.dataset = torchaudio.datasets.TEDLIUM(data_dir, subset=self.split, audio_ext=".flac",download=True)
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
                 N=5,
                 sec_audio = 5): # Length of audio
        """
        Calculates IQR of dataset by random sampling
        
        
        
        """
        start = time()
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
        # Compute global IQR
        q1, q3 = np.percentile(waveform_values, [25, 75])
        iqr = q3 - q1
        print(f"Completed in {(time() - start):.6f} seconds")
        return iqr, q3, q1

    def collate_fn(self, batch): #Collate not the problem
        # Collate function to handle variable-length audio sequences

        audio, sampling_rate,transcript,utt,speak,chap= zip(*batch)
        # print(audio[0].shape)

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
                    num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)  


def _load_audio_patched (self, path: str, start_time: float, end_time: float, sample_rate: int = 16000):

    start_time = int(float(start_time) * sample_rate)
    end_time = int(float(end_time) * sample_rate)

    kwargs = {"frame_offset": start_time, "num_frames": end_time - start_time, "backend":"soundfile"}
    return torchaudio.load(path, **kwargs)
if __name__ == "__main__":
    ""
    # qr = AudioDataModule(dataset_name="tedlium:dev").all_dataloader()
    # qq = AudioDataModule(dataset_name="librispeech:train-clean-100").all_dataloader()
    # qr = torchaudio.datasets.TEDLIUM(data_dir, subset="dev", audio_ext=".flac",download=True)
    # qq = torchaudio.datasets.LIBRISPEECH(data_dir, "dev-clean", download=True)

    # qr._load_audio = types.MethodType(_load_audio_patched, qr)
    # qq[5]
    # print(f"Lib Read Time: {(time() - start):.6f} seconds")
    # qr[5]
    # print(f"Ted Read Time: {(time() - start):.6f} seconds")
    # start = time()
    # for i in tqdm(qq,total=len(qq)):
    #     pass
    # print(f"Lib Read Time: {(time() - start):.6f} seconds")
    # start = time()
    # for i in tqdm(qr,total=len(qr)):
    #     pass
    # print(f"Ted Read Time: {(time() - start):.6f} seconds")



    # waveform, sample_rate, transcript, talk_id, speaker_id, identifier = qr[1000]
    # print(qr._path)

    # p = q.get_IQR(N=5)
    # q = AudioDataModule(dataset_name="librispeech:train-clean-100")
    # p = q.get_IQR(N=5)
    # print(p)
    # dl = q.train_dataloader()
    
    # for i in range(1):
    #     l = iter(dl)
    #     print(next(l)[0].shape)
    # print(librispeech)