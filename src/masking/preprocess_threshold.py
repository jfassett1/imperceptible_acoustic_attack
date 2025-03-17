import torchaudio
from torch.utils.data import DataLoader
from .mask_threshold import generate_th_batch
from src.data import AudioDataModule
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
def preprocess_dataset(dataset,output, batch_size=64):
    datamodule = AudioDataModule(dataset,batch_size,num_workers=4)
    dataloader = datamodule.all_dataloader()
    # (dataset, batch_size=64, shuffle=False, num_workers=2)
    pbar = tqdm(enumerate(dataloader),total=len(dataloader))

    mean_threshold = None
    for i, (waveform, sample_rate, transcript) in pbar:
        # print(waveform.shape)
        # exit()
        # print(waveform.shape)
        waveform = waveform[:,:160_000] #Only getting first 10 seconds to avoid including padding
        if batch_size != 1:
            sample_rate = sample_rate[0]
        
        theta_xs, psd_max, PSD = generate_th_batch(audio=waveform, fs=sample_rate)
        threshold_i = theta_xs.mean(axis=0)
        #Online mean calculation
        if mean_threshold is None:
            mean_threshold = threshold_i
        else:
            a = 1/ (i+1)
            b = 1 - a
            mean_threshold = a * threshold_i + b * mean_threshold
    np.save(output,mean_threshold)
    print(f"Dataset Preprocessed.\nSaving final threshold to {output}")
    return mean_threshold


if __name__ == "__main__":

    # print("s")
    for dset in tqdm(['dev-clean','test-clean','train-clean-100',"train-other-500"]):
        preprocess_dataset(dataset=dset,output=f"/home/jaydenfassett/audioversarial/imperceptible/thresholds/{dset}.np.npy")