import torchaudio
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pathlib import Path
from whisper import pad_or_trim


root_dir = Path(__file__).parent.parent
data_dir = root_dir / "data"


# print(root_dir)
librispeech = torchaudio.datasets.LIBRISPEECH(data_dir,
                                #  'dev-clean',
                                    'train-clean-100',
                                     download = True)

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name="librispeech", batch_size=32, num_workers=4):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.setup()
    def prepare_data(self):
        # Download the dataset if it is not already available
        if self.dataset_name == "librispeech":
            self.dataset = torchaudio.datasets.LIBRISPEECH(data_dir,
                                # 'dev-clean',
                                'train-clean-100',
                                     download = True)
    
    def setup(self, stage=None):
        self.prepare_data()
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

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


if __name__ == "__main__":

    q = AudioDataModule(dataset_name="librispeech")
    dl = q.train_dataloader()
    
    for i in range(1):
        l = iter(dl)
        print(next(l)[0].shape)
    # print(librispeech)