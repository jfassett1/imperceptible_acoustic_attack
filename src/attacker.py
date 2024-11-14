"""
Trainable Attacker class


"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .decoder import RawDecoder
from typing import Union
from whisper import pad_or_trim
from pathlib import Path
from .utils import raw_to_mel,sample_mel
from pytorch_lightning import LightningModule
import whisper


class MelBasedAttackerLightning(LightningModule):
    """
    LightningModule that prepends noise for adversarial attacks, based on
    https://arxiv.org/pdf/2405.06134
    """
    def __init__(self,
                 model:str = "tiny.en",
                 sec: Union[int, float] = 1,
                 prepend: bool = True,
                 target_length: int = 48000):
        super(MelBasedAttackerLightning, self).__init__()

        # Initialize noise as a learnable parameter
        self.noise = nn.Parameter(torch.randn(1, 80, int(sec * 100)), requires_grad=True)
        self.prepend = prepend
        self.target_length = target_length

        self.model = whisper.load_model(model)
        if "en" in model:
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False,task="transcribe")
        else:
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        self.decoder = RawDecoder(self.model)

    def forward(self, x):
        # Optionally prepend the noise
        if self.prepend:
            x = torch.cat([self.noise, x], dim=-1)

        # Pad or trim to the target length
        x = pad_or_trim(x, length=self.target_length)
        return x

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        x, sampling_rate, transcript, sp, utt, chap = batch

        # Generate the adversarial example
        attacked_x = self(x)

        # Define a loss (placeholder)
        loss = nn.functional.mse_loss(attacked_x, x)  # Substitute with a relevant loss function
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def dump(self, path: Union[str, Path], mel: bool = False):
            pass

if __name__ == "__main__":
    device = "cuda"
    model = MelBasedAttackerLightning().to(device)
    
    qq = model(sample_mel)


