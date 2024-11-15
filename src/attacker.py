"""
Trainable Attacker class


"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .decoder import RawDecoder
from typing import Union
from whisper import pad_or_trim, log_mel_spectrogram
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
        self.noise = nn.Parameter(torch.randn(1, 80, int(sec * 100)), requires_grad=True).to(self.device)
        self.prepend = prepend
        self.target_length = target_length
        self.noise_len = int(sec * 100)
        self.model = whisper.load_model(model).to(self.device)

        #Fix sparse tensors for Pytorch Lightning
        alignment_heads_dense = self.model.get_buffer("alignment_heads").to_dense()
        self.model.register_buffer("alignment_heads", alignment_heads_dense, persistent=False)
        del alignment_heads_dense


        if "en" in model:
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False,task="transcribe")
        else:
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        self.decoder = RawDecoder(self.model,self.tokenizer,self.device)
        #Freezing Whisper weights
        for param in self.model.parameters():
            param.requires_grad = False
    def to(self, device):
        #Updated to function
        super().to(device)

        self.noise = self.noise.to(device)

        self.decoder.device = device
        self.decoder.update_device(device)
        return self

    def forward(self, x):
        x = x.to(self.device)
        BATCH_SIZE = x.shape[0]
        # Optionally prepend the noise
        x = pad_or_trim(x)
        x = log_mel_spectrogram(x)
        if self.prepend:
            #Slice noise sized chunk from x & add noise
            x = x[:,:,:-self.noise_len]
            noise = self.noise.repeat(BATCH_SIZE,1,1)
            x = torch.cat([noise, x], dim=-1)
        return self.decoder.get_eot_prob(x)

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        x, sampling_rate, transcript = batch

        x = self.forward(x)

        loss = -torch.log(x + 1e-9).mean()
        # print(f"Training loss: {loss.item()}")  # Debug print
        # print(loss.shape)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def dump(self, path: Union[str, Path], mel: bool = False):
            pass

if __name__ == "__main__":
    device = "cuda"
    model = MelBasedAttackerLightning().to(device)

    x = whisper.load_audio("/home/jaydenfassett/audioversarial/imperceptible/original_audio.wav")
    x = torch.tensor(x).unsqueeze(0)
    x = torch.cat([x,x,x,x],dim=0)
    print(x.shape)
    x = (x,1,1)
    # print(x.shape)

    # x = raw_to_mel(x)
    for name, param in model.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

    # print(model.parameters())
    # qq = model.training_step(x,1)
    # print(qq)
    # qq = model(sample_mel)


