import torch
import torch.nn as nn
from src.decoder import RawDecoder
from typing import Union, Optional, Literal
from whisper import pad_or_trim, log_mel_spectrogram
from pathlib import Path
from pytorch_lightning import LightningModule
from tqdm import tqdm
import numpy as np
import librosa
import whisper




class MelBasedAttackerLightning(LightningModule): 
    """ #NOTE: DEPRECATED
    TODO: Make MEL an option of a general attacker class
    LightningModule that prepends noise for adversarial attacks, based on
    https://arxiv.org/pdf/2405.06134
    """
    def __init__(self,
                 model:str = "tiny.en",
                 sec: Union[int, float] = 1,
                 prepend: bool = True,
                 epsilon: float = 0.02,
                 target_length: int = 48000,
                 batch_size: int = 64,
                 discriminator = Optional[nn.Module],
                 noise_type: str = "uniform", #Options are ['uniform','normal']
                 gamma: float = 1.,
                 no_speech: bool = False,
):
        super(MelBasedAttackerLightning, self).__init__()

        # Initialize noise as a learnable parameter
        self.noise = nn.Parameter(torch.rand(1, 80, int(sec * 100)), requires_grad=True).to(self.device)
        self.prepend = prepend
        self.target_length = target_length
        self.noise_len = int(sec * 100)
        self.model = whisper.load_model(model).to(self.device)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.no_speech = no_speech

        self.discriminator = discriminator


        if self.discriminator is not None:
            for param in self.discriminator.parameters():
                param.requires_grad = False

        #Fix sparse tensors for Pytorch Lightning
        alignment_heads_dense = self.model.get_buffer("alignment_heads").to_dense()
        self.model.register_buffer("alignment_heads", alignment_heads_dense, persistent=False)
        del alignment_heads_dense

        #Different tokenizers for English vs multilingual models
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
        if self.prepend:
            #Slice noise sized chunk from x & add noise
            x = x[:,:-self.noise_len]
            noise = self.noise.repeat(BATCH_SIZE,1,1)
            x = torch.cat([noise, x], dim=-1)



        else: #Adding Noise to tensor
            # pad_size = x.size(2) - noise.size(2)
            padding = torch.zeros_like(x)[:,:,:-self.noise_len]
            noise = self.noise.repeat(BATCH_SIZE,1,1)

            noise_padded = torch.cat([noise,padding],dim=-1)
            x = noise_padded + x
            return self.decoder.get_eot_prob(x)
    
    # Clamps to epsilon before passing to optimizer
    def on_before_optimizer_step(self, optimizer):
        if self.epsilon != -1:
            with torch.no_grad():
                # self.noise.clamp_(min=-80,max=0)
                self.noise.clamp_(min=-self.epsilon, max=self.epsilon)


    def training_step(self, batch, batch_idx):
        # Unpack the batch
        x, sampling_rate, transcript = batch

        x = pad_or_trim(x)
        x = log_mel_spectrogram(x)
        # probs = self.forward(x) #Gets probabilities

        if self.discriminator is not None:
            loss = -torch.log(self.forward(x) + 1e-9).mean() + self.gamma * self.discriminator(self.noise)

        else:
            loss = -torch.log(self.forward(x) + 1e-9).mean()
        self.log("train_loss", loss, batch_size=self.batch_size,prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def dump(self, path: Union[str, Path], mel: bool = False):
            log_mel_spec = self.noise
            mel_spec = torch.exp(log_mel_spec).detach().cpu().numpy()
    # Whisper audio parameters
            sample_rate = 16000
            n_fft = 400
            hop_length = 160
            win_length = 400
            n_mels = 80
            mel_fmin = 0
            mel_fmax = sample_rate / 2  # 8000 Hz
            S = librosa.feature.inverse.mel_to_stft(
                mel_spec,
                sr=sample_rate,
                n_fft=n_fft,
                power=1.0,  # Since we're dealing with magnitude spectrograms
                fmin=mel_fmin,
                fmax=mel_fmax,
                htk=True  # Use HTK formula as Whisper does

    )
            y = librosa.griffinlim(
                S,
                n_iter=60,  # Increased iterations for better quality
                hop_length=hop_length,
                win_length=win_length,
                window='hann',
                n_fft=n_fft
            )
            np.save(path,y)
            return 
    