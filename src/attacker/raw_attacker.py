"""
Trainable Attacker classes


"""
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
from src.mask_threshold import generate_th_batch, compute_PSD_matrix_batch


class RawAudioAttackerLightning(LightningModule):
    """
    LightningModule that prepends noise for adversarial attacks, based on
    https://arxiv.org/pdf/2405.06134

    NOTE: Assumes sample rate of 16000
    """
    def __init__(self,
                 model:str = "tiny.en",
                 sec: Union[int, float] = 1,
                 prepend: bool = True,
                 epsilon: float = 0.02,
                 target_length: int = 48000,
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 discriminator = Optional[nn.Module],
                 noise_type: str = "uniform", #Options are ['uniform','normal']
                 gamma: float = 1.,
                 no_speech: bool = False,
                 frequency_decay: tuple = (None,None),
                 frequency_penalty: bool = False,
                 frequency_masking:bool = False,
                 window_size:int = 2048,
                 
):
        super(RawAudioAttackerLightning, self).__init__()

        # Initialize noise as a learnable parameter
        self.noise = nn.Parameter(torch.rand(1, int(sec*16000)), requires_grad=True).to(self.device)

        #General hyperparameters for attack
        self.prepend = prepend
        self.target_length = target_length
        self.noise_len = int(sec * 16000)
        self.model = whisper.load_model(model).to(self.device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Imperceptibility parameters
        self.epsilon = epsilon
        self.gamma = gamma
        self.freq_decay, self.decay_strength = frequency_decay
        self.no_speech = no_speech
        self.discriminator = discriminator
        self.frequency_penalty = frequency_penalty

        self.frequency_masking = frequency_masking
        self.window_size = window_size

        #Fix sparse tensors for Pytorch Lightning
        alignment_heads_dense = self.model.get_buffer("alignment_heads").to_dense()
        self.model.register_buffer("alignment_heads", alignment_heads_dense, persistent=False)
        del alignment_heads_dense

        # Picking the proper Whisper model
        if "en" in model:
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False,task="transcribe")
        else:
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        self.decoder = RawDecoder(self.model,self.tokenizer,self.device)
        #Freezing Whisper weights
        for param in self.model.parameters():
            param.requires_grad = False

        # if self.discriminator is not None:
        #     for param in self.discriminator.parameters():
        #         param.requires_grad = False

        #Frequency Decay
        if self.freq_decay == "linear":
            self.pattern = torch.linspace(0,1,steps=80).to(self.device) * self.decay_strength

        elif self.freq_decay == 'polynomial':
            self.pattern = torch.pow(torch.linspace(0,1,steps=80).to(self.device),2) * self.decay_strength

        elif self.freq_decay == "logarithmic":
            self.pattern = torch.logspace(0, 2, steps=80).to(self.device) * self.decay_strength
  




    def frequency_decay(self,
                          mel,
                          transformation: Literal['linear','polynomial','exponential'] = None): #TODO: Add support for more than just linspace
        """
        Adds values to gradients to penalize higher frequencies
        Mel Spectrograms are (Batch_size, 80, Time)

        This function is called everytime, so if no transformation is set, then it returns input w/ no changes        
        """

        if transformation is None:
            return mel
        #Finish pattern 
        pattern = self.pattern.resize(1,80,1).to(self.device)
        pattern = pattern.expand_as(mel)
        return mel * pattern

    def to(self, device):
        #Updated 'to' function. Updates device of submodules
        super().to(device)

        self.noise = self.noise.to(device)

        self.decoder.device = device
        self.decoder.update_device(device)
        return self

    def forward(self, x, noise):
        x = x.to(self.device)
        BATCH_SIZE = x.shape[0]

        # Optionally prepend the noise
        
        if self.prepend:
            noise = self.frequency_decay(noise,self.freq_decay)
            noise = noise.repeat(BATCH_SIZE,1,1)
            #Slice noise sized chunk from x & add noise

            x = x[:, :, :-noise.shape[-1]]
            x = torch.cat([noise, x], dim=-1)
            
        else: #Adding Noise to tensor

            noise = self.frequency_decay(noise,self.freq_decay)
            noise = noise.repeat(BATCH_SIZE,1,1)

            padding = torch.zeros_like(x)[:,:,:-noise.shape[-1]]

            noise_padded = torch.cat([noise,padding],dim=-1)
            x = noise_padded + x

        return self.decoder.get_eot_prob(x)
    
    # Clamps to epsilon before passing to optimizer
    def on_before_optimizer_step(self, optimizer):
        if self.epsilon != -1:
            with torch.no_grad():
                self.noise.clamp_(max=0.02)
                # self.noise.clamp_(max=self.epsilon)

    def _mel_difference(self,mel1,mel2):
        mel1 = mel1.mean(dim=2)
        mel2 = mel2.mean(dim=2)

        difference = mel1 - mel2
        difference = torch.abs(difference)
        return difference.mean()

    def on_train_epoch_start(self):
        #TODO: Add discriminator training here
        return
    

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        x, sampling_rate, transcript = batch
        epoch_number = self.current_epoch + 1 # one-indexing epoch num for convenience

        x = pad_or_trim(x)
        if self.frequency_masking: # Saves a copy for PSD calculation if needed
            x_pad = x
   
        x = log_mel_spectrogram(x)
        # print("SHAPE",x.shape)
        noise_mel = log_mel_spectrogram(self.noise)
        # probs = self.forward(x) #Gets probabilities
        eot_prob, no_speech_prob = self.forward(x,noise_mel)

        loss = -torch.log(eot_prob + 1e-9).mean()
        if self.discriminator is not None:
            if epoch_number % 5 == 0: #Lagging the discriminator TODO: Make this a hyperparameter
                pass
            loss += self.gamma * self.discriminator(log_mel_spectrogram(self.noise))

        elif self.frequency_penalty:
            loss += self.gamma * self._mel_difference(x,noise_mel)
            
        elif self.no_speech:
            loss += self.gamma * -torch.log(no_speech_prob + 1e-9).mean() #TODO: Add custom no_speech 
        elif self.frequency_masking:
            # batch_th = self._freq_mask()
            print("aud",x_pad.shape)
            print(x_pad[:,:self.noise.shape[1]].shape)
            l_theta = self._threshold_loss(self.noise,x_pad[:,:self.noise.shape[1]], fs=sampling_rate,window_size = self.window_size)
            loss += self.gamma * l_theta

        
            
        # elif self.bark_penalty:
        #     loss += self.gamma * 


        self.log("train_loss", loss, batch_size=self.batch_size,prog_bar=True, on_step=True, on_epoch=True)
        return loss
    def _threshold_loss(self,noise,audio,fs=16000,window_size = 2048):
        print("Audio min:", audio.min().item(), "Audio max:", audio.max().item())

        theta_xs, psd_max, PSD_x  = generate_th_batch(audio,fs=fs,window_size=window_size)
        audio = torch.from_numpy(audio)
        print("NaNs in audio?", torch.isnan(audio).any().item())
        print("Infs in audio?", torch.isinf(audio).any().item())
         # theta_x is (n, 43, 1025)
        theta_xs = torch.from_numpy(theta_xs).to(self.device)
        print("theta min:", theta_xs.min().item(), "thetao max:", theta_xs.max().item())
        PSD_delta, PSD_max_delta = compute_PSD_matrix_batch(noise,self.window_size,transpose=True)
        # theta_xs = theta_xs[:, :attack_tsteps, :] # Match theta_xs with timesteps in 
        diff = torch.relu(PSD_delta - theta_xs) # 
        sum_over_freq = diff.sum(dim=2).mean(dim=1)
        sum_over_freq = sum_over_freq / theta_xs.shape[2] # final dim of theta_xs is the same as floor(1 / N/2 )
        return sum_over_freq
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def dump(self, path: Union[str, Path], mel: bool = False):
            noise = self.noise.detach().cpu().numpy()
            np.save(path,noise)
            return 


if __name__ == "__main__":
    device = "cuda"
    model = RawAudioAttackerLightning(prepend=False,discriminator=None,frequency_masking=True).to(device)

    x = whisper.load_audio("/home/jaydenfassett/audioversarial/imperceptible/original_audio.wav")
    x = x.reshape(1,-1)
    print(x.shape)
    print(model.noise.shape)
    # qq = log_mel_spectrogram(x)
    r = model.training_step((x,16000,1),1)
    print(r)
    # model.forward(torch.tensor(x).to(device),torch.randn(1,100).to(device))
    # x = torch.tensor(x).unsqueeze(0)
    # x = torch.cat([x,x,x,x],dim=0)
    # x = (x,1,1)

