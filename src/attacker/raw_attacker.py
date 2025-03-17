"""
Trainable Attacker classes


"""
import torch
import torch.nn as nn
from src.decoder import RawDecoder
from src.utils import overlay_torch
from typing import Union, Optional, Literal
from whisper import pad_or_trim, log_mel_spectrogram
from pathlib import Path
from pytorch_lightning import LightningModule
from tqdm import tqdm
import numpy as np
# import librosa
import whisper
from src.masking.mask_threshold import generate_th_batch, compute_PSD_matrix_batch
from torch.nn.functional import relu

import functools

whisper.torch.load = functools.partial(whisper.torch.load, weights_only=True) # For preventing FutureWarning log from whisper.

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
                 epsilon: Union[float,tuple,None] = (-0.02,0.02),
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 discriminator: Optional[nn.Module] = None,
                 noise_type: str = "uniform", #Options are ['uniform','normal']
                 gamma: float = 1.,
                 no_speech: bool = False,
                 frequency_decay: tuple = (None,None),
                 frequency_penalty: bool = False,
                 frequency_masking:bool = False,
                 window_size:int = 2048,
                 masker_cores:int = 16,
                 mask_threshold : Union[np.array,np.ndarray] = None,
                 concat_mel:bool = False,
                 imper_epochs:int = 0,
                 train_epochs:int = 1,
                 debug:bool=False # Prints all special activated sections

                 
):
        super(RawAudioAttackerLightning, self).__init__()

        #Initialize noise. If epsilon exists, it will pre-emptively clamp.
        # print(epsilon)
        
        with torch.no_grad(): #Avoid tracking the initialization
            noise = torch.empty(1, int(sec*16000),device=self.device).uniform_(-1, 1)
            if None not in epsilon:
                noise.clamp_(min=epsilon[0], max=epsilon[1])
            
        # print("Noise strides before:", noise.stride())
        self.noise = nn.Parameter(noise.contiguous().reshape(1, -1))
        # print("Noise strides after:", self.noise.stride())


        # print(self.noise)


        #General hyperparameters for attack
        self.prepend = prepend
        self.noise_len = int(sec * 16000)
        self.model = whisper.load_model(model).to(self.device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.imper_epochs = imper_epochs
        self.train_epochs = train_epochs

        # Imperceptibility parameters
        self.epsilon = epsilon
        self.gamma = gamma
        self.freq_decay, self.decay_strength = frequency_decay
        self.no_speech = no_speech
        self.discriminator = discriminator
        self.frequency_penalty = frequency_penalty
        self.concat_mel = concat_mel
        self.debug = debug

        self.frequency_masking = frequency_masking
        self.window_size = window_size
        if mask_threshold is not None:
            self.mask_threshold = torch.from_numpy(mask_threshold).unsqueeze(0).to(device)
            # self.mask_threshold = self.mask_threshold[:,:self.noise_len,:]

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
        if self.debug:
            print("Frequency Decay")
        pattern = self.pattern.resize(1,80,1).to(self.device)
        pattern = pattern.expand_as(mel)
        return mel * pattern

    def to(self, device):
        #Updated 'to' function. Updates device of submodules
        super().to(device)

        self.noise = self.noise.to(device)

        self.decoder.device = device
        self.decoder.update_device(device)
        self.model.to(device)
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
            if self.concat_mel:
                noise = self.frequency_decay(noise,self.freq_decay)
                noise = noise.repeat(BATCH_SIZE,1,1)

                padding = torch.zeros_like(x)[:,:,:-noise.shape[-1]]

                noise_padded = torch.cat([noise,padding],dim=-1)
                x = noise_padded + x
            else:

                x = overlay_torch(noise,x)
                x = log_mel_spectrogram(x)

            
        return self.decoder.get_eot_prob(x)
    
    # Clamps to epsilon before passing to optimizer
    def on_before_optimizer_step(self, optimizer):
        if None not in self.epsilon:
            with torch.no_grad():                       
                self.noise.clamp_(min=self.epsilon[0], max=self.epsilon[1])

    def _mel_difference(self,mel1,mel2):

        mel1 = mel1.mean(dim=2)
        mel2 = mel2.mean(dim=2)

        difference = mel1 - mel2
        difference = relu(difference)

        return difference.sum()

    def on_train_epoch_start(self):
        #TODO: Add discriminator training here
        return
    def on_train_epoch_end(self):
        # Switch optimizer for fine-tuning
        if self.current_epoch == (self.trainer.max_epochs - self.imper_epochs):
            #Reset the optimizer so it doesn't keep momentum.
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.gamma)            

    
    def _main_training_phase(self, x):
        """
        Main training phase: uses the default forward pass and loss.
        Returns the loss and a dictionary of additional metrics.
        """
        # Indicates main phase (could be logged if needed)
        finetune = False  
        eot_prob, no_speech_prob, total_probs = self.forward(x, self.noise)
        loss = -torch.log(eot_prob + 1e-9).mean()
        
        # Calculate extra metrics
        prob_argmax = total_probs.argmax(dim=-1)
        eot_per = (prob_argmax == self.tokenizer.eot).sum() / len(prob_argmax)
        metrics = {
            "eot_per": round(eot_per.item(), 2),
            "eot_pr": eot_prob.mean()
        }
        return loss, metrics    
    def _fine_tuning_phase(self, x, sampling_rate):
        """
        Fine-tuning phase: chooses different loss functions based on which mode is active.
        Returns the loss and (optionally) a dictionary of additional metrics.
        """
        finetune = True 
        metrics = {} 
        x_pad = x
        # Branch based on fine-tuning mode
        if self.discriminator is not None:
            if self.debug:
                print("Discriminator")
            # Optionally, add logic here if you want to lag the discriminator
            loss = self.discriminator(log_mel_spectrogram(self.noise))

        elif self.frequency_penalty:
            if self.debug:
                print("Frequency Penalty")
            noise_mel = log_mel_spectrogram(self.noise)
            loss = self._mel_difference(log_mel_spectrogram(x), noise_mel)

        elif self.no_speech:
            no_speech_prob = self.forward(x, self.noise)[1] 
            loss = -torch.log(no_speech_prob + 1e-9).mean()

            if self.debug:
                print("NoSpeech")

        elif self.frequency_masking:
            if self.debug:
                print("Frequency Masking")
            # x_pad should have been computed in training_step if frequency_masking is True.
            l_theta = self._threshold_loss(self.noise, x_pad[:, :self.noise.shape[1]],
                                        fs=sampling_rate, window_size=self.window_size)
            loss = l_theta
        else:
            return NameError
        
        return loss, metrics
    
    def training_step(self, batch, batch_idx):
        # Unpack the batch
        x, sampling_rate, transcript = batch
        epoch_number = self.current_epoch + 1 # one-indexing epoch num for convenience

        x = pad_or_trim(x)

        # print(epoch_number,self.train_epochs)
        if epoch_number <= self.train_epochs:
            loss, metrics = self._main_training_phase(x)
        else:
            loss, metrics = self._fine_tuning_phase(x, sampling_rate)

        self.debug = False


        self.log("loss", loss, batch_size=self.batch_size,prog_bar=True, on_step=True, on_epoch=True)
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, on_step=True)
        return loss
    def validation_step(self,batch,batch_idx):
        x, sampling_rate, transcript = batch
        epoch_number = self.current_epoch + 1 # one-indexing epoch num for convenience

        x = pad_or_trim(x)
        if self.frequency_masking: # Saves a copy for PSD calculation if needed
            x_pad = x
        eot_prob, no_speech_prob,total_probs = self.forward(x,self.noise)

        loss = -torch.log(eot_prob + 1e-9).mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def _threshold_loss(self,noise,audio,fs=16000,window_size = 2048):
        # print("Audio min:", audio.min().item(), "Audio max:", audio.max().item())
        # theta_xs, psd_max, PSD_x  = generate_th_batch(audio,
        #                                               fs=fs[0], #Sampling rate returns tuple of all samples
        #                                               window_size=window_size)
        # audio = torch.from_numpy(audio)

         # theta_x is (n, 43, 1025)
        # theta_xs = torch.from_numpy(theta_xs).to(self.device)
        # print("theta min:", theta_xs.min().item(), "theta max:", theta_xs.max().item())
        PSD_delta, PSD_max_delta = compute_PSD_matrix_batch(noise,self.window_size,transpose=True)
        
        t_max = PSD_delta.shape[1]
        theta_xs = self.mask_threshold[:,:t_max,:]
        # print(theta_xs.shape)
        # print(PSD_delta.shape,theta_xs.shape)
        # theta_xs = theta_xs[:, :attack_tsteps, :] # Match theta_xs with timesteps in 
        diff = torch.relu(PSD_delta - theta_xs) # 
        sum_over_freq = diff.sum(dim=2).mean(dim=1)
        sum_over_freq = sum_over_freq / theta_xs.shape[2] # final dim of theta_xs is the same as floor(1 / N/2 )
        # print("q",sum_over_freq)
        return sum_over_freq.mean()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def dump(self, path: Union[str, Path], mel: bool = False):
            noise = self.noise.detach().cpu().numpy()
            np.save(path,noise)
            return 


if __name__ == "__main__":
    device = "cuda"
    # mt = np.load("/home/jaydenfassett/audioversarial/imperceptible/thresholds/train-clean-100.np.npy")
    model = RawAudioAttackerLightning(prepend=False).to(device)


    x = whisper.load_audio("/home/jaydenfassett/audioversarial/imperceptible/original_audio.wav")
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = torch.concat([x,x])
    # print(x.shape)

    # x = torch.tensor([x,x]).to(device)
    # print(x.shape)
    # print(model.noise.shape)
    # qq = log_mel_spectrogram(x)
    r = model.training_step((x,16000,1),1)
    print(r)
    # print(r)
    # model.forward(torch.tensor(x).to(device),torch.randn(1,100).to(device))
    # x = torch.tensor(x).unsqueeze(0)
    # x = torch.cat([x,x,x,x],dim=0)
    # x = (x,1,1)

