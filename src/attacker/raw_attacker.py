"""
Trainable Attacker class


"""
import functools
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import whisper
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch.nn.functional as F
from whisper import log_mel_spectrogram, pad_or_trim

from src.decoder import RawDecoder
from src.masking.mask_threshold import compute_PSD_matrix_batch
from src.utils import overlay_torch
from src.masking.mel_mask import generate_mel_th, log_mel_spectrogram_raw
from pytorch_lightning.callbacks import Callback
import torch.distributed as dist
# For preventing FutureWarning log from whisper.
whisper.torch.load = functools.partial(whisper.torch.load, weights_only=True)

    
class RawAudioAttackerLightning(LightningModule):
    """
    LightningModule that prepends noise for adversarial attacks, based on
    https://arxiv.org/pdf/2405.06134

    NOTE: Assumes sample rate of 16000
    """

    def __init__(self,
                 model: str = "tiny.en",
                 sec: Union[int, float] = 1,
                 prepend: bool = True,
                 epsilon: Union[float, tuple, None] = (-0.02, 0.02),
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 discriminator: Optional[nn.Module] = None,
                 # Options are ['uniform','normal']
                 noise_type: str = "uniform",
                 gamma: float = 1.,
                 no_speech: bool = False,
                 frequency_decay: tuple = (None, None),
                 frequency_penalty: bool = False,
                 frequency_masking: bool = False,
                 window_size: int = 2048,
                 masker_cores: int = 16,
                 mask_threshold: Union[np.array, np.ndarray] = None,
                 concat_mel: bool = False,
                 mel_mask: bool = False,
                 finetune:bool = False,
                 debug: bool = False,  # Prints all special activated sections
                 offset: float = 0
                 ):
        super(RawAudioAttackerLightning, self).__init__()

        # Initialize noise. If epsilon exists, it will pre-emptively clamp.
        # print(epsilon)

        with torch.no_grad():  # Avoid tracking the initialization
            noise = torch.empty(1, int(sec*16000),
                                device=self.device).uniform_(-1, 1)
            if None not in epsilon:
                noise.clamp_(min=epsilon[0], max=epsilon[1])
                # noise.clamp_(min=0.00, max=0.00)
        # # print("Noise strides before:", noise.stride())
        self.noise = nn.Parameter(noise.contiguous().reshape(1, -1))
        # self.noise = nn.Parameter(noise.contiguous().reshape(1, -1))
        # self.noise = nn.Parameter(torch.rand(1, int(sec * 16000)), requires_grad=True).to(self.device)
        # General hyperparameters for attack
        self.prepend = prepend
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
        self.concat_mel = concat_mel
        self.mel_mask = mel_mask
        self.debug = debug
        self.finetune = finetune
        self.last_loss = None
        self.stage = 1
        self.offset = offset

        self.frequency_masking = frequency_masking
        self.window_size = window_size
        if mask_threshold is not None:
            self.mask_threshold = torch.from_numpy(
                mask_threshold).unsqueeze(0).to(device)
            # self.mask_threshold = self.mask_threshold[:,:self.noise_len,:]

        # Fix sparse tensors for Pytorch Lightning
        alignment_heads_dense = self.model.get_buffer(
            "alignment_heads").to_dense()
        self.model.register_buffer(
            "alignment_heads", alignment_heads_dense, persistent=False)
        del alignment_heads_dense

        # Picking the proper Whisper model
        if "en" in model:
            self.tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=False, task="transcribe")
        else:
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True,task="transcribe")


        self.decoder = RawDecoder(self.model, self.tokenizer, self.device)
        # Freezing Whisper weights
        for param in self.model.parameters():
            param.requires_grad = False

        # if self.discriminator is not None:
        #     for param in self.discriminator.parameters():
        #         param.requires_grad = False

        # Frequency Decay
        if self.freq_decay == "linear":
            self.pattern = torch.linspace(0, 1, steps=80).to(
                self.device) * self.decay_strength

        elif self.freq_decay == 'polynomial':
            self.pattern = torch.pow(torch.linspace(0, 1, steps=80).to(
                self.device), 2) * self.decay_strength

        elif self.freq_decay == "logarithmic":
            self.pattern = torch.logspace(0, 2, steps=80).to(
                self.device) * self.decay_strength

    def frequency_decay(self,
                        mel,
                        transformation: Literal['linear', 'polynomial', 'exponential'] = None):  # TODO: Add support for more than just linspace
        """
        Adds values to gradients to penalize higher frequencies
        Mel Spectrograms are (Batch_size, 80, Time)

        This function is called everytime, so if no transformation is set, then it returns input w/ no changes        
        """

        if transformation is None:
            return mel
        # Finish pattern
        if self.debug:
            print("Frequency Decay")
        pattern = self.pattern.resize(1, 80, 1).to(self.device)
        pattern = pattern.expand_as(mel)
        return mel * pattern

    def to(self, device):
        # Updated 'to' function. Updates device of submodules
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
            raise NotImplementedError # NOTE: Takes in Mel Spectrogram, which is currently out of date
            noise = self.frequency_decay(noise, self.freq_decay)
            noise = noise.repeat(BATCH_SIZE, 1, 1)
            # Slice noise sized chunk from x & add noise

            x = x[:, :, :-noise.shape[-1]]
            x = torch.cat([noise, x], dim=-1)

        else:  # Adding Noise to tensor
            if self.concat_mel:
                noise = self.frequency_decay(noise, self.freq_decay)
                noise = noise.repeat(BATCH_SIZE, 1, 1)

                padding = torch.zeros_like(x)[:, :, :-noise.shape[-1]]

                noise_padded = torch.cat([noise, padding], dim=-1)
                x = noise_padded + x
            else:

                x = overlay_torch(noise, x)
                x = log_mel_spectrogram(x)

        return self.decoder.get_eot_prob(x)

    # Clamps to epsilon before passing to optimizer
    def on_before_optimizer_step(self, optimizer):
        if None not in self.epsilon:
            with torch.no_grad():
                self.noise.clamp_(min=self.epsilon[0], max=self.epsilon[1])
                # print(self.epsilon)
                # exit()
    def _mel_difference(self, mel1, mel2):

        mel1 = mel1.mean(dim=2)
        mel2 = mel2.mean(dim=2)

        difference = mel1 - mel2
        difference = F.relu(difference)

        return difference.sum()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs['loss'] < 10:
            return -1
        return super().on_train_batch_end(outputs, batch, batch_idx)
    def on_train_batch_start(self, batch, batch_idx):

        return
        if self.global_step == 0:
            return
        # First stage, modify epsilon if needed
        if self.stage == 1 and self.global_step % 10 == 0 and self.last_eot_per > 0.9:
            if None not in self.epsilon:
                noise_max = self.noise.detach().abs().max()
                if self.epsilon[1] > noise_max:
                    self.epsilon = (-noise_max, noise_max)
                    self.log("eps",noise_max,prog_bar=True)
        if self.stage == 2:
            if self.global_step % 20 == 0:
                if self.last_eot_per > 0.9:
                    self.gamma *= 1.2
            if self.global_step % 50 == 0:
                if self.last_eot_per < 0.8:
                    self.gamma *= 0.8


            



    def reset_optimizer(self):
        # Switch optimizer for fine-tuning
        # Reset the optimizer so it doesn't keep momentum.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.gamma)

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        x, sampling_rate, transcript, lengths = batch
    
        x = pad_or_trim(x)
        x_pad = x
        eot_prob, no_speech_prob, total_probs = self.forward(x, self.noise)

        loss = -torch.log(eot_prob + 1e-9).mean()   
        loss_f = None

        if self.discriminator is not None:
            raise NotImplementedError
            # if self.debug:
            #     print("Discriminator")
            # # Optionally, add logic here if you want to lag the discriminator
            # loss_f = self.discriminator(log_mel_spectrogram(self.noise))

        elif self.frequency_penalty:
            if self.debug:
                print("Frequency Penalty")
            noise_mel = log_mel_spectrogram(self.noise)
            loss_f = self._mel_difference(log_mel_spectrogram(x), noise_mel)

        elif self.no_speech:
            loss_f = -torch.log(no_speech_prob + 1e-9).mean()

            if self.debug:
                print("NoSpeech")

        elif self.frequency_masking:
            if self.debug:
                print("Frequency Masking")
            # x_pad should have been computed in training_step if frequency_masking is True.
            l_theta = self._threshold_loss(self.noise, x_pad[:, :self.noise.shape[1]],
                                           fs=sampling_rate, window_size=self.window_size)
            loss_f = l_theta

        elif self.mel_mask:
            # from src.masking.mel_mask import plotnshow
            OFFSET = 60
            samp_mels = log_mel_spectrogram_raw(x)
            threshold = generate_mel_th(samp_mel=samp_mels,lengths=lengths,offset = OFFSET) + self.offset
            noise_mel = log_mel_spectrogram_raw(self.noise,in_db=True) # convert noise mel to db
            # noise_mel = noise_mel +  (OFFSET - noise_mel.max())
            noise_mel_len = noise_mel.shape[-1]

            # plotnshow(noise_mel.detach().cpu().numpy()[0,:,10],threshold.detach().cpu().numpy()[0,:,10])
            # exit()
            diffs = noise_mel - threshold[:,:,:noise_mel_len]
            margin = 2
            z = F.relu(diffs) # Removing vals lower than the threshold
            # z = F.relu(diffs - margin) # Removing vals lower than the threshold
            # z = F.softplus(diffs) # Removing vals lower than the threshold

            loss_f = z.mean()
            # print("Noise power:", (self.noise**2).mean().item())
            # print("Noise dB range:", noise_mel.min().item(), noise_mel.mean().item(), noise_mel.max().item())
            # print("Threshold dB range:", threshold.min().item(), threshold.mean().item(), threshold.max().item())

        # print(loss_f)
        # print(f"Gamma: {self.gamma}, Loss: {loss.item()}, Loss_f: {loss_f.item()}")
        if loss_f is not None:
            loss = (loss * (1 - self.gamma) + loss_f * self.gamma)
            if abs(self.gamma - 1.0) < 1e-6:
                assert abs(loss.item() - loss_f.item()) < 1e-3, "Loss != Loss_f even though gamma == 1"
        prob_argmax = total_probs.argmax(dim=-1)
        eot_per = (prob_argmax == self.tokenizer.eot).sum() / len(prob_argmax)
        pred = prob_argmax.mode()[0].item() # Most likely value


        self.trainer.progress_bar_callback.set_token(self.tokenizer.decode([pred]))
        self.debug = False
        metrics = {
            "eot_per": round(eot_per.item(), 2),
            "eot_pr": eot_prob.mean(),
            "pred":pred
        }

        self.log("loss", loss, batch_size = self.batch_size,
                 prog_bar=True, on_step=True, on_epoch=True)
        self.log("gamma",self.gamma,prog_bar=True,on_step=True)
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, batch_size=self.batch_size, on_step=True)
        self.last_loss = loss.detach()
        self.last_eot_per = eot_per

        return loss
    def validation_step(self, batch, batch_idx):
        x, sampling_rate, transcript, lengths = batch
        epoch_number = self.current_epoch + 1  # one-indexing epoch num for convenience
        x = pad_or_trim(x)
        if self.frequency_masking:  # Saves a copy for PSD calculation if needed
            x_pad = x
        eot_prob, no_speech_prob, total_probs = self.forward(x, self.noise)
        prob_argmax = total_probs.argmax(dim=-1)
        eot_per = (prob_argmax == self.tokenizer.eot).sum() / len(prob_argmax)
        loss = -torch.log(eot_prob + 1e-9).mean()
        self.log("val_loss", loss, batch_size=self.batch_size,prog_bar=True)
        self.log("val_per", eot_per, batch_size=self.batch_size,prog_bar=True)


        # if eot_per.item() > 0.95:
        #     self.stage = 2
        #     self.gamma = 0.05
        return loss

    def _threshold_loss(self, noise, audio, fs=16000, window_size=2048):

        PSD_delta, PSD_max_delta = compute_PSD_matrix_batch(
            noise, self.window_size, transpose=True)

        t_max = PSD_delta.shape[1]
        theta_xs = self.mask_threshold[:, :t_max, :]

        diff = torch.relu(PSD_delta - theta_xs)
        sum_over_freq = diff.sum(dim=2).mean(dim=1)
        # final dim of theta_xs is the same as floor(1 / N/2 )
        sum_over_freq = sum_over_freq / theta_xs.shape[2]
        # print("q",sum_over_freq)
        return sum_over_freq.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        return optimizer

    def dump(self, path: Union[str, Path], mel: bool = False):
        noise = self.noise.detach().cpu().numpy()
        np.save(path, noise)
        return


if __name__ == "__main__":
    device = "cuda"
    # mt = np.load("/home/jaydenfassett/audioversarial/imperceptible/thresholds/train-clean-100.np.npy")
    model = RawAudioAttackerLightning(prepend=False,mel_mask=True,gamma=0.5).to(device)

    x = whisper.load_audio(
        "/home/jaydenfassett/audioversarial/imperceptible/original_audio.wav")
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = torch.concat([x, x])
    # print(x.shape)

    # x = torch.tensor([x,x]).to(device)
    # print(x.shape)
    # print(model.noise.shape)
    # qq = log_mel_spectrogram(x)
    r = model.training_step((x, 16000, 1), 1)
    print(r)
    # print(r)
    # model.forward(torch.tensor(x).to(device),torch.randn(1,100).to(device))
    # x = torch.tensor(x).unsqueeze(0)
    # x = torch.cat([x,x,x,x],dim=0)
    # x = (x,1,1)
