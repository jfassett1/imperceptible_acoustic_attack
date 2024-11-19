"""
Modified decoder file from original whisper repo.
"""
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import whisper

if TYPE_CHECKING:
    from whisper.model import Whisper




class RawDecoder():

    """
    Modified Whisper decoder which pulls out target given sequence
    NOTE: Doesn't support beamsearch


    """
    def __init__(self,model,tokenizer,device):
        super(RawDecoder, self).__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.sot_tokens = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=0).to(self.device)
    def update_device(self,device):
        self.model = self.model.to(device)

    def forward(self,mel,tokens=None):
        if type(tokens) == None:
            tokens = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=1).to(self.device)
            
        return self.model.forward(mel,tokens)
    
    def get_eot_prob(self,mel):
        eot_id = self.tokenizer.eot
        sot_tokens = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=0).to(self.device)
        logits = self.forward(mel,sot_tokens)
        probs = logits.softmax(dim=-1)
        eot_probs = probs[:,0,eot_id]
        # print("PROBS",eot_probs)
        return eot_probs



    def autoregressive(self,mel,n=15):
        batch_size = mel.shape[0]
        sot_tokens = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=0).to(self.device)
        tokens = torch.zeros(size=(batch_size,n + len(sot_tokens)),dtype=torch.int64).to(self.device)
        
        tokens[:,:sot_tokens.shape[1]] = sot_tokens
        eot_id = self.tokenizer.eot
        

        for i in range(n):
            logits = self.forward(mel,tokens[:,:i+len(sot_tokens)])
            pred = logits[:,-1].argmax(dim=-1)
            tokens[0:,i+len(sot_tokens)] = pred
            if n == None:
                if pred == eot_id:
                    break
                
        return tokens

            # likelihoods.append(softmax[eot_id])
        # print(likelihoods)
        # print(tokens)
        
if __name__ == "__main__":
    x = whisper.load_audio("/home/jaydenfassett/audioversarial/imperceptible/original_audio.wav")
    q = whisper.load_audio("/home/jaydenfassett/audioversarial/imperceptible/audio_with_attack.wav")
    device = "cuda"
    # print("before:",x.shape)
    # x = np.concatenate([x,x,x])
    # print("after",x.shape)
    def prep(x):
        x = whisper.pad_or_trim(x)
        mel = whisper.log_mel_spectrogram(x).to(device).unsqueeze(dim=0)
        return mel

    # mels = torch.cat((prep(q),prep(x)),dim=0)
    

    mels = torch.cat((prep(q),prep(x)),dim=0).to(device)
    print(mels.shape)

    NAME = "tiny.en"
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False,task="transcribe")
    model = whisper.load_model(NAME)

    sot_tokens = torch.tensor(tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=1).to(device)
    qq = RawDecoder(model=model,tokenizer=tokenizer,device=device)

    ll = qq.get_eot_prob(mels)
    # print(ll)

    