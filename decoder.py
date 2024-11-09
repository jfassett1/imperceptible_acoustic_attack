"""
Modified decoder file from original whisper repo.
"""
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import whisper

if TYPE_CHECKING:
    from whisper.model import Whisper




class EvilDecoder():

    """
    Modified Whisper decoder which pulls out target given sequence
    NOTE: Doesn't support beamsearch


    """
    def __init__(self,model,tokenizer,device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def forward(self,mel,tokens):

        return self.model.forward(mel,tokens)
    def get_eot_prob(self,mel):
        print("Input mel shape:", mel.shape)
        eot_id = self.tokenizer.eot
        sot_tokens = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=1).to(self.device)
        logits = self.forward(mel,sot_tokens)
        print(logits.shape)
        probs = logits.softmax(dim=-1)
        eot_probs = probs[:,:,eot_id]

        print("EOT probs shape:", eot_probs.shape)
        print("EOT probs:", eot_probs)

        return eot_probs



    def autoregressive(self,mel,n=5):
        sot_tokens = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=1).to(self.device)
        tokens = torch.zeros(size=(n + len(sot_tokens),),dtype=torch.int64).to(self.device)
        tokens[range(len(sot_tokens))] = sot_tokens.reshape(2)
        eot_id = self.tokenizer.eot
        
        likelihoods = []

        for i in range(n):
            logits = self.forward(mel,tokens)

            pred = logits.argmax()
            tokens[i+len(sot_tokens)] = pred
            softmax = logits.softmax(dim=2)



            likelihoods.append(softmax[eot_id])
        print(likelihoods)
        print(tokens)
        
if __name__ == "__main__":
    x = whisper.load_audio("/home/jaydenfassett/audioversarial/imperceptible/original_audio.wav")
    q = whisper.load_audio("/home/jaydenfassett/audioversarial/imperceptible/audio_with_attack.wav")
    device = "cuda"

    def prep(x):
        x = whisper.pad_or_trim(x)
        mel = whisper.log_mel_spectrogram(x).to(device).unsqueeze(dim=0)
        return mel

    # mels = torch.cat((prep(q),prep(x)),dim=0)
    


    NAME = "tiny.en"
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False,task="transcribe")
    model = whisper.load_model(NAME)


    qq = EvilDecoder(model=model,tokenizer=tokenizer,device=device)

    print(qq.get_eot_prob(prep(q)), "%")
    

    