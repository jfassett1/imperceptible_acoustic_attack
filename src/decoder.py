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
        self.sot_tokens = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=0).to(self.device)

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
        return eot_probs



    def autoregressive(self,mel,n=15):
        batch_size = mel.shape[0]
        sot_tokens = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=0).to(self.device)
        tokens = torch.zeros(size=(batch_size,n + len(sot_tokens)),dtype=torch.int64).to(self.device)
        
        tokens[:,:sot_tokens.shape[1]] = sot_tokens
        # tokens[]
        # tokens = torch.transpose(tokens,0,1)
        eot_id = self.tokenizer.eot
        

        for i in range(n):
            logits = self.forward(mel,tokens[:,:i+len(sot_tokens)])
            # print(logits.shape)
            # print(logits.shape)
            pred = logits[:,-1].argmax(dim=-1)

            # print(pred == eot_id)
            tokens[0:,i+len(sot_tokens)] = pred
            if n == None:
                if pred == eot_id:
                    break
            # softmax = logits.softmax(dim=2)
            # if i == n:
            #     break
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

    NAME = "tiny.en"
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False,task="transcribe")
    model = whisper.load_model(NAME)

    sot_tokens = torch.tensor(tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=1).to(device)
    qq = EvilDecoder(model=model,tokenizer=tokenizer,device=device)

    # seq= qq.get_eot_prob(mels)
    # print("EOT",tokenizer.eot)
    # seq = qq.autoregressive(mels,n=3)

    print(prep(q).shape)
    # print(seq)
    # print(qq.get_eot_prob(prep(q)))
    # print(tokenizer.decode(seq.squeeze()))
    # print(prep(x).shape)
    # pred = qq.forward(prep(x),sot_tokens)
    # print("SOT tokens:",tokenizer.decode(sot_tokens))
    # print(pred.argmax())
    # print(tokenizer.decode([pred.argmax()]))
    # eot_prob1 = qq.get_eot_prob(prep(x))
    # eot_prob2 = qq.get_eot_prob(prep(q))

    # print(eot_prob1,eot_prob2)
    

    