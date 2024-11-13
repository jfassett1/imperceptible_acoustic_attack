"""
Trainable Attacker class


"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .decoder import EvilDecoder
from typing import Union
from whisper import pad_or_trim
from pathlib import Path
from .utils import raw_to_mel,sample_mel


class MelBasedAttacker(nn.Module):
    """
    Prepends noise based on https://arxiv.org/pdf/2405.06134
    
    """
    def __init__(self,
                 sec:Union[int,float] = 1):
        super(MelBasedAttacker, self).__init__()
        
        self.noise = nn.Parameter(torch.randn(80,sec*100)).unsqueeze(dim=0)
    def forward(self,x):

        x = pad_or_trim(torch.cat([self.noise,x],dim=-1))
        return x
    def dump(self,
             path:Union[str,Path], mel:bool = False):
        pass
        #TODO: Dump raw audio or mel representation of trained attack


class MelBasedAttackerLightning(pl.LightningModule):
    """
    Prepends noise based on https://arxiv.org/pdf/2405.06134
    
    """
    def __init__(self,
                 sec:Union[int,float] = 1,
                 prepend:bool = True):
        
        super(MelBasedAttackerLightning, self).__init__()

        self.attacker = MelBasedAttacker(sec)

    def forward(self,x):
        return self.attacker(x)
    def training_step(self,batch,batch_idx):
        pass
if __name__ == "__main__":
    device = "cuda"
    model = MelBasedAttacker().to(device)
    
    qq = model(sample_mel)


    print(qq)