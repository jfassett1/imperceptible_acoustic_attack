import whisper
from src.data import AudioDataModule
import argparse
import torch
from tqdm import tqdm
import numpy as np
from src.utils import overlay_torch
parser = argparse.ArgumentParser(description="Training Script Arguments")

# General training settings
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--dataset',type = str,default='dev-clean',choices=['dev-clean','train-clean-100','train-other-500'])
parser.add_argument("--no_train",action="store_true",default=False,help="Whether to train noise. Used for testing pathing & saving")
parser.add_argument("--whisper_model",choices=['tiny.en','base.en','small.en','medium.en'], default='tiny.en', help='Which Whisper model to use')
# Attack Settings
parser.add_argument('--domain',type=str,default="raw_audio",choices=['raw_audio',"mel"],help="Whether to attack in mel or audio space")
parser.add_argument('--attack_length',type=float,default=1.,help = "Length of attack in seconds")
parser.add_argument('--prepend',action="store_true", default = False,help="Whether to prepend or not")
parser.add_argument('--noise_dir',type=str,default=None,help="Where to save noise outputs")
parser.add_argument('--clip_val',type=float,default = -1,help="Clamping Value")
parser.add_argument('--gamma',type=float,default = 1., help= "Gamma value for scaling penalty")
parser.add_argument('--num_workers',type=float,default = 0,help="Clamping Value")

args = parser.parse_args()


class RawEvaluator:
    def __init__(self,
                 noise,
                 model:str,
                 dataset='dev-clean',
                 device='cuda',
                 attack = True,
                 prepend = False):
            self.dataloader = AudioDataModule(dataset_name=args.dataset,
                                               batch_size=1,
                                               num_workers=args.num_workers).train_dataloader()
            self.device = device
            self.model = whisper.load_model(model).to(self.device)
            self.noise = noise
            self.attack = attack
            if self.noise.ndim > 1:
                self.noise = self.noise.squeeze()

            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False,task="transcribe")
            self.prepend = prepend

    def _attack(self,
                x):
        """
        Applies attack
        """
        if not self.attack:
            return x
        if self.prepend:
            x = torch.cat([self.noise,x]).to(self.device)
        else:
            x = overlay_torch(self.noise,x)
        

        return x

                   
                   
        
    def calc_nsl(self):
        """
        Calculates negative sequence length
        
        """
        i = 0
        nsl = 0
        for batch in tqdm(self.dataloader):
            x, sampling_rate, transcript = batch
            x = x.to(self.device)
            x = self._attack(x.squeeze())
            
            output = self.model.transcribe(x)['text']
            # print(output)
            i+=1
            nsl += len(output)
            if i == 30:
                break
        nsl /= i
        print(nsl)
        return nsl         

if __name__ == "__main__":


    noise = torch.randn(16000).to('cuda')
    qq = RawEvaluator(noise,"tiny.en",prepend=True,device='cuda')
    # qq.model.transcribe(noise)
    qq.calc_nsl()
    pass
