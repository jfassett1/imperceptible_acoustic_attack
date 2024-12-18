import whisper
from src.data import AudioDataModule
import argparse
import torch
from tqdm import tqdm
import numpy as np
from src.utils import overlay_torch

def get_args():
    parser = argparse.ArgumentParser(description="Training Script Arguments")

    parser.add_argument('--dataset',type = str,default='dev-clean',choices=['dev-clean','train-clean-100','train-other-500'])
    parser.add_argument("--whisper_model",choices=['tiny.en','base.en','small.en','medium.en'], default='tiny.en', help='Which Whisper model to use')
    parser.add_argument('--domain',type=str,default="raw_audio",choices=['raw_audio',"mel"],help="Whether to attack in mel or audio space")
    parser.add_argument('--prepend',action="store_true", default = False,help="Whether to prepend")
    parser.add_argument('--noise_path',type=str,help="Path of noise .npy")
    parser.add_argument('--num_workers', type=int,default = 0, help="Number of data loading workers")
    parser.add_argument('--no_attack', action="store_true", default=False, help = "Whether to use attack or not")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'], help='Device to use for training')
    args = parser.parse_args()
    return args


class RawEvaluator:
    def __init__(self,
                 noise,
                 model:str,
                 dataset='dev-clean',
                 device='cuda',
                 no_attack = False,
                 prepend = False):
            self.dataloader = AudioDataModule(dataset_name=dataset,
                                               batch_size=1,
                                               num_workers=args.num_workers).train_dataloader()
            self.device = device
            self.model = whisper.load_model(model).to(self.device)
            self.noise = torch.from_numpy(np.load(noise)).to(self.device) # Load noise & convert to tensor
            self.no_attack = no_attack
            if self.noise.ndim > 1:
                self.noise = self.noise.squeeze()

            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False,task="transcribe")
            self.prepend = prepend

    def _attack(self,
                x):
        """
        Applies attack
        """
        if self.no_attack:
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
        nsl /= i
        print(nsl)
        return nsl         

if __name__ == "__main__":

    args = get_args()
    # noise = torch.randn(16000).to('cuda')

    qq = RawEvaluator(args.noise_path,
                      args.whisper_model,
                      args.dataset,
                      args.device,
                      args.no_attack,
                      args.prepend,
                      )
    # qq.model.transcribe(noise)
    qq.calc_nsl()