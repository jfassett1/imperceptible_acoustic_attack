import whisper
from src.data import AudioDataModule
import argparse
import torch
from tqdm import tqdm
import numpy as np
from src.utils import overlay_torch
# from pesq import pesq_batch
def get_args():
    parser = argparse.ArgumentParser(description="Training Script Arguments")

    parser.add_argument('--dataset',type = str,default='dev-clean',choices=['dev-clean','train-clean-100','train-other-500'])
    parser.add_argument("--whisper_model",choices=['tiny.en','base.en','small.en','medium.en'], default='tiny.en', help='Which Whisper model to use')
    parser.add_argument('--domain',type=str,default="raw_audio",choices=['raw_audio',"mel"],help="Whether to attack in mel or audio space")
    parser.add_argument('--prepend',action="store_true", default = False,help="Whether to prepend")
    parser.add_argument('--noise_path',type=str,required = True, help="Path of noise .npy")
    parser.add_argument('--num_workers', type=int,default = 0, help="Number of data loading workers")
    parser.add_argument('--no_attack', action="store_true", default=False, help = "Whether to use attack or not")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'], help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for eval')

    args = parser.parse_args()
    return args


def evaluate(noise,data_module,args):
    noise = noise
    evaluator = RawEvaluator(noise,
                                args.whisper_model,
                                num_workers=args.num_workers,
                                data_module=data_module)
    asl = evaluator.calc_asl()


class RawEvaluator:
    def __init__(self,
                 noise,
                 model:str,
                 data_module,
                 dataset='dev-clean',
                 device='cuda',
                 no_attack = False,
                 prepend = False,
                 batch_size = 1,
                 num_workers=0,
                 ):
            self.dataloader = data_module.val_dataloader(batch_size=1)
            self.device = device
            self.model = whisper.load_model(model).to(self.device)
            self.noise = noise.to(device) if isinstance(noise,torch.Tensor) else noise # Load noise & convert to tensor
            self.no_attack = no_attack
            self.batch_size = batch_size
            if self.noise.ndim > 1:
                self.noise = self.noise.squeeze()

            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False,task="transcribe")
            self.prepend = prepend

    def _attack(self,
                x,
                noise):
        """
        Applies attack
        """
        if self.no_attack:
            return x
        if self.prepend:
            # noise = noise.repeat(self.batch_size,1)
            # print(x.shape,noise.shape)
            # x = x.unsqueeze(dim=0)
            x = torch.cat([noise,x],dim=-1).to(self.device)
        else:
            x = overlay_torch(noise,x)

        return x

                   
                   
        
    def calc_asl(self):
        """
        Calculates negative sequence length and logs the running average to the progress bar.
        """
        i = 0
        asl = 0
        with tqdm(self.dataloader, desc="Calculating ASL") as pbar:
            for batch in pbar:
                x, sampling_rate, transcript = batch
                x = x.to(self.device)
                x = self._attack(x.squeeze(), self.noise)
                output = self.model.transcribe(x)['text']
                i += 1
                asl += len(output)
                running_avg = asl / i
                
                # Update progress bar with the running average
                pbar.set_postfix(running_avg=running_avg)
        
        asl /= i
        print("Average Sequence Length:", asl)
        return asl
    

    # def calc_pesq(self,fs=16000):
    #     i = 0
    #     avg_pesq = 0
    #     noise = self.noise.unsqueeze(0)
    #     for batch in tqdm(self.dataloader):

    #         x, sampling_rate, transcript = batch
    #         # x = x.squeeze()

    #         deg = self._attack(x.to(self.device),noise)
    #         deg = deg.cpu().numpy()
    #         x = x.cpu().numpy()
    #         # print(deg.shape,x.shape)
    #         result = pesq_batch(16000,x,deg,mode="wb")
    #         i+=1
    #         avg_pesq += result
    #     avg_pesq /= i
    #     print("PESQ:", avg_pesq)

if __name__ == "__main__":

    args = get_args()
    # noise = torch.randn(16000).to('cuda')

    qq = RawEvaluator(args.noise_path,
                      args.whisper_model,
                      args.dataset,
                      args.device,
                      args.no_attack,
                      args.prepend,
                      args.batch_size
                      )
    # qq.model.transcribe(noise)
    qq.calc_asl()