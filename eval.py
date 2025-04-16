import whisper
from whisper import log_mel_spectrogram, pad_or_trim
from src.data import AudioDataModule
from src.decoder import RawDecoder
import argparse
import torch
from tqdm import tqdm
import numpy as np
import functools
from src.utils import overlay_torch
# from pesq import pesq_batch
whisper.torch.load = functools.partial(whisper.torch.load, weights_only=True)

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
    results = evaluator.calc_asl()
    return results



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
            self.asl_dataloader = data_module.test_dataloader(batch_size=1)
            self.dataloader = data_module.test_dataloader(batch_size=batch_size)
            self.device = device
            self.model = whisper.load_model(model).to(self.device)
            if "en" in model:
                self.tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=False)
                self.multi = False
            else:
                self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
                self.multi = True
            self.decoder = RawDecoder(self.model,self.tokenizer,self.device)
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
        per_muted = 0
        eot_id = self.tokenizer.eot
        avg_snr = 0

        if self.multi:
            eot_id += 1
        with tqdm(self.asl_dataloader, desc="Calculating ASL") as pbar:

            assert len(self.noise.shape) == 1, "Incorrect noise length. Must be 1D"
    
            noise_len = self.noise.shape[0]
            noise_pow = (self.noise.pow(2).sum() + 1e-12).log10() * 10

            for batch in pbar:
                x, sampling_rate, transcript,length = batch
                
                x = x.to(self.device)
                x = x.squeeze()

                samp_pow = (x[:noise_len].pow(2).sum() + 1e-12).log10() * 10
                avg_snr += samp_pow - noise_pow
                # import matplotlib.pyplot as plt

                # plt.plot(x[:noise_len].detach().cpu().numpy(), label="Clean")
                # plt.plot(self.noise.detach().cpu().numpy(), label="Noise", alpha=0.7)
                # plt.savefig("/home/jaydenfassett/audioversarial/imperceptible/eval.png")


                x = self._attack(x, self.noise) # Apply noise to the sample


                if len(sampling_rate) == 1: # If only one sample
                    x_mel = log_mel_spectrogram(x).unsqueeze(0)
                else:
                    x_mel = log_mel_spectrogram(x)
                # print(x.shape)
                eot, _, total_probs = self.decoder.get_eot_prob(x_mel)
                prob_argmax = total_probs.argmax(dim=-1)
                # print(prob_argmax,self.tokenizer.eot)
                # print(prob_argmax.item(),self.tokenizer.eot)
                # vals = self.tokenizer.decode([prob_argmax,self.tokenizer.eot])
                # print(vals)
                # exit()

                if prob_argmax == eot_id: #NOTE: If per_muted breaks, check the indexing in the tokenizer. During testing, we noticed that the tokenizer prediction, and its position are different.
                    asl += 0
                    per_muted += 1
                    # print("h")
                else:
                    output = self.model.transcribe(x)['text']
                    asl += len(output)

                # print(x.shape)
                # print(type(x))
                # exit()

                i += 1
                running_avg = asl / i
                
                # Update progress bar with the running average
                pbar.set_postfix(running_avg=running_avg, num_muted=per_muted)
        
        asl /= i
        per_muted /= i

        avg_snr /= i
        print(f"Average Sequence Length: {asl:.2f}")
        print(f"Percent Muted: {per_muted:.2f}")
        print(f"Mean SNR: {avg_snr:.2f} db")
        return asl,per_muted, avg_snr.item()
    
    # def num_silenced(self):
    #     with tqdm(self.dataloader, desc="Calculating # of Silenced") as pbar:
    #         muted = 0
    #         for batch in pbar:
    #             x, sampling_rate, transcript,length = batch

    #             x = x.to(self.device)
    #             x = self._attack(x.squeeze(), self.noise)
    #             if len(sampling_rate) == 1: # If only one sample
    #                 x_mel = log_mel_spectrogram(x).unsqueeze(0)
    #             else:
    #                 x_mel = log_mel_spectrogram(x)
    #             eot, _, total_probs = self.decoder.get_eot_prob(x_mel)
    #             prob_argmax = total_probs.argmax(dim=-1)
    #             muted += (prob_argmax == self.tokenizer.eot).sum().item()
    #             pbar.set_postfix(num_muted = muted)
    #     print(muted / len(self.dataloader))

    #     return muted / len(self.dataloader)




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
    # attack = torch.tensor(np.load("/home/jaydenfassett/audioversarial/imperceptible/noise/tiny/raw_audio/overlay/tedlium/GammaTest11/mel_mask/length_2.0/gamma_0.1/noise.np.npy")).unsqueeze(dim=0)
    # attack = torch.tensor(np.load("/home/jaydenfassett/audioversarial/imperceptible/noise/tiny.en/raw_audio/overlay/tedlium/GammaTester2/length_2.0/noise.np.npy"))
    attack = torch.tensor(np.load("/home/jaydenfassett/audioversarial/imperceptible/noise/tiny/raw_audio/overlay/tedlium/GammaTestMain/mel_mask/length_2.0/gamma_0.35/noise.np.npy"))
    print(attack.shape)
    # attack = torch.randn(32000) * 0.01
    # attack = torch.zeros(1,16000)
    dm = AudioDataModule("tedlium:",attack_len=2,batch_size=1)
    qq = RawEvaluator(attack,"tiny",data_module=dm)


    qq.calc_asl()
    # qq.num_silenced()
    exit()

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