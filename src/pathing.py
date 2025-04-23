"""
Handles pathing for training, eval, and demos.
Used to be in train_attack.py

"""
from pathlib import Path
import sys
import shlex
import pandas as pd
import torch
ROOT_DIR = Path(__file__).parent.parent

NOISE_DIR = ROOT_DIR / "noise"
EXAMPLE_DIR = ROOT_DIR / "examples"
DATA_DIR = ROOT_DIR / "data" 
DISCRIM_PATH = ROOT_DIR / "discriminator" 


def log_path(PATHS,asl):

    command = shlex.join(sys.argv)
    txt_file = ROOT_DIR / "paths.txt"
    row = f"{command} | {str(PATHS.noise_path)} |  {asl}\n"
    with open(txt_file, "a") as file:
        file.write(row)
    return
def unpack_metrics(metrics):
    unpacked = {}
    for key, tensor in metrics.items():
        if isinstance(tensor, torch.Tensor):
            # If tensor is a single element, use .item(), otherwise convert to list.
            unpacked[key] = tensor.item() if tensor.numel() == 1 else tensor.detach().cpu().tolist()
        else:
            unpacked[key] = tensor
    return unpacked

def log_path_pd(PATHS, asl, per_muted, snr, clip_val, attack_length, name, metrics: dict):

    log_file = ROOT_DIR / "paths.csv"
    command = shlex.join(sys.argv)
    noise_path = str(PATHS.noise_path)
    metrics = unpack_metrics(metrics)
    # Combine fixed fields with metrics
    new_entry = {
        "Name": name,
        "command": command,
        "noise_path": noise_path,
        "clip_val": clip_val,
        "asl": asl,
        "per_muted": per_muted,
        "attack_length":attack_length,
        "snr":snr,
        **metrics  # Unpack additional metrics into the row
    }

    if log_file.exists():
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])

    df.to_csv(log_file, index=False)


class AttackPath:
    """
    Class that has all potential paths as attributes
    
    """
    def __init__(self,
                 args,
                 root_dir=ROOT_DIR,
                 noise_dir=NOISE_DIR,
                 example_dir = EXAMPLE_DIR,
                 data_dir = DATA_DIR,
                 discrim_dir = DISCRIM_PATH):
        
        self.root_dir    = root_dir 
        self.example_dir = example_dir 
        self.data_dir    = data_dir 
        self.discrim_dir = discrim_dir
        if ":" in args.dataset:
            args.dataset, _ = args.dataset.split(":")

        self.DIR_LIST = [] # Provides structure for any given sample. Can be re-used to noise, images, etc. Use looks like ROOT_DIR / DIRECTORY_STRUCTURE / example.png

        ATTACK_LEN_SEC = args.attack_length
        # Basic Unchanging dir structure
        #------------------------------------------------------------------------------------------#

        self.add_dir(args.whisper_model)
        self.add_dir(args.domain)
        self.add_dir(("prepend" if args.prepend else "overlay"))
        self.add_dir(args.dataset)

        #Penalty Args
        #------------------------------------------------------------------------------------------#

        num_constraints = 0
        if args.test_name:
            base = args.test_name
            runs_parent = self.root_dir / "noise"
            runs_parent.mkdir(parents=True, exist_ok=True)

            # find existing run‑dirs that start with exactly that base
            # siblings = [d for d in runs_parent.iterdir()
            #             if d.is_dir() and d.name.startswith(base)]

            # first run is "base", later ones "base_2", "base_3", …
            # suffix = "" if idx == 1 else f"_{idx}"
            run_dir = runs_parent / f"{base}"
            run_dir.mkdir(exist_ok=True,parents=True)
            siblings = list(run_dir.glob("*.npy"))
            # print(siblings)
            # print(run_dir)
            idx = len(siblings) + 1
            # now enumerate the three artifacts themselves
            ext = ".np.npy" if args.domain == "raw_audio" else ".pth"
            # now enumerate purely by number, e.g. 1.np.npy, 2.np.npy, or 1.pth, 2.pth, …
            self.noise_path = run_dir / f"{idx}{ext}"

            # leave your img/audio dirs & filenames exactly as before…
            self.img_dir    = self.example_dir / "images" / base
            self.img_dir.mkdir(parents=True, exist_ok=True)
            self.img_path   = self.img_dir / f"{idx}.png"

            self.audio_dir  = self.example_dir / "audio" / base
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            self.audio_path = self.audio_dir / f"{idx}.wav"
            return

        if args.use_discriminator:
            self.add_dir("discriminator")
            num_constraints +=1
        if args.freq_decay:
            self.add_dir("frequency_decay")
            self.add_dir(str(args.freq_decay))
            self.add_dir(str(args.decay_strength))
            num_constraints +=1
        if args.frequency_penalty:
            self.add_dir("frequency_penalty")
            num_constraints +=1
        if args.mel_mask:
            self.add_dir("mel_mask")
            num_constraints +=1
        if args.no_speech:
            self.add_dir("nospeech")
            num_constraints +=1
        if args.frequency_masking:
            self.add_dir("frequency_masking")

        if args.adaptive_clip:
            self.add_dir("clip_val_adaptive")
        else:
            if args.clip_val != -1:
                self.add_dir(f"clip_val_{round(args.clip_val,5)}")

        self.add_dir(f"length_{args.attack_length}")
        # self.add_dir(f"steps_{args.epochs}")
        
        if num_constraints > 0: #Conditional because gamma is meaningless when not using a constraint
            #TODO: Make different gammas
            self.add_dir(f"gamma_{args.gamma}")
            if num_constraints > 1:
                print("WARNING. Be careful using more than one constraint! Code may not work yet.")

        # self.add_dir("frequency_penalty")
        self.DIRECTORY_STRUCTURE = Path("")
        for path in self.DIR_LIST:
            self.DIRECTORY_STRUCTURE /= path


        #Final attributes for use:
        
        self.noise_dir = self.root_dir / "noise" / self.DIRECTORY_STRUCTURE
        self.noise_dir.mkdir(parents=True,exist_ok=True)

        self.img_dir = self.example_dir / "images" / self.DIRECTORY_STRUCTURE 
        self.audio_dir = self.example_dir / "audio" / self.DIRECTORY_STRUCTURE
        if args.show:
            self.img_dir.mkdir(parents=True,exist_ok=True)
            self.audio_dir.mkdir(parents=True,exist_ok=True)
        self.noise_path = self.noise_dir / ("noise.np.npy" if args.domain == "raw_audio" else "noise.pth")
        self.img_path = self.img_dir  / "plot.png"
        self.audio_path = self.audio_dir / "audio_with_attack.wav"
        return

    def add_dir(self,
                curr,
                base=None):
        if base is None:
            base = self.DIR_LIST
        base.append(curr)


    #------------------------------------------------------------------------------------------#
   

if __name__ ==  "__main__":
    import argparse
    def get_args():
        parser = argparse.ArgumentParser(description="Training Script Arguments")

        # General training settings
        parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('--dataset',type = str,choices=['dev-clean','train-clean-100','train-other-500'])
        parser.add_argument("--no_train",action="store_true",default=False,help="Whether to train noise. Used for testing pathing & saving")
        parser.add_argument("--whisper_model",choices=['tiny.en','base.en','small.en','medium.en'], default='tiny.en', help='Which Whisper model to use')
        # Attack Settings
        parser.add_argument('--domain',type=str,default="raw_audio",choices=['raw_audio',"mel"],help="Whether to attack in mel or audio space")
        parser.add_argument('--attack_length',type=float,default=1.,help = "Length of attack in seconds")
        parser.add_argument('--prepend',action="store_true", default = False,help="Whether to prepend or not")
        parser.add_argument('--noise_dir',type=str,default=None,help="Where to save noise outputs")
        parser.add_argument('--clip_val',type=float,default = -1,help="Clamping Value")
        parser.add_argument('--gamma',type=float,default = 1., help= "Gamma value for scaling penalty")
        parser.add_argument("--no_speech",action="store_true",default=False,help="Whether to use Nospeech in loss function")

        # Data processing
        parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
        # parser.add_argument('--dataset',type=str, default="librispeech",choices=['librispeech'], help="Which dataset to use") #TODO: Add support for more datasets.
        parser.add_argument("--root_dir",type=str,default=None,help="Path of Root Directory")

        # PENALTY ARGS:
        #----------------------------------------------------------------------------------------------------------------#
        # Arguments for Discriminator TODO: Either make better or remove
        parser.add_argument("--use_discriminator", action="store_true",default=False,help="Whether to use discriminator")
        parser.add_argument("--use_pretrained_discriminator",type=bool,default=True,help="Whether to use pretrained discriminator. Will find pre-trained automatically") #TODO: Set up pathing for pretrained discriminators
        # parser.add_argument("--lambda",type=float,default=1.,help="Lambda value. Represents strength of discriminator during training") 

        #Arguments for frequency decay
        parser.add_argument("--frequency_decay", type = str, choices = ['linear','polynomial','logarithmic','exponential'], default=None, help="Whether to use frequency decay, and what pattern of frequency decay") #TODO: Replace default with whatever works best for final code submission
        parser.add_argument("--decay_strength",type = float, default = 1., help = "Weight of the frequency decay")

        parser.add_argument("--frequency_penalty", action="store_true", default=False, help="Toggle for MSE frequency penalty")
        #NOTE: For controlling strength, use gamma
        # Optimizer and scheduler settings #TODO: Implement these arguments
        #----------------------------------------------------------------------------------------------------------------#

        parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer type')
        parser.add_argument('--scheduler', type=str, default=None, choices=['step', 'cosine'], help='Learning rate scheduler')
        parser.add_argument('--scheduler_step_size', type=int, default=10, help='Step size for StepLR')
        parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Gamma for StepLR')

        # GPU settings
        parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'], help='Device to use for training')
        parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs to use for training')

        #Saving Settings
        parser.add_argument('--show',action="store_true",default=False,help="Whether to save image")
        parser.add_argument('--save_ppt',action="store_true",default=False,help="Whether to save powerpoint with examples")

        # Debugging and testing
        # parser.add_argument('--debug', action='store_true', help='Run in debug mode with minimal data')
        # parser.add_argument('--test_only', action='store_true', help='Run only in evaluation mode (no training)')

        args = parser.parse_args()
        return args
    
    qq = AttackPath(get_args())
    print(qq.noise_path)