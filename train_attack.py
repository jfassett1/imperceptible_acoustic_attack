import torch
import pytorch_lightning
import argparse
import numpy as np
from tqdm import tqdm
from pytorch_lightning import Trainer
from pathlib import Path
from typing import Union, Optional, Literal
import time
from src.data import AudioDataModule
from src.attacker import MelBasedAttackerLightning, RawAudioAttackerLightning
from src.discriminator import MelDiscriminator






def get_args():
    parser = argparse.ArgumentParser(description="Training Script Arguments")

    # General training settings
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

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
    parser.add_argument('--dataset',type=str, default="librispeech",choices=['librispeech'], help="Which dataset to use") #TODO: Add support for more datasets.
    parser.add_argument("--root_dir",type=str,default=None,help="Path of Root Directory")

    # PENALTY ARGS:
    #----------------------------------------------------------------------------------------------------------------#
    # Arguments for Discriminator TODO: Either make better or remove
    parser.add_argument("--use_discriminator", action="store_true",default=False,help="Whether to use discriminator")
    parser.add_argument("--use_pretrained_discriminator",type=bool,default=True,help="Whether to use pretrained discriminator. Will find pre-trained automatically") #TODO: Set up pathing for pretrained discriminators
    # parser.add_argument("--lambda",type=float,default=1.,help="Lambda value. Represents strength of discriminator during training") 
    #Arguments for frequency decay
    parser.add_argument("--frequency_decay", type = str, choices = ['linear','polynomial','exponential'], default=None, help="Whether to use frequency decay, and what pattern of frequency decay") #TODO: Replace default with whatever works best for final code submission
    parser.add_argument("--decay_strength",type = float, default = 1., help = "Weight of the frequency decay")
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

    # Debugging and testing
    # parser.add_argument('--debug', action='store_true', help='Run in debug mode with minimal data')
    # parser.add_argument('--test_only', action='store_true', help='Run only in evaluation mode (no training)')

    args = parser.parse_args()
    return args


def main(args):


    #PATHING
    #------------------------------------------------------------------------------------------#
    if args.root_dir is None:
        ROOT_DIR = Path(__file__).parent
    else:
        ROOT_DIR = args.root_dir
    # DATA_DIR = ROOT_DIR / "data"
    if args.noise_dir is None:
        NOISE_DIR = ROOT_DIR / "noise"
        NOISE_DIR.mkdir(exist_ok=True) if not NOISE_DIR.exists() else None

    else:
        NOISE_DIR = args.noise_dir

    DIRECTORY_STRUCTURE = Path("") # General directory structure which is re-used for saving different files like noise, image, etc

    ATTACK_LEN_SEC = args.attack_length
    DISCRIM_PATH = ROOT_DIR / "discriminator"

    DIRECTORY_STRUCTURE = DIRECTORY_STRUCTURE / args.domain
    # NOISE_SAVEPATH.mkdir(exist_ok=True) if not NOISE_SAVEPATH.exists() else None
    DIRECTORY_STRUCTURE = DIRECTORY_STRUCTURE/"prepend" if args.prepend else DIRECTORY_STRUCTURE/"overlay"
    # NOISE_SAVEPATH.mkdir(exist_ok=True)

    if not DISCRIM_PATH.exists():
        DISCRIM_PATH.mkdir()
    if args.frequency_decay is not None:
        DIRECTORY_STRUCTURE = DIRECTORY_STRUCTURE / str(args.frequency_decay)
        DIRECTORY_STRUCTURE = DIRECTORY_STRUCTURE / str(args.decay_strength)

    #OBJECTIVE FUNCTION ARGS
    #------------------------------------------------------------------------------------------#

        assert sum([args.use_discriminator, args.no_speech]) <= 1, \
            "Can use EITHER discriminator or chunkloss or no speech"    
    if args.use_discriminator:
        discriminator = MelDiscriminator()
        DIRECTORY_STRUCTURE = DIRECTORY_STRUCTURE / "discriminator"
    else:
        discriminator = None

    if args.use_pretrained_discriminator and args.use_discriminator:
        discrim_weights = torch.load(ROOT_DIR/"discriminator" / f"discriminator_{int(ATTACK_LEN_SEC*100)}tsteps.pth",weights_only=True)

        discriminator.load_state_dict(discrim_weights)

    #MODULE SETUP
    #------------------------------------------------------------------------------------------#
    gpu_list = [int(gpu) for gpu in args.gpus.split(',')]

    if args.domain == "mel":
        attacker = MelBasedAttackerLightning(sec=ATTACK_LEN_SEC,
                                            prepend=args.prepend,
                                            batch_size=args.batch_size,
                                            discriminator=discriminator,
                                            epsilon=args.clip_val,
                                            gamma = args.gamma,
                                            no_speech=args.no_speech)
    elif args.domain == "raw_audio":
        attacker = RawAudioAttackerLightning(sec=ATTACK_LEN_SEC,
                                            prepend=args.prepend,
                                            batch_size=args.batch_size,
                                            discriminator=discriminator,
                                            epsilon=args.clip_val,
                                            gamma = args.gamma,
                                            no_speech=args.no_speech,
                                            frequency_decay=(args.frequency_decay,args.decay_strength)
                                            )
    

    data_module = AudioDataModule(dataset_name=args.dataset,batch_size=args.batch_size,num_workers=args.num_workers)

    trainer = Trainer(max_epochs=args.epochs,devices=gpu_list)
    trainer.fit(attacker,data_module)

    file_name = f"noise_{int(ATTACK_LEN_SEC*100)}tsteps"

    NOISE_SAVEPATH = NOISE_DIR / DIRECTORY_STRUCTURE
    NOISE_SAVEPATH.mkdir(exist_ok=True,parents=True)

    print(f"Saving to {NOISE_SAVEPATH}/{file_name}.pth")
    print(f"Saving to {NOISE_SAVEPATH}/{file_name}.np.npy")

    torch.save(attacker.noise,f"{NOISE_SAVEPATH}/noise_{int(ATTACK_LEN_SEC*100)}tsteps.pth")
    attacker.dump(f"{NOISE_SAVEPATH}/noise_{int(ATTACK_LEN_SEC*100)}tsteps.np.npy")
    if args.show:
        from src.utils import save_photo_overlay # Jack would hate me for this, but I like importing things as I need them
        from scipy.io import wavfile
        EXAMPLE_SAVEPATH = ROOT_DIR / "examples" / DIRECTORY_STRUCTURE
        EXAMPLE_SAVEPATH.mkdir(exist_ok=True,parents=True)
        noise = attacker.noise.detach().cpu().numpy().squeeze()
        sample_audio = wavfile.read(ROOT_DIR/"original_audio.wav")[1]
        # print("AUDIO SHAPE",sample_audio.shape)
        # print("NOISE SHAPE",noise.shape)
        print(f"Saving image to {EXAMPLE_SAVEPATH/'plot.png'}")
        save_photo_overlay(noise,sample_audio,EXAMPLE_SAVEPATH/"plot.png")



if __name__ == "__main__":
    args  = get_args()
    main(args)
