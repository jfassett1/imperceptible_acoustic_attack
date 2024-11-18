import torch
import pytorch_lightning
import argparse
from tqdm import tqdm
from pytorch_lightning import Trainer
from pathlib import Path
from typing import Union
import time
from src.data import AudioDataModule
from src.attacker import MelBasedAttackerLightning






def get_args():
    parser = argparse.ArgumentParser(description="Training Script Arguments")

    # General training settings
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Attack Settings
    parser.add_argument('--attack_length',type=float,default=1.,help = "Length of attack in seconds")
    parser.add_argument('--prepend',type=bool, default = False,help="Whether to prepend or not")

    # Data processing
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--dataset',type=str, default="librispeech",choices=['librispeech'], help="Which dataset to use") #TODO: Add support for more datasets.
    parser.add_argument("--root_dir",type=str,default=None,help="Path of Root Directory")

    # Arguments for Discriminator
    parser.add_argument("--use_discriminator", type=bool, default=False,help="Whether to use discriminator")
    parser.add_argument("--use_pretrained_discriminator",type=bool,default=True,help="Whether to use pretrained discriminator. Will find pre-trained automatically") #TODO: Set up pathing for pretrained discriminators
    parser.add_argument("--lambda",type=float,default=1.,help="Lambda value. Represents strength of discriminator during training") 

    # Optimizer and scheduler settings #TODO: Implement these arguments
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default=None, choices=['step', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--scheduler_step_size', type=int, default=10, help='Step size for StepLR')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Gamma for StepLR')

    # GPU settings
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'], help='Device to use for training')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs to use for training')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')

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
    DATA_DIR = ROOT_DIR / "data"
    ATTACK_LEN_SEC = args.attack_length
    DISCRIM_PATH = ROOT_DIR / "discriminator"
    if not DISCRIM_PATH.exists():
        DISCRIM_PATH.mkdir()


    #MODULE SETUP
    #------------------------------------------------------------------------------------------#
    gpu_list = [int(gpu) for gpu in args.gpus.split(',')]

    attacker = MelBasedAttackerLightning(sec=ATTACK_LEN_SEC,prepend=args.prepend)
    data_module = AudioDataModule(dataset_name=args.dataset,batch_size=args.batch_size,num_workers=args.num_workers)

    trainer = Trainer(max_epochs=args.epochs,devices=gpu_list)
    trainer.fit(attacker,data_module)


    torch.save(attacker.noise,"/home/jaydenfassett/audioversarial/imperceptible/tempattacks/noise.pth")



if __name__ == "__main__":
    args  = get_args()
    print(args)
    main(args)
