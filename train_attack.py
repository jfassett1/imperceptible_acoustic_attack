import argparse
import os
import sys

import torch
import torch.distributed as dist
from pytorch_lightning import Trainer

from eval import evaluate
from src.attacker.mel_attacker import MelBasedAttackerLightning
from src.attacker.raw_attacker import RawAudioAttackerLightning
from src.data import AudioDataModule, clipping
from src.masking.mask_utils import masking
from src.pathing import ROOT_DIR, AttackPath, log_path

# from src.discriminator import MelDiscriminator
from src.visual_utils import show


def get_args():
    parser = argparse.ArgumentParser(description="Training Script Arguments")

    # General training settings
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                        default=0.005, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default="librispeech:dev-clean",
                        help="Which dataset to use. Should look like `dataset:split` ex. librispeech:dev-clean")
    parser.add_argument("--no_train", action="store_true", default=False,
                        help="Whether to train noise. Used for testing pathing & saving")
    parser.add_argument("--whisper_model", choices=['tiny.en', 'base.en', 'small.en',
                        'medium.en'], default='tiny.en', help='Which Whisper model to use')
    # Attack Settings
    parser.add_argument('--domain', type=str, default="raw_audio", choices=[
                        'raw_audio', "mel"], help="Whether to attack in mel or audio space")
    parser.add_argument('--attack_length', type=float,
                        default=1., help="Length of attack in seconds")
    parser.add_argument('--prepend', action="store_true",
                        default=False, help="Whether to prepend or not")
    parser.add_argument('--noise_dir', type=str, default=None,
                        help="Where to save noise outputs")
    parser.add_argument('--gamma', type=float, default=1.,
                        help="Gamma value for scaling penalty")
    parser.add_argument("--no_speech", action="store_true",
                        default=False, help="Whether to use Nospeech in loss function")

    # Epsilon Constraints
    parser.add_argument('--clip_val', type=float,
                        default=None, help="Clamping Value")
    parser.add_argument('--adaptive_clip', action="store_true", default=False,
                        help="Whether to adapt the clipping value to the dataset.")

    # Data processing
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    # parser.add_argument('--dataset',type=str, default="librispeech",choices=['librispeech'], help="Which dataset to use") #TODO: Add support for more datasets.
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Path of Root Directory")

    # PENALTY ARGS:
    # ----------------------------------------------------------------------------------------------------------------#
    # Arguments for Discriminator TODO: Either make better or remove
    parser.add_argument("--use_discriminator", action="store_true",
                        default=False, help="Whether to use discriminator")
    parser.add_argument("--use_pretrained_discriminator", type=bool, default=True,
                        help="Whether to use pretrained discriminator. Will find pre-trained automatically")  # TODO: Set up pathing for pretrained discriminators
    # parser.add_argument("--lambda",type=float,default=1.,help="Lambda value. Represents strength of discriminator during training")

    # Arguments for frequency decay

    parser.add_argument("--freq_decay", action="store_true",
                        default=False, help="Whether to use frequency decay")
    parser.add_argument("--decay_pattern", type=str, choices=['linear', 'polynomial', 'logarithmic', 'exponential'], default=None,
                        help="Whether to use frequency decay, and what pattern of frequency decay")  # TODO: Replace default with whatever works best for final code submission
    parser.add_argument("--decay_strength", type=float,
                        default=1., help="Weight of the frequency decay")

    parser.add_argument("--frequency_penalty", action="store_true",
                        default=False, help="Toggle for MSE frequency penalty")
    # Arguments for frequency masking
    parser.add_argument("--frequency_masking", action="store_true",
                        default=False, help="Toggle for frequency masking")
    parser.add_argument("--masker_cores", type=int, default=0,
                        help="Number of cores allocated to masking threshold.")
    parser.add_argument("--window_size", type=int,
                        default=2048, help="Window Size for FFT")

    # NOTE: For controlling strength, use gamma
    # Optimizer and scheduler settings #TODO: Implement these arguments
    # ----------------------------------------------------------------------------------------------------------------#

    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['step', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--scheduler_step_size', type=int,
                        default=10, help='Step size for StepLR')
    parser.add_argument('--scheduler_gamma', type=float,
                        default=0.1, help='Gamma for StepLR')

    # GPU settings
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'], help='Device to use for training')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated list of GPU IDs to use for training')

    # Saving Settings
    parser.add_argument('--show', action="store_true",
                        default=False, help="Whether to save image")
    parser.add_argument('--save_ppt', action="store_true", default=False,
                        help="Whether to save powerpoint with examples")
    parser.add_argument('--log_path', action="store_true",
                        default=False, help="Whether to save paths in CSV")
    parser.add_argument('--debug', action="store_true",
                        default=False, help="Print when modules activate")
    parser.add_argument('--eval', action="store_true",
                        default=False, help="Evaluation of model")

    # Debugging and testing
    # parser.add_argument('--debug', action='store_true', help='Run in debug mode with minimal data')
    # parser.add_argument('--test_only', action='store_true', help='Run only in evaluation mode (no training)')

    args = parser.parse_args()
    return args


def main(args):
    # ARG HANDLING
    # ------------------------------------------------------------------------------------------#
    args.dataset = args.dataset.lower()
    args.root_dir = ROOT_DIR
    imper_epochs = 0

    threshold = masking(args)

    discriminator = None
    if args.use_discriminator:
        raise NotImplementedError
    # MODULE SETUP
    # ------------------------------------------------------------------------------------------#
    try:
        gpu_list = [int(gpu)
                    for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
        gpu_list = list(range(len(gpu_list)))
        print(gpu_list)
    except KeyError:
        gpu_list = []
        print("Running on CPU")

    # gpu_list = [int(gpu) for gpu in args.gpus.split(",")]
    ATTACK_LEN_SEC = args.attack_length
    threshold = None

    data_module = AudioDataModule(dataset_name=args.dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  attack_len=args.attack_length)
    # print([args.frequency_decay, args.frequency_penalty, args.frequency_masking, args.use_discriminator])
    if any(x is not False for x in [args.frequency_penalty, args.frequency_masking, args.use_discriminator]):
        print("Adding fine-tuning epoch(s)")
        imper_epochs = 1

    # Handle clipping args
    clipping(data_module, args)

    # print(args.clip_val)
    if args.domain == "mel":
        raise NotImplementedError  # NOTE: Deprecated
        attacker = MelBasedAttackerLightning(sec=ATTACK_LEN_SEC,
                                             prepend=args.prepend,
                                             batch_size=args.batch_size,
                                             discriminator=discriminator,
                                             epsilon=args.clip_val,
                                             gamma=args.gamma,
                                             no_speech=args.no_speech)

    elif args.domain == "raw_audio":
        # signature = inspect.signature(RawAudioAttackerLightning.__init__)
        # parameters = set(signature.parameters.keys())
        # t_args = set(vars(args).keys()).intersection(parameters)
        # print(t_args)
        # exit(
        # valid_args = args - parameters
        # attacker = RawAudioAttackerLightning(sec=ATTACK_LEN_SEC,
        #                                     discriminator=discriminator,
        #                                     frequency_decay=(args.decay_pattern,args.decay_strength),
        #                                     mask_threshold = threshold,
        #                                     imper_epochs=imper_epochs,
        #                                     **vars(args)
        #                                     )
        attacker = RawAudioAttackerLightning(sec=ATTACK_LEN_SEC,
                                             model=args.whisper_model,
                                             prepend=args.prepend,
                                             batch_size=args.batch_size,
                                             discriminator=discriminator,
                                             epsilon=args.clip_val,
                                             gamma=args.gamma,
                                             no_speech=args.no_speech,
                                             frequency_decay=(
                                                 args.decay_pattern, args.decay_strength),
                                             learning_rate=args.learning_rate,
                                             frequency_masking=args.frequency_masking,
                                             window_size=args.window_size,
                                             masker_cores=args.masker_cores,
                                             mask_threshold=threshold,
                                             debug=args.debug,
                                             frequency_penalty=args.frequency_penalty,
                                             train_epochs=args.epochs,
                                             imper_epochs=imper_epochs,
                                             )

    trainer = Trainer(max_epochs=args.epochs + imper_epochs,
                      val_check_interval=0.8,
                      devices=gpu_list,
                      enable_progress_bar=True)

    if not args.no_train:
        trainer.fit(attacker, data_module.train_dataloader(),
                    data_module.val_dataloader())

    # Ending the
    if dist.is_initialized():
        dist.barrier()
    if trainer.global_rank != 0:
        sys.exit(0)

    PATHS = AttackPath(args, ROOT_DIR)
    print(f"Saving to {PATHS.noise_path}")

    if args.domain == "raw_audio":
        attacker.dump(PATHS.noise_path)
    else:
        torch.save(attacker.noise, PATHS.noise_path)

    # audio_sample = wavfile.read("/home/jaydenfassett/audioversarial/imperceptible/original_audio.wav")[1]
    asl = None
    # torch.cuda.set_device()
    # data_module.device = int(args.gpus[0])

    if args.eval:
        # torch.cuda.set_device(int())

        evaluate(attacker.noise.detach(),
                 data_module,
                 args,)

    if args.show:
        show(attacker.noise.detach().cpu().numpy().squeeze(),
             data_module.sample[0].cpu().squeeze(0).numpy(),
             PATHS,
             args.prepend)

    if args.log_path:
        log_path(PATHS, asl)
        return


if __name__ == "__main__":
    args = get_args()
    main(args)
