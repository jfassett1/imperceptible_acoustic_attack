Acoustic Attack on Whisper

Based on https://arxiv.org/abs/2405.06134


Full README coming soon!

Currently this implementation should work out of the box. But if not, please email me at jfassett1@student.gsu.edu

# Training Script Arguments

To run the attack, use the `train_attack.py` file, with the following args: 

## General Training Settings

- `--epochs`: Number of training epochs (default: `1`).
- `--batch_size`: Batch size for training (default: `128`).
- `--learning_rate`: Learning rate for the optimizer (default: `0.005`).
- `--weight_decay`: Weight decay (L2 regularization) applied to the optimizer (default: `1e-5`).
- `--seed`: Random seed for reproducibility (default: `42`).
- `--dataset`: Dataset to use, options include `dev-clean`, `train-clean-100`, `train-other-500`.
- `--no_train`: If set, disables training and is used for testing pathing and saving (default: `False`).
- `--whisper_model`: Specifies which Whisper model to use. Options: `tiny.en`, `base.en`, `small.en`, `medium.en` (default: `tiny.en`).

## Attack Settings

- `--domain`: Defines whether to attack in raw audio or mel space. Options: `raw_audio`, `mel` (default: `raw_audio`).
- `--attack_length`: Length of the attack in seconds (default: `1.0`).
- `--prepend`: Whether to prepend the attack (default: `False`).
- `--noise_dir`: Directory to save noise outputs.
- `--gamma`: Scaling penalty value (default: `1.0`).
- `--no_speech`: If set, uses "nospeech" in the loss function (default: `False`).

## Epsilon Constraints

- `--clip_val`: Clamping value for limiting attack strength (default: `-1`).
- `--adaptive_clip`: Whether to adapt the clipping value based on the dataset (default: `False`).

## Data Processing

- `--num_workers`: Number of workers for data loading (default: `0`).
- `--root_dir`: Path to the root directory of the dataset. 

### Discriminator Settings

- `--use_discriminator`: If set, enables the discriminator (default: `False`).
- `--use_pretrained_discriminator`: Whether to use a pre-trained discriminator (default: `True`).

### Frequency Decay
- `--frequency_decay`: Type of frequency decay to apply. Options: `linear`, `polynomial`, `logarithmic`,

### Frequency Masking

- `--frequency_masking`: If set, enables frequency masking (default: `False`).
- `--masker_cores`: Number of cores allocated for masking threshold (default: `0`).
- `--window_size`: Window size for FFT (default: `2048`).


# Organization

## Path Logging

Generated noise will be saved according to the arguments. Its path will be printed at the end of training.
It will also be saved in paths.txt in the main directory (created when running the script).