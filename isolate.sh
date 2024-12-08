#!/bin/bash
conda activate imperceptible

cd  /home/jaydenfassett/audioversarial/imperceptible

python train_attack.py --domain raw_audio --epochs 2 --clip_val 0.02 --gpus 2,3 --no_speech --show --frequency_decay polynomial --decay_strength 1 --dataset train-other-500
python train_attack.py --domain raw_audio --epochs 2 --clip_val 0.02 --gpus 2,3 --no_speech --show --frequency_decay polynomial --decay_strength 5 --dataset train-other-500
python train_attack.py --domain raw_audio --epochs 2 --clip_val 0.02 --gpus 2,3 --no_speech --show --frequency_decay polynomial --decay_strength 10 --dataset train-other-500
python train_attack.py --domain raw_audio --epochs 2 --clip_val 0.02 --gpus 2,3 --no_speech --show --frequency_decay polynomial --decay_strength 100 --dataset train-other-500