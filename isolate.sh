#!/bin/bash
conda activate imperceptible

cd /home/jaydenfassett/audioversarial/imperceptible

gamma_values=(1 5 10 20)

for gamma in "${gamma_values[@]}"; do
    python train_attack.py \
        --domain raw_audio \
        --epochs 15 \
        --clip_val 0.02 \
        --gpus 2,3 \
        --attack_length 2 \
        --show \
        --frequency_penalty \
        --gamma "$gamma" \
        --dataset train-clean-100 \
        --num_workers 4

    echo "Train with gamma $gamma complete"
done