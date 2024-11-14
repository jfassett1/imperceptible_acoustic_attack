#!/bin/bash

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n imperceptible python=3.11 -y
conda activate imperceptible

# Install PyTorch with CUDA (if required) and other packages
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install matplotlib -y
pip install openai-whisper==20240930

