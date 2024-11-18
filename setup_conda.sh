#!/bin/bash
# This script uses CUDA 12.1. You can swap with CUDA 11.8.
conda create --name strata \
    python=3.10 \
    pytorch-cuda=12.1 \
    pytorch=2.3.0 cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate strata

pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git@0376b81364ad036d8fc2236c9e9fc13c9b1afe3d"

pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

pip install -e . -v

# To remove the env:
# conda remove -n strata --all
