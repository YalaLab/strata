#!/bin/bash
cd /root/strata

# Uncomment to use other cuda versions (11.8 or 12.1) and unsloth versions (for pre-Ampere GPUs)
# pip3 install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu118
pip install pip==24.0 packaging==24.1
pip install torch==2.3.0 torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
pip install trl==0.9.6 peft==0.11.1 accelerate==0.32.1 bitsandbytes==0.43.1 transformers==4.42.4 triton==2.3.0
MAX_JOBS=4 pip install flash-attn==2.5.8 --no-build-isolation --no-binary :all:

# pip install "unsloth[cu118-torch230] @ git+https://github.com/unslothai/unsloth.git@0376b81364ad036d8fc2236c9e9fc13c9b1afe3d"
# pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git@0376b81364ad036d8fc2236c9e9fc13c9b1afe3d"
# pip install "unsloth[cu118-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git@0376b81364ad036d8fc2236c9e9fc13c9b1afe3d"
pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git@0376b81364ad036d8fc2236c9e9fc13c9b1afe3d"

pip install -e . -v

# nvcc -v
# python -m xformers.info
# python -m bitsandbytes
