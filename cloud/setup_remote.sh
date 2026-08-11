#!/bin/bash
set -euo pipefail

# Run this on the EC2 instance after code has been copied over. The AMI
# (Deep Learning Base OSS Nvidia Driver GPU AMI) already has NVIDIA
# drivers installed -- we build our own clean venv on top rather than
# relying on a pre-baked conda environment, so it matches what we tested
# locally.

python3 -m venv ~/aurora-venv
source ~/aurora-venv/bin/activate

pip install --upgrade pip

# CUDA-enabled torch build.
pip install torch --index-url https://download.pytorch.org/whl/cu126

pip install transformers datasets tokenizers boto3 tqdm psutil

echo "=== GPU check ==="
python3 -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "Environment ready. Activate with: source ~/aurora-venv/bin/activate"
