#!/bin/bash
#SBATCH --job-name=cd_patch_mil_single
#SBATCH --output=logs/patch_mil_single_%j.out
#SBATCH --error=logs/patch_mil_single_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

echo "==== SLURM JOB START ===="
date
echo "JobID: ${SLURM_JOB_ID:-unknown}"
echo "Host:  $(hostname)"

module purge
module load Miniforge3
module load CUDA/12.4.0

source activate infra_env

cd /mnt/scratch/kodumuru/pari/chicken_disease
mkdir -p logs
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==== PREFLIGHT ===="
echo "PWD: $(pwd)"
echo "Python: $(which python3)"
python3 -V

python3 - <<'PY'
import numpy as np
import sklearn
import timm
import torch
import torchvision
from chicken_disease.utils.params import read_params

p = read_params()
assert "patch_mil" in p, "Missing patch_mil section in params.yaml"

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("sklearn:", sklearn.__version__)
print("numpy:", np.__version__)
print("patch_mil:", p["patch_mil"])
PY

test -d data/processed/single/train || { echo "Missing data/processed/single/train. Run Stage 01+02 first."; exit 2; }

echo "==== TRAIN (STAGE 04B) ===="
python3 -u src/chicken_disease/pipeline/stage_04b_train_patch_mil_single.py

echo "==== SLURM JOB END ===="
date
