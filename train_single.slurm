#!/bin/bash
#SBATCH --job-name=cd_train_single
#SBATCH --output=logs/train_single_%j.out
#SBATCH --error=logs/train_single_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

module purge
module load Miniforge3
module load CUDA/12.4.0

source activate infra_env

cd /mnt/scratch/kodumuru/pari/chicken_disease
mkdir -p logs
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python3 -u src/chicken_disease/pipeline/stage_03_train_single.py
