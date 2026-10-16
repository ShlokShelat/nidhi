#!/bin/bash
#SBATCH --job-name=factors-focal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python src/train_factors.py --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --model microsoft/deberta-v3-large --outdir runs/factors_focal \
    --pos-weighting --focal-gamma 1.5
