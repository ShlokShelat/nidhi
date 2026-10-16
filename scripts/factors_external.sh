#!/bin/bash
#SBATCH --job-name=factors-ext
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python src/train_factors_cv.py \
    --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --model microsoft/deberta-v3-large --outdir runs/factors_external \
    --external artifacts/external_factor_aug.csv --save-oof
