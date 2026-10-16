#!/bin/bash
#SBATCH --job-name=blend-fac-xl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python src/blend_factors_xl.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --v3-runs runs/factors_nli,runs/factors_nli_s2 \
    --xl-run runs/factors_nli_xl \
    --force-weight 0.50 \
    --out artifacts/factors_blend_xl.csv
