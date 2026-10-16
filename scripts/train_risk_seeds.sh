#!/bin/bash
#SBATCH --job-name=risk-seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
source scripts/env.sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
set -e
python src/train_risk.py --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --model microsoft/deberta-v3-large --outdir runs/risk_v1_s14 --seed 14
python src/train_risk.py --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --model microsoft/deberta-v3-large --outdir runs/risk_v1_s15 --seed 15
