#!/bin/bash
#SBATCH --job-name=fac-mbert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python src/train_factors_nli.py \
    --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --model "mental/mental-bert-base-uncased" --outdir runs/factors_nli_mbert \
    --max-len 256 --neg-ratio 3 --epochs 3
