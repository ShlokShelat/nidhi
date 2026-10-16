#!/bin/bash
#SBATCH --job-name=factors-synthetic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python src/train_factors_nli.py \
    --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --external artifacts/gpt5_generated_all_fixed.csv \
    --model microsoft/deberta-v3-large --outdir runs/factors_nli_synthetic_final \
    --neg-ratio 3 --epochs 3
