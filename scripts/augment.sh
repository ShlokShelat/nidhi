#!/bin/bash
#SBATCH --job-name=augment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_env
export TOKENIZERS_PARALLELISM=false
set -e

python src/augment_factors.py \
    --train artifacts/train_split.csv \
    --model Qwen/Qwen2.5-14B-Instruct \
    --target-labels "substance use,exposure to others' suicide,sexual orientation related issues,cognitive deficits,poor school performance,sense of responsibility,meaning in life,traumatic experience,low socio-economic status" \
    --per-post 3 --backend hf --out artifacts/train_aug.csv

# retrain factors on the augmented set (riskcup for stable DeBERTa training)
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
python src/train_factors.py --train artifacts/train_aug.csv \
    --val artifacts/val_split.csv --model microsoft/deberta-v3-large \
    --outdir runs/factors_v3_aug
