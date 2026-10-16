#!/bin/bash
#SBATCH --job-name=augment2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
source scripts/env.sh qwen_env
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
set -e
python src/augment_factors.py \
    --train artifacts/train_split.csv \
    --model Qwen/Qwen2.5-14B-Instruct \
    --target-labels "substance use,exposure to others' suicide,sexual orientation related issues,poor school performance,traumatic experience,cognitive deficits,low socio-economic status" \
    --per-post 6 --backend hf --out artifacts/train_aug2.csv
