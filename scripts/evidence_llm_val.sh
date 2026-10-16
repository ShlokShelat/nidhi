#!/bin/bash
#SBATCH --job-name=ev-llm-val
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python src/evidence_llm.py --split artifacts/val_split.csv \
    --qwen "Qwen/Qwen2.5-14B-Instruct" --out artifacts/evidence_llm_val.csv --score
