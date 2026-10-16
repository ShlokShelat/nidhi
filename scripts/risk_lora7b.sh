#!/bin/bash
#SBATCH --job-name=risk-lora7b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python src/train_risk_lora.py \
    --train artifacts/train_split.csv --val artifacts/val_split.csv \
    --model Qwen/Qwen2.5-7B-Instruct \
    --outdir runs/risk_lora7b --epochs 3
