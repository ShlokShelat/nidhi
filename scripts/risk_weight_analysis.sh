#!/bin/bash
#SBATCH --job-name=risk-weight-analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

echo "================================================================================"
echo "WEIGHT SWEEP"
echo "================================================================================"
python src/risk_ensemble_weight_sweep_v2.py \
    --val artifacts/val_split.csv \
    --val-probs runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv

echo ""
echo "================================================================================"
echo "DISAGREEMENT ANALYSIS"
echo "================================================================================"
python src/risk_disagreement_analysis.py \
    --val artifacts/val_split.csv \
    --val-probs runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
    --names risk_v1,risk_v1_s14,risk_v1_s15
