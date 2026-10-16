#!/bin/bash
#SBATCH --job-name=submit-fac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python src/predict.py --test data/test.csv \
    --risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
    --risk-override artifacts/risk_blend.csv \
    --evidence-run runs/evidence_v2 \
    --factors-run runs/factors_cv_final \
    --factors-override artifacts/factors_blend_xl.csv \
    --team NIDHI
cp submissions/NIDHI.csv submissions/archive/NIDHI_facblendxl.csv
