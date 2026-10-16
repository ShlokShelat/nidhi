#!/bin/bash
#SBATCH --job-name=submit-selftrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python src/predict_factors_nli.py \
    --test data/test.csv \
    --nli-run runs/factors_nli_selftrain,runs/factors_nli_s2 \
    --out artifacts/factors_selftrain_ens.csv
python src/predict.py --test data/test.csv \
    --risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
    --risk-override artifacts/risk_blend.csv \
    --evidence-run runs/evidence_v2 \
    --factors-run runs/factors_cv_final \
    --factors-override artifacts/factors_selftrain_ens.csv \
    --team NIDHI
cp submissions/NIDHI.csv submissions/archive/NIDHI_selftrain.csv
