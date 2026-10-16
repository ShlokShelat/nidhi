#!/bin/bash
#SBATCH --job-name=risk-weight-variants
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

echo "================================================================================"
echo "VARIANT A: (0.5, 0.25, 0.25)"
echo "================================================================================"
python src/ensemble_risk.py --test data/test.csv \
    --our-risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
    --friend-probs artifacts/friend_risk_probs.csv \
    --weight-ours 0.5 \
    --our-weights 0.5,0.25,0.25 \
    --out artifacts/ensembled_risk.csv

python src/blend_risk_saved.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --ens-val runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
    --ens-test artifacts/ensembled_risk.csv \
    --force-weight 0.30 \
    --out artifacts/risk_blend.csv

python src/predict.py --test data/test.csv \
    --risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 --risk-override artifacts/risk_blend.csv \
    --evidence-run runs/evidence_v2 --factors-run runs/factors_cv_final \
    --factors-override artifacts/nli_ensemble_factors.csv --team NIDHI

cp submissions/NIDHI.csv submissions/archive/NIDHI_riskweight_050_025_025.csv
echo "Variant A archived."

echo ""
echo "================================================================================"
echo "VARIANT B: (0.5, 0.3, 0.2)"
echo "================================================================================"
python src/ensemble_risk.py --test data/test.csv \
    --our-risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
    --friend-probs artifacts/friend_risk_probs.csv \
    --weight-ours 0.5 \
    --our-weights 0.5,0.3,0.2 \
    --out artifacts/ensembled_risk.csv

python src/blend_risk_saved.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --ens-val runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
    --ens-test artifacts/ensembled_risk.csv \
    --force-weight 0.30 \
    --out artifacts/risk_blend.csv

python src/predict.py --test data/test.csv \
    --risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 --risk-override artifacts/risk_blend.csv \
    --evidence-run runs/evidence_v2 --factors-run runs/factors_cv_final \
    --factors-override artifacts/nli_ensemble_factors.csv --team NIDHI

cp submissions/NIDHI.csv submissions/archive/NIDHI_riskweight_050_030_020.csv
echo "Variant B archived."

echo ""
echo "================================================================================"
echo "BOTH VARIANTS COMPLETE"
echo "================================================================================"
ls -la submissions/archive/NIDHI_riskweight_*.csv
