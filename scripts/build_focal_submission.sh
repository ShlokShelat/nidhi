#!/bin/bash
#SBATCH --job-name=build-focal-submission
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
echo "STEP 1: rebuild base risk ensemble using the 3 FOCAL checkpoints (equal weights)"
echo "================================================================================"
python src/ensemble_risk.py --test data/test.csv \
    --our-risk-run runs/risk_focal_rerun_g2_0,runs/risk_focal_rerun_g2_0_s14,runs/risk_focal_rerun_g2_0_s15 \
    --friend-probs artifacts/friend_risk_probs.csv \
    --weight-ours 0.5 \
    --out artifacts/ensembled_risk_focal.csv

echo ""
echo "================================================================================"
echo "STEP 2: blend with risk-NLI, same weight as champion (0.30)"
echo "================================================================================"
python src/blend_risk_saved.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --ens-val runs/risk_focal_rerun_g2_0/val_predictions.csv,runs/risk_focal_rerun_g2_0_s14/val_predictions.csv,runs/risk_focal_rerun_g2_0_s15/val_predictions.csv \
    --ens-test artifacts/ensembled_risk_focal.csv \
    --force-weight 0.30 \
    --out artifacts/risk_blend_focal.csv

echo ""
echo "================================================================================"
echo "STEP 3: apply the validated rules (urgency, completed_act) on top"
echo "================================================================================"
python src/apply_rules_to_test.py \
    --test data/test.csv \
    --risk-blend artifacts/risk_blend_focal.csv \
    --rules urgency,completed_act \
    --out artifacts/risk_blend_focal_with_rules.csv

echo ""
echo "================================================================================"
echo "STEP 4: build the final submission"
echo "================================================================================"
python src/predict.py --test data/test.csv \
    --risk-run runs/risk_focal_rerun_g2_0,runs/risk_focal_rerun_g2_0_s14,runs/risk_focal_rerun_g2_0_s15 \
    --risk-override artifacts/risk_blend_focal_with_rules.csv \
    --evidence-run runs/evidence_v2 --factors-run runs/factors_cv_final \
    --factors-override artifacts/nli_ensemble_factors.csv --team NIDHI

cp submissions/NIDHI.csv submissions/archive/NIDHI_focal_ensemble.csv

echo ""
echo "================================================================================"
echo "DONE — submissions/archive/NIDHI_focal_ensemble.csv ready to upload"
echo "================================================================================"
ls -la submissions/archive/NIDHI_focal_ensemble.csv
