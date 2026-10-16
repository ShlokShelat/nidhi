#!/bin/bash
#SBATCH --job-name=rebuild-rules
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
echo "STEP 1: regenerate ensembled_risk.csv with EQUAL weights"
echo "================================================================================"
python src/ensemble_risk.py --test data/test.csv \
    --our-risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
    --friend-probs artifacts/friend_risk_probs.csv \
    --weight-ours 0.5 \
    --out artifacts/ensembled_risk.csv

echo ""
echo "================================================================================"
echo "STEP 2: regenerate risk_blend.csv at the reference 0.30 blend weight"
echo "================================================================================"
python src/blend_risk_saved.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --ens-val runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
    --ens-test artifacts/ensembled_risk.csv \
    --force-weight 0.30 \
    --out artifacts/risk_blend_equal.csv

echo ""
echo "================================================================================"
echo "STEP 3: verify this reproduces the archived reference submission exactly"
echo "================================================================================"
python src/predict.py --test data/test.csv \
    --risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 --risk-override artifacts/risk_blend_equal.csv \
    --evidence-run runs/evidence_v2 --factors-run runs/factors_cv_final \
    --factors-override artifacts/nli_ensemble_factors.csv --team NIDHI

echo ""
echo "diff against the archived reference (empty output below = VERIFIED IDENTICAL):"
diff submissions/NIDHI.csv submissions/archive/NIDHI_20260730_2009_riskblend030.csv
DIFF_STATUS=$?

if [ $DIFF_STATUS -ne 0 ]; then
    echo ""
    echo "!!! DIFF FAILED — reconstructed baseline does NOT match the archived reference. STOPPING before applying rules. !!!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Verification passed — proceeding to rule application"
echo "================================================================================"

echo ""
echo "================================================================================"
echo "STEP 4: apply the two validated rules on top of the confirmed baseline"
echo "================================================================================"
python src/apply_rules_to_test.py \
    --test data/test.csv \
    --risk-blend artifacts/risk_blend_equal.csv \
    --rules urgency,completed_act \
    --out artifacts/risk_blend_with_rules.csv

echo ""
echo "================================================================================"
echo "STEP 5: build the final submission"
echo "================================================================================"
python src/predict.py --test data/test.csv \
    --risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 --risk-override artifacts/risk_blend_with_rules.csv \
    --evidence-run runs/evidence_v2 --factors-run runs/factors_cv_final \
    --factors-override artifacts/nli_ensemble_factors.csv --team NIDHI

cp submissions/NIDHI.csv submissions/archive/NIDHI_rules_urgency_completedact.csv
echo ""
echo "Archived: submissions/archive/NIDHI_rules_urgency_completedact.csv"
echo "================================================================================"
echo "DONE"
echo "================================================================================"
