#!/bin/bash
#SBATCH --job-name=risk-nofriend
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Ablate the friend (MentalRoBERTa) model out of the risk ensemble, to measure
# what the second architecture contributes at the point where it acts: the
# 0.5/0.5 average, before the 0.30 risk-NLI blend.
#
# The ablation is judged on author-disjoint halves by src/friend_ablation.py,
# in terms of the posts the friend uniquely FIXES against the posts it uniquely
# BREAKS. Aggregate accuracy alone cannot separate a member that adds
# independent signal from one that merely agrees with the majority, and a
# member must help on both author halves to be trusted on unseen authors.
#
# Deliberately NOT re-weighted. Dropping a component is a structural decision
# with one degree of freedom; reading a weight off the val sweep fits a
# continuous parameter to 372 posts and will track that split's noise.
#
# Only STEP 1 differs from the deployed chain: --weight-ours 1.0 gives the
# friend zero weight. Everything downstream is byte-identical, so the friend is
# the only variable.
#
# The patch base is the same submission the other risk experiment was built
# from, so the two risk changes remain independently attributable.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

echo "================================================================"
echo "STEP 1: risk ensemble with the friend at ZERO weight"
echo "================================================================"
python src/ensemble_risk.py --test data/test.csv \
    --our-risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
    --friend-probs artifacts/friend_risk_probs.csv \
    --weight-ours 1.0 \
    --out artifacts/ensembled_risk_nofriend.csv || exit 1
# "changed 0/378 vs the local-seed-ensemble alone" is the EXPECTED output here --
# with weight 1.0 the blend IS the seed ensemble.

echo ""
echo "================================================================"
echo "STEP 2: same 0.30 risk-NLI blend as the deployed chain"
echo "================================================================"
python src/blend_risk_saved.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --ens-val runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
    --ens-test artifacts/ensembled_risk_nofriend.csv \
    --force-weight 0.30 \
    --out artifacts/risk_blend_nofriend.csv || exit 1

echo ""
echo "================================================================"
echo "STEP 3: identical rule layer"
echo "================================================================"
python src/apply_rules_to_test.py --test data/test.csv \
    --risk-blend artifacts/risk_blend_nofriend.csv \
    --rules urgency,completed_act \
    --out artifacts/risk_blend_nofriend_rules.csv || exit 1

echo ""
echo "================================================================"
echo "STEP 4: sanity gate (no honest val score exists for a blend change)"
echo "================================================================"
python src/risk_candidate_check.py \
    --candidate artifacts/risk_blend_nofriend_rules.csv \
    --champion  artifacts/risk_blend_scheduledfix.csv

echo ""
echo "================================================================"
echo "STEP 5: risk-only patch onto the base submission"
echo "================================================================"
python src/patch_submission_risk.py \
    --base submissions/NIDHI_evidence_vote2.csv \
    --risk artifacts/risk_blend_nofriend_rules.csv \
    --out submissions/NIDHI_risk_nofriend.csv \
    --max-changes 150

echo ""
echo "================================================================"
echo "DONE -> submissions/NIDHI_risk_nofriend.csv"
echo ""
echo "  The friend carries 35% of the pre-NLI blend, so removing it should"
echo "  move a substantial number of posts. A count near the 150 patch cap"
echo "  means something other than the friend weight changed -- stop and"
echo "  find it before reading the result."
echo "================================================================"
