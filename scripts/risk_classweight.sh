#!/bin/bash
#SBATCH --job-name=riskcw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out

# Turn on class weighting for the risk loss. ONE variable against the deployed
# risk chain.
#
# WHY. The friend's model -- mental/mental-roberta-base, a 125M BASE encoder --
# outscores the 435M deberta-v3-large member on held-out OOF, 0.7522 against
# 0.7311, and the gap is concentrated on the rarest class:
#     class        gold   friend F1   deberta member
#     attempt       110      0.615        0.457
#     behavior      388      0.694
#     ideation      499      0.720
#     indicator     593      0.842
# `attempt` is 7% of the data. Under an unweighted cross-entropy the loss is
# dominated by the three common classes, so the decision boundary drifts away
# from the rare one -- the standard failure mode of an imbalanced 4-way head,
# and exactly the shape of the gap above.
#
# The tri-head pipeline (vendored at src/friend_pipeline/) differs on three axes:
# multi-task heads (risk + evidence tagging + factors on one encoder),
# mental-health domain pretraining, and CLASS WEIGHTS on the risk loss. The
# first is tested separately by scripts/risk_rationale_deploy.sh.
#
# This tests the third. The two implementations use an IDENTICAL formula:
#     len(tr_df) / (len(RISK_LEVELS) * counts[label])
# inverse frequency, normalised. src/train_risk.py already exposes it as
# --class-weighting; the deployed runs all have class_weighting=False.
#
# Every other hyperparameter below is copied verbatim from
# runs/risk_v1/run_config.json: 5 epochs, lr 1e-5, batch 8, grad_accum 2,
# max_len 512, alpha_ordinal 0.5, seeds 13/14/15. Their val weighted-F1 is
# 0.7576 / 0.7511 / 0.7129, which is what the weighted runs are compared to.
#
# One training flag changes and nothing else -- same architecture, same three
# seeds, same data, same ensemble, same blend, same rules -- so the result is
# attributable to class weighting alone.
#
# NOTE this deliberately does NOT set RISK_CAT_ONLY. Class weighting should
# improve the categorical head's `attempt` performance, which is the same thing
# the cat-only change exploits; testing both at once would make a negative
# result uninterpretable. Measure them separately, combine after.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${NTHREADS}" MKL_NUM_THREADS="${NTHREADS}"
unset RISK_CAT_ONLY || true

TAG="${TAG:-cw}"
# BASE must be the CURRENT best complete submission. patch_submission_risk.py
# replaces only the risk column and copies the rest verbatim, so patching onto
# a superseded base silently discards every improvement that lives in the other
# columns of the newer one, and the result is then not a one-variable change.
BASE="${BASE:-submissions/NIDHI_recond.csv}"
SEEDS="${SEEDS:-13 14 15}"
MODEL="${MODEL:-microsoft/deberta-v3-large}"

for f in artifacts/train_split.csv artifacts/val_split.csv data/test.csv \
         artifacts/friend_risk_probs.csv "${BASE}"; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing"; exit 1; }
done
echo "class weighting ON; every other hyperparameter mirrors runs/risk_v1"
echo "unweighted reference val weighted-F1: 0.7576 (s13) / 0.7511 (s14) / 0.7129 (s15)"

for SEED in ${SEEDS}; do
    OUT="runs/risk_${TAG}_s${SEED}"
    if [ -f "${OUT}/model.pt" ]; then
        echo "--- ${OUT} exists, skipping ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "seed ${SEED} -> ${OUT}"
    echo "================================================================"
    python src/train_risk.py \
        --train artifacts/train_split.csv --val artifacts/val_split.csv \
        --model "${MODEL}" --outdir "${OUT}" \
        --epochs 5 --lr 1e-5 --batch 8 --grad-accum 2 --max-len 512 \
        --alpha-ordinal 0.5 --class-weighting --seed "${SEED}" \
        || { echo "FAILED seed ${SEED}"; exit 1; }
done

# ---------------------------------------------------------------------------
# Assembly needs ALL THREE seeds. When this is fired as three parallel
# single-seed jobs (SEEDS=13 / 14 / 15) for queue backfill, the first two to
# finish must stop here rather than ensemble an incomplete set.
# ---------------------------------------------------------------------------
# Accept EITHER name. The trainer writes model.pt, but a completed run that
# already went through the rename step below may have only model_final.pt, so
# testing for model.pt alone would report a finished seed as untrained and
# abort the assembly chain.
MISSING=0
for S in 13 14 15; do
    if [ ! -f "runs/risk_${TAG}_s${S}/model.pt" ] \
       && [ ! -f "runs/risk_${TAG}_s${S}/model_final.pt" ]; then
        MISSING=$((MISSING+1))
    fi
done
if [ "${MISSING}" -gt 0 ]; then
    echo ""
    echo "================================================================"
    echo "${MISSING} of 3 seeds not yet trained -- stopping before assembly."
    echo "Re-run this script with no SEEDS override once all three exist;"
    echo "the training loop skips finished seeds and goes straight to the"
    echo "ensemble/blend/rules/patch chain."
    echo "================================================================"
    exit 0
fi

# src/predict.py reads model_final.pt; the trainer writes model.pt.
echo ""
echo "=== rename model.pt -> model_final.pt for predict.py ==="
TAG="${TAG}" python - <<'PY' || exit 1
import os
import shutil
TAG = os.environ["TAG"]
for s in (13, 14, 15):
    d = f"runs/risk_{TAG}_s{s}"
    src, dst = os.path.join(d, "model.pt"), os.path.join(d, "model_final.pt")
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    assert os.path.exists(dst), f"{dst} missing after rename step"
    print(f"  {d}: model_final.pt ready")
PY

RUNS="runs/risk_${TAG}_s13,runs/risk_${TAG}_s14,runs/risk_${TAG}_s15"

echo ""
echo "=== 0.5/0.5 with the friend ==="
python src/ensemble_risk.py \
    --test data/test.csv --our-risk-run "${RUNS}" \
    --friend-probs artifacts/friend_risk_probs.csv \
    --out "artifacts/ensembled_risk_${TAG}.csv" \
    || { echo "FAILED ensemble"; exit 1; }

echo ""
echo "=== blend with the SAME two NLI members at a FORCED 0.30 ==="
python src/blend_risk_saved.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --ens-val "runs/risk_${TAG}_s13/val_predictions.csv,runs/risk_${TAG}_s14/val_predictions.csv,runs/risk_${TAG}_s15/val_predictions.csv" \
    --ens-test "artifacts/ensembled_risk_${TAG}.csv" \
    --force-weight 0.30 \
    --out "artifacts/risk_blend_${TAG}.csv" \
    || { echo "FAILED blend"; exit 1; }

echo ""
echo "=== the same two rules ==="
python src/apply_rules_to_test.py \
    --test data/test.csv \
    --risk-blend "artifacts/risk_blend_${TAG}.csv" \
    --rules urgency,completed_act \
    --out "artifacts/risk_blend_${TAG}_rules.csv" \
    || { echo "FAILED rules"; exit 1; }

echo ""
echo "=== patch the risk column onto the base submission ==="
python src/patch_submission_risk.py \
    --base "${BASE}" --risk "artifacts/risk_blend_${TAG}_rules.csv" \
    --max-changes 150 --out "submissions/NIDHI_risk_${TAG}.csv" \
    || { echo "FAILED patch"; exit 1; }

echo ""
BASE="${BASE}" TAG="${TAG}" python - <<'PY'
import os
from collections import Counter
import pandas as pd
b = pd.read_csv(os.environ["BASE"], keep_default_na=False)
n = pd.read_csv(f"submissions/NIDHI_risk_{os.environ['TAG']}.csv",
                keep_default_na=False)
print(f"evidence identical {b.evidence.equals(n.evidence)}   "
      f"factors identical {b.factors.equals(n.factors)}   "
      f"<- both must be True, or this is not a risk-only change")
print(f"  (base {os.environ['BASE']})")
ch = int((b.risk_level.str.lower() != n.risk_level.str.lower()).sum())
print(f"risk changed on {ch} of {len(n)} posts")
ob, nb = Counter(b.risk_level.str.lower()), Counter(n.risk_level.str.lower())
print(f"\n{'class':<12}{'old':>6}{'new':>6}{'delta':>7}{'gold %':>8}")
for k, g in (("indicator", 37.4), ("ideation", 24.5), ("behavior", 23.9),
             ("attempt", 7.0)):
    print(f"{k:<12}{ob[k]:>6}{nb[k]:>6}{nb[k]-ob[k]:>+7}{g:>8.1f}")
print("\nclass weighting should raise `attempt` toward its 7.0% gold share;")
print("the unweighted chain emits 17 of 378 = 4.5%. The friend, which does")
print("use class weights, reaches attempt F1 0.615 against the deberta")
print("member's 0.457 -- that is the gap this change targets.")
PY

echo ""
echo "DONE -> submissions/NIDHI_risk_${TAG}.csv"
