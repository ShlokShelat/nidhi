#!/bin/bash
#SBATCH --job-name=riskhard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.out

# Hard-example mining for risk: upweight the training posts the model gets
# WRONG, but only the ones that are demonstrably LEARNABLE.
#
# THE IDEA. Train, find the mistakes, train again weighting those examples up.
# Standard practice; the codebase already does a version of it for factors via
# artifacts/confusable_mined.json.
#
# THE TRAP IT AVOIDS. Finding the mistakes on val_split would destroy the only
# held-out set available -- the model would look excellent on it and fail on
# test, the exact shape of a submission that overfits its own validation
# signal. Every error used here is OUT-OF-FOLD: each post scored by a model
# that never trained on it.
#
# THE REFINEMENT THAT MAKES IT WORTH DOING. Not every error is learnable. A
# second model with a completely different architecture is available, so
# errors can be split:
#     encoder wrong, friend RIGHT  -> learnable; the signal is in the text
#     encoder wrong, friend wrong  -> suspect; two unrelated architectures
#                                  failing together, and the organizers
#                                  report an inter-annotator Kappa of 0.79
# Only the first group is upweighted. Upweighting the second would memorise
# annotation noise. --noise-weight stays at 1.0 so this arm tests ONE thing.
#
# WHY RISK RATHER THAN FACTORS. For risk the errors are demonstrably learnable:
# gold evidence spans classify the confident ideation<->behavior errors at 0.692
# against a 0.538 majority baseline, which is why rationale supervision showed
# a large gain on the member. For factors, the encoder and the friend are both
# wrong 3.74x above chance, which points at label ambiguity instead.
#
# EVERYTHING ELSE MIRRORS runs/risk_v1: 5 epochs, lr 1e-5, batch 8, grad_accum
# 2, max_len 512, alpha_ordinal 0.5, seeds 13/14/15. Val weighted-F1 to beat:
# 0.7576 / 0.7511 / 0.7129.
#
# Weights apply to TRAINING ONLY -- the val loader never sees them, so epoch
# selection stays on the plain objective and cannot be gamed by the weighting.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${NTHREADS}" MKL_NUM_THREADS="${NTHREADS}"
unset RISK_CAT_ONLY || true

TAG="${TAG:-hard}"
HW="${HW:-3.0}"
NW="${NW:-1.0}"
WEIGHTS="${WEIGHTS:-artifacts/risk_hard_weights.csv}"
BASE="${BASE:-submissions/NIDHI_canon_on_nlimnli.csv}"
MODEL="${MODEL:-microsoft/deberta-v3-large}"

for f in artifacts/train_split.csv artifacts/val_split.csv data/test.csv \
         artifacts/friend_oof_export.csv artifacts/friend_risk_probs.csv \
         "${BASE}"; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing"; exit 1; }
done

echo "=== build the weights from OUT-OF-FOLD errors ==="
python src/make_hard_weights.py \
    --hard-weight "${HW}" --noise-weight "${NW}" --out "${WEIGHTS}" \
    || { echo "FAILED weight build"; exit 1; }

for SEED in 13 14 15; do
    OUT="runs/risk_${TAG}_s${SEED}"
    if [ -f "${OUT}/model.pt" ]; then
        echo "--- ${OUT} exists, skipping ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "seed ${SEED} -> ${OUT}   (hard weight ${HW})"
    echo "================================================================"
    python src/train_risk.py \
        --train artifacts/train_split.csv --val artifacts/val_split.csv \
        --model "${MODEL}" --outdir "${OUT}" \
        --epochs 5 --lr 1e-5 --batch 8 --grad-accum 2 --max-len 512 \
        --alpha-ordinal 0.5 --seed "${SEED}" \
        --sample-weights "${WEIGHTS}" \
        || { echo "FAILED seed ${SEED}"; exit 1; }
done

echo ""
echo "=== rename model.pt -> model_final.pt for predict.py ==="
TAG="${TAG}" python - <<'PY' || exit 1
import os
import shutil
TAG = os.environ["TAG"]
for s in (13, 14, 15):
    d = f"runs/risk_{TAG}_s{s}"
    dst = os.path.join(d, "model_final.pt")
    if not os.path.exists(dst):
        shutil.copyfile(os.path.join(d, "model.pt"), dst)
    print(f"  {d}: model_final.pt ready")
PY

RUNS="runs/risk_${TAG}_s13,runs/risk_${TAG}_s14,runs/risk_${TAG}_s15"

echo ""
echo "=== 0.5/0.5 friend, forced 0.30 NLI, same two rules ==="
python src/ensemble_risk.py --test data/test.csv --our-risk-run "${RUNS}" \
    --friend-probs artifacts/friend_risk_probs.csv \
    --out "artifacts/ensembled_risk_${TAG}.csv" || exit 1
python src/blend_risk_saved.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --ens-val "runs/risk_${TAG}_s13/val_predictions.csv,runs/risk_${TAG}_s14/val_predictions.csv,runs/risk_${TAG}_s15/val_predictions.csv" \
    --ens-test "artifacts/ensembled_risk_${TAG}.csv" \
    --force-weight 0.30 --out "artifacts/risk_blend_${TAG}.csv" || exit 1
python src/apply_rules_to_test.py --test data/test.csv \
    --risk-blend "artifacts/risk_blend_${TAG}.csv" \
    --rules urgency,completed_act \
    --out "artifacts/risk_blend_${TAG}_rules.csv" || exit 1
python src/patch_submission_risk.py --base "${BASE}" \
    --risk "artifacts/risk_blend_${TAG}_rules.csv" --max-changes 150 \
    --out "submissions/NIDHI_risk_${TAG}.csv" || exit 1

echo ""
BASE="${BASE}" TAG="${TAG}" python - <<'PY'
import os
from collections import Counter
import pandas as pd
b = pd.read_csv(os.environ["BASE"], keep_default_na=False)
n = pd.read_csv(f"submissions/NIDHI_risk_{os.environ['TAG']}.csv",
                keep_default_na=False)
print(f"evidence identical {b.evidence.equals(n.evidence)}   "
      f"factors identical {b.factors.equals(n.factors)}   <- S2 must be 0.6323")
print(f"risk changed on "
      f"{int((b.risk_level.str.lower() != n.risk_level.str.lower()).sum())} "
      f"of {len(n)} posts")
ob, nb = Counter(b.risk_level.str.lower()), Counter(n.risk_level.str.lower())
print(f"\n{'class':<12}{'old':>6}{'new':>6}{'gold %':>8}")
for k, g in (("indicator", 37.4), ("ideation", 24.5), ("behavior", 23.9),
             ("attempt", 7.0)):
    print(f"{k:<12}{ob[k]:>6}{nb[k]:>6}{g:>8.1f}")
PY

echo ""
echo "DONE -> submissions/NIDHI_risk_${TAG}.csv"
