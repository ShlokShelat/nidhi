#!/bin/bash
#SBATCH --job-name=riskcat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# Drop the ordinal head at inference. NO RETRAINING -- this reuses the
# champion's own three risk checkpoints and only changes which logits they
# return.
#
# THE MEASUREMENT. src/risk_head_diagnosis.py dumped the two heads separately
# (nothing had ever done this; _combine() only ever exposes the merged 50/50
# mixture). On the val split:
#     categorical alone   0.7579
#     ordinal alone       0.7363
#     shipped 50/50 mix   0.7498
# The gap is driven by `attempt`: cat F1 0.457 against ordinal 0.200. And
# _combine() hardcodes 50/50 no matter what --alpha-ordinal says -- alpha only
# ever weighted the LOSS -- so the mixture was never a tuned decision, it was a
# default nobody revisited.
#
# WHY SHIP SOMETHING THAT MOVES ONLY ~7 POSTS. Because its correctness does not
# depend on an offline estimate. Nine submissions in, the two that WON were the
# two whose correctness was arguable without a number (an encoder swap confirmed
# by two unrelated metrics, and a deterministic string fix); all seven losses
# were chosen BY an offline number, and the winner's-curse arithmetic matched
# our best reading to four decimals. Cat-only beats the mixture on the val split
# AND on the head diagnosis, on the same checkpoints that ship. That is the
# filter, and this passes it.
#
# It should be folded into the pipeline permanently rather than treated as a
# one-off arm.
#
# RISK_CAT_ONLY is read inside DualHeadModel.forward, so it must be exported for
# every step that loads a risk model -- ensemble_risk and blend_risk_saved both
# do. It is NOT set during training anywhere; doing so would also change epoch
# selection, which is a different experiment.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${NTHREADS}" MKL_NUM_THREADS="${NTHREADS}"

export RISK_CAT_ONLY=1
echo "RISK_CAT_ONLY=1 -- categorical logits only, ordinal head ignored"

TAG="${TAG:-catonly}"
BASE="${BASE:-submissions/NIDHI_canon_on_nlimnli.csv}"
# the champion's own three risk members, unchanged
RUNS="${RUNS:-runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15}"

for d in $(echo "${RUNS}" | tr ',' ' '); do
    [ -f "${d}/model_final.pt" ] || { echo "ERROR: ${d}/model_final.pt missing"; \
        exit 1; }
done
for p in "${BASE}" data/test.csv artifacts/friend_risk_probs.csv \
         artifacts/val_split.csv; do
    [ -f "${p}" ] || { echo "ERROR: ${p} missing"; exit 1; }
done
echo "members: ${RUNS}"
echo "base   : ${BASE}"

echo ""
echo "=== 0.5/0.5 with the friend, champion members, cat-only logits ==="
python src/ensemble_risk.py \
    --test data/test.csv --our-risk-run "${RUNS}" \
    --friend-probs artifacts/friend_risk_probs.csv \
    --out "artifacts/ensembled_risk_${TAG}.csv" \
    || { echo "FAILED ensemble"; exit 1; }

echo ""
echo "=== blend with the SAME two NLI members at a FORCED 0.30 ==="
# The NLI members are plain classifiers, not DualHeadModel, so RISK_CAT_ONLY
# does not touch them -- only our three DeBERTa members change.
python src/blend_risk_saved.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --ens-val runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
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
echo "=== patch the risk column onto the champion ==="
python src/patch_submission_risk.py \
    --base "${BASE}" \
    --risk "artifacts/risk_blend_${TAG}_rules.csv" \
    --max-changes 60 \
    --out "submissions/NIDHI_risk_${TAG}.csv" \
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
      f"factors identical {b.factors.equals(n.factors)}   <- S2 must be 0.6323")
ch = int((b.risk_level.str.lower() != n.risk_level.str.lower()).sum())
print(f"risk changed on {ch} of {len(n)} posts")
print("  The diagnosis predicted ~7, all behavior->attempt. A much larger")
print("  number means RISK_CAT_ONLY reached something it should not have.")
print(f"old {dict(Counter(b.risk_level.str.lower()))}")
print(f"new {dict(Counter(n.risk_level.str.lower()))}")
PY

echo ""
echo "DONE -> submissions/NIDHI_risk_${TAG}.csv"
