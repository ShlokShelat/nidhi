#!/bin/bash
#SBATCH --job-name=riskmnlisub
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Build the risk column with MNLI-pretrained NLI members. ONE variable: the
# encoder those two members start from. Blend weight, ensemble members, rules
# and every other component are untouched, so the encoder is the only thing a
# change in the result can be attributed to.
#
# Risk is an ARGMAX over 4 classes, so a member swap has no per-class threshold
# to mis-transfer; the decision depends only on the ordering of the four
# probabilities, not on where any of them sits on an absolute scale.
#
# NOT INCLUDED, deliberately:
#   * w_nli = 1.0, which drops the deberta and friend members. That is a second
#     independent change; bundling it with the encoder swap would make a
#     negative result uninterpretable, since neither half could be blamed.
#   * a second NLI architecture. A member is only added when it improves BOTH
#     author halves of the held-out data -- a gain on one half alone is within
#     what author-level resampling produces by chance.
#
# --force-weight 0.30 keeps the blend weight FIXED. Re-tuning the weight after
# swapping a member fits a parameter to the same data used to judge the swap,
# which flatters the swap and does not carry over to unseen authors.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

BASE="${BASE:-submissions/NIDHI_evidence_vote8.csv}"
NLI="${NLI:-runs/risk_nli_mnli,runs/risk_nli_mnli_s2}"
TAG="${TAG:-mnli}"
# 0.30 is the deployed weight. 1.00 drops the deberta and friend members
# entirely -- a STRUCTURAL choice about which members exist, not a tuned value.
# Do NOT pass the peak of the weight sweep: held-out and out-of-fold sweeps put
# the peak in different places, so its location is noise, and selecting it is
# fitting the weight to whichever split was swept.
WEIGHT="${WEIGHT:-0.30}"

for d in $(echo "${NLI}" | tr ',' ' '); do
    [ -f "${d}/model.pt" ] || { echo "ERROR: ${d}/model.pt missing"; exit 1; }
done
for f in "${BASE}" data/test.csv artifacts/ensembled_risk.csv \
         artifacts/val_split.csv; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing"; exit 1; }
done

echo "base   : ${BASE}"
echo "nli    : ${NLI}"
echo "weight : ${WEIGHT}   $([ "${WEIGHT}" = "1.00" ] || [ "${WEIGHT}" = "1.0" ] \
    && echo '(NLI-only -- deberta and friend dropped)' \
    || echo '(champion weight)')"

echo ""
echo "================================================================"
echo "STEP 1  blend, weight FORCED to ${WEIGHT} (nothing fitted)"
echo "================================================================"
python src/blend_risk_saved.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --nli-run "${NLI}" \
    --ens-val runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
    --ens-test artifacts/ensembled_risk.csv \
    --force-weight "${WEIGHT}" \
    --out "artifacts/risk_blend_${TAG}.csv" \
    || { echo "FAILED step 1"; exit 1; }

echo ""
echo "================================================================"
echo "STEP 2  the same two rules the deployed chain uses"
echo "================================================================"
python src/apply_rules_to_test.py \
    --test data/test.csv \
    --risk-blend "artifacts/risk_blend_${TAG}.csv" \
    --rules urgency,completed_act \
    --out "artifacts/risk_blend_${TAG}_rules.csv" \
    || { echo "FAILED step 2"; exit 1; }

echo ""
echo "================================================================"
echo "STEP 3  patch onto the base submission -- risk column ONLY"
echo "================================================================"
# --max-changes defaults to 25, which is meant for a rule tweak. An encoder
# swap legitimately moves more posts than that, so the cap is raised and the
# actual count is printed for inspection rather than silently accepted.
python src/patch_submission_risk.py \
    --base "${BASE}" \
    --risk "artifacts/risk_blend_${TAG}_rules.csv" \
    --max-changes 150 \
    --out "submissions/NIDHI_risk_${TAG}.csv" \
    || { echo "FAILED step 3"; exit 1; }

echo ""
echo "================================================================"
echo "STEP 4  sanity"
echo "================================================================"
BASE="${BASE}" TAG="${TAG}" python - <<'PY'
import os
import pandas as pd
from collections import Counter

b = pd.read_csv(os.environ["BASE"], keep_default_na=False)
n = pd.read_csv(f"submissions/NIDHI_risk_{os.environ['TAG']}.csv",
                keep_default_na=False)
print(f"rows {len(n)}   row_id order identical "
      f"{b.row_id.equals(n.row_id)}")
print(f"evidence identical {b.evidence.equals(n.evidence)}   "
      f"factors identical {b.factors.equals(n.factors)}")
ch = int((b.risk_level.str.lower() != n.risk_level.str.lower()).sum())
print(f"risk_level changed on {ch} of {len(n)} posts ({100*ch/len(n):.1f}%)")
print(f"\nold distribution {dict(Counter(b.risk_level.str.lower()))}")
print(f"new distribution {dict(Counter(n.risk_level.str.lower()))}")
print("\nA swap of one member at 30% weight should move a modest slice of")
print("posts. If the distribution has shifted sharply toward one class,")
print("something is wrong -- check before uploading.")
PY

echo ""
echo "DONE -> submissions/NIDHI_risk_${TAG}.csv"
echo "Evidence and factors are byte-identical to the base submission, so any"
echo "movement is attributable to the risk encoder alone."
