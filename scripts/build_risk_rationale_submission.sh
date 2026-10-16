#!/bin/bash
#SBATCH --job-name=riskratsub
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# Build the risk column from the rationale-supervised risk bag. ONE variable
# changes: the DeBERTa member becomes a 5-fold rationale-trained bag instead of
# a set of train_split seeds. Friend member, NLI member, blend weights and rules
# are untouched, so any movement is attributable to that member alone.
#
# WHY AVERAGING FIVE FOLD MODELS IS SAFE HERE. Risk is an ARGMAX over 4 classes
# with no per-class thresholds, so averaging more models can only stabilise the
# decision boundary. (A thresholded head is different: there, averaging shifts
# the probability distribution out from under a threshold fitted elsewhere.)
#
# WHY AUTHOR-DISJOINT FOLDS. A fixed train/val split spends its validation posts
# on epoch selection and never trains on them. Five author-disjoint folds train
# on every post across the bag while still selecting each epoch on authors the
# model has not seen, which is the regime the final evaluation uses.
#
# THE PROBABILITY DEFINITION IS VERIFIED IDENTICAL, not assumed -- a member that
# emits a differently-normalised quantity blends silently and wrongly:
#   DualHeadModel.forward returns logits = log(0.5*cat_p + 0.5*ord_p)
#   predict.py:_risk_probs_one  -> softmax(logits) = softmax(log(mix)) = mix
#                                  (mix already sums to 1)
#   train_risk_rationale.py     -> val_probs.npz stores np.exp(P) = the same mix
# So the offline instrument and the deployed path score the SAME quantity.
#
# TWO PLUMBING FACTS, handled below:
#   * src/predict.py:load_state reads model_final.pt; the trainer wrote model.pt
#   * it RAISES on unexpected state-dict keys, and the rationale checkpoints
#     carry span_head.* -- which is inference-irrelevant but would abort
# Step 1 writes a stripped model_final.pt beside each checkpoint, leaving
# predict.py -- which is on the deployed path -- unmodified.
#
# The class_weights buffer needs no handling: it is in predict.py's
# TRAINING_ONLY_BUFFERS, so it is tolerated as an unexpected key.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

TAG="${TAG:-rat}"
BASE="${BASE:-submissions/NIDHI_canon_on_nlimnli.csv}"
K="${K:-5}"

RUNS=""
for f in $(seq 0 $((K - 1))); do
    d="runs/risk_${TAG}_f${f}"
    [ -f "${d}/model.pt" ] || { echo "ERROR: ${d}/model.pt missing"; exit 1; }
    RUNS="${RUNS:+${RUNS},}${d}"
done
for p in "${BASE}" data/test.csv artifacts/friend_risk_probs.csv \
         artifacts/val_split.csv; do
    [ -f "${p}" ] || { echo "ERROR: ${p} missing"; exit 1; }
done
echo "bag  : ${RUNS}"
echo "base : ${BASE}"

echo ""
echo "================================================================"
echo "STEP 1  strip span_head -> model_final.pt (predict.py compatible)"
echo "================================================================"
TAG="${TAG}" K="${K}" python - <<'PY' || exit 1
import json, os
import torch
TAG, K = os.environ["TAG"], int(os.environ["K"])
for f in range(K):
    d = f"runs/risk_{TAG}_f{f}"
    sd = torch.load(os.path.join(d, "model.pt"), map_location="cpu")
    dropped = [k for k in sd if k.startswith("span_head")]
    for k in dropped:
        sd.pop(k)
    if not dropped:
        raise SystemExit(f"{d}: no span_head.* keys -- is this really a "
                         f"rationale checkpoint?")
    # predict.py loads with weights_only=True, so the payload must be a plain
    # tensor dict. best_state already is (detach().cpu().clone()).
    torch.save(sd, os.path.join(d, "model_final.pt"))
    # predict.py rebuilds the architecture from run_config.json, so every key it
    # reads must be present. Fail loudly rather than silently defaulting.
    p = os.path.join(d, "run_config.json")
    cfg = json.load(open(p))
    for need in ("model", "max_len", "alpha_ordinal", "user_context"):
        if need not in cfg:
            raise SystemExit(f"{d}: run_config.json lacks {need!r}")
    # --user-context is type=int, so it lands as 1; build_text takes a bool and
    # 1 is truthy, so this is a cosmetic no-op written only if it changes.
    if not isinstance(cfg["user_context"], bool):
        cfg["user_context"] = bool(cfg["user_context"])
        json.dump(cfg, open(p, "w"), indent=2)
    print(f"  {d}: dropped {dropped}")
    print(f"      model={cfg['model']}  max_len={cfg['max_len']}  "
          f"alpha_ordinal={cfg['alpha_ordinal']}  "
          f"user_context={cfg['user_context']}  "
          f"lambda_span={cfg.get('lambda_span')}  "
          f"best_val={cfg.get('best_val_weighted_f1')}")
PY

echo ""
echo "================================================================"
echo "STEP 2  the bag as the DeBERTa member, 0.5/0.5 with the friend"
echo "================================================================"
python src/ensemble_risk.py \
    --test data/test.csv \
    --our-risk-run "${RUNS}" \
    --friend-probs artifacts/friend_risk_probs.csv \
    --out "artifacts/ensembled_risk_${TAG}.csv" \
    || { echo "FAILED step 2"; exit 1; }

echo ""
echo "================================================================"
echo "STEP 3  blend with the SAME two NLI members at a FORCED 0.30"
echo "================================================================"
# NOTE the val sweep this prints is computed from the risk_v1 val predictions.
# The rationale folds never scored artifacts/val_split.csv, and could not do so
# honestly, because val posts sit inside some folds' training data. The sweep is
# therefore REPORTING ONLY; --force-weight 0.30 fixes the weight so the sweep
# cannot influence the output. Judge the member on out-of-fold predictions.
python src/blend_risk_saved.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --ens-val runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
    --ens-test "artifacts/ensembled_risk_${TAG}.csv" \
    --force-weight 0.30 \
    --out "artifacts/risk_blend_${TAG}.csv" \
    || { echo "FAILED step 3"; exit 1; }

echo ""
echo "================================================================"
echo "STEP 4  the same two rules"
echo "================================================================"
python src/apply_rules_to_test.py \
    --test data/test.csv \
    --risk-blend "artifacts/risk_blend_${TAG}.csv" \
    --rules urgency,completed_act \
    --out "artifacts/risk_blend_${TAG}_rules.csv" \
    || { echo "FAILED step 4"; exit 1; }

echo ""
echo "================================================================"
echo "STEP 5  patch the risk column onto the champion"
echo "================================================================"
python src/patch_submission_risk.py \
    --base "${BASE}" \
    --risk "artifacts/risk_blend_${TAG}_rules.csv" \
    --max-changes 150 \
    --out "submissions/NIDHI_risk_${TAG}.csv" \
    || { echo "FAILED step 5"; exit 1; }

echo ""
echo "================================================================"
echo "STEP 6  sanity"
echo "================================================================"
BASE="${BASE}" TAG="${TAG}" python - <<'PY'
import os
from collections import Counter
import pandas as pd
b = pd.read_csv(os.environ["BASE"], keep_default_na=False)
n = pd.read_csv(f"submissions/NIDHI_risk_{os.environ['TAG']}.csv",
                keep_default_na=False)
print(f"rows {len(n)}   row_id order identical {b.row_id.equals(n.row_id)}")
print(f"evidence identical {b.evidence.equals(n.evidence)}   "
      f"factors identical {b.factors.equals(n.factors)}")
print("  Both must be True: only the risk column may move.")
ch = int((b.risk_level.str.lower() != n.risk_level.str.lower()).sum())
print(f"risk_level changed on {ch} of {len(n)} posts ({100*ch/len(n):.1f}%)")
print("  A change touching fewer than ~40 of 378 posts is below what a test")
print("  set this size can resolve; the justification is then mechanism only.")
print(f"\nold {dict(Counter(b.risk_level.str.lower()))}")
print(f"new {dict(Counter(n.risk_level.str.lower()))}")
print("\ngold train prevalence: indicator 37.4  ideation 24.5  behavior 23.9  "
      "attempt 7.0")
print("Compare the new distribution against that. Test prevalence need not")
print("match train, so this is a direction check, NOT a pass/fail gate -- but a")
print("sharp collapse toward one class means the member is mis-calibrated.")
PY

echo ""
echo "DONE -> submissions/NIDHI_risk_${TAG}.csv"
