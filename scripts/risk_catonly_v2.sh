#!/bin/bash
#SBATCH --job-name=riskcat2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# DROP THE ORDINAL HEAD AT INFERENCE. No retraining -- the champion's own three
# checkpoints, only which logits they return.
#
# THE MEASUREMENT (src/risk_head_diagnosis.py, val split, same checkpoints):
#     categorical alone   0.7579
#     ordinal alone       0.7363
#     shipped 50/50 mix   0.7498
# The gap is `attempt`: cat F1 0.457 vs ordinal 0.200 -- the weakest class and a
# pure recall failure (0.591 recall, 84 predicted where 110 exist).
#
# AND THE MIXTURE WAS NEVER A DECISION. train_risk.py:188 hardcodes
# log(0.5*cat_p + 0.5*ord_p). `--alpha-ordinal` only ever weighted the LOSS. So
# 50/50 at inference is a default that was never revisited, the same class of
# oversight as the friend blend weight.
#
# ---------------------------------------------------------------------------
# WHY THIS REPLACES scripts/risk_catonly_submission.sh, WHICH IS BROKEN.
#
# That script's chain is ensemble -> blend_risk_saved -> apply_rules -> patch.
# It has NO route_attempt step, so that chain's output ships a strictly
# weaker risk column than one that includes routing. It also defaults BASE to
# an older, superseded submission file, predating the current champion.
# submissions/NIDHI_risk_catonly.csv on disk was built that way: it differs
# from the champion on 7 posts but carries a lower attempt count (19) than the
# champion's (22). DO NOT UPLOAD THAT FILE.
#
# This version routes the cat-only probabilities through
# champion_risk_inject.py --alt-ensembled, which is the SHIPPED chain:
#   0.70*ensembled + 0.30*nli -> argmax -> rules(urgency,completed_act)
#   -> route_attempt
# It gates on a 378/378 rebuild first and refuses to write if any post crosses
# the indicator boundary. Same harness that validates provenance before any
# risk-column change.
#
# RISK_CAT_ONLY is read inside DualHeadModel.forward, so it only affects
# ensemble_risk.py (which loads the three DeBERTa members). champion_risk_inject
# reads CSVs and loads no model, so the flag is irrelevant downstream.
#
# EXPECTATION: ~7 posts, behavior->attempt. A much larger number means
# RISK_CAT_ONLY reached something it should not have.
#
# NOTHING IS SUBMITTED HERE.
# ===========================================================================

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

TAG="${TAG:-catonly2}"
CHAMP="${CHAMP:-submissions/NIDHI_recond.csv}"
RUNS="${RUNS:-runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15}"
mkdir -p logs

echo "node $(hostname)  $(date)"
for d in $(echo "${RUNS}" | tr ',' ' '); do
    [ -f "${d}/model_final.pt" ] || { echo "ERROR: ${d}/model_final.pt missing"; exit 1; }
done
for p in "${CHAMP}" data/test.csv artifacts/friend_risk_probs.csv \
         artifacts/nli_risk_test_probs.csv; do
    [ -f "${p}" ] || { echo "ERROR: ${p} missing"; exit 1; }
done
echo "champion ${CHAMP}   members ${RUNS}"

# ---------------------------------------------------------------------------
# GATE. The unchanged chain must rebuild the champion's risk column 378/378
# before anything is attributed to the head change.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "GATE: unchanged chain reproduces the champion?"
echo "################################################################"
unset RISK_CAT_ONLY || true
python src/champion_risk_inject.py --champion "${CHAMP}" --verify \
    || { echo "GATE FAILED -- fix the chain before changing the head"; exit 1; }

# ---------------------------------------------------------------------------
# CAT-ONLY probabilities from the same three checkpoints
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "RISK_CAT_ONLY=1 -- categorical logits only, ordinal head ignored"
echo "################################################################"
export RISK_CAT_ONLY=1
python src/ensemble_risk.py \
    --test data/test.csv --our-risk-run "${RUNS}" \
    --friend-probs artifacts/friend_risk_probs.csv \
    --out "artifacts/ensembled_risk_${TAG}.csv" \
    || { echo "FAILED ensemble"; exit 1; }
unset RISK_CAT_ONLY

# sanity: cat-only must actually differ from the shipped ensemble
python - <<PY
import numpy as np, pandas as pd
C = ["p_Indicator","p_Ideation","p_Behavior","p_Attempt"]
a = pd.read_csv("artifacts/ensembled_risk.csv").set_index("row_id")
b = pd.read_csv("artifacts/ensembled_risk_${TAG}.csv").set_index("row_id")
i = [r for r in a.index if r in b.index]
d = np.abs(a.loc[i,C].to_numpy(float) - b.loc[i,C].to_numpy(float))
print(f"  max|dp| {d.max():.4f}   mean|dp| {d.mean():.4f}   rows {len(i)}")
print("  If max|dp| is ~0 the env var never reached the model -- STOP.")
PY

# ---------------------------------------------------------------------------
# through the SHIPPED chain: nli blend -> rules -> route_attempt
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "INJECT through the champion chain (incl. route_attempt)"
echo "################################################################"
python src/champion_risk_inject.py \
    --champion "${CHAMP}" \
    --alt-ensembled "artifacts/ensembled_risk_${TAG}.csv" \
    --out "submissions/NIDHI_${TAG}.csv" \
    || { echo ""; \
         echo "inject refused. If it is the indicator boundary, the change is"; \
         echo "NOT single-variable -- evidence is suppressed there. Re-run with"; \
         echo "--out-risk to dump labels and rebuild evidence, as wsplit did."; \
         exit 0; }

CHAMP="${CHAMP}" OUT="submissions/NIDHI_${TAG}.csv" python - <<'PY'
import os
import sys
from collections import Counter
import pandas as pd
sys.path.insert(0, "src")
from scoring import parse_evidence, normalize_label
a = pd.read_csv(os.environ["CHAMP"], keep_default_na=False)
b = pd.read_csv(os.environ["OUT"], keep_default_na=False)
assert list(a.row_id.astype(str)) == list(b.row_id.astype(str)), "ROW ORDER"
ok = True
for c in ("evidence", "factors"):
    n = int((a[c].astype(str) != b[c].astype(str)).sum())
    print(f"  {c:<11} differs {n}   {'OK' if n == 0 else 'FAIL'}")
    ok &= n == 0
la = [normalize_label(x) for x in a.risk_level]
lb = [normalize_label(x) for x in b.risk_level]
nr = sum(x != y for x, y in zip(la, lb))
cross = sum(1 for x, y in zip(la, lb) if (x == "indicator") != (y == "indicator"))
ea = sum(1 for x in a.evidence if not parse_evidence(x))
eb = sum(1 for x in b.evidence if not parse_evidence(x))
print(f"  risk_level  differs {nr} of {len(a)}   (diagnosis predicted ~7)")
print(f"  crossing indicator {cross}   {'OK' if cross == 0 else 'FAIL'}")
print(f"  empty evidence {ea} -> {eb}   {'OK' if ea == eb else 'FAIL'}")
print(f"  old {dict(Counter(la))}")
print(f"  new {dict(Counter(lb))}")
print(f"  attempt {Counter(la)['attempt']} -> {Counter(lb)['attempt']}  "
      f"(gold prevalence ~7.0% = ~26 of 378)")
print(f"\n  {'SHIPPABLE as one variable' if (ok and cross == 0 and nr) else 'DO NOT SHIP'}")
PY

cat <<'NEXT'

################################################################
DONE. Nothing submitted.
################################################################
Upload submissions/NIDHI_catonly2.csv only if evidence and factors differ on 0
rows, nothing crosses the indicator boundary, and the empty-evidence count is
unchanged. The evidence sub-score (S2) must remain exactly 0.6908.

This is not judged on an offline number alone -- cat-only beats the mixture on
the val split and on the head diagnosis, using the same checkpoints that ship,
and the 50/50 mixing weight was never a tuned choice.
NEXT
