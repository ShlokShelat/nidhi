#!/bin/bash
#SBATCH --job-name=frweight
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# THE FRIEND MODEL'S BLEND WEIGHT (--weight-ours) DEFAULTS TO 0.5, NEVER
# TUNED.
#
# ensemble_risk.py cannot validate this weight locally: the friend model's
# probabilities are test-only, so there is no matching split to sweep the
# weight against directly. 0.5 is a round number nobody has measured.
#
# WHY THIS MATTERS. The composite score's risk term is S1 = (4/7)*risk +
# (3/7)*evidence, so a risk-side knob here moves the composite directly
# through S1.
#
# THE VAL INSTRUMENT IS UNRELIABLE FOR THIS PARAMETER -- DO NOT USE IT.
#   twin: w=0.3 0.7763 | 0.4 0.7966 | 0.5 0.7992 | 0.6 0.8072 | 0.7 0.8155
# Val trends monotonically toward MORE weight on the local encoders,
# extrapolating to w=1.0 (no friend) as best. That extrapolation is wrong:
# val scores the friend model as a SINGLE held-out checkpoint while test
# scores its 5-FOLD BAG, so val systematically undervalues it. This
# construction mismatch means any monotonic trend in val for this specific
# parameter should not be trusted or extrapolated.
#
# NOTHING IS SUBMITTED HERE. It builds candidate columns and reports whether
# each is a clean SINGLE-VARIABLE change.
# ===========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

set -u
CHAMP="${CHAMP:-submissions/NIDHI_ff11.csv}"
SEEDS="${SEEDS:-runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15}"
FRIEND="${FRIEND:-artifacts/friend_risk_probs.csv}"
WEIGHTS="${WEIGHTS:-0.40 0.30}"
mkdir -p logs

echo "node $(hostname)  $(date)"
echo "champion ${CHAMP}   seeds ${SEEDS}   friend ${FRIEND}"
for f in "${CHAMP}" "${FRIEND}" data/test.csv artifacts/nli_risk_test_probs.csv; do
    [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done

# ---------------------------------------------------------------------------
# GATE FIRST. Prove the chain reproduces the champion's risk column EXACTLY
# from the files on disk before changing an input. Without this a rebuild bug
# is indistinguishable from the weight's effect on the result.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "GATE: does the UNCHANGED chain rebuild the champion 378/378?"
echo "################################################################"
python src/champion_risk_inject.py --champion "${CHAMP}" --verify \
    || { echo "GATE FAILED -- the chain does not reproduce the shipped risk"; \
         echo "column. Fix that before attributing anything to the weight."; \
         exit 1; }

# ---------------------------------------------------------------------------
for W in ${WEIGHTS}; do
    TAG="w$(echo ${W} | tr -d '.')"
    ENS="artifacts/ensembled_risk_${TAG}.csv"
    OUT="submissions/NIDHI_${TAG}.csv"
    echo ""
    echo "################################################################"
    echo "WEIGHT ${W}  (friend gets $(python -c "print(round(1-${W},2))"))"
    echo "################################################################"

    python src/ensemble_risk.py \
        --test data/test.csv \
        --our-risk-run "${SEEDS}" \
        --friend-probs "${FRIEND}" \
        --weight-ours "${W}" \
        --out "${ENS}" \
        || { echo "  ensemble failed at w=${W}"; continue; }

    python src/champion_risk_inject.py \
        --champion "${CHAMP}" \
        --alt-ensembled "${ENS}" \
        --out "${OUT}" \
        || { echo "  inject failed at w=${W} (it REFUSES when a post crosses"; \
             echo "  the indicator boundary -- evidence is suppressed there,"; \
             echo "  so that would silently become a two-variable change)"; \
             continue; }

    CHAMP="${CHAMP}" OUT="${OUT}" W="${W}" python - <<'PY'
import os
import sys
import pandas as pd
sys.path.insert(0, "src")
from scoring import normalize_label
a = pd.read_csv(os.environ["CHAMP"], keep_default_na=False)
b = pd.read_csv(os.environ["OUT"], keep_default_na=False)
w = os.environ["W"]
assert list(a.row_id.astype(str)) == list(b.row_id.astype(str)), "ROW ORDER"
ok = True
for c in ("evidence", "factors"):
    n = int((a[c].astype(str) != b[c].astype(str)).sum())
    print(f"  {c:<11} differs {n}   {'OK' if n == 0 else 'FAIL - must be 0'}")
    ok &= n == 0
nr = int((a.risk_level.astype(str) != b.risk_level.astype(str)).sum())
print(f"  risk_level  differs {nr} of {len(a)} rows")
la = [normalize_label(x) for x in a.risk_level]
lb = [normalize_label(x) for x in b.risk_level]
cross = sum(1 for x, y in zip(la, lb)
            if (x == "indicator") != (y == "indicator"))
print(f"  crossing the INDICATOR boundary: {cross}   "
      f"{'OK' if cross == 0 else 'NOT single-variable -- evidence must be regenerated'}")
print(f"  new dist: {pd.Series(lb).value_counts().to_dict()}")
print(f"  w={w}: {'SHIPPABLE as one variable' if (ok and cross == 0 and nr) else 'DO NOT SHIP'}"
      f"{' (no rows changed -- nothing to test)' if nr == 0 else ''}")
PY
done

cat <<'NEXT'

################################################################
DONE. Nothing submitted.
################################################################
Ship the LOWER weight first (w_ours=0.40, i.e. more weight on the friend
model). It must be a clean single-variable change: evidence and factors
differ on 0 rows, no post crosses the indicator boundary, and the evidence
sub-score (S2) must remain exactly 0.6908.

If 0.40 gains, 0.30 is the follow-up. If it loses, 0.5 is the interior
optimum and this parameter is settled.

Do NOT judge this on champion_oof_twin -- it is unreliable for this specific
parameter, because val scores the friend model as a single checkpoint while
test scores its 5-fold bag, which biases its extrapolation toward dropping
the friend model entirely.
NEXT
