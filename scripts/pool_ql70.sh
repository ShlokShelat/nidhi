#!/bin/bash
# LOGIN-NODE script (CPU only, seconds):   bash scripts/pool_ql70.sh
#
# Pools the five 70B QLoRA folds into one honest OOF file and judges it paired
# against the SAME folds' encoder OOF.
#
# JUDGE AGAINST oof_base, NEVER champion_oof. champion_oof is a weaker column on
# these posts than a single fresh risk-NLI model, so a delta measured against it
# can be an artefact of the baseline rather than a property of the challenger.
# oof_base is built from these same artifacts/folds/, which makes the pairing
# structural rather than assumed.

set -u
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export PYTHONNOUSERSITE=1

TAG="${TAG:-ql70}"
BASE="${BASE:-artifacts/nligaur/oof_base.csv}"

MISSING=""
for f in 0 1 2 3 4; do
    [ -f "runs/risk_${TAG}_f${f}/val_predictions.csv" ] || MISSING="${MISSING} ${f}"
done
if [ -n "$MISSING" ]; then
    echo "not ready -- missing folds:${MISSING}"
    echo "squeue -u \$USER"
    exit 1
fi
[ -f "$BASE" ] || { echo "ERROR: $BASE missing"; exit 1; }

TAG="${TAG}" python - <<'PY'
import os
import pandas as pd
tag = os.environ["TAG"]
fr = []
for f in range(5):
    d = pd.read_csv(f"runs/risk_{tag}_f{f}/val_predictions.csv")
    d["fold"] = f
    print(f"  fold {f}: {len(d)} posts")
    fr.append(d)
d = pd.concat(fr, ignore_index=True)
d["row_id"] = d["row_id"].astype(str)
n0 = len(d)
d = d.drop_duplicates("row_id")
out = f"artifacts/lora_{tag}_oof.csv"
d.to_csv(out, index=False)
print(f"{n0} rows -> {len(d)} unique posts   wrote {out}")
if len(d) != 1635:
    print(f"  WARNING: expected 1635, got {len(d)} -- the folds do not tile "
          f"full_train.csv and the pairing is broken")
PY

echo ""
echo "================================================================"
echo "JUDGE -- paired against the same folds' encoder OOF"
echo "================================================================"
python src/risk_lora_member.py \
    --lora-csv "artifacts/lora_${TAG}_oof.csv" \
    --enc "$BASE" --val artifacts/full_train.csv \
    || { echo "eval FAILED"; exit 1; }

cat <<'EOF'

================================================================
THE BAR -- fixed before the run
================================================================
  pooled OOF wF1 >= 0.8150
  must beat oof_base with a 95% paired-AUTHOR CI excluding zero

Single-fold numbers are not admissible here: folds differ in
difficulty, so only the pooled OOF over all five is comparable to
the encoder baseline.

IF IT CLEARS -- REPLACE the champion's risk column with the FULL-DATA
model from job ql70full (artifacts/lora_ql70full_test.csv). A single
model, not a vote, a routing rule, or a blend: a replacement avoids
the OOF-versus-deployment mismatch that a combination rule inherits.

  Two things a risk REPLACEMENT drags with it:

  1. EVIDENCE MUST BE REBUILT, WITH THE _rc RUNS. Gold evidence is
     empty exactly on indicator posts and the champion suppresses
     spans there, so every post whose risk changes also changes its
     evidence score. Rebuilding with the ORIGINAL eight taggers moves
     119 of 378 posts at identical risk and silently reverts the
     risk-conditioned taggers -- rebuild with the _rc runs below so
     that cannot happen.

       RC8=evidence_v2_rc,evidence_v1_rc,evidence_s14_rc,evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,evidence_lr2_rc,evidence_ep9_rc
       python src/evidence_vote_test.py --runs "$RC8" \
           --dir artifacts/evidence_runs --k 2 --cluster --rep modal \
           --test data/test.csv --risk-from <new risk csv> \
           --out artifacts/evvote_test_ql70.csv

  2. CHECK THE MARGINALS against the probe 90% intervals before
     uploading: indicator 112-170, ideation 99-142, behavior 69-113,
     attempt 13-41. Also check whether it still under-calls attempt:
     route_attempt exists to correct that, and a replacement may or
     may not still need it.

IF IT MISSES: scale is not the lever for risk, and the remaining
headroom is in the factor column.
EOF
