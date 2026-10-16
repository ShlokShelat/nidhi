#!/bin/bash
#SBATCH --job-name=cwfix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# MAKE THE CLASS-WEIGHTED RISK COLUMN COHERENT SO IT CAN BE TESTED.
#
# artifacts/risk_blend_cw_rules.csv contains 14 risk labels that differ from
# the champion. It is NOT shippable as-is because THREE of them cross the
# indicator boundary --
#     P01780  indicator -> ideation
#     P01991  indicator -> ideation
#     P02181  indicator -> ideation
# -- and evidence was carried over unchanged, so those posts would ship as
# `ideation` with EMPTY evidence cells. patch_submission_risk permits it (only
# one COLUMN moved) but it is semantically broken.
#
# This rebuilds evidence conditioned on the class-weighted risk column, exactly
# as `recond` did, then patches BOTH columns.
#
# WHY TEST IT AT ALL, given the mechanism looks refuted. Class weighting was
# supposed to raise `attempt` toward its 7.0% gold share; it went 22 -> 20,
# the opposite direction, so the stated mechanism did not fire. But 14 labels
# is a large change and the sign of its effect is not established from that
# alone. Since only the best-scoring submission is kept, testing a clean
# candidate costs nothing beyond a submission slot.
#
# NOTHING IS SUBMITTED HERE.
# ===========================================================================

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

CHAMP="${CHAMP:-submissions/NIDHI_catonly2.csv}"
RISK="${RISK:-artifacts/risk_blend_cw_rules.csv}"
NEWDIR="${NEWDIR:-artifacts/evidence_runs_cw}"
RC7="evidence_ep9_rc,evidence_lr2_rc,evidence_s14_rc,evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,evidence_v1_rc"
mkdir -p "${NEWDIR}" logs

for f in "${CHAMP}" "${RISK}" data/test.csv; do
    [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done
echo "node $(hostname)  $(date)"
echo "champion ${CHAMP}   cw risk ${RISK}"

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "STEP 1: redump the 7 taggers conditioned on the CW risk column"
echo "################################################################"
python src/evidence_dump_all.py \
    --runs "${RC7}" --test data/test.csv --val artifacts/val_split.csv \
    --outdir "${NEWDIR}" --risk-cond \
    --test-risk "${RISK}" \
    --val-risk artifacts/nligaur/oof_base.csv \
    || { echo "REDUMP FAILED"; exit 1; }

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "STEP 2: vote with the SHIPPED decode (k2 cluster modal cap3 gate .65)"
echo "################################################################"
python src/evidence_vote_test.py \
    --runs "${RC7}" --dir "${NEWDIR}" --split test --test data/test.csv \
    --risk-from "${RISK}" \
    --k 2 --cluster --rep modal --cap 3 --min-conf-override 0.65 \
    --out artifacts/evidence_cw_test.csv \
    || { echo "VOTE FAILED"; exit 1; }

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "STEP 3: patch BOTH columns (risk first, then evidence onto that)"
echo "################################################################"
python src/patch_submission_risk.py \
    --base "${CHAMP}" --risk "${RISK}" --max-changes 60 \
    --out artifacts/_cw_stage1.csv || { echo "risk patch failed"; exit 1; }
# --max-changes defaults to 25; a reconditioned evidence column moves far more
# than that (recond alone moved 4, but this conditions on 14 changed labels and
# re-runs the whole vote), so the default would refuse.
python src/patch_submission_risk.py \
    --base artifacts/_cw_stage1.csv --evidence artifacts/evidence_cw_test.csv \
    --max-changes 400 \
    --out submissions/NIDHI_cwfix.csv || { echo "evidence patch failed"; exit 1; }

CHAMP="${CHAMP}" python - <<'PY'
import os
import sys
from collections import Counter
import pandas as pd
sys.path.insert(0, "src")
from scoring import parse_evidence, normalize_label
a = pd.read_csv(os.environ["CHAMP"], keep_default_na=False)
b = pd.read_csv("submissions/NIDHI_cwfix.csv", keep_default_na=False)
assert list(a.row_id.astype(str)) == list(b.row_id.astype(str)), "ROW ORDER"
n = int((a.factors.astype(str) != b.factors.astype(str)).sum())
print(f"  factors     differs {n}   {'OK' if n == 0 else 'FAIL - S2 must stay 0.6908'}")
la = [normalize_label(x) for x in a.risk_level]
lb = [normalize_label(x) for x in b.risk_level]
print(f"  risk_level  differs {sum(x != y for x, y in zip(la, lb))} of {len(a)}")
print(f"  evidence    differs {int((a.evidence.astype(str) != b.evidence.astype(str)).sum())}")
ea = sum(1 for x in a.evidence if not parse_evidence(x))
eb = sum(1 for x in b.evidence if not parse_evidence(x))
print(f"  empty evidence {ea} -> {eb}   (expect {eb} == indicator count {Counter(lb)['indicator']})")
ok = Counter(lb)['indicator'] == eb
print(f"  CONSISTENT: {'YES' if ok else 'NO -- suppression set does not match the risk column'}")
print(f"  old {dict(Counter(la))}")
print(f"  new {dict(Counter(lb))}")
PY

cat <<'NEXT'

################################################################
DONE. Nothing submitted.
################################################################
Ship submissions/NIDHI_cwfix.csv only if `factors` differs on 0 rows AND the
empty-evidence count EQUALS the indicator count -- that equality is the whole
point of this job. The evidence sub-score (S2) must remain exactly 0.6908.

This is a TWO-column change (risk + evidence), so it is not single-variable.
That is forced by the pipeline, not a second knob: evidence is suppressed on
indicator, and three posts leave indicator. If it wins, the two contributions
cannot be separated -- acceptable here only because only the best-scoring
submission counts, and the alternative is not testing a 14-label change at
all.
NEXT
