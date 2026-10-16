#!/bin/bash
#SBATCH --job-name=recond2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# RE-CONDITION THE EVIDENCE CANDIDATES ON THE RISK COLUMN THAT IS SUBMITTED.
#
# The seven RC7 taggers are RISK-CONDITIONED -- the predicted level is part of
# their input, so a candidate set is only valid for the risk column it was
# generated against. Whenever the risk column moves after the candidates were
# dumped, the affected posts carry spans generated under a level that is not
# the one being submitted, and the taggers must be re-run against the current
# column to restore the match between their input and the output.
#
# STEP 0 enforces the single-variable property. Evidence is suppressed exactly
# on `indicator`, so a risk change that crosses the indicator boundary also
# changes which evidence cells are emptied -- two effects in one measurement.
# The script only proceeds when no changed label crosses that boundary, so
# risk and factors must come back byte-identical and only evidence may move.
#
# SAFETY: writes to a NEW outdir. The existing candidates are never touched.
#
# NOTHING IS SUBMITTED HERE.
# ===========================================================================

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

CHAMP="${CHAMP:-submissions/NIDHI_catonly2.csv}"
OLD="${OLD:-submissions/NIDHI_recond.csv}"
NEWDIR="${NEWDIR:-artifacts/evidence_runs_cat2}"
RC7="evidence_ep9_rc,evidence_lr2_rc,evidence_s14_rc,evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,evidence_v1_rc"
mkdir -p "${NEWDIR}" logs

for f in "${CHAMP}" "${OLD}" data/test.csv; do
    [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done
echo "node $(hostname)  $(date)"
echo "champion ${CHAMP}   old conditioning ${OLD}"

# ---------------------------------------------------------------------------
# STEP 0 -- is there anything to fix, and does it stay single-variable?
# ---------------------------------------------------------------------------
CHAMP="${CHAMP}" OLD="${OLD}" python - <<'PY'
import os
import sys
import pandas as pd
sys.path.insert(0, "src")
from scoring import normalize_label
a = pd.read_csv(os.environ["OLD"], keep_default_na=False)
b = pd.read_csv(os.environ["CHAMP"], keep_default_na=False)
for d in (a, b):
    d["row_id"] = d["row_id"].astype(str)
m = a[["row_id", "risk_level"]].merge(b[["row_id", "risk_level"]],
                                      on="row_id", suffixes=("_o", "_n"))
diff = m[[normalize_label(x) != normalize_label(y)
          for x, y in zip(m.risk_level_o, m.risk_level_n)]]
print(f"{len(m)} shared rows, risk differs on {len(diff)}")
for r in diff.itertuples():
    print(f"    {r.row_id}  {normalize_label(r.risk_level_o)} -> "
          f"{normalize_label(r.risk_level_n)}")
cross = sum(1 for r in diff.itertuples()
            if (normalize_label(r.risk_level_o) == "indicator")
            != (normalize_label(r.risk_level_n) == "indicator"))
print(f"  crossing the indicator boundary: {cross}  "
      f"{'OK -- stays single-variable' if cross == 0 else 'NOT single-variable'}")
sys.exit(0 if len(diff) and cross == 0 else 3)
PY
rc=$?
[ $rc -eq 3 ] && { echo "nothing to fix, or it would not be single-variable"; exit 0; }
[ $rc -ne 0 ] && { echo "comparison failed"; exit 1; }

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "REDUMP conditioned on ${CHAMP} -> ${NEWDIR}"
echo "################################################################"
python src/evidence_dump_all.py \
    --runs "${RC7}" --test data/test.csv --val artifacts/val_split.csv \
    --outdir "${NEWDIR}" --risk-cond \
    --test-risk "${CHAMP}" --val-risk artifacts/nligaur/oof_base.csv \
    || { echo "REDUMP FAILED"; exit 1; }

echo ""
echo "################################################################"
echo "VOTE with the SHIPPED decode (k2 cluster modal cap3 gate 0.65)"
echo "################################################################"
python src/evidence_vote_test.py \
    --runs "${RC7}" --dir "${NEWDIR}" --split test --test data/test.csv \
    --risk-from "${CHAMP}" \
    --k 2 --cluster --rep modal --cap 3 --min-conf-override 0.65 \
    --out artifacts/evidence_cat2_test.csv \
    || { echo "VOTE FAILED"; exit 1; }

python src/patch_submission_risk.py \
    --base "${CHAMP}" --evidence artifacts/evidence_cat2_test.csv \
    --max-changes 60 --out submissions/NIDHI_recond2.csv \
    || { echo "PATCH FAILED"; exit 1; }

CHAMP="${CHAMP}" python - <<'PY'
import os
import sys
import pandas as pd
sys.path.insert(0, "src")
from scoring import parse_evidence
a = pd.read_csv(os.environ["CHAMP"], keep_default_na=False)
b = pd.read_csv("submissions/NIDHI_recond2.csv", keep_default_na=False)
assert list(a.row_id.astype(str)) == list(b.row_id.astype(str)), "ROW ORDER"
ok = True
for c in ("risk_level", "factors"):
    n = int((a[c].astype(str) != b[c].astype(str)).sum())
    print(f"  {c:<11} differs {n}   {'OK' if n == 0 else 'FAIL'}")
    ok &= n == 0
print(f"  evidence    differs "
      f"{int((a.evidence.astype(str) != b.evidence.astype(str)).sum())}")
ea = sum(1 for x in a.evidence if not parse_evidence(x))
eb = sum(1 for x in b.evidence if not parse_evidence(x))
print(f"  empty       {ea} -> {eb}   {'OK' if ea == eb else 'FAIL'}")
print(f"  spans       {sum(len(parse_evidence(x)) for x in a.evidence)} -> "
      f"{sum(len(parse_evidence(x)) for x in b.evidence)}")
print(f"\n  {'SHIPPABLE' if ok and ea == eb else 'DO NOT SHIP'}")
PY

cat <<'NEXT'

################################################################
DONE. Nothing submitted.
################################################################
submissions/NIDHI_recond2.csv is shippable only if risk_level and factors
differ on 0 rows and the empty-evidence count is unchanged: an unchanged empty
count is what proves the suppression set did not move, which is what keeps
this a single-variable change with the factor sub-score fixed.
NEXT
