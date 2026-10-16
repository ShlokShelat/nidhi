#!/bin/bash
#SBATCH --job-name=evrecond
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# RE-CONDITION THE EVIDENCE CANDIDATES ON THE RISK COLUMN THAT IS SUBMITTED.
#
# The seven RC7 taggers are RISK-CONDITIONED: the predicted level is part of
# their input, so a candidate set is only valid for the risk column it was
# generated against. When the submitted risk column moves after the candidates
# were dumped, some posts carry spans generated under a level that is not the
# one being submitted. Regenerating the candidates against the submitted
# column restores the invariant that the tagger input matches the output.
#
# This is a correctness property, not a tuned parameter: the conditioning input
# and the submitted label must be the same object. Wrong-level conditioning
# measurably degrades span quality on an affected post.
#
# Posts labelled `indicator` have their evidence suppressed, so a risk change
# that crosses the indicator boundary also moves the suppression set. When the
# indicator count is unchanged, the suppression set is unaffected and only the
# conditioning is stale.
#
# SAFETY: writes to a NEW outdir. The candidates in artifacts/evidence_runs are
# never touched, so a bad result costs nothing and the existing column stays
# rebuildable from the artefacts on disk.
#
# NOTHING IS SUBMITTED HERE. It ends with a built column and a diff.
# ===========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

set -u
CHAMP="${CHAMP:-submissions/NIDHI_ff11.csv}"
OLDRISK="${OLDRISK:-submissions/NIDHI_cluster.csv}"
OLDDIR="${OLDDIR:-artifacts/evidence_runs}"
NEWDIR="${NEWDIR:-artifacts/evidence_runs_ff11}"
RC7="evidence_ep9_rc,evidence_lr2_rc,evidence_s14_rc,evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,evidence_v1_rc"
mkdir -p "${NEWDIR}" logs

echo "node $(hostname)  $(date)"
echo "champion ${CHAMP}   old conditioning ${OLDRISK}"
df -h . | tail -1

for f in "${CHAMP}" "${OLDRISK}" data/test.csv; do
    [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done

# ---------------------------------------------------------------------------
# STEP 0 -- is there anything to fix? Do not spend a GPU hour on a no-op.
# ---------------------------------------------------------------------------
CHAMP="${CHAMP}" OLDRISK="${OLDRISK}" python - <<'PY'
import os, sys
import pandas as pd
a = pd.read_csv(os.environ["OLDRISK"], keep_default_na=False)
b = pd.read_csv(os.environ["CHAMP"], keep_default_na=False)
for d in (a, b):
    d["row_id"] = d["row_id"].astype(str)
m = a[["row_id", "risk_level"]].merge(
    b[["row_id", "risk_level"]], on="row_id", suffixes=("_old", "_new"))
diff = m[m.risk_level_old.str.lower() != m.risk_level_new.str.lower()]
print(f"{len(m)} shared rows, risk differs on {len(diff)}")
print("  old dist:", a.risk_level.str.lower().value_counts().to_dict())
print("  new dist:", b.risk_level.str.lower().value_counts().to_dict())
if len(diff):
    print("  transitions:",
          diff.groupby([diff.risk_level_old.str.lower(),
                        diff.risk_level_new.str.lower()]).size().to_dict())
    ind = diff[(diff.risk_level_old.str.lower() == "indicator") |
               (diff.risk_level_new.str.lower() == "indicator")]
    print(f"  of those, {len(ind)} cross the INDICATOR boundary "
          f"(these also change which posts are suppressed)")
sys.exit(0 if len(diff) else 3)
PY
rc=$?
if [ $rc -eq 3 ]; then
    echo ""
    echo "The two risk columns are IDENTICAL -- the candidates are already"
    echo "conditioned correctly and there is nothing to fix. Exiting."
    exit 0
elif [ $rc -ne 0 ]; then
    echo "risk comparison failed"; exit 1
fi

# ---------------------------------------------------------------------------
# STEP 1 -- redump the seven taggers, conditioned on the CHAMPION's risk
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "REDUMP: 7 RC7 taggers conditioned on ${CHAMP} -> ${NEWDIR}"
echo "################################################################"
python src/evidence_dump_all.py \
    --runs "${RC7}" \
    --test data/test.csv \
    --val artifacts/val_split.csv \
    --outdir "${NEWDIR}" \
    --risk-cond \
    --test-risk "${CHAMP}" \
    --val-risk artifacts/nligaur/oof_base.csv \
    || { echo "REDUMP FAILED"; exit 1; }
df -h . | tail -1

# ---------------------------------------------------------------------------
# STEP 2 -- how much actually moved?
# ---------------------------------------------------------------------------
echo ""
NEWDIR="${NEWDIR}" OLDDIR="${OLDDIR}" python - <<'PY'
import os
import pandas as pd
new, old = os.environ["NEWDIR"], os.environ["OLDDIR"]
runs = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
        "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
        "evidence_v1_rc"]
tot_posts = set()
for r in runs:
    po = os.path.join(old, f"{r}_test_candidates.csv")
    pn = os.path.join(new, f"{r}_test_candidates.csv")
    if not (os.path.exists(po) and os.path.exists(pn)):
        print(f"  {r}: missing a dump, skipping compare")
        continue
    a = pd.read_csv(po, keep_default_na=False)
    b = pd.read_csv(pn, keep_default_na=False)
    sa = a.groupby(a.row_id.astype(str)).span.apply(lambda s: tuple(sorted(s)))
    sb = b.groupby(b.row_id.astype(str)).span.apply(lambda s: tuple(sorted(s)))
    ix = sorted(set(sa.index) | set(sb.index))
    ch = [r_ for r_ in ix if sa.get(r_, ()) != sb.get(r_, ())]
    tot_posts |= set(ch)
    print(f"  {r:<20} {len(a):>5} -> {len(b):>5} spans   "
          f"candidates changed on {len(ch):>3} posts")
print(f"\nposts whose candidate set moved on ANY tagger: {len(tot_posts)}")
print("Near 0 means the taggers are insensitive to the changed levels and the")
print("redump is a no-op; tens of posts means the conditioning was material.")
PY

# ---------------------------------------------------------------------------
# STEP 3 -- vote with the shipped decode, on the new candidates
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "VOTE: shipped decode (k=2, cluster, modal) on the new candidates"
echo "################################################################"
python src/evidence_vote_test.py \
    --runs "${RC7}" \
    --dir "${NEWDIR}" \
    --split test \
    --test data/test.csv \
    --risk-from "${CHAMP}" \
    --k 2 --cluster --rep modal \
    --out artifacts/evidence_recond_test.csv \
    || { echo "VOTE FAILED"; exit 1; }

python src/patch_submission_risk.py \
    --base "${CHAMP}" \
    --evidence artifacts/evidence_recond_test.csv \
    --out submissions/NIDHI_recond.csv \
    || { echo "PATCH FAILED"; exit 1; }

CHAMP="${CHAMP}" python - <<'PY'
import os
import sys
import pandas as pd
sys.path.insert(0, "src")
from scoring import parse_evidence
a = pd.read_csv(os.environ["CHAMP"], keep_default_na=False)
b = pd.read_csv("submissions/NIDHI_recond.csv", keep_default_na=False)
assert list(a.row_id.astype(str)) == list(b.row_id.astype(str)), "ROW ORDER"
for c in ("risk_level", "factors"):
    n = int((a[c].astype(str) != b[c].astype(str)).sum())
    print(f"{c:<11} differs {n}   {'OK' if n == 0 else 'FAIL - must be 0'}")
n = int((a.evidence.astype(str) != b.evidence.astype(str)).sum())
print(f"evidence    differs {n} of {len(a)} rows")
print(f"spans {sum(len(parse_evidence(x)) for x in a.evidence)} -> "
      f"{sum(len(parse_evidence(x)) for x in b.evidence)}")
print(f"empty {sum(1 for x in a.evidence if not parse_evidence(x))} -> "
      f"{sum(1 for x in b.evidence if not parse_evidence(x))}")
PY

cat <<'NEXT'

################################################################
DONE. Nothing submitted.
################################################################
`risk_level` and `factors` MUST both differ on 0 rows -- only the evidence
column may move, so the factor sub-score has to come back unchanged.

The candidates are now conditioned on the risk column that is submitted.
Note there is no offline instrument for this change: the validation-split
candidates were always conditioned on the validation risk column, so the
validation score cannot see the difference at all.
NEXT
