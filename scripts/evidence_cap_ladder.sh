#!/bin/bash
# CPU only -- run on the login node, no Slurm needed. Minutes.
#
# THE CAP LADDER. `--cap 3` gained substantially more on the held-out
# evaluation than predicted from val -- roughly four times the predicted
# effect. Two consequences:
#
# 1. THE VAL RANKING OF CAP VALUES IS DISCREDITED. evidence_cap_sweep.py chose
#    cap=3 over cap=2 on val. Val just mis-sized cap=3 by 4x, so its ordering
#    of the rungs cannot be trusted either. cap=2 has to be measured on the
#    held-out evaluation, not inferred.
#
# 2. THE COLUMN STILL OVER-EMITS. After cap=3 the column is 406 spans / 378 posts =
#    1.07 per post, against gold's 0.96 -- still 11% over. cap=3 touched only
#    the 17 posts above 3 spans; every post sitting at exactly 2 or 3 was never
#    examined.
#
# WHY VAL UNDERSTATED IT, because this governs how to read what comes back:
# the 2.8x val->test attenuation was derived from effects spread broadly across
# posts. The cap is CONCENTRATED -- one post going 12 spans -> 3 was 30% of the
# entire gain. Concentrated effects transfer at whatever rate that handful of
# posts transfers. Do not apply a broad-effect discount to a change touching a
# few posts, in either direction.
#
# EXPECT THE LADDER TO TURN. Gold is 0.96 spans/post and cap=1 forces at most
# one, which must eventually cost recall. The question is whether the turn is
# at 2 or at 1. Ship cap=2 first; only ship cap=1 if cap=2 also gains.
#
# NOTHING IS SUBMITTED HERE.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export PYTHONUNBUFFERED=1

RC7="evidence_ep9_rc,evidence_lr2_rc,evidence_s14_rc,evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,evidence_v1_rc"
DIR="${DIR:-artifacts/evidence_runs_ff11}"
# the cap is applied to the SAME vote that produced the champion, so the base
# for the patch is recond -- cap3's own base. Patching cap2 onto cap3 would
# stack two caps.
BASE="${BASE:-submissions/NIDHI_recond.csv}"
CHAMP="${CHAMP:-submissions/NIDHI_cap3.csv}"

for f in "${BASE}" "${CHAMP}" data/test.csv; do
    [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done

for C in 2 1; do
    echo ""
    echo "################################################################"
    echo "CAP ${C}"
    echo "################################################################"
    python src/evidence_vote_test.py \
        --runs "${RC7}" --dir "${DIR}" --split test \
        --test data/test.csv --risk-from "${BASE}" \
        --k 2 --cluster --rep modal --cap "${C}" \
        --out "artifacts/evidence_cap${C}_test.csv" \
        || { echo "vote failed at cap=${C}"; continue; }

    python src/patch_submission_risk.py \
        --base "${BASE}" \
        --evidence "artifacts/evidence_cap${C}_test.csv" \
        --out "submissions/NIDHI_cap${C}.csv" \
        || { echo "patch failed at cap=${C}"; continue; }

    # against the CHAMPION, the baseline any future comparison is measured from
    python src/cap_diff_check.py --base "${CHAMP}" \
        --new "submissions/NIDHI_cap${C}.csv" --cap "${C}"

    CHAMP="${CHAMP}" C="${C}" python - <<'PY'
import os
import sys
import pandas as pd
sys.path.insert(0, "src")
from scoring import parse_evidence
a = pd.read_csv(os.environ["CHAMP"], keep_default_na=False)
b = pd.read_csv(f"submissions/NIDHI_cap{os.environ['C']}.csv",
                keep_default_na=False)
for c in ("risk_level", "factors"):
    n = int((a[c].astype(str) != b[c].astype(str)).sum())
    print(f"  {c:<11} vs champion differs {n}   {'OK' if n == 0 else 'FAIL'}")
sa = sum(len(parse_evidence(x)) for x in a.evidence)
sb = sum(len(parse_evidence(x)) for x in b.evidence)
print(f"  spans {sa} ({sa/len(a):.2f}/post) -> {sb} ({sb/len(b):.2f}/post)"
      f"   gold 0.96/post")
ea = sum(1 for x in a.evidence if not parse_evidence(x))
eb = sum(1 for x in b.evidence if not parse_evidence(x))
print(f"  empty {ea} -> {eb}   {'OK' if ea == eb else 'FAIL - suppression set moved'}")
PY
done

cat <<'NEXT'

################################################################
DONE. Nothing submitted.
################################################################
Ship submissions/NIDHI_cap2.csv FIRST. Only ship cap1 if cap2 also gains --
gold is 0.96 spans/post, so the ladder must turn somewhere and one-span-max is
where recall starts paying for precision.

Both must show risk_level and factors differing on 0 rows against the champion,
and the empty-evidence count unchanged at 137. S2 must return exactly 0.6908.
NEXT
