#!/bin/bash
#SBATCH --job-name=evcanon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Re-run the evidence taggers on CANONICALIZED test text.
#
# THE DEFECT. src/evidence_dump_all.py read data/test.csv with a plain
# pd.read_csv, while every split the taggers trained and validated on came
# through data_audit.load(), which applies clean_text. Measured:
#
#     artifacts/full_train.csv   0.1% of posts change under clean_text
#     artifacts/val_split.csv    0.3%
#     data/test.csv             29.9%   -- 113 of 378
#
# So the taggers were trained on canonical text and deployed on raw text for
# nearly a third of the test set, where a curly apostrophe tokenizes
# differently from a straight one. No offline instrument can see this because
# the folds are already canonical.
#
# The other two components already route test text the same way they route
# training text -- predict_factors_nli.py and blend_risk_saved.py both import
# data_audit.load. evidence_dump_all.py now does the same, so all three read
# canonical text everywhere.
#
# The correction is verifiable by reading the code rather than by a score, and
# it touches 113 posts -- above the ~40-post floor at which a 378-post test set
# can resolve a difference at all.
#
# NON-DESTRUCTIVE. Writes to artifacts/evidence_runs_canon so the existing
# candidate dumps are untouched and the two can be diffed before shipping.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

OUT="${OUT:-artifacts/evidence_runs_canon}"
VOTEK="${VOTEK:-2}"
RISK="${RISK:-artifacts/risk_blend_with_rules.csv}"
BASE="${BASE:-submissions/NIDHI_evidence_vote8.csv}"

if [ ! -f "${RISK}" ]; then
    echo "ERROR: ${RISK} not found. Available risk artifacts:"
    ls -1 artifacts/risk_blend*.csv 2>/dev/null || echo "  (none)"
    echo "Pass the champion's rules-applied risk file as RISK=..."
    exit 1
fi
[ -f "${BASE}" ] || { echo "ERROR: ${BASE} missing"; exit 1; }

# same discovery logic as scripts/evidence_dump.sh -- evidence_nli is a span
# SCORER, not a TaggerModel, so collect_candidates cannot load it
RUNS=""
for d in runs/evidence_*; do
    n=$(basename "$d")
    [ "$n" = "evidence_nli" ] && continue
    if [ -f "$d/model_final.pt" ] || [ -f "$d/model.pt" ]; then
        RUNS="${RUNS}${RUNS:+ }${n}"
    fi
done
echo "taggers: ${RUNS}"
echo "count  : $(echo ${RUNS} | wc -w)   (the shipped ensemble is 8)"
[ -n "${RUNS}" ] || { echo "ERROR: no tagger checkpoints found"; exit 1; }

echo ""
echo "================================================================"
echo "STEP 1  re-dump candidates on canonicalized text (GPU)"
echo "================================================================"
mkdir -p "${OUT}"
python src/evidence_dump_all.py \
    --runs "$(echo ${RUNS} | tr ' ' ',')" \
    --test data/test.csv \
    --val artifacts/val_split.csv \
    --outdir "${OUT}" \
    || { echo "FAILED step 1"; exit 1; }

echo ""
echo "================================================================"
echo "STEP 2  vote, k=${VOTEK}, same rule as the shipped ensemble (CPU)"
echo "================================================================"
python src/evidence_vote_test.py \
    --dir "${OUT}" \
    --test data/test.csv \
    --k "${VOTEK}" \
    --allow-missing \
    --out artifacts/evidence_vote_canon_test.csv \
    || { echo "FAILED step 2"; exit 1; }

echo ""
echo "================================================================"
echo "STEP 3  the same risk-consistency rule (CPU)"
echo "================================================================"
python src/evidence_risk_consistency.py --apply \
    --evidence artifacts/evidence_vote_canon_test.csv \
    --risk "${RISK}" \
    --fallback "${OUT}" \
    --tag test \
    --out artifacts/evidence_consistent_canon_test.csv \
    || { echo "FAILED step 3"; exit 1; }

echo ""
echo "================================================================"
echo "STEP 4  diff against the base submission's evidence column"
echo "================================================================"
BASE="${BASE}" python - <<'PY'
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from text_norm import clean_text

base = pd.read_csv(os.environ["BASE"], keep_default_na=False).set_index("row_id")
new = pd.read_csv("artifacts/evidence_consistent_canon_test.csv",
                  keep_default_na=False).set_index("row_id")
if set(base.index) != set(new.index):
    raise SystemExit("row_id sets differ -- refusing to compare")
new = new.loc[base.index]

b = base["evidence"].astype(str)
n = new["evidence"].astype(str)
ch = int((b != n).sum())
print(f"evidence changed on {ch} of {len(b)} posts ({100.0*ch/len(b):.1f}%)")
print("  Under ~40 changed posts a test set this size cannot resolve the")
print("  difference; the justification is then mechanism only, which still")
print("  holds for the held-out final evaluation.")

def nspans(s):
    return 0 if not s.strip() else len([x for x in s.split(";") if x.strip()])
bs, ns = [nspans(x) for x in b], [nspans(x) for x in n]
print(f"\nspans/post   champion {np.mean(bs):.2f}   new {np.mean(ns):.2f}")
print(f"empty posts  champion {sum(1 for x in bs if x==0)}   "
      f"new {sum(1 for x in ns if x==0)}")

TYPO = "‘’“”–—…"
print(f"\ncells with typographic chars   champion "
      f"{sum(1 for x in b if any(c in TYPO for c in x))}   "
      f"new {sum(1 for x in n if any(c in TYPO for c in x))}")
print("  new should be 0 -- spans now come from canonical text, and gold uses")
print("  straight characters 794 of 804 times on posts that contain curly ones.")

print("\nfirst 10 changed posts:")
k = 0
for rid in base.index:
    if b[rid] != n[rid]:
        print(f"  {rid}")
        print(f"    - {b[rid][:150]!r}")
        print(f"    + {n[rid][:150]!r}")
        k += 1
        if k >= 10:
            break
PY

echo ""
echo "DONE -> artifacts/evidence_consistent_canon_test.csv"
echo ""
echo "If the diff looks sane, patch it on. The evidence column ONLY is"
echo "replaced, so risk and factors stay byte-identical and the evidence"
echo "sub-score is the only one that may move:"
echo ""
echo "  python src/patch_submission_risk.py \\"
echo "      --base ${BASE} \\"
echo "      --evidence artifacts/evidence_consistent_canon_test.csv \\"
echo "      --out submissions/NIDHI_evidence_canonfull.csv"
