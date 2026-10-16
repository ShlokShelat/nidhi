#!/bin/bash
#SBATCH --job-name=evidence-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Swap evidence_v2 -> evidence_full inside the deployed vote>=2 ensemble.
#
# evidence_full is the same BIO tagger architecture trained on
# artifacts/full_train.csv (1635 posts) instead of artifacts/train_split.csv
# (1263) -- +29% data. It already exists, so this costs one inference pass.
#
# THIS CANNOT BE VALIDATED. full_train contains all 372 val posts, so any val
# score for evidence_full is memorisation, exactly like runs/risk_full_* which
# score 0.95-0.97 on val by recall. STEP 1 below scores it on val ONLY as a
# sanity check: a high number confirms the checkpoint trained on the expected
# data. A number near 0.69 would mean something is wrong with the run.
#
# Minimal change on purpose: the ensemble stays vote>=2 over three taggers,
# with one member upgraded. Adding evidence_full as a FOURTH member was
# rejected: going from 3 to 4 members measurably hurts vote quality
# independent of member quality (with s15 as the fourth member, val moved
# 0.7098 -> 0.7009), because it changes the vote arithmetic rather than the
# model quality.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

echo "================================================================"
echo "STEP 1: SANITY ONLY -- score evidence_full on val"
echo "  Expect a HIGH F1 (memorisation). That confirms the checkpoint"
echo "  really did train on full_train. It is NOT evidence of quality."
echo "================================================================"
python src/evidence_vote_ensemble.py \
    --test artifacts/val_split.csv --eval-gold \
    --runs runs/evidence_full --k 1 \
    --out artifacts/_evfull_val_sanity.csv || exit 1

echo ""
echo "================================================================"
echo "STEP 2: vote>=2 over evidence_full + v1 + s14 on TEST"
echo "  (v1 and s14 test predictions are already cached; only"
echo "   evidence_full needs scoring)"
echo "================================================================"
python src/evidence_vote_ensemble.py \
    --test data/test.csv \
    --runs runs/evidence_full,runs/evidence_v1,runs/evidence_s14 \
    --k 2 \
    --out artifacts/evidence_vote2_full_test.csv || exit 1

echo ""
echo "================================================================"
echo "STEP 3: how far does this move the submission?"
echo "================================================================"
python - <<'PY'
import sys
sys.path.insert(0, "src")
import pandas as pd
from scoring import parse_evidence, _span_pair_matches

cur = pd.read_csv("artifacts/evidence_vote2_test.csv", keep_default_na=False)
new = pd.read_csv("artifacts/evidence_vote2_full_test.csv", keep_default_na=False)
m = cur.merge(new, on="row_id", suffixes=("_cur", "_new"))

def equivalent(a, b):
    A, B = parse_evidence(a), parse_evidence(b)
    return len(A) == len(B) and all(_span_pair_matches(x, y) for x, y in zip(A, B))

eq = sum(equivalent(a, b) for a, b in zip(m.evidence_cur, m.evidence_new))
n_cur = sum(len(parse_evidence(x)) for x in cur.evidence)
n_new = sum(len(parse_evidence(x)) for x in new.evidence)
empty_cur = sum(1 for x in cur.evidence if not parse_evidence(x))
empty_new = sum(1 for x in new.evidence if not parse_evidence(x))
print(f"posts                 : {len(m)}")
print(f"metric-equivalent     : {eq}")
print(f"materially different  : {len(m) - eq}")
print(f"total spans   cur {n_cur} -> new {n_new}")
print(f"empty posts   cur {empty_cur} -> new {empty_new}")
PY

echo ""
echo "================================================================"
echo "STEP 4: evidence-only patch onto the scored 0.7340 submission"
echo "================================================================"
python src/patch_submission_risk.py \
    --base submissions/NIDHI_evidence_vote2.csv \
    --evidence artifacts/evidence_vote2_full_test.csv \
    --out submissions/NIDHI_evidence_full.csv

echo ""
echo "================================================================"
echo "DONE -> submissions/NIDHI_evidence_full.csv"
echo ""
echo "  Read STEP 3 before submitting. If 'materially different' is"
echo "  under ~20 posts this is a near-no-op and probably not worth a"
echo "  submission slot. If it is over ~150, evidence_full behaves very"
echo "  differently from v2 and the variance is larger than the +29%"
echo "  data argument justifies -- review manually before uploading."
echo "================================================================"
