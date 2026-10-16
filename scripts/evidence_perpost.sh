#!/bin/bash
#SBATCH --job-name=evpp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# ONE FLAG AGAINST ITS OWN TWIN: does per-post loss normalisation help?
#
# THE MISMATCH. The scorer MACRO-averages phrase-F1 over POSTS -- a 40-token
# post and a 400-token post each count once. `train_evidence.py` used
# CrossEntropyLoss with the default token-mean reduction, so the 400-token post
# contributes ~10x the gradient. The taggers are trained on a length-weighted
# mixture and scored on an unweighted one.
#
# This is the same defect class already addressed once in the factors
# component (a metric weighting all 24 labels equally against a loss
# weighting them by frequency), in the component examined least so far.
# Evidence is 0.3 of the composite and sits near 0.784.
#
# WHY A STANDALONE SCRIPT. evidence_5fold.sh works and reproduces the five
# shipped OOF taggers exactly; it has no EXTRA passthrough, and modifying a
# file whose output five other measurements depend on is avoided here by using
# a standalone script instead. This trains ONLY the `v2` config with
# `--per-post` added, into tag `v2pp`, and dumps candidates in the identical
# schema so it drops straight into the existing vote machinery beside its
# twin.
#
# EVERYTHING ELSE IS HELD FIXED: same folds, same seed 13, same max-len 512,
# same 6 epochs, same lr 1e-5, same batch 8 / accum 2, same base model. One
# variable.
#
# THE VERDICT, and it is a SOLO comparison not a bag one. Replacing 1 of 5
# taggers dilutes any gain fivefold, and the absorption law has hidden six real
# member gains that way. So compare v2pp ALONE against v2 ALONE:
#
#     python src/evidence_vote_oof.py --average macro
#         -> `v2 alone` = 0.7540   (the incumbent, already measured)
#     python src/evidence_vote_oof.py --average macro --tags v2pp,ep12,lr3,bs16,ml256
#         -> `v2pp alone` = ?
#
#   --average macro is MANDATORY. The script defaults to micro, and the scorer
#   is macro -- tuning the vote on micro produces a metric mismatch that has
#   previously caused a shipped submission to underperform its expected gain.
#
# IF v2pp BEATS v2 SOLO: retrain the other four the same way (~16 GPU-hours) and
# the whole evidence bag lifts together. IF NOT: the length weighting was not
# binding and the evidence side is finished.
#
# Usage:
#   sbatch --cpus-per-task=4 scripts/evidence_perpost.sh

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TAG="${TAG:-v2pp}"
BASE="${BASE:-microsoft/deberta-v3-large}"
OUT=artifacts/evidence_oof
mkdir -p "${OUT}" logs

[ -f artifacts/folds/train_f0.csv ] || {
    echo "ERROR: artifacts/folds/ missing"; exit 1; }
grep -q "per-post" src/train_evidence.py || {
    echo "ERROR: src/train_evidence.py has no --per-post flag -- the patched"
    echo "       copy did not reach the cluster. Transfer it first."; exit 1; }

echo "tag ${TAG}   base ${BASE}   v2 config + --per-post"
echo "twin to beat: v2 alone = 0.7540 macro on 1102 scorable OOF posts"
echo ""

for f in 0 1 2 3 4; do
    d="runs/ev5f_${TAG}_f${f}"
    if [ -f "${d}/model_final.pt" ]; then
        echo "--- ${TAG} fold ${f}: exists ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "${TAG} fold ${f}"
    echo "================================================================"
    python src/train_evidence.py \
        --train "artifacts/folds/train_f${f}.csv" \
        --val   "artifacts/folds/val_f${f}.csv" \
        --model "${BASE}" \
        --outdir "${d}" --num-workers 0 \
        --seed 13 --max-len 512 --epochs 6 --lr 1e-5 --batch 8 --grad-accum 2 \
        --per-post \
        || { echo "FAILED ${TAG} fold ${f}"; exit 1; }
done

echo ""
echo "================================================================"
echo "DUMP OOF CANDIDATE SPANS  (each fold scored by its own model)"
echo "================================================================"
TAG="${TAG}" python - <<'PY'
import json
import os
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, "src")
from train_evidence import TaggerModel, collect_candidates

TAG = os.environ["TAG"]
OUT = "artifacts/evidence_oof"
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

dst = f"{OUT}/{TAG}_oof_candidates.csv"
if os.path.exists(dst):
    print(f"--- {dst} exists ---")
    raise SystemExit(0)

rows = []
for f in range(5):
    d = f"runs/ev5f_{TAG}_f{f}"
    cfg = json.load(open(f"{d}/run_config.json"))
    ml = int(cfg.get("max_len", 512))
    name = cfg.get("model", "microsoft/deberta-v3-large")
    va = pd.read_csv(f"artifacts/folds/val_f{f}.csv")
    tok = AutoTokenizer.from_pretrained(name)
    # per_post only affects the LOSS, so inference needs no special handling
    m = TaggerModel(name)
    m.load_state_dict(torch.load(f"{d}/model_final.pt", map_location="cpu"))
    m.to(dev).eval()
    cands = collect_candidates(m, tok, [str(x) for x in va["post"]], ml, dev)
    for i, rid in enumerate(va["row_id"]):
        for s, c in cands[i]:
            rows.append((rid, s, float(c), cfg.get("evidence_min_conf", 0.0)))
    del m
    torch.cuda.empty_cache()
    print(f"  {TAG} fold {f}: {len(va)} posts")

pd.DataFrame(rows, columns=["row_id", "span", "conf", "min_conf"]).to_csv(
    dst, index=False)
print(f"wrote {dst}  ({len(rows)} candidates)")
PY

cat <<'EOF'

================================================================
JUDGE IT SOLO, NOT IN THE BAG
================================================================
  python src/evidence_vote_oof.py --average macro --tags v2pp,ep12,lr3,bs16,ml256

Read the FIRST row, `v2pp alone`, against the incumbent `v2 alone` = 0.7540.
--average macro is mandatory: the script defaults to micro and the scorer is
macro. Tuning on micro produces a metric mismatch that has previously caused
a shipped submission to underperform its expected gain.

Beats 0.7540 clearly -> retrain ep12, lr3, bs16, ml256 with --per-post
                        (~16 GPU-hours) and the whole evidence bag lifts.
Does not              -> length weighting was not binding; evidence is done.

DO NOT judge by bag contribution first. Swapping 1 of 5 taggers dilutes the
effect fivefold, and diluting an effect fivefold routinely hides real
member-level gains in bag-level comparisons.
EOF
