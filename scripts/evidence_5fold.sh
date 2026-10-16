#!/bin/bash
#SBATCH --job-name=ev5f
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x_%j.out

# Validate the EVIDENCE VOTE on 1635 posts instead of 372.
#
# WHY THIS MATTERS. Voting across evidence taggers is the primary technique
# used for evidence performance. Its entire evidence base is the 372-post val
# split -- every one of the 17 taggers trains on train_split and scores on
# those same 372 posts.
#
# That ruler has proven unreliable more than once: offline gains measured on
# the 372-post val split have differed substantially in magnitude, and at
# times in sign, from the corresponding held-out result.
#
# The factor work fixed this by going five-fold and fitting/validating on 1635
# out-of-fold posts, which caught bad changes before they were shipped.
# Evidence has never had that treatment, so the one lever that is currently
# working is being steered with the same instrument that misread the others.
#
# WHAT THIS ANSWERS
#   - does vote>=k actually beat the single tagger on 1635 posts?
#   - is k=3 (chosen on 372 posts) the right count, or an artefact?
#   - how much of the val-measured vote gain survives 4.4x the data?
#
# FIVE CONFIGS, not seventeen. Seventeen taggers x 5 folds is 85 runs. The
# earlier sweep showed seed variants cluster tightly and add almost nothing to
# the vote, while regime changes carry it -- so this takes the shipped config
# plus the four most distinct regimes, by measured val F1 and by how different
# their training is:
#
#     v2     lr 1e-5  ep 6   len 512   the deployed tagger        0.6902
#     ep12   lr 1e-5  ep 12  len 512   best single tagger         0.7096
#     lr3    lr 3e-5  ep 6   len 512   highest learning rate      0.7078
#     bs16   lr 2e-5  ep 6   len 512   large batch                0.6979
#     ml256  lr 1e-5  ep 6   len 256   short context, a genuinely
#                                      different view of the post 0.6722
#
# 25 runs at ~35 min each, so roughly 15 hours. Long, but it is the difference
# between shipping the evidence vote on faith and shipping it on evidence.
#
# Reuses artifacts/folds/{train,val}_f*.csv -- the same author-disjoint split
# every other experiment uses.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

if [ ! -f artifacts/folds/train_f0.csv ]; then
    echo "ERROR: artifacts/folds/ missing (scripts/nli_risk_5fold.sh writes it)."
    exit 1
fi

OUT=artifacts/evidence_oof
mkdir -p "${OUT}"

run_cv () {
    local tag=$1; shift
    for f in 0 1 2 3 4; do
        local dir="runs/ev5f_${tag}_f${f}"
        if [ ! -f "${dir}/model_final.pt" ]; then
            echo ""
            echo "================================================================"
            echo "${tag} fold ${f}   $*"
            echo "================================================================"
            python src/train_evidence.py \
                --train "artifacts/folds/train_f${f}.csv" \
                --val   "artifacts/folds/val_f${f}.csv" \
                --model microsoft/deberta-v3-large \
                --outdir "${dir}" --num-workers 0 \
                "$@" || { echo "FAILED ${tag} fold ${f}"; return 1; }
        else
            echo "--- ${tag} fold ${f}: exists ---"
        fi
    done
}

run_cv v2    --seed 13 --max-len 512 --epochs 6  --lr 1e-5 --batch 8  --grad-accum 2
run_cv ep12  --seed 37 --max-len 512 --epochs 12 --lr 1e-5 --batch 8  --grad-accum 2
run_cv lr3   --seed 32 --max-len 512 --epochs 6  --lr 3e-5 --batch 8  --grad-accum 2
run_cv bs16  --seed 35 --max-len 512 --epochs 6  --lr 2e-5 --batch 16 --grad-accum 1
run_cv ml256 --seed 33 --max-len 256 --epochs 6  --lr 1e-5 --batch 16 --grad-accum 1

echo ""
echo "================================================================"
echo "DUMP OOF CANDIDATE SPANS  (each fold scored by its own model)"
echo "================================================================"
python - <<'PY'
import json
import os
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, "src")
from train_evidence import TaggerModel, collect_candidates

OUT = "artifacts/evidence_oof"
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

for tag in ("v2", "ep12", "lr3", "bs16", "ml256"):
    dst = f"{OUT}/{tag}_oof_candidates.csv"
    if os.path.exists(dst):
        print(f"--- {dst} exists ---")
        continue
    rows = []
    for f in range(5):
        d = f"runs/ev5f_{tag}_f{f}"
        cfg = json.load(open(f"{d}/run_config.json"))
        ml = int(cfg.get("max_len", 512))
        va = pd.read_csv(f"artifacts/folds/val_f{f}.csv")
        tok = AutoTokenizer.from_pretrained(cfg.get(
            "model", "microsoft/deberta-v3-large"))
        m = TaggerModel(cfg.get("model", "microsoft/deberta-v3-large"))
        m.load_state_dict(torch.load(f"{d}/model_final.pt", map_location="cpu"))
        m.to(dev).eval()
        cands = collect_candidates(m, tok, [str(x) for x in va["post"]], ml, dev)
        for i, rid in enumerate(va["row_id"]):
            for s, c in cands[i]:
                rows.append((rid, s, float(c), cfg.get("evidence_min_conf", 0.0)))
        del m
        torch.cuda.empty_cache()
        print(f"  {tag} fold {f}: {len(va)} posts")
    pd.DataFrame(rows, columns=["row_id", "span", "conf", "min_conf"]).to_csv(
        dst, index=False)
    print(f"wrote {dst}  ({len(rows)} candidates)")
PY

echo ""
echo "Next (CPU): score the vote on 1635 OOF posts with"
echo "  python src/evidence_vote_oof.py"
echo ""
echo "The number to beat is the single deployed tagger on the SAME 1635 posts."
echo "The val-measured vote gain may shrink the way the factor numbers did"
echo "when moved from a 372-post val split to the 1635-post OOF -- if it"
echo "shrinks the same way here, the vote is worth far less than currently"
echo "assumed and k=3 was fitted to the split."
