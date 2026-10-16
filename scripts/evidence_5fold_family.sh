#!/bin/bash
#SBATCH --job-name=ev5ffam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

# A BIO span tagger from a DIFFERENT ARCHITECTURE FAMILY, trained on the same
# five author-disjoint folds, so its OOF candidates can join the vote honestly.
#
# WHY A DIFFERENT FAMILY, AND WHY AS AN ADDED VOTER. An encoder swap judged as
# a REPLACEMENT has to be better outright; a weaker member can still help a bag,
# because what a bag needs from a member is decorrelated errors. Changing the
# architecture family changes tokenization and pretraining objective together,
# which decorrelates span boundaries far more than changing pretraining data
# alone.
#
# This is only usable because CLUSTER-VOTING counts agreement under the
# SCORER'S containment rule rather than exact string equality. Under an
# exact-string vote a tagger with different boundary conventions cannot cast a
# countable vote AT ALL -- its spans fail to key-match anyone else's and are
# discarded -- so boundary-diverse members contribute nothing by construction.
#
# ELECTRA is the principled first choice: its replaced-token-detection
# pretraining IS token-level classification, the closest objective match to BIO
# tagging available. RoBERTa is the second, for its byte-BPE tokenizer, which
# produces genuinely different boundary hypotheses.
#
# TWO GATES, both pre-stated:
#   PARITY  the member's own val phrase-F1 must land within ~0.02 of its
#           DeBERTa twin, so a member that is simply broken is caught before it
#           is judged on bag contribution.
#   BAG     adding it must lift cluster>=2 macro on the 1635 OOF posts,
#           positive on BOTH author halves. Member quality is not the
#           metric -- bag contribution is.
#
# Usage:
#   sbatch --export=ALL,MODEL=google/electra-large-discriminator,TAG=elec \
#       scripts/evidence_5fold_family.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

MODEL="${MODEL:?set MODEL=...}"
TAG="${TAG:?set TAG=...}"
OUT=artifacts/evidence_oof
mkdir -p "${OUT}" logs

# Mirror the DEPLOYED tagger's recipe exactly (v2: seed 13, len 512, 6 epochs,
# lr 1e-5, batch 8, grad-accum 2) so the ENCODER is the only variable and the
# parity gate compares like with like.
for f in 0 1 2 3 4; do
    dir="runs/ev5f_${TAG}_f${f}"
    if [ -f "${dir}/model_final.pt" ]; then
        echo "--- ${TAG} fold ${f}: exists, skipping ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "${TAG} fold ${f}   ${MODEL}"
    echo "================================================================"
    python src/train_evidence.py \
        --train "artifacts/folds/train_f${f}.csv" \
        --val   "artifacts/folds/val_f${f}.csv" \
        --model "${MODEL}" \
        --outdir "${dir}" --num-workers 0 \
        --seed 13 --max-len 512 --epochs 6 --lr 1e-5 --batch 8 --grad-accum 2 \
        || { echo "FAILED ${TAG} fold ${f}"; exit 1; }
done

echo ""
echo "================================================================"
echo "DUMP OOF CANDIDATES  (each fold scored by its own model)"
echo "================================================================"
TAG="${TAG}" python - <<'PY' || exit 1
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
    # read the model from the run config, never a hardcoded default -- loading
    # a checkpoint under the wrong tokenizer silently produces garbage spans
    name = cfg["model"]
    ml = int(cfg.get("max_len", 512))
    va = pd.read_csv(f"artifacts/folds/val_f{f}.csv")
    tok = AutoTokenizer.from_pretrained(name)
    m = TaggerModel(name)
    m.load_state_dict(torch.load(f"{d}/model_final.pt", map_location="cpu"))
    m.to(dev).eval()
    cands = collect_candidates(m, tok, [str(x) for x in va["post"]], ml, dev)
    for i, rid in enumerate(va["row_id"]):
        for s, c in cands[i]:
            rows.append((rid, s, float(c), cfg.get("evidence_min_conf", 0.0)))
    del m
    torch.cuda.empty_cache()
    print(f"  {TAG} fold {f}: {len(va)} posts, {len(rows)} candidates so far")

pd.DataFrame(rows, columns=["row_id", "span", "conf", "min_conf"]).to_csv(
    dst, index=False)
print(f"wrote {dst}  ({len(rows)} candidates)")
PY

echo ""
echo "================================================================"
echo "DONE. Judge it as an ADDED VOTER, not a replacement:"
echo "================================================================"
echo "  python src/evidence_vote_oof.py --average macro \\"
echo "      --tags v2,ep12,lr3,bs16,ml256,${TAG}"
echo ""
echo "Compare cluster>=2 modal WITH ${TAG} against the same vote without it."
echo "It must gain macro, positive on BOTH author halves."
echo "Also check this member's own val phrase-F1 above: within ~0.02 of the"
echo "DeBERTa twin is parity, which is the gate for judging it as a voter."
