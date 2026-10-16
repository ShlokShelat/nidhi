#!/bin/bash
#SBATCH --job-name=nli5f
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# Out-of-fold risk-NLI predictions on all 1635 labelled posts.
#
# WHY. A single 372-post validation split holds only ~32 authors, and posts by
# one author are highly correlated, so a symbolic rule or blend weight can look
# strong on it while its entire effect comes from two authors. Judging a change
# on so few authors cannot distinguish a real gain from author-level noise.
#
# Three of the four risk probability sources ALREADY have 5-fold out-of-fold
# predictions covering ~1590-1635 posts:
#
#     v1-architecture DeBERTa   v2_rebuild/runs/risk_ctrl_f*      1635
#     friend model              artifacts/friend_oof_export.csv   1590
#     ordinal Qwen              v2_rebuild/runs/risk_llm2_ord_f*  1635
#
# Only risk-NLI is missing -- it exists solely as a val-scored model. This job
# trains it on the SAME author-disjoint folds as everything else, so a
# full ensemble can be reconstructed over ~1590 posts rather than 354. That is
# roughly 4.5x the posts and ~5x the authors, which cuts the standard error of
# every rule, blend-weight and threshold verdict by about half, and makes a
# rule driven by two authors visible as such.
#
# CAVEAT: this reconstruction uses ONE v1-architecture seed where the deployed
# chain averages three, so it is a close analogue of that chain, not the chain
# itself. That is adequate for RANKING rules by their error profile; a final
# ship decision still needs a comparison against the deployed chain.
#
# ~45 min per fold.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

mkdir -p artifacts/folds

# Fold CSVs, built with the byte-identical author-disjoint policy every other
# component uses. If the fold assignment drifts, out-of-fold predictions from
# different components stop being comparable and cannot be pooled.
python - <<'PY'
import os
import pandas as pd

df = pd.read_csv("artifacts/full_train.csv")

def author_folds(df, k=5, seed=13):
    counts = df.groupby("anon_user_id").size().sort_values(
        ascending=False, kind="mergesort")
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    sizes = {i: 0 for i in range(k)}
    assign = {}
    for a in order:
        f = min(sizes, key=lambda i: sizes[i])
        assign[a] = f
        sizes[f] += counts[a]
    return df["anon_user_id"].map(assign)

fold = author_folds(df)
for f in range(5):
    tr, va = df[fold != f], df[fold == f]
    assert not (set(tr.anon_user_id) & set(va.anon_user_id)), "author leak"
    tr.to_csv(f"artifacts/folds/train_f{f}.csv", index=False)
    va.to_csv(f"artifacts/folds/val_f{f}.csv", index=False)
    print(f"fold {f}: train {len(tr)} / val {len(va)} "
          f"({va.anon_user_id.nunique()} held-out authors)")
PY

for f in 0 1 2 3 4; do
    OUT="runs/risk_nli_cv_f${f}"
    if [ ! -f "${OUT}/model.pt" ]; then
        echo ""
        echo "================================================================"
        echo "risk-NLI fold ${f}"
        echo "================================================================"
        python src/train_risk_nli.py \
            --train "artifacts/folds/train_f${f}.csv" \
            --val   "artifacts/folds/val_f${f}.csv" \
            --model microsoft/deberta-v3-large \
            --outdir "${OUT}" \
            --max-len 384 --epochs 4 --lr 1e-5 \
            --batch 4 --grad-accum 4 --seed 13 \
            || { echo "FAILED fold ${f}"; exit 1; }
    else
        echo "--- fold ${f}: model exists ---"
    fi

    DST="artifacts/nli_risk_oof_f${f}.csv"
    if [ ! -f "${DST}" ]; then
        python src/dump_nli_probs.py \
            --data "artifacts/folds/val_f${f}.csv" \
            --nli-run "${OUT}" \
            --max-len 384 \
            --out "${DST}" || echo "  dump failed fold ${f}"
    fi
done

echo ""
echo "================================================================"
echo "POOLED OUT-OF-FOLD RISK-NLI"
echo "================================================================"
python - <<'PY'
import glob, os, sys
import pandas as pd
sys.path.insert(0, "src")
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1

parts = [pd.read_csv(p) for p in sorted(glob.glob("artifacts/nli_risk_oof_f*.csv"))]
if not parts:
    raise SystemExit("no fold dumps")
oof = pd.concat(parts, ignore_index=True)
oof.to_csv("artifacts/nli_risk_oof_all.csv", index=False)

lab = pd.read_csv("artifacts/full_train.csv")
g = dict(zip(lab.row_id, lab.risk_level))
cols = [f"prob_{c}" for c in RISK_LEVELS]
pred = [RISK_LEVELS[i] for i in oof[cols].to_numpy().argmax(1)]
gold = [normalize_label(g[r]) for r in oof.row_id]
print(f"{len(oof)} posts   pooled OOF weighted-F1 = "
      f"{risk_weighted_f1(gold, pred):.4f}")
print("  (Pooled out-of-fold is a harder instrument than a single held-out")
print("   split -- every post is scored by a model that never saw its author,")
print("   so a somewhat lower figure here is expected, not a defect.)")
print("\nwrote artifacts/nli_risk_oof_all.csv")
print("\nAll four risk components now have out-of-fold predictions, so the")
print("ensemble and the symbolic layer can be evaluated over ~1590 posts.")
PY
