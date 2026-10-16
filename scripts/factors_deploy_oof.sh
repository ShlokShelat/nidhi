#!/bin/bash
#SBATCH --job-name=facdoof
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# Build factor probabilities for labelled posts under the deployment averaging
# regime, so thresholds can be fitted in the same regime they are applied in.
#
# NOT ON THE SUBMISSION PATH. Nothing in build_submission.py reads this
# script's output, and its historical outputs are not used.
#
# Standard OOF scores each post with ONE checkpoint per arm (the fold that held
# it out), while the test set is scored by ALL checkpoints averaged. Averaging
# more models narrows each probability distribution, so a threshold fitted on
# single-checkpoint OOF probabilities is applied to a differently-shaped
# distribution at test time; rare labels can fall to a positive rate of zero.
#
# This builds, for every labelled post, a probability vector averaged over many
# checkpoints. Leakage is avoided by excluding, for each post, every checkpoint
# whose training fold contained it: a post in fold f is scored by the 4 other
# folds of each arm -- 12 checkpoints for a 3-arm bag, against 15 at test.
#
# Cost: 15 checkpoints x 1635 posts x 24 factors = 588,600 pairs, plus 15 model
# loads. Scored once per arm and reused by every later comparison.
#
# Usage:
#   sbatch --export=ALL,"ARMS=mnli dmnli cmine" scripts/factors_deploy_oof.sh

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

unset FACTOR_DEFS_JSON || true
unset CONFUSABLE_JSON || true
unset PROTECTIVE_TEMPLATE || true

ARMS="${ARMS:?set ARMS=\"mnli dmnli cmine\"}"
MAXLEN="${MAXLEN:-320}"
OUT=artifacts/factor_deploy
mkdir -p "${OUT}"

# template arms need the flag ON at inference; mixing is unscoreable
TEMPLATE_ARMS=" prot mnlip dmnlip cminep mnlifep "
n_tpl=0; n_plain=0
for a in ${ARMS}; do
    case "${TEMPLATE_ARMS}" in
        *" ${a} "*) n_tpl=$((n_tpl + 1)) ;;
        *)          n_plain=$((n_plain + 1)) ;;
    esac
done
if [ "${n_tpl}" -gt 0 ] && [ "${n_plain}" -gt 0 ]; then
    echo "ERROR: mixed template/untemplated bag -- unscoreable in one pass."
    exit 1
fi
[ "${n_tpl}" -gt 0 ] && export PROTECTIVE_TEMPLATE=1
echo "arms: ${ARMS}   protective template: ${PROTECTIVE_TEMPLATE:-off}"

# ---------------------------------------------------------------- score once
# Each checkpoint scores ALL 1635 labelled posts. The leakage mask is applied
# afterwards on CPU, so no post is ever averaged with a model that trained on
# it -- but each model only runs once instead of once per fold.
if [ ! -f artifacts/full_train.csv ]; then
    echo "ERROR: artifacts/full_train.csv missing"; exit 1
fi

for a in ${ARMS}; do
    for f in 0 1 2 3 4; do
        d="runs/factors_${a}_cv_f${f}"
        dst="${OUT}/${a}_f${f}_on_all.npz"
        if [ -f "${dst}" ]; then
            echo "--- ${dst} exists ---"
            continue
        fi
        if [ ! -f "${d}/model.pt" ]; then
            echo "MISSING ${d}/model.pt"; exit 1
        fi
        echo ""
        echo "=== ${a} fold ${f} -> all 1635 posts ==="
        python src/predict_factors_nli.py \
            --test artifacts/full_train.csv --nli-run "${d}" \
            --max-len "${MAXLEN}" \
            --dump-probs "${dst}" \
            --out "${OUT}/${a}_f${f}_on_all_labels.csv" \
            || { echo "FAILED ${a} f${f}"; exit 1; }
    done
done

# ------------------------------------------- assemble the leakage-free average
echo ""
echo "================================================================"
echo "assembling deployment-regime OOF (CPU)"
echo "================================================================"
ARMS="${ARMS}" python - <<'PY'
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY

ARMS = os.environ["ARMS"].split()
OUT = "artifacts/factor_deploy"
N = len(FACTOR_TAXONOMY)

# which fold each labelled post was held out in
fold_of = {}
for f in range(5):
    va = pd.read_csv(f"artifacts/folds/val_f{f}.csv")
    for r in va["row_id"]:
        fold_of[r] = f

base = np.load(f"{OUT}/{ARMS[0]}_f0_on_all.npz", allow_pickle=True)
rid = list(base["row_id"])
own = np.array([fold_of[r] for r in rid])

acc = np.zeros((len(rid), N), dtype=np.float64)
cnt = np.zeros(len(rid), dtype=np.int32)
for a in ARMS:
    for f in range(5):
        d = np.load(f"{OUT}/{a}_f{f}_on_all.npz", allow_pickle=True)
        r = list(d["row_id"])
        if r != rid:
            o = {x: i for i, x in enumerate(r)}
            P = d["probs"][[o[x] for x in rid]]
        else:
            P = d["probs"]
        # Keep only posts this checkpoint did NOT train on. `own` is the fold a
        # post was HELD OUT in, so checkpoint f trained on every post with
        # `own != f`; selecting `own == f` picks exactly the held-out posts.
        m = own == f
        acc[m] += P[m]
        cnt[m] += 1
if cnt.min() != cnt.max():
    raise SystemExit(f"uneven checkpoint counts {cnt.min()}..{cnt.max()}")
print(f"{len(rid)} posts, each averaged over {cnt[0]} leakage-free checkpoints "
      f"(single-checkpoint OOF uses {len(ARMS)})")
P = (acc / cnt[:, None]).astype(np.float32)

# write in the same per-fold shape arm_robustness.py and compare_factor_arms.py
# already consume, so every existing instrument works unchanged
tag = "deploy_" + "".join(a[0] for a in ARMS) + str(len(ARMS))
for f in range(5):
    m = own == f
    np.savez(f"artifacts/factor_runs/{tag}_cv_oof_f{f}.npz",
             probs=P[m], thresholds=np.full(N, 0.5),
             row_id=np.array([r for r, k in zip(rid, m) if k], dtype=object))
print(f"wrote artifacts/factor_runs/{tag}_cv_oof_f{{0..4}}.npz")
print(f"\nNext:  python src/arm_robustness.py --splits 50 --vs {tag}")
PY
