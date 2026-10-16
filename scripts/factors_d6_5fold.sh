#!/bin/bash
#SBATCH --job-name=facd6cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

# Give an alternative factor regime the INCUMBENT'S threshold protocol.
#
# THE PROBLEM THIS SOLVES. A 24-label factor column needs 24 thresholds, one
# free parameter per label. Fitting them on the 372-post val split gives some
# labels only a handful of positives to fit against, so the chosen cutoff
# tracks that sample rather than the label. The signature is visible before any
# upload: a threshold lands near the bottom of the grid and the label then
# fires on a share of test posts far above its training prevalence.
#
# Fitting the same 24 thresholds on the 1635 out-of-fold posts uses 4.4x the
# data for the same number of parameters, which is why the incumbent's protocol
# is 5-fold and not single-split. Any candidate model has to be measured under
# that protocol before its val advantage means anything.
#
# THE DESIGN. Train the d6 regime (seed 21, lr 2e-5, 3 epochs, max_len 384) on
# the same five author-disjoint folds the incumbent used, so its thresholds are
# fitted on 1635 out-of-fold posts. The ONLY difference from the deployed
# system is then the model's hyperparameters -- same architecture, same 5-fold
# protocol, same threshold fitting, everything else held. One variable.
#
# d4 (lr 1e-5, 5 epochs) is trained alongside as a second regime, so a
# two-model ensemble can be tested afterwards under the same protocol. d6 alone
# is the primary comparison because it adds no selection parameter.
#
# Reuses artifacts/folds/{train,val}_f*.csv written by scripts/nli_risk_5fold.sh
# -- the same author-disjoint split every other experiment uses.
#
# ~40 min per fold per model, so about 7 hours for both.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

if [ ! -f artifacts/folds/train_f0.csv ]; then
    echo "ERROR: artifacts/folds/ missing. Run scripts/nli_risk_5fold.sh first"
    echo "(it writes the shared author-disjoint fold CSVs)."
    exit 1
fi

OUT=artifacts/factor_runs
mkdir -p "${OUT}"

# d6: --seed 21 --lr 2e-5 --epochs 3 --max-len 384
# d4: --seed 13 --lr 1e-5 --epochs 5 --max-len 320
run_cv () {
    local tag=$1; shift
    for f in 0 1 2 3 4; do
        local dir="runs/factors_${tag}_cv_f${f}"
        if [ ! -f "${dir}/model.pt" ]; then
            echo ""
            echo "================================================================"
            echo "${tag} fold ${f}   $*"
            echo "================================================================"
            python src/train_factors_nli.py \
                --train "artifacts/folds/train_f${f}.csv" \
                --val   "artifacts/folds/val_f${f}.csv" \
                --model microsoft/deberta-v3-large \
                --outdir "${dir}" \
                "$@" || { echo "FAILED ${tag} fold ${f}"; return 1; }
        else
            echo "--- ${tag} fold ${f}: exists ---"
        fi

        local dst="${OUT}/${tag}_cv_oof_f${f}.npz"
        if [ ! -f "${dst}" ]; then
            python src/predict_factors_nli.py \
                --test "artifacts/folds/val_f${f}.csv" \
                --nli-run "${dir}" \
                --dump-probs "${dst}" \
                --out "${OUT}/${tag}_cv_oof_f${f}_labels.csv" \
                || echo "  dump failed ${tag} fold ${f}"
        fi
    done

    # test predictions: bag all five fold models, matching how the incumbent's
    # test column is produced from its CV models
    local dst="${OUT}/${tag}_cv_test.npz"
    if [ ! -f "${dst}" ]; then
        local runs=""
        for f in 0 1 2 3 4; do
            runs="${runs}${runs:+,}runs/factors_${tag}_cv_f${f}"
        done
        python src/predict_factors_nli.py \
            --test data/test.csv --nli-run "${runs}" \
            --dump-probs "${dst}" \
            --out "${OUT}/${tag}_cv_test_labels.csv" \
            || echo "  test dump failed ${tag}"
    fi
}

run_cv d6 --seed 21 --lr 2e-5 --epochs 3 --max-len 384
run_cv d4 --seed 13 --lr 1e-5 --epochs 5 --max-len 320

echo ""
echo "================================================================"
echo "POOLED OOF -- thresholds can now be fitted on 1635 posts"
echo "================================================================"
python - <<'PY'
import glob, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

lab = pd.read_csv("artifacts/full_train.csv").set_index("row_id")
idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)

for tag in ("d6", "d4"):
    paths = sorted(glob.glob(f"artifacts/factor_runs/{tag}_cv_oof_f*.npz"))
    if len(paths) < 5:
        print(f"{tag}: only {len(paths)}/5 folds dumped")
        continue
    rids, mats = [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        rids.extend(list(d["row_id"]))
        mats.append(d["probs"])
    PR = np.concatenate(mats, axis=0)
    Y = np.zeros(PR.shape, dtype=int)
    for i, r in enumerate(rids):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1

    th = np.full(PR.shape[1], 0.5)
    for j in range(PR.shape[1]):
        best, bt = -1.0, 0.5
        for t in GRID:
            pred = (PR[:, j] >= t).astype(int)
            tp = int((pred & Y[:, j]).sum())
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + int((pred & (1 - Y[:, j])).sum())
                           + int(((1 - pred) & Y[:, j]).sum()))
            if f1 > best:
                best, bt = f1, t
        th[j] = bt
    P = (PR >= th).astype(int)
    f1s = []
    for j in range(P.shape[1]):
        tp = int((P[:, j] & Y[:, j]).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp /
                   (2 * tp + int((P[:, j] & (1 - Y[:, j])).sum())
                    + int(((1 - P[:, j]) & Y[:, j]).sum())))
    np.save(f"artifacts/factor_thresholds_{tag}_oof.npy", th)
    print(f"\n{tag}: {len(rids)} OOF posts   macro-F1 {np.mean(f1s):.4f}")
    print(f"   thresholds min {th.min():.2f} median {np.median(th):.2f} "
          f"max {th.max():.2f}   -> artifacts/factor_thresholds_{tag}_oof.npy")
    print(f"   Compare against the incumbent on the SAME 1635 OOF posts "
          f"(fac_oof_f*.npz, deployed thresholds).")
    print(f"   Only a win under the same protocol shows a better model rather "
          f"than a luckier threshold fit.")
PY
