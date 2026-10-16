#!/bin/bash
#SBATCH --job-name=facenc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out

# Swap the ENCODER. One variable: everything else is the incumbent recipe.
#
# WHY THIS IS ONE OF THE LAST JUSTIFIED GPU EXPERIMENTS. src/factor_diagnosis.py
# attributed the factor deficit:
#
#     thresholds   small oracle ceiling; per-label and global tuning tie
#     more data    AUC FLAT across prevalence bands (0.932 for <40 positives,
#                  0.908 for >400) -- the model is not data-starved
#     ranking      where the deficit actually lives
#
# Ranking is an encoder property. Nine factor experiments changed the data, the
# thresholds, the definitions or the decision rule; none changed the encoder.
# DeBERTa-v3-large is a general-purpose model being asked whether a post shows
# "a sense of purpose, goals, or reasons for living".
#
# AIMH/mental-roberta-large is RoBERTa-large pretrained on mental-health text.
# It brings different priors AND a different architecture, so either outcome is
# useful: better ranking means swap it in, different-but-comparable ranking
# means ensemble it (and unlike the d1-d8 sweep, which only varied
# hyperparameters within one architecture, this is genuine diversity).
#
# READ THE AP COLUMN, NOT F1. Average precision measures ranking without any
# threshold, and it is what the diagnosis showed F1 tracks. AUC is misleading
# on labels this imbalanced -- `meaning in life` has AUC 0.817 and AP 0.173.
#
# Submit one encoder at a time:
#   sbatch --export=ALL,MODEL=AIMH/mental-roberta-large,TAG=mrob scripts/factors_encoder.sh
#   sbatch --export=ALL,MODEL=microsoft/deberta-v2-xlarge,TAG=dv2xl,LR=5e-6 scripts/factors_encoder.sh

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# Published definitions, so the ENCODER is the only thing that differs from the
# incumbent. Explicitly unset in case it leaked in from the shell.
unset FACTOR_DEFS_JSON || true

# The protective template rewrites the hypothesis for the five protective
# factors from "This post shows X" to "The person in this post has X".
# Measured over 50 author partitions, it reliably helps on its own, so it is
# now an opt-in rather than something to suppress.
#
# It MUST be identical in training and inference -- train_factors_nli.py builds
# the hypothesis in both paths through hypothesis_for(), and a mismatch would
# train on one question and ask another with no error raised. Exporting it here
# covers both the training and the predict step below.
if [ "${PROTECTIVE_TEMPLATE:-0}" = "1" ]; then
    export PROTECTIVE_TEMPLATE=1
    echo "protective template: ON"
else
    unset PROTECTIVE_TEMPLATE || true
    echo "protective template: off"
fi

MODEL="${MODEL:-microsoft/deberta-v3-large}"
TAG="${TAG:?set TAG=...}"
LR="${LR:-1e-5}"
MAXLEN="${MAXLEN:-320}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-16}"
ACC="${ACC:-2}"
SEED="${SEED:-13}"
# extra flags for train_factors_nli.py, e.g. "--full-enum --loss asl".
# Quote the WHOLE assignment when submitting, because sbatch --export splits on
# commas and the shell would otherwise split on the space:
#   sbatch --export=ALL,TAG=fe,"EXTRA=--full-enum --loss asl" scripts/factors_encoder.sh
EXTRA="${EXTRA:-}"
# captured ONCE, because the fold loop overwrites CONFUSABLE_JSON in place
CONFUSABLE_JSON_TMPL="${CONFUSABLE_JSON:-}"

echo "encoder : ${MODEL}"
echo "tag     : ${TAG}"
echo "lr      : ${LR}   max_len ${MAXLEN}   epochs ${EPOCHS}   seed ${SEED}"
echo "extra   : ${EXTRA:-(none)}"
echo "confusable: ${CONFUSABLE_JSON:-hand-written}"

if [ ! -f artifacts/folds/train_f0.csv ]; then
    echo "ERROR: artifacts/folds/ missing (scripts/nli_risk_5fold.sh writes it)."
    exit 1
fi

OUT=artifacts/factor_runs
mkdir -p "${OUT}"

for f in 0 1 2 3 4; do
    # CONFUSABLE_JSON may contain the literal {f}, in which case each fold gets
    # its own hard-negative map mined from the OTHER four folds only. Training
    # is the only consumer -- predict_factors_nli.py never reads it -- so unlike
    # FACTOR_DEFS_JSON there is no train/serve skew to guard against here.
    if [ -n "${CONFUSABLE_JSON_TMPL}" ]; then
        export CONFUSABLE_JSON="${CONFUSABLE_JSON_TMPL//\{f\}/$f}"
        if [ ! -f "${CONFUSABLE_JSON}" ]; then
            echo "ERROR: ${CONFUSABLE_JSON} missing (run src/confusion_audit.py --per-fold)"
            exit 1
        fi
        echo "fold ${f} confusable map: ${CONFUSABLE_JSON}"
    fi
    dir="runs/factors_${TAG}_cv_f${f}"
    if [ ! -f "${dir}/model.pt" ]; then
        echo ""
        echo "================================================================"
        echo "${TAG} fold ${f}   ${MODEL}"
        echo "================================================================"
        python src/train_factors_nli.py \
            --train "artifacts/folds/train_f${f}.csv" \
            --val   "artifacts/folds/val_f${f}.csv" \
            --model "${MODEL}" \
            --outdir "${dir}" \
            --lr "${LR}" --max-len "${MAXLEN}" --epochs "${EPOCHS}" \
            --batch "${BATCH}" --grad-accum "${ACC}" --seed "${SEED}" \
            ${EXTRA} \
            || { echo "FAILED fold ${f}"; exit 1; }
    else
        echo "--- fold ${f}: exists ---"
    fi
    dst="${OUT}/${TAG}_cv_oof_f${f}.npz"
    [ -f "${dst}" ] || python src/predict_factors_nli.py \
        --test "artifacts/folds/val_f${f}.csv" --nli-run "${dir}" \
        --max-len "${MAXLEN}" \
        --dump-probs "${dst}" --out "${OUT}/${TAG}_cv_oof_f${f}_labels.csv" \
        || echo "  dump failed fold ${f}"
done

echo ""
echo "================================================================"
echo "RANKING (AP) AND F1 vs THE INCUMBENT"
echo "================================================================"
TAG="${TAG}" python - <<'PY'
import glob, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

TAG = os.environ["TAG"]
lab = pd.read_csv("artifacts/full_train.csv").set_index("row_id")
idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def gold(rids):
    Y = np.zeros((len(rids), 24), dtype=int)
    for i, r in enumerate(rids):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    return Y


def load(pattern):
    paths = sorted(glob.glob(pattern))
    if len(paths) < 5:
        return None, None, None
    r, P, T = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        r.extend(list(d["row_id"]))
        P.append(d["probs"])
        T.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
    return r, np.concatenate(P), np.concatenate(T)


def f1s(P, Y):
    o = []
    for j in range(24):
        tp = int((P[:, j] & Y[:, j]).sum())
        o.append(0.0 if tp == 0 else 2 * tp /
                 (2 * tp + int((P[:, j] & (1 - Y[:, j])).sum())
                  + int(((1 - P[:, j]) & Y[:, j]).sum())))
    return o


ir, IP, ITH = load("artifacts/fac_oof_f*.npz")
IY = gold(ir)
inc_f1 = f1s((IP >= ITH).astype(int), IY)
inc_ap = [average_precision_score(IY[:, j], IP[:, j]) for j in range(24)]

nr, NP, _ = load(f"artifacts/factor_runs/{TAG}_cv_oof_f*.npz")
if nr is None:
    raise SystemExit("fewer than 5 folds dumped")
NY = gold(nr)
th = np.full(24, 0.5)
for j in range(24):
    best, bt = -1.0, 0.5
    for t in GRID:
        p = (NP[:, j] >= t).astype(int)
        tp = int((p & NY[:, j]).sum())
        if tp == 0:
            continue
        v = 2 * tp / (2 * tp + int((p & (1 - NY[:, j])).sum())
                      + int(((1 - p) & NY[:, j]).sum()))
        if v > best:
            best, bt = v, t
    th[j] = bt
new_f1 = f1s((NP >= th).astype(int), NY)
new_ap = [average_precision_score(NY[:, j], NP[:, j]) for j in range(24)]

print(f"{'factor':<44}{'AP inc':>8}{'AP new':>8}{'dAP':>8}"
      f"{'F1 inc':>8}{'F1 new':>8}{'dF1':>8}")
print("-" * 92)
for j, t in enumerate(FACTOR_TAXONOMY):
    print(f"{t:<44}{inc_ap[j]:>8.3f}{new_ap[j]:>8.3f}{new_ap[j]-inc_ap[j]:>+8.3f}"
          f"{inc_f1[j]:>8.3f}{new_f1[j]:>8.3f}{new_f1[j]-inc_f1[j]:>+8.3f}")
print(f"\n{'MEAN':<44}{np.mean(inc_ap):>8.4f}{np.mean(new_ap):>8.4f}"
      f"{np.mean(new_ap)-np.mean(inc_ap):>+8.4f}"
      f"{np.mean(inc_f1):>8.4f}{np.mean(new_f1):>8.4f}"
      f"{np.mean(new_f1)-np.mean(inc_f1):>+8.4f}")
better = [FACTOR_TAXONOMY[j] for j in range(24)
          if new_ap[j] > inc_ap[j] + 0.02]
print(f"\nranks BETTER (AP +0.02 or more) on {len(better)}/24: {better}")
np.save(f"artifacts/factor_thresholds_{TAG}_oof.npy", th)
print(f"\nBAR: macro-F1 0.6673.")
print("If mean AP is up but F1 is not, the ranking improved and the")
print("thresholds are lagging. If AP is up on labels where DeBERTa is weak")
print("even with a lower mean, ENSEMBLE the two rather than swapping.")
PY
