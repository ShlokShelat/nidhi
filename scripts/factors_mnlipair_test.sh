#!/bin/bash
#SBATCH --job-name=facmp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Test-set predictions for `mnlipair` -- the first factor arm to beat the
# incumbent under every threshold regime measured (per-fold, pooled, and
# cross-fitted by author half).
#
# mnlipair is the average of two MNLI-pretrained encoders:
#   mnli   MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
#   dmnli  microsoft/deberta-large-mnli
#
# The incumbent is deliberately NOT in the average. mnli3, which adds it,
# scores worse than the pair. The non-MNLI model is now the weakest member
# and dilutes the mean.
#
# WHY TWO DIFFERENT BASE MODELS MATTER. dmnli is DeBERTa v1, mnli is v3. They
# share only the MNLI pretraining, and both beat the incumbent by a similar
# margin, so the cause is the entailment pretraining rather than either
# architecture. That is independent corroboration, not one lucky run.
#
# NO TRAINING HERE. The ten fold checkpoints already exist. This is pure
# inference: 378 posts x 24 factors x 10 models = 90,720 pairs, a few minutes.
#
# THRESHOLDS. Fitted on all 1635 out-of-fold posts at once, then applied
# unchanged to the test set. That is the most data available and therefore the
# most stable estimate -- src/threshold_rule_robustness.py showed at this
# sample size the plain argmax rule already beats shrinkage, plateau and
# prevalence matching, so no clever rule is used.
#
# The comma-joined run list is built INSIDE this script on purpose. Passing it
# through `sbatch --export` would silently truncate at the first comma, which
# is exactly how evidence_dump.sh once dumped 1 model instead of 8.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# published definitions, hand-written confusable map -- the arms were trained
# that way and inference must match
unset FACTOR_DEFS_JSON || true
unset PROTECTIVE_TEMPLATE || true
unset CONFUSABLE_JSON || true

TEST="${TEST:-data/test.csv}"
OUT=artifacts/factor_runs
mkdir -p "${OUT}"

[ -f "${TEST}" ] || { echo "ERROR: ${TEST} not found"; exit 1; }

# ---------------------------------------------------------------- checkpoints
RUNS=""
MISSING=0
for arm in mnli dmnli; do
    for f in 0 1 2 3 4; do
        d="runs/factors_${arm}_cv_f${f}"
        if [ -f "${d}/model.pt" ]; then
            RUNS="${RUNS:+${RUNS},}${d}"
        else
            echo "MISSING ${d}/model.pt"
            MISSING=$((MISSING + 1))
        fi
    done
done
NRUNS=$(echo "${RUNS}" | tr ',' '\n' | grep -c .)
echo "found ${NRUNS} checkpoints, ${MISSING} missing"
if [ "${NRUNS}" -ne 10 ]; then
    echo "ERROR: expected 10 checkpoints (5 mnli + 5 dmnli), refusing to run."
    echo "A partial bag is a DIFFERENT model from the one already measured."
    exit 1
fi

# ------------------------------------------------- thresholds from the OOF fit
echo ""
echo "================================================================"
echo "STEP 1  thresholds, fitted on all 1635 out-of-fold posts (CPU)"
echo "================================================================"
python - <<'PY' || exit 1
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

N = len(FACTOR_TAXONOMY)
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
lab = pd.read_csv("artifacts/full_train.csv").set_index("row_id")
idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

rid, P = [], []
for f in range(5):
    d = np.load(f"artifacts/factor_runs/mnlipair_cv_oof_f{f}.npz", allow_pickle=True)
    rid.extend(list(d["row_id"]))
    P.append(d["probs"])
P = np.concatenate(P)

Y = np.zeros((len(rid), N), dtype=int)
for i, r in enumerate(rid):
    for x in parse_factors(lab.loc[r, "factors"]):
        j = idx.get(str(x).strip().lower())
        if j is not None:
            Y[i, j] = 1

th = np.full(N, 0.5)
for j in range(N):
    best, bt = -1.0, 0.5
    for t in GRID:
        p = (P[:, j] >= t).astype(int)
        tp = int((p & Y[:, j]).sum())
        if tp == 0:
            continue
        f1 = 2 * tp / (2 * tp + int((p & (1 - Y[:, j])).sum())
                       + int(((1 - p) & Y[:, j]).sum()))
        if f1 > best:
            best, bt = f1, t
    th[j] = bt

np.save("artifacts/factor_thresholds_mnlipair_ship.npy", th)
on = (P >= th).astype(int)
v = []
for j in range(N):
    tp = int((on[:, j] & Y[:, j]).sum())
    v.append(0.0 if tp == 0 else 2 * tp /
             (2 * tp + int((on[:, j] & (1 - Y[:, j])).sum())
              + int(((1 - on[:, j]) & Y[:, j]).sum())))
print(f"{len(rid)} OOF posts, pooled macro-F1 {np.mean(v):.4f}")
print(f"thresholds  min {th.min():.2f}  max {th.max():.2f}  "
      f"median {np.median(th):.2f}")
print("wrote artifacts/factor_thresholds_mnlipair_ship.npy")
print("\nNOTE this pooled number is IN-SAMPLE and inflated. The honest")
print("estimate is the xfit-half column, 0.6400 against 0.6084.")
PY

# ------------------------------------------------------------ test prediction
echo ""
echo "================================================================"
echo "STEP 2  score the test set with all 10 checkpoints (GPU)"
echo "================================================================"
python src/predict_factors_nli.py \
    --test "${TEST}" \
    --nli-run "${RUNS}" \
    --thresholds artifacts/factor_thresholds_mnlipair_ship.npy \
    --dump-probs "${OUT}/mnlipair_test.npz" \
    --out artifacts/mnlipair_test_factors.csv \
    || { echo "FAILED"; exit 1; }

# --------------------------------------------------------------- sanity check
echo ""
echo "================================================================"
echo "STEP 3  does the test output look like the training data?"
echo "================================================================"
python - <<'PY'
import ast
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

new = pd.read_csv("artifacts/mnlipair_test_factors.csv", keep_default_na=False)
cnt = [len(ast.literal_eval(x)) for x in new["factors"]]
print(f"test posts {len(new)}   mean {np.mean(cnt):.2f} factors/post   "
      f"empty {sum(1 for c in cnt if c == 0)}")

lab = pd.read_csv("artifacts/full_train.csv")
g = [len(parse_factors(x)) for x in lab["factors"]]
print(f"gold train                mean {np.mean(g):.2f} factors/post")

from collections import Counter
c = Counter()
for x in new["factors"]:
    c.update(ast.literal_eval(x))
print("\nper-label rate, test prediction vs gold training prevalence:")
print(f"{'factor':<44}{'pred %':>9}{'gold %':>9}{'ratio':>8}")
print("-" * 72)
for t in FACTOR_TAXONOMY:
    pr = 100.0 * c.get(t, 0) / len(new)
    gp = 100.0 * sum(1 for x in lab["factors"] if t in parse_factors(x)) / len(lab)
    r = pr / gp if gp > 0 else float("nan")
    flag = "   <-- check" if (gp > 1 and (r > 2.0 or r < 0.4)) else ""
    print(f"{t:<44}{pr:>9.1f}{gp:>9.1f}{r:>8.2f}{flag}")
print("\nA label predicted far more or far less often than it occurs in")
print("training is the signature of a bad threshold. That is what inverted")
print("a prior factor submission between val and test.")
PY

echo ""
echo "================================================================"
echo "DONE -> artifacts/mnlipair_test_factors.csv"
echo "================================================================"
echo "This is the FACTOR COLUMN ONLY. Nothing has been submitted."
echo "Assemble it onto the champion, changing nothing else, so the"
echo "submission stays a single-variable test."
