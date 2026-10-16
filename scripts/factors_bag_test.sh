#!/bin/bash
#SBATCH --job-name=facbag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# Test-set predictions for an ENSEMBLE of factor arms, bagged over their five
# CV checkpoints each.
#
# Generalises scripts/factors_mnlipair_test.sh, which hardcoded two arms. Use:
#   sbatch --export=ALL,TAG=paircm,"ARMS=mnli dmnli cmine" scripts/factors_bag_test.sh
#
# Quote the WHOLE assignment: sbatch --export splits on commas and the shell
# would split on the space, so an unquoted multi-arm ARMS silently reaches the
# job as its first word alone and the bag is built from one arm's checkpoints.
#
# WHAT THE ARMS VARY. Each is a separate axis, so bagging them pools models
# that fail differently rather than models that fail together:
#   mnli/dmnli  MNLI-pretrained encoders -- the task IS entailment, and a
#               checkpoint that has never seen an entailment example is being
#               asked to make one. Two different base models (v3 and v1) show
#               the same effect, so the cause is the pretraining, not the
#               architecture.
#   cmine       hard negatives mined from the model's own confusions. The
#               hand-written CONFUSABLE map matches the model's real confusions
#               only 35% of the time (src/confusion_audit.py), so mined pairs
#               put the training pressure where the errors actually are.
#
# THE INCUMBENT IS DELIBERATELY EXCLUDED. Members are averaged with equal
# weight, so a member weaker than the rest of the bag pulls the mean down
# rather than adding diversity.
#
# WHY NOT pairprot. It needs PROTECTIVE_TEMPLATE=1 for `prot` and OFF for the
# MNLI models. That variable is read at inference, it is global, and a wrong
# setting silently asks most of the bag the wrong question with no error. This
# script REFUSES to include `prot` for that reason. cmine has no such problem:
# its mined map is consumed only during training -- predict_factors_nli.py
# never reads CONFUSABLE_JSON -- so every checkpoint here is scored under
# identical inference settings in a single pass.
#
# NO TRAINING. The checkpoints exist; this is inference plus a CPU threshold
# fit.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

unset FACTOR_DEFS_JSON || true
# CONFUSABLE_JSON is training-only (predict_factors_nli.py never reads it), so
# it is always cleared here regardless of how the members were trained.
unset CONFUSABLE_JSON || true
# PROTECTIVE_TEMPLATE is NOT training-only -- hypothesis_for() reads it at
# inference too. It is set further down from the arm names, never by hand.
unset PROTECTIVE_TEMPLATE || true

TAG="${TAG:?set TAG=... (must match an existing <TAG>_cv_oof_f*.npz)}"
ARMS="${ARMS:?set ARMS=\"mnli dmnli cmine\"}"
TEST="${TEST:-data/test.csv}"
MAXLEN="${MAXLEN:-320}"
OUT=artifacts/factor_runs

mkdir -p "${OUT}"
[ -f "${TEST}" ] || { echo "ERROR: ${TEST} not found"; exit 1; }

echo "tag  : ${TAG}"
echo "arms : ${ARMS}"
echo "test : ${TEST}"

# --------------------------------------------- derive the template setting
# PROTECTIVE_TEMPLATE rewrites the hypothesis for the five protective factors
# and is read by hypothesis_for() at INFERENCE as well as training. One pass
# has one setting, so a bag mixing template-trained and untemplated members is
# unscoreable: whichever way the flag goes, some members are asked a question
# they were never trained on, silently and with no error.
#
# Deriving it from the arm names rather than accepting a flag removes the
# footgun instead of adding another way to get it wrong. Keep this list in
# sync with any new template-trained arm.
TEMPLATE_ARMS=" prot mnlip dmnlip cminep mnlifep "
n_tpl=0
n_plain=0
for a in ${ARMS}; do
    case "${TEMPLATE_ARMS}" in
        *" ${a} "*) n_tpl=$((n_tpl + 1)) ;;
        *)          n_plain=$((n_plain + 1)) ;;
    esac
done
if [ "${n_tpl}" -gt 0 ] && [ "${n_plain}" -gt 0 ]; then
    echo ""
    echo "ERROR: mixed bag -- ${n_tpl} template-trained arm(s) and ${n_plain}"
    echo "untemplated. PROTECTIVE_TEMPLATE is global at inference, so one of"
    echo "the two groups would be scored on a question it never saw. Bag"
    echo "template arms together (mnlip dmnlip cminep) or plain arms together"
    echo "(mnli dmnli cmine), not both."
    exit 1
fi
if [ "${n_tpl}" -gt 0 ]; then
    export PROTECTIVE_TEMPLATE=1
    echo "protective template: ON  (all ${n_tpl} arms are template-trained)"
else
    echo "protective template: off (all ${n_plain} arms are untemplated)"
fi

# --------------------------------------------------------------- checkpoints
RUNS=""
for a in ${ARMS}; do
    for f in 0 1 2 3 4; do
        d="runs/factors_${a}_cv_f${f}"
        if [ -f "${d}/model.pt" ]; then
            RUNS="${RUNS:+${RUNS},}${d}"
        else
            echo "MISSING ${d}/model.pt"
        fi
    done
done
NARM=$(echo "${ARMS}" | wc -w)
WANT=$((NARM * 5))
GOT=$(echo "${RUNS}" | tr ',' '\n' | grep -c .)
echo "checkpoints ${GOT}/${WANT}"
if [ "${GOT}" -ne "${WANT}" ]; then
    echo "ERROR: incomplete bag. A partial bag is a DIFFERENT model from the"
    echo "one measured, and its OOF score would not apply. Refusing to run."
    exit 1
fi

# Equal weight per checkpoint equals equal weight per arm, because every arm
# contributes exactly five. That reproduces how average_factor_seeds.py built
# the OOF arm these thresholds come from.

# ------------------------------------------------- thresholds from the OOF fit
echo ""
echo "================================================================"
echo "STEP 1  thresholds, fitted on all 1635 out-of-fold posts (CPU)"
echo "================================================================"
TAG="${TAG}" python - <<'PY' || exit 1
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

TAG = os.environ["TAG"]
N = len(FACTOR_TAXONOMY)
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
lab = pd.read_csv("artifacts/full_train.csv").set_index("row_id")
idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

rid, P = [], []
for f in range(5):
    p = f"artifacts/factor_runs/{TAG}_cv_oof_f{f}.npz"
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- build it with average_factor_seeds.py")
    d = np.load(p, allow_pickle=True)
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

np.save(f"artifacts/factor_thresholds_{TAG}_ship.npy", th)
on = (P >= th).astype(int)
v = []
for j in range(N):
    tp = int((on[:, j] & Y[:, j]).sum())
    v.append(0.0 if tp == 0 else 2 * tp /
             (2 * tp + int((on[:, j] & (1 - Y[:, j])).sum())
              + int(((1 - on[:, j]) & Y[:, j]).sum())))
print(f"{len(rid)} OOF posts, pooled macro-F1 {np.mean(v):.4f}  (IN-SAMPLE)")
print(f"thresholds  min {th.min():.2f}  max {th.max():.2f}  "
      f"median {np.median(th):.2f}")
print(f"wrote artifacts/factor_thresholds_{TAG}_ship.npy")
PY

# ------------------------------------------------------------ test prediction
echo ""
echo "================================================================"
echo "STEP 2  score the test set with all ${GOT} checkpoints (GPU)"
echo "================================================================"
python src/predict_factors_nli.py \
    --test "${TEST}" \
    --nli-run "${RUNS}" \
    --max-len "${MAXLEN}" \
    --thresholds "artifacts/factor_thresholds_${TAG}_ship.npy" \
    --dump-probs "${OUT}/${TAG}_test.npz" \
    --out "artifacts/${TAG}_test_factors.csv" \
    || { echo "FAILED"; exit 1; }

# --------------------------------------------------------------- sanity check
echo ""
echo "================================================================"
echo "STEP 3  does the test output look like the training data?"
echo "================================================================"
TAG="${TAG}" python - <<'PY'
import ast, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

TAG = os.environ["TAG"]
new = pd.read_csv(f"artifacts/{TAG}_test_factors.csv", keep_default_na=False)
cnt = [len(ast.literal_eval(x)) for x in new["factors"]]
lab = pd.read_csv("artifacts/full_train.csv")

# Gold cells carry ONE TAG PER EVIDENCE SPAN, so 812 of 1635 repeat a factor
# and a raw len() reports 5.30 factors/post against a true 2.92. Predictions
# are de-duplicated, so gold must be de-duplicated too; comparing the two
# directly makes a correctly-calibrated model look like it is missing 43% of
# the labels.
TAX = set(FACTOR_TAXONOMY)
g = [len({str(f).strip().lower() for f in parse_factors(x)} & TAX)
     for x in lab["factors"]]
print(f"test  {len(new)} posts   mean {np.mean(cnt):.2f} factors/post   "
      f"empty {sum(1 for c in cnt if c == 0)} "
      f"({100.0 * sum(1 for c in cnt if c == 0) / len(cnt):.1f}%)")
print(f"gold  {len(lab)} posts   mean {np.mean(g):.2f} factors/post   "
      f"empty {sum(1 for c in g if c == 0)} "
      f"({100.0 * sum(1 for c in g if c == 0) / len(g):.1f}%)   "
      f"[unique factors, not raw tags]")

from collections import Counter
c = Counter()
for x in new["factors"]:
    c.update(ast.literal_eval(x))
print("\nper-label rate, test prediction vs gold training prevalence:")
print(f"{'factor':<44}{'pred %':>9}{'gold %':>9}{'ratio':>8}")
print("-" * 72)
bad = 0
for t in FACTOR_TAXONOMY:
    pr = 100.0 * c.get(t, 0) / len(new)
    gp = 100.0 * sum(1 for x in lab["factors"] if t in parse_factors(x)) / len(lab)
    r = pr / gp if gp > 0 else float("nan")
    # Every label is checked, including those under 1% gold prevalence:
    # macro-F1 weights all 24 labels equally, so a rare label predicted zero
    # times scores F1 = 0 outright and costs a full 1/24 of the metric. A
    # prevalence floor on this guard would skip exactly those labels.
    flag = ""
    if c.get(t, 0) == 0 and gp > 0:
        flag, bad = "   <-- ZERO PREDICTIONS", bad + 1
    elif r > 1.5 or r < 0.6:
        flag, bad = "   <-- CHECK", bad + 1
    print(f"{t:<44}{pr:>9.1f}{gp:>9.1f}{r:>8.2f}{flag}")
print(f"\n{bad} label(s) flagged.")
print("A label predicted far more or far less often than it occurs in")
print("training is the signature of a badly placed threshold: the ranking can")
print("be fine while the operating point is wrong, and macro-F1 punishes that")
print("hardest on the rare labels.")
PY

echo ""
echo "================================================================"
echo "DONE -> artifacts/${TAG}_test_factors.csv"
echo "================================================================"
echo "FACTOR COLUMN ONLY. Nothing submitted. Patch it onto the champion:"
echo ""
echo "  python src/patch_submission_risk.py \\"
echo "      --base submissions/NIDHI_evidence_consistent.csv \\"
echo "      --factors artifacts/${TAG}_test_factors.csv \\"
echo "      --out submissions/NIDHI_factors_${TAG}.csv"
