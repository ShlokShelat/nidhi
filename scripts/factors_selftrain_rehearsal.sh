#!/bin/bash
#SBATCH --job-name=strehear
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

# REHEARSE TRANSDUCTIVE SELF-TRAINING, HONESTLY.
#
# THE PLAN IT REHEARSES. At submission time: predict the test set, keep the
# high-confidence factor predictions as pseudo-labels, retrain on
# train + pseudo(test), re-predict test. The 2024 winner of this competition
# family used exactly this procedure (arXiv 2410.04501).
#
# WHY TRANSDUCTIVE PSEUDO-LABELS ARE NOT JUST MORE EXTERNAL DATA. Crawled or
# generated posts import DISTRIBUTION-SHIFTED text, and a model trained on them
# has to bridge that shift at no benefit to the scored posts. Test posts are the
# same subreddit, the same author space, the same annotation regime -- and they
# are the thing being scored.
#
# THE REHEARSAL IS HONEST. For fold f, val_f plays "the test set":
#   teacher    = runs/factors_${ARM}_cv_f${f}, which never saw val_f
#   pseudo     = its own high-confidence predictions on val_f -- NO GOLD
#   student    = retrained on train_f + pseudo(val_f)
#   scored     = student's predictions on val_f, against gold
# Gold for val_f is used ONLY at scoring, so every input the student sees is one
# that would also be available at test time.
#
# BOTH ARMS USE FIXED EPOCHS. The baseline fold models were epoch-selected on
# val_f, but at deployment there is no clean val, so the student cannot be. To
# keep the ONLY difference the pseudo-labels, the baseline is retrained here
# with the same fixed epochs rather than reusing the val-selected checkpoint.
# That is why this is 10 runs per arm, not 5.
#
# COST. 10 runs x ~35 min = ~6h for one arm. Start with ONE arm; if the
# rehearsal is positive, extend.
#
# Usage:
#   sbatch scripts/factors_selftrain_rehearsal.sh
#   sbatch --export=ALL,ARM=mnli,GATE=0.85 scripts/factors_selftrain_rehearsal.sh

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

ARM="${ARM:-mnli}"
GATE="${GATE:-0.85}"
CAP="${CAP:-60}"
EPOCHS="${EPOCHS:-3}"
TEACH="${TEACH:-runs/factors_${ARM}_cv_f}"
OUT=artifacts/pseudo
mkdir -p "${OUT}" artifacts/selftrain

for f in 0 1 2 3 4; do
    [ -f "artifacts/folds/val_f${f}.csv" ] || {
        echo "ERROR: artifacts/folds/val_f${f}.csv missing"; exit 1; }
    [ -d "${TEACH}${f}" ] || {
        echo "ERROR: teacher ${TEACH}${f} missing"; exit 1; }
done

for f in 0 1 2 3 4; do
    echo ""
    echo "================================================================"
    echo "FOLD ${f}  --  val_f${f} plays the test set"
    echo "================================================================"

    # 1. teacher scores its own held-out fold -> probs (no gold involved)
    if [ ! -f "${OUT}/${ARM}_probs_f${f}.npz" ]; then
        python src/predict_factors_nli.py \
            --test "artifacts/folds/val_f${f}.csv" \
            --nli-run "${TEACH}${f}" \
            --dump-probs "${OUT}/${ARM}_probs_f${f}.npz" \
            || { echo "FAILED teacher fold ${f}"; exit 1; }
    fi

    # 2. high-confidence pseudo-labels
    python src/pseudo_label_factors.py \
        --probs "${OUT}/${ARM}_probs_f${f}.npz" \
        --posts "artifacts/folds/val_f${f}.csv" \
        --out "${OUT}/${ARM}_pseudo_f${f}.csv" \
        --gate "${GATE}" --cap "${CAP}" \
        || { echo "FAILED pseudo fold ${f}"; exit 1; }

    # 3a. BASELINE at fixed epochs -- the control. Same recipe, no pseudo rows.
    if [ ! -f "runs/st_base_${ARM}_f${f}/thresholds.npy" ]; then
        python src/train_factors_nli.py \
            --train "artifacts/folds/train_f${f}.csv" \
            --val   "artifacts/folds/val_f${f}.csv" \
            --outdir "runs/st_base_${ARM}_f${f}" \
            --epochs "${EPOCHS}" \
            || { echo "FAILED baseline fold ${f}"; exit 1; }
    fi

    # 3b. STUDENT -- identical except for --external
    if [ ! -f "runs/st_self_${ARM}_f${f}/thresholds.npy" ]; then
        python src/train_factors_nli.py \
            --train "artifacts/folds/train_f${f}.csv" \
            --val   "artifacts/folds/val_f${f}.csv" \
            --external "${OUT}/${ARM}_pseudo_f${f}.csv" \
            --outdir "runs/st_self_${ARM}_f${f}" \
            --epochs "${EPOCHS}" \
            || { echo "FAILED student fold ${f}"; exit 1; }
    fi

    # 4. both score val_f
    for k in base self; do
        python src/predict_factors_nli.py \
            --test "artifacts/folds/val_f${f}.csv" \
            --nli-run "runs/st_${k}_${ARM}_f${f}" \
            --dump-probs "artifacts/selftrain/${ARM}_${k}_f${f}.npz" \
            || { echo "FAILED score ${k} fold ${f}"; exit 1; }
    done
done

echo ""
echo "================================================================"
echo "POOLED RESULT -- student vs baseline on 1635 held-out posts"
echo "================================================================"
ARM="${ARM}" python - <<'PY'
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

ARM = os.environ["ARM"]
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
lab = pd.read_csv("artifacts/full_train.csv", keep_default_na=False)
lab["row_id"] = lab["row_id"].astype(str)
lab = lab.set_index("row_id")

def pool(kind):
    P, R = [], []
    for f in range(5):
        z = np.load(f"artifacts/selftrain/{ARM}_{kind}_f{f}.npz",
                    allow_pickle=True)
        P.append(z["probs"].astype(float))
        R.extend(str(x) for x in z["row_id"])
    return np.concatenate(P), R

Pb, rid = pool("base")
Ps, rid2 = pool("self")
assert rid == rid2
Y = np.zeros((len(rid), 24), dtype=int)
for i, r in enumerate(rid):
    for x in set(parse_factors(str(lab.loc[r, "factors"]))):
        j = IDX.get(str(x).strip().lower())
        if j is not None:
            Y[i, j] = 1

def ap(p, y):
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-p, kind="stable"); ys = y[o]
    return float((np.cumsum(ys)/np.arange(1, len(ys)+1)*ys).sum()/ys.sum())

auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rid])
ua = pd.unique(auth); by = {a: np.flatnonzero(auth == a) for a in ua}
rng = np.random.default_rng(13)
mb = lambda P, ix: float(np.nanmean([ap(P[ix, j], Y[ix, j])
                                     for j in range(24)]))
allix = np.arange(len(rid))
print(f"{len(rid)} posts   mean AP  baseline {mb(Pb, allix):.4f}   "
      f"self-trained {mb(Ps, allix):.4f}")
d = []
for _ in range(400):
    ix = np.concatenate([by[a] for a in rng.choice(ua, len(ua), replace=True)])
    d.append(mb(Ps, ix) - mb(Pb, ix))
d = np.array(d); sd = d.std(ddof=1)
print(f"delta {d.mean():+.4f}  win {(d>0).mean():.0%}  sd {sd:.4f}  "
      f"one-draw [{d.mean()-1.96*sd:+.4f},{d.mean()+1.96*sd:+.4f}]")
print("\nper-label (gold n, baseline AP, self AP, delta):")
for j in np.argsort(-(np.array([ap(Ps[:, j], Y[:, j]) for j in range(24)])
                      - np.array([ap(Pb[:, j], Y[:, j]) for j in range(24)]))):
    a, b = ap(Pb[:, j], Y[:, j]), ap(Ps[:, j], Y[:, j])
    print(f"  {FACTOR_TAXONOMY[j]:<44}{int(Y[:, j].sum()):>5}"
          f"{a:>8.3f}{b:>8.3f}{b-a:>+8.3f}")
print("\nSelf-training helps if the delta is positive with a high win rate.")
print("Watch the RARE labels: a teacher at gate 0.85 gives them few or no")
print("pseudo-positives, so they cannot improve and may drift.")
PY
