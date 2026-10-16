#!/bin/bash
#SBATCH --job-name=rat3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# RATIONALE-SPAN SUPERVISION, REBUILT IN THE SHIPPABLE CONSTRUCTION.
#
# WHY THIS AND NOT THE OLD FOLD RUN. The rationale head recorded a large
# standalone member weighted-F1 gain, but that number came from a SINGLE-FOLD
# estimator while the artefact shipped was a 5-FOLD BAG in which four of five
# models had trained on every scored post -- so the measurement did not
# describe the object that was deployed, and the deployed result could not be
# attributed to it. It was INVALIDATED, never refuted: no instrument that
# could judge the bag correctly ever existed.
#
# The fix is not a cleverer instrument for the bag. It is to build the member
# in the SAME SHAPE as the deployed encoder seeds -- three seeds on
# `artifacts/train_split.csv` with `artifacts/val_split.csv` held out -- so
# val_split is an honest holdout and the thing measured is the thing shipped.
# That is the property (and the ONLY property) shared by the two prior
# changes that generalized best from offline measurement: `ev7` and `ltail9`,
# the latter of which was offline UNDERSTATED relative to what it delivered.
#
# HOW IT MUST BE JUDGED, AND WHY NOT ON THE FULL CHAIN.
# `champion_oof_twin.py`'s repair proved the twin cannot be made
# level-accurate: with the friend model included it reads 0.7992, without it
# 0.8294, and the deployed value is 0.8203. The friend model helps risk
# weighted-F1 on test while hurting it on val -- a contradiction that means
# `friend_risk_probs_val.csv` is a different, weaker model than what actually
# ships, and the friend model has no checkpoint or code in this repo, so it
# cannot be re-dumped for a matched comparison.
#
# Therefore: **judge this arm FRIEND-FREE, at MEMBER level** -- the mean of the
# three new seeds against the mean of `runs/risk_v1{,_s14,_s15}`, on the same
# 372 val posts. The friend and the NLI term cancel out of a paired member
# comparison. Then discount analytically for the chain: x0.5 for friend
# dilution, x0.7 for the NLI weight, ~x0.35 overall. The member-level prior
# sits well above the twin's detectability floor, which is what makes this
# one of the very few risk questions the available instruments can still
# decide.
#
# EPOCH SELECTION MUST BE SYMMETRIC. `runs/risk_v1*` are best-on-val-epoch
# checkpoints, so these arms are val-selected too (no --no-val). Comparing a
# fixed-epoch arm against val-selected incumbents would bias against the new
# arm; this keeps the bias identical on both sides.
#
# COST ~6-9 GPU-hours (3 seeds x 5 epochs on 1263 posts at max-len 512).
# NOTHING SHIPS FROM HERE. It ends in a CPU comparison that can still say no.
# ===========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

# RISK_CAT_ONLY is read INSIDE the model at inference; the factor vars are
# cleared on principle. sbatch defaults to --export=ALL, so a dirty login shell
# would otherwise silently change what both arms compute.
unset RISK_CAT_ONLY RISK_DEFS FACTOR_DEFS_JSON FACTOR_ASPECTS_JSON \
      PROTECTIVE_TEMPLATE CONFUSABLE_JSON FACTOR_SUBSET 2>/dev/null || true

set -u
TAG="${TAG:-rat3}"
MODEL="${MODEL:-microsoft/deberta-v3-large}"
SEEDS="${SEEDS:-13 14 15}"
OUT=artifacts/rationale
mkdir -p "$OUT" logs runs
FAIL=0

echo "node $(hostname)  $(date)"
echo "tag ${TAG}  seeds ${SEEDS}  model ${MODEL}"
df -h . | tail -1

for f in artifacts/train_split.csv artifacts/val_split.csv data/test.csv; do
    [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done

# TAG-COLLISION GUARD. A dir written here carries "lambda_span" in its
# run_config.json (train_risk_rationale.py records argparse). A finished
# checkpoint without it belongs to another experiment.
# NOTE ON FILENAMES, which cost a run: train_risk_rationale.py:306 saves
# **model.pt**, while predict.py:53 loads **model_final.pt**. The first version
# of this script tested for model_final.pt everywhere, so the strip loop
# matched nothing, the dump had no inputs, and three trained seeds sat unusable
# on disk. Source is model.pt; the stripped copy is written as model_final.pt
# precisely because that is the name the deployed inference path expects.
for d in "runs/risk_${TAG}_s"*; do
    [ -d "$d" ] || continue
    if [ -f "${d}/model.pt" ] \
       && ! grep -q 'lambda_span' "${d}/run_config.json" 2>/dev/null; then
        echo "ERROR: ${d} holds a finished checkpoint not written by this"
        echo "script. Another experiment owns TAG=${TAG}. Refusing."; exit 1
    fi
done

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "TRAIN -- 3 seeds mirroring runs/risk_v1* (train_split, val held out)"
echo "################################################################"
# Effective batch 16 (4 x 4), matching the deployed seeds' 8 x 2. max-len 512,
# lr 1e-5, 5 epochs, alpha_ordinal 0.5 -- the risk_v1 recipe. lambda_span is
# FIXED at 0.5 a priori and never swept: one fitted parameter on a 372-post
# instrument is one too many.
for S in ${SEEDS}; do
    d="runs/risk_${TAG}_s${S}"
    if [ ! -f "${d}/model.pt" ]; then
        echo ""
        echo "=== seed ${S} -> ${d} ==="
        python src/train_risk_rationale.py \
            --train artifacts/train_split.csv \
            --val   artifacts/val_split.csv \
            --model "${MODEL}" --outdir "${d}" \
            --seed "${S}" --epochs 5 --lr 1e-5 \
            --batch 4 --grad-accum 4 --max-len 512 \
            --alpha-ordinal 0.5 --lambda-span 0.5 \
            || { echo "FAILED seed ${S}"; FAIL=1; continue; }
    else
        echo "--- ${d}: exists ---"
    fi
    df -h . | tail -1
done

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "STRIP -- span_head must not reach predict.py"
echo "################################################################"
# RECORDED GOTCHA: predict.py:load_state reads model_final.pt and RAISES on
# unexpected keys, so `span_head.*` breaks the deployed inference path. The
# span head is a training-time auxiliary only -- it is not part of the risk
# decision -- so a stripped copy is the honest deployable artefact, and
# stripping it here means the test dump uses the SAME code path deployment
# uses rather than a bespoke one.
for S in ${SEEDS}; do
    src="runs/risk_${TAG}_s${S}"
    dst="runs/risk_${TAG}p_s${S}"
    [ -f "${src}/model.pt" ] || { echo "  ${src}/model.pt missing"; continue; }
    if [ ! -f "${dst}/model_final.pt" ]; then
        mkdir -p "${dst}"
        cp "${src}/run_config.json" "${dst}/run_config.json" 2>/dev/null || true
        python - "${src}" "${dst}" <<'PY'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(f"{src}/model.pt", map_location="cpu")
keep = {k: v for k, v in sd.items() if not k.startswith("span_head")}
dropped = sorted(set(sd) - set(keep))
torch.save(keep, f"{dst}/model_final.pt")
print(f"  {src} -> {dst}: kept {len(keep)}, dropped {dropped}")
if not dropped:
    print("  WARNING: no span_head.* keys found -- was the span head trained?")
PY
    else
        echo "--- ${dst}: exists ---"
    fi
done

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "DUMP -- val AND test from the SAME checkpoints, in one pass"
echo "################################################################"
# This is the whole point of the rebuild. Measuring one artefact and shipping
# a different one invalidates any conclusion drawn from the measurement.
RUNS=""
for S in ${SEEDS}; do
    d="runs/risk_${TAG}p_s${S}"
    [ -f "${d}/model_final.pt" ] && RUNS="${RUNS}${RUNS:+,}${d}"
done
[ -n "${RUNS}" ] || { echo "no stripped runs; nothing to dump"; exit 1; }
echo "runs: ${RUNS}"

# `<<'PY' || FAIL=1` on one line parses, but it hides the heredoc from every
# static check (the terminator regex needs the redirect at end of line), so the
# exit status is taken on the next line instead.
python - "${RUNS}" "${OUT}" <<'PY'
import sys, os
sys.path.insert(0, "src")
import pandas as pd, torch
from data_audit import load
from predict import predict_risk
from scoring import RISK_LEVELS
runs, out = sys.argv[1], sys.argv[2]
dev = "cuda" if torch.cuda.is_available() else "cpu"
for split, path in (("val", "artifacts/val_split.csv"),
                    ("test", "data/test.csv")):
    dst = os.path.join(out, f"risk_rat3_{split}_probs.csv")
    if os.path.exists(dst):
        print(f"--- {dst} exists ---"); continue
    df = load(path)
    labels, P = predict_risk(runs, df, dev)
    o = pd.DataFrame({"row_id": df["row_id"].astype(str),
                      "pred_risk_level": labels})
    for i, l in enumerate(RISK_LEVELS):
        o[f"prob_{l}"] = P[:, i]
    o.to_csv(dst, index=False)
    print(f"wrote {dst}  {P.shape}  "
          f"{pd.Series(labels).value_counts().to_dict()}")
PY
[ $? -eq 0 ] || FAIL=1

echo ""
echo "################################################################"
echo "DONE (FAIL=${FAIL}).  CPU from here."
echo "################################################################"
cat <<'NEXT'

JUDGE FRIEND-FREE, AT MEMBER LEVEL. Do NOT judge on the full chain: the twin
cannot be made level-accurate (friend in 0.7992 / friend out 0.8294 / deployed
0.8203) because friend_risk_probs_val.csv is a different, weaker model than the
friend's test artefact and there is no checkpoint to re-dump it from.

  # the paired comparison the friend and NLI terms cancel out of:
  #   new  = mean(artifacts/rationale/risk_rat3_val_probs.csv)
  #   base = mean(runs/risk_v1{,_s14,_s15}/val_predictions.csv)
  # on the same 372 val posts, author-resampled, weighted F1.
  python src/risk_member_forward.py     # reads both as members
  python src/champion_oof_twin.py --ens-val \
      artifacts/rationale/risk_rat3_val_probs.csv --no-friend-restrict

BAR, pre-registered:
  * member level: the prior effect sits above the twin's detectability floor,
    so a real effect must be VISIBLE here. If the new mean does not beat the
    incumbent mean standalone, the lane closes on one paired read -- do not
    proceed to chain estimates, whatever they say.
  * chain estimate = member delta x ~0.35 (x0.5 friend dilution, x0.7 NLI
    weight), x0.4 for composite -- a small member-level gain becomes a much
    smaller composite one.
  * epoch selection is symmetric (both sides val-selected), so the comparison
    is fair even though both are mildly in-sample on val_split.

IF it clears: ensembled_risk_rat = 0.5*mean(risk_rat3_test_probs) + 0.5*friend,
then src/champion_risk_inject.py --verify (378/378 gate on the UNCHANGED chain
first), then --alt-ensembled. Regenerate evidence for any post that crosses the
indicator boundary -- the inject script refuses otherwise. One single-variable
change; S2 must return EXACTLY the incumbent's factor score.
NEXT
