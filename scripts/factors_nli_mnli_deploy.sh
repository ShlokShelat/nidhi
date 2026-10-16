#!/bin/bash
#SBATCH --job-name=facnlimnli
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out

# The deployed factor pipeline, EXACTLY -- with one variable changed: the
# encoder checkpoint.
#
# PROCEDURE IS HELD FIXED ON PURPOSE. An offline comparison is only informative
# when the candidate and the incumbent are built the same way. A cross-fitted
# reconstruction of the incumbent is a weaker object than the deployed 2-seed
# model, so any arm measured against that reconstruction is flattered by the
# reconstruction gap rather than by its own merit. This script therefore
# changes nothing about the procedure: same script bodies as
# scripts/factors_nli.sh and scripts/nli_seed2.sh -- same split, same
# neg-ratio, the same asymmetric epoch counts (6 and 3), the same seeds
# (13 and 14), and the same inference path that produced
# artifacts/nli_ensemble_factors.csv: average the two runs' probabilities and
# average their own val-fitted thresholds. Only --model differs.
#
# WHY THE ENCODER IS THE RIGHT VARIABLE. The task scores
# (post, "This post shows <factor>: <definition>") pairs -- an entailment
# judgement -- and the incumbent checkpoint has never seen an entailment
# example. Starting from an MNLI-pretrained checkpoint matches the pretraining
# objective to the inference objective. The effect reproduces across two
# different base architectures, so it comes from the pretraining rather than
# from one particular checkpoint.
#
# The gain to look for is in RANKING (threshold-free average precision), not in
# threshold placement: thresholds are fitted on a small validation split and do
# not transfer, so an arm whose entire advantage sits in threshold space is not
# a candidate here even when its cross-fitted macro-F1 is higher.
#
# The risk column is untouched, so a factors-only patch isolates the factor
# sub-score.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# published definitions, hand-written confusable map, no protective template --
# exactly what the deployed pair was trained with
unset PROTECTIVE_TEMPLATE || true

# FACTOR_DEFS_JSON is opt-in. It overrides individual factor DEFINITIONS, which
# `hypothesis_for()` reads at BOTH training and inference, so it must be set for
# both -- exporting it here covers the training runs and the predict step below.
#
# WHY IT MATTERS. Three of the 24 definitions in FACTOR_DEFINITIONS are NOT the
# organizers' Table III wording (arXiv 2507.10008):
#   meaning in life        code "a sense of purpose, goals, or reasons for living"
#                          paper "Involves cognitive component, motivational
#                                 component and affective component"
#   psychological capital  code adds "such as resilience or optimism", which
#                          appears nowhere in the source
#   sexual orientation     code "gender or sexual identity issues"
#                          paper "Having gender/sexual disorder"
# The ANNOTATORS labelled against the paper's strings, so a paraphrased
# hypothesis asks a different question than the one that was annotated. A
# label whose hypothesis drifts from the annotation guideline is indis-
# tinguishable from a label that is simply hard, which is why the two are
# separated by swapping the definitions rather than by inspecting scores.
#
#   sbatch --export=ALL,TAG=mnlidef,FACTOR_DEFS_JSON=artifacts/defs_official.json \
#          scripts/factors_nli_mnli_deploy.sh
#
# CAUTION: the paper's `meaning in life` wording is abstract psychometric jargon
# with little lexical contact with post text, so it may score WORSE as a
# cross-encoder hypothesis even though it is what the annotators used. That is
# the question this arm answers; it is not a foregone win.
if [ -n "${FACTOR_DEFS_JSON:-}" ]; then
    [ -f "${FACTOR_DEFS_JSON}" ] || {
        echo "ERROR: FACTOR_DEFS_JSON=${FACTOR_DEFS_JSON} not found"; exit 1; }
    export FACTOR_DEFS_JSON
    echo "definitions: ${FACTOR_DEFS_JSON} (override)"
else
    unset FACTOR_DEFS_JSON || true
    echo "definitions: published (FACTOR_DEFINITIONS)"
fi

# CONFUSABLE_JSON is opt-in. Training-only -- predict_factors_nli.py never
# reads it -- so it cannot cause train/serve skew the way the protective
# template can, and a bag may mix mined and hand-written members safely.
#
# The mined map supplies hard negatives drawn from the model's own confusions,
# so training pressure lands on the factor pairs the model actually mixes up
# rather than on pairs a human guessed would be confusable.
#
#   sbatch --export=ALL,TAG=mnlicm,CONFUSABLE_JSON=artifacts/confusable_mined.json \
#          scripts/factors_nli_mnli_deploy.sh
#
# NOTE the aggregate map is mined from all 1635 OOF posts, which include
# val_split -- so the val-fitted thresholds see a map informed by their own
# posts. The channel is 24 rows of factor NAMES, not labels, so the leakage is
# small, but it is real and is recorded here rather than left implicit.
if [ -n "${CONFUSABLE_JSON:-}" ]; then
    [ -f "${CONFUSABLE_JSON}" ] || {
        echo "ERROR: CONFUSABLE_JSON=${CONFUSABLE_JSON} not found"; exit 1; }
    export CONFUSABLE_JSON
    echo "confusable map: ${CONFUSABLE_JSON} (mined)"
else
    unset CONFUSABLE_JSON || true
    echo "confusable map: hand-written"
fi

MODEL="${MODEL:-MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"
TAG="${TAG:-mnli}"

for f in artifacts/train_split.csv artifacts/val_split.csv data/test.csv; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing"; exit 1; }
done

echo "model : ${MODEL}"
echo "mirrors scripts/factors_nli.sh (seed 13, 6 epochs)"
echo "    and scripts/nli_seed2.sh   (seed 14, 3 epochs)"

# ---- seed 13, 6 epochs -- mirrors scripts/factors_nli.sh --------------------
if [ -f "runs/factors_nli_${TAG}/model.pt" ]; then
    echo "--- runs/factors_nli_${TAG} exists, skipping ---"
else
    echo ""
    echo "================================================================"
    echo "seed 13, 6 epochs -> runs/factors_nli_${TAG}"
    echo "================================================================"
    python src/train_factors_nli.py \
        --train artifacts/train_split.csv --val artifacts/val_split.csv \
        --model "${MODEL}" --outdir "runs/factors_nli_${TAG}" \
        --neg-ratio 3 --epochs 6 \
        || { echo "FAILED seed 13"; exit 1; }
fi

# ---- seed 14, 3 epochs -- mirrors scripts/nli_seed2.sh ---------------------
if [ -f "runs/factors_nli_${TAG}_s2/model.pt" ]; then
    echo "--- runs/factors_nli_${TAG}_s2 exists, skipping ---"
else
    echo ""
    echo "================================================================"
    echo "seed 14, 3 epochs -> runs/factors_nli_${TAG}_s2"
    echo "================================================================"
    python src/train_factors_nli.py \
        --train artifacts/train_split.csv --val artifacts/val_split.csv \
        --model "${MODEL}" --outdir "runs/factors_nli_${TAG}_s2" \
        --neg-ratio 3 --epochs 3 --seed 14 \
        || { echo "FAILED seed 14"; exit 1; }
fi

# ---- test prediction, the deployed way -------------------------------------
# No --thresholds override. predict_factors_nli.py averages the two runs'
# probabilities AND averages their own val-fitted thresholds -- byte-for-byte
# the path that produced artifacts/nli_ensemble_factors.csv.
echo ""
echo "================================================================"
echo "test prediction -- averaged probs, averaged val-fitted thresholds"
echo "================================================================"
python src/predict_factors_nli.py \
    --test data/test.csv \
    --nli-run "runs/factors_nli_${TAG},runs/factors_nli_${TAG}_s2" \
    --dump-probs "artifacts/factor_runs/nli_ensemble_${TAG}_test.npz" \
    --out "artifacts/nli_ensemble_factors_${TAG}.csv" \
    || { echo "FAILED prediction"; exit 1; }

# ---- compare against the deployed factor column ----------------------------
echo ""
echo "================================================================"
echo "vs the deployed factor column"
echo "================================================================"
TAG="${TAG}" python - <<'PY'
import ast, os, sys
import numpy as np
import pandas as pd
from collections import Counter
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

TAG = os.environ["TAG"]
old = pd.read_csv("artifacts/nli_ensemble_factors.csv", keep_default_na=False)
new = pd.read_csv(f"artifacts/nli_ensemble_factors_{TAG}.csv", keep_default_na=False)
old = old.set_index("row_id").loc[new["row_id"]].reset_index()

co = [len(ast.literal_eval(x)) for x in old["factors"]]
cn = [len(ast.literal_eval(x)) for x in new["factors"]]
lab = pd.read_csv("artifacts/full_train.csv")
TAX = set(FACTOR_TAXONOMY)
g = [len({str(f).strip().lower() for f in parse_factors(x)} & TAX)
     for x in lab["factors"]]
print(f"deployed  mean {np.mean(co):.2f} factors/post   empty {sum(1 for c in co if c==0)}")
print(f"new       mean {np.mean(cn):.2f} factors/post   empty {sum(1 for c in cn if c==0)}")
print(f"gold      mean {np.mean(g):.2f} factors/post   "
      f"empty {sum(1 for c in g if c==0)}  [unique factors]")
diff = int((old["factors"] != new["factors"]).sum())
print(f"\nposts whose factor set changed: {diff} of {len(new)} "
      f"({100.0*diff/len(new):.1f}%)")
print("  Under ~40 and the change cannot be resolved on a 378-post test set.")

A, B = Counter(), Counter()
for x in old["factors"]:
    A.update(ast.literal_eval(x))
for x in new["factors"]:
    B.update(ast.literal_eval(x))
print(f"\n{'factor':<44}{'gold%':>7}{'depl':>6}{'new':>6}{'delta':>7}")
print("-" * 72)
zero = []
for t in FACTOR_TAXONOMY:
    gp = 100.0 * sum(1 for x in lab["factors"] if t in parse_factors(x)) / len(lab)
    flag = ""
    if B[t] == 0 and gp > 0:
        flag = "  <-- ZERO, F1=0 outright"
        zero.append(t)
    print(f"{t:<44}{gp:>7.1f}{A[t]:>6}{B[t]:>6}{B[t]-A[t]:>+7}{flag}")
if zero:
    print(f"\n*** {len(zero)} label(s) predicted zero times. Each costs 1/24 of")
    print("    macro-F1 outright. Do not ship without checking these.")
else:
    print("\nno label predicted zero times.")
PY

echo ""
echo "DONE -> artifacts/nli_ensemble_factors_${TAG}.csv"
echo ""
echo "Patch onto the base submission (risk and evidence untouched, so the risk"
echo "sub-score must return unchanged and any factor move is this change alone):"
echo ""
echo "  python src/patch_submission_risk.py \\"
echo "      --base submissions/NIDHI_evidence_vote8.csv \\"
echo "      --factors artifacts/nli_ensemble_factors_${TAG}.csv \\"
echo "      --out submissions/NIDHI_factors_nlimnli.csv"
