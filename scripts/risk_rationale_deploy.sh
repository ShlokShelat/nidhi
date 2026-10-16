#!/bin/bash
#SBATCH --job-name=ratdep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# CPUS: src/train_risk_rationale.py builds BOTH DataLoaders with num_workers=0
# (lines 256-257), so everything loads in the main process and one core is
# enough. It does NOT inherit train_risk.py's --num-workers, so requesting more
# cores only makes the job queue for an allocation it cannot use. The factor
# scripts are different -- train_factors_nli.py uses num_workers=2, so those
# need 3-4.

# Rationale supervision in the DEPLOYED shape: three seeds on train_split,
# mirroring risk_v1 / risk_v1_s14 / risk_v1_s15 exactly. Only the loss changes.
#
# WHY THREE SEEDS AND NOT A 5-FOLD BAG. A bag of fold models cannot be
# validated out of fold: for any post, four of the five members trained on it,
# so no held-out reading of the shipped object exists. The three-seed shape
# trains every member on the same train_split, which keeps val_split held out
# for all of them and keeps the shipped object the same object that was
# measured.
#
# WHY SPAN SUPERVISION. Restricted to the gold evidence spans, the confident
# ideation<->behavior confusions are separable at 0.692 accuracy where the full
# post reaches 0.308: the discriminative text is present but diluted by the
# rest of the post. An auxiliary span-BCE head pushes the encoder to attend to
# it.
#
# This changes ONE THING against the deployed pipeline:
# `loss = (1-a)CE + a*CORN` becomes `loss = (1-a)CE + a*CORN + lambda*spanBCE`.
# Same architecture, same three seeds, same train_split, same epoch selection
# on val_split, same 0.5/0.5 friend ensemble, same 0.70/0.30 NLI blend, same
# two rules -- one component swapped inside a frozen pipeline, so any change in
# the result is attributable to the loss term alone.
#
# NO OFFLINE NUMBER IS PRODUCED FOR IT, DELIBERATELY. val_split drives epoch
# selection here exactly as it does for the deployed runs, so a val score would
# be in-sample for both arms. With 153 authors the out-of-fold instrument has a
# +/-0.0157 null band, which is wider than the effect being looked for, so a
# reading from it would select on noise.
#
# Span masks: 96.0% of gold spans locate in the post (96-98% for
# ideation/behavior/attempt). `indicator` posts get near-empty masks, which is
# correct supervision -- 95.4% of them have gold evidence "none".

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# Without this, torch sizes its intra-op thread pool from the machine's total
# core count (48) regardless of how few cores Slurm actually granted, and the
# threads thrash against each other on a 2-CPU allocation.
NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${NTHREADS}" MKL_NUM_THREADS="${NTHREADS}"

LAM="${LAM:-0.5}"
TAG="${TAG:-ratdep}"
MODEL="${MODEL:-microsoft/deberta-v3-large}"

for f in artifacts/train_split.csv artifacts/val_split.csv data/test.csv; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing"; exit 1; }
done
echo "lambda_span=${LAM}   model=${MODEL}"
echo "mirrors the deployed risk_v1 / _s14 / _s15 (seeds 13/14/15)"

# ---- three seeds, exactly the deployed trio ---------------------------------
for SEED in 13 14 15; do
    OUT="runs/risk_${TAG}_s${SEED}"
    if [ -f "${OUT}/model.pt" ]; then
        echo "--- ${OUT} exists, skipping ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "seed ${SEED} -> ${OUT}"
    echo "================================================================"
    python src/train_risk_rationale.py \
        --train artifacts/train_split.csv --val artifacts/val_split.csv \
        --model "${MODEL}" --outdir "${OUT}" \
        --lambda-span "${LAM}" --seed "${SEED}" \
        || { echo "FAILED seed ${SEED}"; exit 1; }
done

# ---- src/predict.py needs model_final.pt and rejects span_head.* ------------
# Same two facts as scripts/build_risk_rationale_submission.sh: load_state reads
# model_final.pt while the trainer writes model.pt, and it RAISES on unexpected
# state-dict keys. predict.py sits on the deployed path and is NOT edited.
echo ""
echo "=== strip span_head -> model_final.pt ==="
TAG="${TAG}" python - <<'PY' || exit 1
import json, os
import torch
TAG = os.environ["TAG"]
for s in (13, 14, 15):
    d = f"runs/risk_{TAG}_s{s}"
    sd = torch.load(os.path.join(d, "model.pt"), map_location="cpu")
    dropped = [k for k in sd if k.startswith("span_head")]
    if not dropped:
        raise SystemExit(f"{d}: no span_head.* -- not a rationale checkpoint?")
    for k in dropped:
        sd.pop(k)
    torch.save(sd, os.path.join(d, "model_final.pt"))
    p = os.path.join(d, "run_config.json")
    cfg = json.load(open(p))
    for need in ("model", "max_len", "alpha_ordinal", "user_context"):
        if need not in cfg:
            raise SystemExit(f"{d}: run_config.json lacks {need!r}")
    if not isinstance(cfg["user_context"], bool):
        cfg["user_context"] = bool(cfg["user_context"])
        json.dump(cfg, open(p, "w"), indent=2)
    print(f"  {d}: dropped {dropped}  best_val="
          f"{cfg.get('best_val_weighted_f1')}")
PY

RUNS="runs/risk_${TAG}_s13,runs/risk_${TAG}_s14,runs/risk_${TAG}_s15"

echo ""
echo "=== 0.5/0.5 with the friend, exactly as the deployed chain does ==="
python src/ensemble_risk.py \
    --test data/test.csv --our-risk-run "${RUNS}" \
    --friend-probs artifacts/friend_risk_probs.csv \
    --out "artifacts/ensembled_risk_${TAG}.csv" \
    || { echo "FAILED ensemble"; exit 1; }

echo ""
echo "=== blend with the SAME two NLI members at a FORCED 0.30 ==="
# --ens-val gets THESE runs' own val predictions, so the printed sweep refers
# to the models actually being blended -- but the weight is still FORCED, never
# fitted, because fitting it on val would tune a second knob at the same time.
python src/blend_risk_saved.py \
    --val artifacts/val_split.csv --test data/test.csv \
    --nli-run runs/risk_nli,runs/risk_nli_s2 \
    --ens-val "runs/risk_${TAG}_s13/val_predictions.csv,runs/risk_${TAG}_s14/val_predictions.csv,runs/risk_${TAG}_s15/val_predictions.csv" \
    --ens-test "artifacts/ensembled_risk_${TAG}.csv" \
    --force-weight 0.30 \
    --out "artifacts/risk_blend_${TAG}.csv" \
    || { echo "NOTE blend failed -- if val_predictions.csv is missing, rerun"; \
         echo "with --ens-val pointing at runs/risk_v1/... (reporting only)"; \
         exit 1; }

echo ""
echo "=== the same two rules ==="
python src/apply_rules_to_test.py \
    --test data/test.csv \
    --risk-blend "artifacts/risk_blend_${TAG}.csv" \
    --rules urgency,completed_act \
    --out "artifacts/risk_blend_${TAG}_rules.csv" \
    || { echo "FAILED rules"; exit 1; }

echo ""
echo "=== patch the risk column onto the base submission ==="
python src/patch_submission_risk.py \
    --base submissions/NIDHI_canon_on_nlimnli.csv \
    --risk "artifacts/risk_blend_${TAG}_rules.csv" \
    --max-changes 150 \
    --out "submissions/NIDHI_risk_${TAG}.csv" \
    || { echo "FAILED patch"; exit 1; }

echo ""
echo "DONE -> submissions/NIDHI_risk_${TAG}.csv"
echo "The factor sub-score must come back unchanged and evidence must be"
echo "untouched, so any risk move is this loss-function change alone."
