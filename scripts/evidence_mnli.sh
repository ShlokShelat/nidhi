#!/bin/bash
#SBATCH --job-name=evmnli
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

# Train evidence taggers on an MNLI-pretrained encoder instead of plain DeBERTa.
#
# WHY THIS CHANGE. The evidence bag is eight taggers on plain
# microsoft/deberta-v3-large, four of which (v2, v1, s14, s15) share one
# configuration -- 6 epochs, lr 1e-5, batch 8, max_len 512 -- and differ only by
# seed, so the bag's diversity is seed noise. An ensemble gains from members
# whose errors are decorrelated, and different pretraining is a stronger source
# of decorrelation than a different seed. MNLI pretraining also matches the task
# shape: span selection is an entailment-like judgement about whether a clause
# supports a label.
#
# WHY STAGED. Replacing all eight means eight training runs. This trains THREE
# that mirror the three strongest existing taggers exactly -- same epochs, lr,
# batch, max_len, seed -- changing only --model, so the encoder is the single
# free variable. Each new run's best_val_phrase_f1 is then directly comparable
# to its matched twin:
#     evidence_v2   seed 13   0.6902
#     evidence_s14  seed 14   0.6969
#     evidence_s15  seed 15   0.6749
#
# These are head-to-head MEMBER comparisons on 372 val posts, not a claim about
# the bag. The vote is what is scored, and vote>=2 over eight members is the
# established decode, so member quality is the lever this script tests.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${NTHREADS}" MKL_NUM_THREADS="${NTHREADS}"

MODEL="${MODEL:-MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"
TAG="${TAG:-mnli}"
SEEDS="${SEEDS:-13 14 15}"

for f in artifacts/train_split.csv artifacts/val_split.csv; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing"; exit 1; }
done
echo "model: ${MODEL}"
echo "mirroring evidence_v2 / _s14 / _s15 exactly, changing only --model"
echo "their val phrase-F1 to beat: 0.6902 / 0.6969 / 0.6749"

for SEED in ${SEEDS}; do
    OUT="runs/evidence_${TAG}_s${SEED}"
    if [ -f "${OUT}/model.pt" ]; then
        echo "--- ${OUT} exists, skipping ---"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "seed ${SEED} -> ${OUT}"
    echo "================================================================"
    # every hyperparameter below is copied from runs/evidence_v2/run_config.json
    python src/train_evidence.py \
        --train artifacts/train_split.csv --val artifacts/val_split.csv \
        --model "${MODEL}" --outdir "${OUT}" \
        --epochs 6 --lr 1e-5 --batch 8 --grad-accum 2 --max-len 512 \
        --seed "${SEED}" \
        || { echo "FAILED seed ${SEED}"; exit 1; }
done

echo ""
echo "================================================================"
echo "HEAD TO HEAD -- same config, same val, only the encoder differs"
echo "================================================================"
TAG="${TAG}" python - <<'PY'
import json
import os
TAG = os.environ["TAG"]
base = {13: ("evidence_v2", 0.6902), 14: ("evidence_s14", 0.6969),
        15: ("evidence_s15", 0.6749)}
print(f"{'seed':>5}{'plain deberta':>16}{'mnli':>9}{'delta':>9}")
print("-" * 40)
wins = 0
n = 0
for s, (ref, f1) in base.items():
    p = f"runs/evidence_{TAG}_s{s}/run_config.json"
    if not os.path.exists(p):
        print(f"{s:>5}{f1:>16.4f}{'--':>9}   not trained")
        continue
    got = json.load(open(p)).get("best_val_phrase_f1")
    if got is None:
        print(f"{s:>5}{f1:>16.4f}{'--':>9}   no val score recorded")
        continue
    n += 1
    wins += got > f1
    print(f"{s:>5}{f1:>16.4f}{got:>9.4f}{got - f1:>+9.4f}   vs {ref}")
if n:
    print(f"\nMNLI wins {wins} of {n} matched pairs.")
    print("372 val posts, so a single pair proves nothing -- look for a")
    print("CONSISTENT sign across all three. 2 of 3 is a coin flip; 3 of 3")
    print("with a mean gain above ~0.01 is worth training the other five.")
    print("\nIf it wins: train seeds 16, 17 and the lr2/ep9 variants, then")
    print("  python src/evidence_vote_test.py --runs <the 8 new ones> ...")
    print("If it loses: keep the existing plain-DeBERTa taggers; the encoder")
    print("  swap was settled with three runs instead of eight.")
PY
