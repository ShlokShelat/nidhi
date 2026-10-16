#!/bin/bash
#SBATCH --job-name=evmore
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# Train four more evidence taggers so the VOTE has more voters.
#
# WHY MORE VOTERS. Span agreement across taggers is what lifts evidence F1, and
# it does so by RAISING PRECISION -- agreement discards more bad spans than good
# ones. On val, across four models:
#
#     evidence_v2 alone   0.7118   P 0.7516  R 0.6760
#     vote>=2             0.7221   P 0.7862  R 0.6676
#     union all           0.6926
#     vote>=3             0.6678
#
# The threshold is coarse at four voters: vote>=3 is a 75% quorum and recall
# collapses to 0.55. With eight voters a 3-or-4 threshold is a far better
# resolved consensus, and the vote count stops being a choice between two
# extremes.
#
# DIVERSITY, NOT JUST SEEDS. Taggers that share a recipe and differ only by seed
# contribute few genuinely new spans, because agreement between near-identical
# models measures initialisation noise rather than evidence. Two of these vary
# the seed and two vary the training regime (learning rate, epochs/length), so
# the added voters disagree for substantive reasons.
#
# Base config, from runs/evidence_v2/run_config.json:
#     deberta-v3-large, max_len 512, epochs 6, lr 1e-5, batch 8, grad_accum 2
#
# ~30-40 min each, so about 2.5 hours total.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

TRAIN=artifacts/train_split.csv
VAL=artifacts/val_split.csv

run () {
    local name=$1; shift
    if [ -f "runs/${name}/val_predictions.csv" ]; then
        echo "--- ${name}: already done, skipping ---"
        return 0
    fi
    echo ""
    echo "================================================================"
    echo "${name}   $*"
    echo "================================================================"
    python src/train_evidence.py \
        --train "${TRAIN}" --val "${VAL}" \
        --model microsoft/deberta-v3-large \
        --outdir "runs/${name}" \
        --batch 8 --grad-accum 2 --num-workers 0 \
        "$@" || { echo "FAILED ${name}"; return 1; }
}

# two more seeds of the base recipe
run evidence_s16 --seed 16 --max-len 512 --epochs 6 --lr 1e-5
run evidence_s17 --seed 17 --max-len 512 --epochs 6 --lr 1e-5

# two genuine regime changes -- these should disagree with the others for
# reasons other than initialisation
run evidence_lr2 --seed 13 --max-len 512 --epochs 6 --lr 2e-5
run evidence_ep9 --seed 21 --max-len 512 --epochs 9 --lr 1e-5

echo ""
echo "================================================================"
echo "VAL PHRASE-F1 PER RUN"
echo "================================================================"
python - <<'PY'
import glob, json, os
rows = []
for p in sorted(glob.glob("runs/evidence_*/run_config.json")):
    c = json.load(open(p))
    f1 = c.get("best_val_phrase_f1")
    if f1 is None:
        continue
    rows.append((os.path.basename(os.path.dirname(p)), f1,
                 c.get("evidence_min_conf"), c.get("seed"), c.get("lr"),
                 c.get("epochs")))
print(f"{'run':<18}{'val F1':>9}{'min_conf':>10}{'seed':>6}{'lr':>9}{'ep':>4}")
for n, f, mc, s, lr, ep in sorted(rows, key=lambda r: -r[1]):
    print(f"{n:<18}{f:>9.4f}{str(mc):>10}{str(s):>6}{str(lr):>9}{str(ep):>4}")
print("\nA weak member is NOT a reason to exclude it from the vote -- the vote")
print("uses agreement, not quality. Judge the ensemble, not the members.")
PY

echo ""
echo "Next (GPU, dump candidates for the new runs too):"
echo "  sbatch --export=ALL,RUNS=evidence_v2,evidence_v1,evidence_s14,evidence_s15,evidence_s16,evidence_s17,evidence_lr2,evidence_ep9 scripts/evidence_dump.sh"
echo "Then (CPU): python src/evidence_union.py --runs <same list>"
