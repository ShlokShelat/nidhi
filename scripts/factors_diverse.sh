#!/bin/bash
#SBATCH --job-name=facdiv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

# Eight DIVERSE factor models, then dump their val and test probabilities.
#
# WHY AN ENSEMBLE OF REGIMES, NOT OF SEEDS. An ensemble only helps to the extent
# its members make DIFFERENT errors. Seed variants of one recipe converge to
# nearly the same decision surface and their votes are close to duplicates, so
# averaging them mostly cancels sampling noise and leaves the systematic errors
# in place. artifacts/ensembled_factors_seeds.csv and nli_ens3_factors.csv are
# seed ensembles and behave that way. Changing the training REGIME -- objective,
# learning rate, sequence length, negative sampling -- moves the decision
# surface itself, and those members land outside the pack.
#
# The same structure holds on the evidence taggers, where the members that
# differ by regime rather than by seed are the ones that change the vote.
#
# The --loss axis matters most here: bce, per-label-weighted bce and asymmetric
# loss weight rare positives differently, so they disagree about rare labels for
# structural rather than random reasons -- and rare labels are where the factor
# deficit is concentrated.
#
# Defaults, from train_factors_nli.py:
#   max_len 320, epochs 3, lr 1e-5, batch 16, grad_accum 2,
#   neg_ratio 3, hard_frac 0.5, loss bce
#
# ~40 min each plus dumps, so roughly 7 hours.

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

TRAIN=artifacts/train_split.csv
VAL=artifacts/val_split.csv
TEST=data/test.csv
OUT=artifacts/factor_runs
mkdir -p "${OUT}"

train_one () {
    local name=$1; shift
    if [ ! -f "runs/${name}/model.pt" ]; then
        echo ""
        echo "================================================================"
        echo "TRAIN ${name}   $*"
        echo "================================================================"
        python src/train_factors_nli.py \
            --train "${TRAIN}" --val "${VAL}" \
            --model microsoft/deberta-v3-large \
            --outdir "runs/${name}" \
            "$@" || { echo "FAILED ${name} (continuing)"; return 0; }
    else
        echo "--- ${name}: model exists, skipping training ---"
    fi

    # probabilities for both splits, so the ensemble work afterwards is CPU
    for split in val test; do
        local src="${VAL}"; [ "${split}" = "test" ] && src="${TEST}"
        local dst="${OUT}/${name}_${split}.npz"
        [ -f "${dst}" ] && { echo "  ${dst} exists"; continue; }
        python src/predict_factors_nli.py \
            --test "${src}" --nli-run "runs/${name}" \
            --dump-probs "${dst}" \
            --out "${OUT}/${name}_${split}_labels.csv" \
            || echo "  dump failed: ${name} ${split}"
    done
}

# --- regime changes, NOT seeds -------------------------------------------
train_one factors_d1 --seed 13 --lr 2e-5 --epochs 3 --max-len 320
train_one factors_d2 --seed 13 --lr 1e-5 --epochs 3 --max-len 320 --loss asl
train_one factors_d3 --seed 13 --lr 1e-5 --epochs 3 --max-len 320 --loss bce_perlabel
train_one factors_d4 --seed 13 --lr 1e-5 --epochs 5 --max-len 320
train_one factors_d5 --seed 13 --lr 1e-5 --epochs 3 --max-len 320 --neg-ratio 5
train_one factors_d6 --seed 21 --lr 2e-5 --epochs 3 --max-len 384
train_one factors_d7 --seed 22 --lr 1e-5 --epochs 3 --max-len 320 --hard-frac 0.8
train_one factors_d8 --seed 23 --lr 3e-5 --epochs 4 --max-len 256 --loss asl

echo ""
echo "================================================================"
echo "DUMPED PROBABILITY FILES"
echo "================================================================"
ls -1 "${OUT}"/*_val.npz 2>/dev/null | wc -l
ls -lh "${OUT}"/*.npz 2>/dev/null | tail -20

echo ""
echo "Also dump the INCUMBENT so the ensemble has the shipped model in it:"
echo "  python src/predict_factors_nli.py --test artifacts/val_split.csv \\"
echo "      --nli-run runs/factors_nli --dump-probs ${OUT}/factors_nli_val.npz \\"
echo "      --out ${OUT}/factors_nli_val_labels.csv"
echo ""
echo "Then (CPU): python src/factor_ensemble.py"
