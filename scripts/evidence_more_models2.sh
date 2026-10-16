#!/bin/bash
#SBATCH --job-name=evmore2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# Eight MORE evidence taggers, to widen the vote pool.
#
# THE VOTE RULE IS SIZE-INDEPENDENT. The optimum is vote>=2 at four voters (50%)
# and still vote>=2 at eight (25%), so the rule is not a fraction of the pool --
# it is "DROP SPANS NO OTHER MODEL CORROBORATES". One second opinion suffices;
# vote>=3 and above begin discarding good spans, with recall falling
# 0.7291 -> 0.6704 -> 0.6285 across thresholds 2, 3 and 4.
#
# THAT RULE IMPROVES WITH POOL SIZE, in both directions at once. A bad span
# survives only if two models independently make the SAME mistake, and the
# chance of a shared mistake falls as the pool grows and diversifies. A good span
# needs only one corroborator, which gets EASIER as the pool grows. Adding voters
# therefore tightens precision and loosens recall simultaneously.
#
# WEIGHTED TOWARD REGIME CHANGES, NOT SEEDS. Seed-only variants cluster tightly
# in val phrase-F1, so they corroborate each other's mistakes; regime changes
# (learning rate, sequence length, batch size, epochs) land outside that cluster
# and are the variants that make agreement informative.
#
# ~35 min each, so roughly 5 hours.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
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
        --num-workers 0 \
        "$@" || { echo "FAILED ${name} (continuing)"; return 0; }
}

# higher learning rate was the strongest single-model regime -- explore around it
run evidence_lr2b --seed 31 --max-len 512 --epochs 6 --lr 2e-5 --batch 8 --grad-accum 2
run evidence_lr3  --seed 32 --max-len 512 --epochs 6 --lr 3e-5 --batch 8 --grad-accum 2

# length regime: shorter context is a genuinely different view of the post
run evidence_ml256 --seed 33 --max-len 256 --epochs 6 --lr 1e-5 --batch 16 --grad-accum 1
run evidence_ml384 --seed 34 --max-len 384 --epochs 6 --lr 2e-5 --batch 8  --grad-accum 2

# batch/effective-LR regime
run evidence_bs16 --seed 35 --max-len 512 --epochs 6 --lr 2e-5 --batch 16 --grad-accum 1
run evidence_bs4  --seed 36 --max-len 512 --epochs 6 --lr 1e-5 --batch 4  --grad-accum 4

# training length
run evidence_ep12 --seed 37 --max-len 512 --epochs 12 --lr 1e-5 --batch 8 --grad-accum 2
run evidence_ep4  --seed 38 --max-len 512 --epochs 4  --lr 2e-5 --batch 8 --grad-accum 2

echo ""
echo "================================================================"
echo "ALL EVIDENCE RUNS"
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
                 c.get("evidence_min_conf"), c.get("lr"), c.get("epochs"),
                 c.get("max_len")))
print(f"{'run':<20}{'val F1':>9}{'min_conf':>10}{'lr':>9}{'ep':>5}{'len':>6}")
for n, f, mc, lr, ep, ml in sorted(rows, key=lambda r: -r[1]):
    print(f"{n:<20}{f:>9.4f}{str(mc):>10}{str(lr):>9}{str(ep):>5}{str(ml):>6}")
names = ",".join(n for n, *_ in rows if n != "evidence_nli")
print(f"\n{len(rows)} runs. Next:\n  python src/evidence_union.py --runs {names}")
PY
