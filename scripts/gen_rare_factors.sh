#!/bin/bash
#SBATCH --job-name=genrare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Generate synthetic posts for the THIRTEEN starved factor labels.
#
# The previous synthetic run (gpt5_generated_all_targeted.csv, 1720 posts)
# covered only the nine factors that already had 200-745 real examples, and
# generated ZERO for the thirteen with 8-118. That is why it did not help.
#
# S2 is macro-F1: `sexual orientation related issues` has EIGHT examples and
# costs the same 1/24 of the score as hopelessness with 745.
#
# Generation runs on the LOCAL Qwen-14B, so nothing leaves the cluster and no
# API key is involved.

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
if [ ! -f "${MODEL}/config.json" ]; then
    echo "ERROR: ${MODEL}/config.json not found (compute nodes are offline)."
    exit 1
fi

python src/gen_rare_factors.py \
    --labeled artifacts/full_train.csv \
    --model "${MODEL}" \
    --target-count 150 \
    --max-per-factor 150 \
    --batch 8 --max-new 220 --temperature 0.95 \
    --out artifacts/syn_rare_factors.csv

echo ""
echo "================================================================"
echo "COVERAGE CHECK -- real vs synthetic, per factor"
echo "================================================================"
python - <<'PY'
import collections, sys
import pandas as pd
sys.path.insert(0, "src")
from scoring import FACTOR_TAXONOMY, parse_factors

real = pd.read_csv("artifacts/full_train.csv")
rc = collections.Counter(f for c in real["factors"]
                         for f in set(parse_factors(c)) & set(FACTOR_TAXONOMY))
syn = pd.read_csv("artifacts/syn_rare_factors.csv")
sc = collections.Counter(syn["target_factor"])
print(f"{'factor':<44}{'real':>6}{'syn':>6}{'total':>7}")
for f in sorted(FACTOR_TAXONOMY, key=lambda x: rc.get(x, 0)):
    print(f"{f:<44}{rc.get(f,0):>6}{sc.get(f,0):>6}{rc.get(f,0)+sc.get(f,0):>7}")
print("\nThe previous attempt left every one of the bottom thirteen at zero.")
PY

echo ""
echo "NEXT: retrain d6 five-fold WITH the synthetic data as --external,"
echo "fit thresholds on REAL out-of-fold posts only, and compare the"
echo "per-label F1 of the starved labels against the incumbent."
echo "Synthetic data in training only -- never in thresholds or evaluation."
