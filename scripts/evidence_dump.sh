#!/bin/bash
#SBATCH --job-name=evdump
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# Dump every evidence run's candidate spans WITH confidences, on val and test.
#
# RUNS ARE DISCOVERED FROM DISK, NOT PASSED IN. An earlier version took a
# comma-separated RUNS variable, and
#     sbatch --export=ALL,RUNS=evidence_v2,evidence_v1,...
# silently dumped ONLY evidence_v2 -- because sbatch --export uses commas as
# its own separator, so everything after the first comma was parsed as
# separate (invalid) variable assignments. The job succeeded, the vote was
# built from 4 runs instead of 8, and nothing complained.
#
# Discovering the list here removes the failure mode entirely. To restrict it,
# set RUNS with SPACES (not commas):
#     sbatch --export=ALL,RUNS="evidence_v2 evidence_lr2" scripts/evidence_dump.sh

set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

if [ -z "${RUNS:-}" ]; then
    # every evidence_* run with a BIO-tagger checkpoint.
    # evidence_nli is excluded: it is a span SCORER (max_len 64), not a
    # TaggerModel, so collect_candidates cannot load it.
    RUNS=""
    for d in runs/evidence_*; do
        n=$(basename "$d")
        [ "$n" = "evidence_nli" ] && continue
        if [ -f "$d/model_final.pt" ] || [ -f "$d/model.pt" ]; then
            RUNS="${RUNS}${RUNS:+ }${n}"
        fi
    done
fi

echo "runs to dump: ${RUNS}"
echo "count: $(echo ${RUNS} | wc -w)"
echo ""

python src/evidence_dump_all.py \
    --runs "$(echo ${RUNS} | tr ' ' ',')" \
    --test data/test.csv \
    --val artifacts/val_split.csv \
    --outdir artifacts/evidence_runs

echo ""
echo "candidate files now present:"
ls artifacts/evidence_runs/*_test_candidates.csv 2>/dev/null | wc -l
