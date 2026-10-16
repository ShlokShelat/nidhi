#!/bin/bash
#SBATCH --job-name=reproduce-compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

python3 <<'PYEOF'
import sys
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from train_factors_nli import NLIScorer, score_all, tune_thresholds, gold_matrix, FACTOR_TAXONOMY, N

device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
va_df = pd.read_csv("artifacts/val_split.csv")
gold = gold_matrix(va_df)

def score_run(run_dir):
    mn = open(f"{run_dir}/tok").read().strip()
    tok = AutoTokenizer.from_pretrained(mn)
    m = NLIScorer(mn).to(device)
    m.load_state_dict(torch.load(f"{run_dir}/model.pt", map_location=device), strict=False)
    m.eval()
    probs = score_all(m, va_df, tok, 320, device, use_bf16)
    ths = np.load(f"{run_dir}/thresholds.npy")
    del m
    if device == "cuda": torch.cuda.empty_cache()
    per_factor = {}
    for j in range(N):
        pred = (probs[:, j] >= ths[j]).astype(int)
        tp = int(((pred==1)&(gold[:,j]==1)).sum()); fp = int(((pred==1)&(gold[:,j]==0)).sum()); fn = int(((pred==0)&(gold[:,j]==1)).sum())
        f1 = (2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
        per_factor[FACTOR_TAXONOMY[j]] = f1
    return per_factor

TARGETED = {"poor social support","psychological capital","interpersonal difficulty","hopelessness",
            "prior self-harm or suicidal thought/attempt","suicide means (with access)",
            "mental health issues","stressful life event"}

runs = {"baseline": "runs/factors_nli", "seed42": "runs/factors_nli_reproduce_s42",
        "seed123": "runs/factors_nli_reproduce_s123", "seed999": "runs/factors_nli_reproduce_s999"}
results = {name: score_run(path) for name, path in runs.items()}

print(f"{'factor':<48}{'baseline':>9}{'s42':>9}{'s123':>9}{'s999':>9}{'mean(seeds)':>12}{'std':>8}")
for f in TARGETED:
    vals = [results[s][f] for s in ["seed42","seed123","seed999"]]
    print(f"{f:<48}{results['baseline'][f]:>9.4f}{vals[0]:>9.4f}{vals[1]:>9.4f}{vals[2]:>9.4f}{np.mean(vals):>12.4f}{np.std(vals):>8.4f}")
PYEOF
