"""
Per-factor F1 comparison — baseline (no synthetic data) vs the
synthetic-augmented model, on the SAME val_split gold. Macro-F1 is a
mean over labels, so it hides per-factor gains that are offset by
losses elsewhere; splitting it per factor shows whether the factors
that received synthetic data are the ones that moved.

No training involved — pure evaluation of two existing checkpoints.

Usage:
  python src/compare_factor_models.py \
      --val artifacts/val_split.csv \
      --baseline-run runs/factors_nli \
      --synthetic-run runs/factors_nli_synthetic_final
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from train_factors_nli import (NLIScorer, score_all, tune_thresholds,
                               gold_matrix, FACTOR_TAXONOMY, N)

# the 8 factors that received synthetic data — everything else is a
# control group that should be roughly UNCHANGED if the augmentation
# is working as intended (not causing collateral damage elsewhere)
TARGETED = {
    "poor social support", "psychological capital", "interpersonal difficulty",
    "hopelessness", "prior self-harm or suicidal thought/attempt",
    "suicide means (with access)", "mental health issues", "stressful life event",
}


def per_factor_f1(probs, ths, gold):
    """Return {factor_name: f1} using the SAME binary-F1 logic tune_thresholds uses."""
    out = {}
    for j in range(N):
        pred = (probs[:, j] >= ths[j]).astype(int)
        tp = int(((pred == 1) & (gold[:, j] == 1)).sum())
        fp = int(((pred == 1) & (gold[:, j] == 0)).sum())
        fn = int(((pred == 0) & (gold[:, j] == 1)).sum())
        support = int(gold[:, j].sum())
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        out[FACTOR_TAXONOMY[j]] = (f1, support)
    return out


def load_and_score(run_dir, va_df, max_len, device, use_bf16):
    mn = open(os.path.join(run_dir, "tok")).read().strip()
    tok = AutoTokenizer.from_pretrained(mn)
    m = NLIScorer(mn).to(device)
    m.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"),
                                 map_location=device), strict=False)
    m.eval()
    probs = score_all(m, va_df, tok, max_len, device, use_bf16)
    ths = np.load(os.path.join(run_dir, "thresholds.npy"))
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return probs, ths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--baseline-run", required=True)
    ap.add_argument("--synthetic-run", required=True)
    ap.add_argument("--max-len", type=int, default=320)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")

    va_df = pd.read_csv(args.val)
    gold = gold_matrix(va_df)

    print(f"\nscoring baseline ({args.baseline_run})...")
    base_probs, base_ths = load_and_score(args.baseline_run, va_df, args.max_len,
                                          device, use_bf16)
    print(f"scoring synthetic-augmented ({args.synthetic_run})...")
    syn_probs, syn_ths = load_and_score(args.synthetic_run, va_df, args.max_len,
                                        device, use_bf16)

    base_f1 = per_factor_f1(base_probs, base_ths, gold)
    syn_f1 = per_factor_f1(syn_probs, syn_ths, gold)

    rows = []
    for factor in FACTOR_TAXONOMY:
        bf1, support = base_f1[factor]
        sf1, _ = syn_f1[factor]
        rows.append((factor, support, bf1, sf1, sf1 - bf1,
                    "TARGETED" if factor in TARGETED else ""))

    rows.sort(key=lambda r: r[4])  # sort by delta, worst regressions first

    print(f"\n{'factor':<48}{'support':>8}{'baseline':>10}{'synthetic':>10}{'delta':>9}  flag")
    print("-" * 95)
    for factor, support, bf1, sf1, delta, flag in rows:
        marker = " *" if abs(delta) >= 0.05 else ""
        print(f"{factor:<48}{support:>8}{bf1:>10.4f}{sf1:>10.4f}{delta:>+9.4f}  {flag}{marker}")

    targeted_deltas = [d for f, s, b, sy, d, fl in rows if fl == "TARGETED"]
    other_deltas = [d for f, s, b, sy, d, fl in rows if fl != "TARGETED"]

    print(f"\n{'='*95}")
    print(f"TARGETED factors (n=8):     mean delta = {np.mean(targeted_deltas):+.4f}"
          f"   (positive={sum(1 for d in targeted_deltas if d>0)}/8)")
    print(f"NON-targeted factors (n={len(other_deltas)}): mean delta = {np.mean(other_deltas):+.4f}"
          f"   (positive={sum(1 for d in other_deltas if d>0)}/{len(other_deltas)})")
    print(f"\nOverall macro-F1: baseline={np.mean([b for _,_,b,_,_,_ in rows]):.4f}"
          f"  synthetic={np.mean([sy for _,_,_,sy,_,_ in rows]):.4f}")


if __name__ == "__main__":
    main()
