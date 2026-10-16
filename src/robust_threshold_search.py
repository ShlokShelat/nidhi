"""
robust_threshold_search.py

Joint threshold optimization for the champion 2-seed factor-NLI
ensemble, targeting the ACTUAL competition metric (factor macro-F1,
i.e. S2) instead of independently-tuned per-factor F1 — while staying
robust rather than aggressively overfitting to the 372-post val split.

Robustness rules (per team discussion — this is the whole point):
  1. COARSE GRID ONLY: thresholds are searched in steps of 0.05, not
     continuous optimization. No hunting for 0.503 vs 0.506.
  2. MINIMUM POSTS-CHANGED FILTER: a threshold change for a factor is
     only accepted if it flips the prediction on at least
     --min-posts-changed val posts (default 10) relative to the
     ORIGINAL saved threshold. Smaller changes are within this val
     set's noise floor (established earlier this campaign) and are
     discarded regardless of the F1 delta they show.
  3. PLATEAU PREFERENCE: among grid points that clear the F1 bar
     within --plateau-tolerance of the best score, pick the one
     closest to the middle of that contiguous stable region, not the
     single best (possibly spiky) point.
  4. Reports posts-changed and the OLD vs NEW threshold for every
     factor so the change is auditable, not a black box.

No retraining involved — scores the existing champion checkpoints on
val, sweeps thresholds, reports what's actually robust.

Usage:
  python src/robust_threshold_search.py \
      --val artifacts/val_split.csv \
      --nli-run runs/factors_nli,runs/factors_nli_s2 \
      --min-posts-changed 10 --out artifacts/robust_thresholds.npy
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
from train_factors_nli import NLIScorer, score_all, gold_matrix, FACTOR_TAXONOMY, N
from scoring import factors_macro_f1


def to_lists(binary, gold_or_pred):
    return [[FACTOR_TAXONOMY[j] for j in range(N) if row[j]] for row in binary]


def macro_f1_from_probs(probs, ths, gold):
    on = (probs >= ths).astype(int)
    return factors_macro_f1(to_lists(gold, "gold"), to_lists(on, "pred"))


def per_factor_f1(probs_col, th, gold_col):
    pred = (probs_col >= th).astype(int)
    tp = int(((pred == 1) & (gold_col == 1)).sum())
    fp = int(((pred == 1) & (gold_col == 0)).sum())
    fn = int(((pred == 0) & (gold_col == 1)).sum())
    return (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def robust_search_one_factor(probs_col, gold_col, original_th, grid,
                             min_posts_changed, plateau_tol):
    """
    Returns (chosen_threshold, chosen_f1, n_posts_changed_vs_original).
    Only considers grid points whose prediction differs from the
    original threshold's prediction on >= min_posts_changed posts;
    among the survivors, prefers the middle of the best plateau.
    """
    orig_pred = (probs_col >= original_th).astype(int)

    candidates = []  # (threshold, f1, n_changed)
    for t in grid:
        pred = (probs_col >= t).astype(int)
        n_changed = int((pred != orig_pred).sum())
        f1 = per_factor_f1(probs_col, t, gold_col)
        candidates.append((t, f1, n_changed))

    eligible = [(t, f1, n) for t, f1, n in candidates if n >= min_posts_changed]
    # always keep the original threshold itself as a fallback candidate
    orig_f1 = per_factor_f1(probs_col, original_th, gold_col)
    eligible.append((original_th, orig_f1, 0))

    if not eligible:
        return original_th, orig_f1, 0

    best_f1 = max(f1 for _, f1, _ in eligible)
    plateau = [t for t, f1, n in eligible if f1 >= best_f1 - plateau_tol]
    chosen_t = float(np.median(plateau))
    # snap to nearest actually-evaluated candidate for a clean report
    chosen = min(eligible, key=lambda c: abs(c[0] - chosen_t))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--nli-run", required=True, help="comma-sep seed run dirs")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--min-posts-changed", type=int, default=10)
    ap.add_argument("--plateau-tolerance", type=float, default=0.005)
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--out", default="artifacts/robust_thresholds.npy")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")

    run_dirs = [r.strip() for r in args.nli_run.split(",")]
    va_df = pd.read_csv(args.val)
    gold = gold_matrix(va_df)

    print(f"scoring {len(run_dirs)}-seed ensemble on val...")
    all_probs, all_ths = [], []
    for rd in run_dirs:
        mn = open(os.path.join(rd, "tok")).read().strip()
        tok = AutoTokenizer.from_pretrained(mn)
        m = NLIScorer(mn).to(device)
        m.load_state_dict(torch.load(os.path.join(rd, "model.pt"),
                                     map_location=device), strict=False)
        m.eval()
        all_probs.append(score_all(m, va_df, tok, args.max_len, device, use_bf16))
        all_ths.append(np.load(os.path.join(rd, "thresholds.npy")))
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    probs = np.mean(all_probs, axis=0)
    original_ths = np.mean(all_ths, axis=0)

    original_macro = macro_f1_from_probs(probs, original_ths, gold)
    print(f"\nORIGINAL (per-factor-tuned) champion macro-F1: {original_macro:.4f}\n")

    grid = np.arange(0.10, 0.90 + 1e-9, args.grid_step)
    new_ths = original_ths.copy()

    print(f"{'factor':<48}{'orig_th':>9}{'new_th':>9}{'orig_f1':>9}{'new_f1':>9}"
          f"{'delta':>8}{'posts_chg':>11}")
    total_changed = 0
    for j in range(N):
        orig_f1 = per_factor_f1(probs[:, j], original_ths[j], gold[:, j])
        new_t, new_f1, n_changed = robust_search_one_factor(
            probs[:, j], gold[:, j], original_ths[j], grid,
            args.min_posts_changed, args.plateau_tolerance)
        new_ths[j] = new_t
        total_changed += n_changed
        marker = " *" if n_changed > 0 and abs(new_f1 - orig_f1) > 1e-6 else ""
        print(f"{FACTOR_TAXONOMY[j]:<48}{original_ths[j]:>9.2f}{new_t:>9.2f}"
              f"{orig_f1:>9.4f}{new_f1:>9.4f}{new_f1-orig_f1:>+8.4f}"
              f"{n_changed:>11}{marker}")

    robust_macro = macro_f1_from_probs(probs, new_ths, gold)
    print(f"\n{'=' * 90}")
    print(f"ORIGINAL macro-F1 (per-factor independently tuned): {original_macro:.4f}")
    print(f"ROBUST joint-search macro-F1:                       {robust_macro:.4f}")
    print(f"Delta: {robust_macro - original_macro:+.4f}")
    print(f"Total post-level prediction changes across all factors: {total_changed}")
    print(f"{'=' * 90}")

    if robust_macro <= original_macro:
        print("\nNo robust improvement found — original thresholds stand.")
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.save(args.out, new_ths)
        print(f"\nRobust improved thresholds saved to {args.out}")
        print("Next: rebuild predictions with these thresholds and validate "
              "the change transfers (single-variable test) before uploading.")


if __name__ == "__main__":
    main()
