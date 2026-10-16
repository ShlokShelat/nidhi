"""
Factor seed ensemble: probability averaging across differently-seeded,
identical-architecture models, applied to factors.

Unlike the cross-model (friend) blends, this combines models of EQUAL
quality (same architecture, same training data, just different init/order).
Averaging models of equal quality is a lower-risk pattern than weighting
two DIFFERENT-quality models, which requires care to avoid the weaker
model dragging down the stronger one.

This VALIDATES locally on val_split before spending an upload — reports
macro-F1 for the ensemble vs the single best model, so the gain is
confirmed offline before a submission is spent on it.

Usage:
  python src/ensemble_factors_seeds.py \
      --val artifacts/val_split.csv --test data/test.csv \
      --runs runs/factors_cv_final,runs/factors_seed14,runs/factors_seed15 \
      --out artifacts/ensembled_factors_seeds.csv
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

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from scoring import FACTOR_TAXONOMY, parse_factors, factors_macro_f1
from predict import _factor_probs
from train_factors import tune_thresholds

N = len(FACTOR_TAXONOMY)


def to_lists(binary):
    return [[FACTOR_TAXONOMY[j] for j in range(N) if row[j]] for row in binary]


def gold_matrix(df):
    Y = np.zeros((len(df), N), dtype=int)
    for r, cell in enumerate(df["factors"]):
        for f in parse_factors(cell):
            if f in FACTOR_TAXONOMY:
                Y[r, FACTOR_TAXONOMY.index(f)] = 1
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--runs", required=True,
                    help="comma-separated factor run dirs to ensemble")
    ap.add_argument("--out", default="artifacts/ensembled_factors_seeds.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dirs = [r.strip() for r in args.runs.split(",")]
    print(f"ensembling {len(run_dirs)} factor models: {run_dirs}\n")

    # ---- local validation on val_split (honest — none trained on it as
    # final data, only used it for epoch-selection like every model here) ----
    val_df = load(args.val)
    gold = gold_matrix(val_df)

    val_probs_per_model = []
    solo_macros = []
    for rd in run_dirs:
        p = _factor_probs(rd, val_df, device)
        val_probs_per_model.append(p)
        ths = np.load(os.path.join(rd, "thresholds.npy"))
        on = p >= ths
        macro = factors_macro_f1(to_lists(gold), to_lists(on.astype(int)))
        solo_macros.append(macro)
        print(f"  {rd}: solo val macro-F1 = {macro:.4f}")

    val_ens_probs = np.mean(val_probs_per_model, axis=0)
    ens_ths = tune_thresholds(val_ens_probs, gold)
    ens_on = val_ens_probs >= ens_ths
    ens_macro = factors_macro_f1(to_lists(gold), to_lists(ens_on.astype(int)))

    best_solo = max(solo_macros)
    print(f"\n  best SOLO model : {best_solo:.4f}")
    print(f"  ENSEMBLE (avg)  : {ens_macro:.4f}  (delta {ens_macro-best_solo:+.4f})")

    if ens_macro <= best_solo:
        print("\n  *** ENSEMBLE DOES NOT BEAT BEST SOLO MODEL LOCALLY ***")
        print("  *** recommend NOT submitting this — save the upload ***")
    else:
        print("\n  ensemble validated locally — safe to generate test predictions")

    # ---- test-set predictions using the ensemble (only meaningful if the
    # local check above was positive, but we generate regardless so you can
    # decide) ----
    test_df = load(args.test)
    test_probs_per_model = [_factor_probs(rd, test_df, device) for rd in run_dirs]
    test_ens_probs = np.mean(test_probs_per_model, axis=0)
    on = test_ens_probs >= ens_ths
    fac_lists = to_lists(on.astype(int))
    out = pd.DataFrame({
        "row_id": test_df["row_id"].tolist(),
        "factors": [str(list(f)) for f in fac_lists],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} (mean factors/post: {on.sum(1).mean():.2f})")


if __name__ == "__main__":
    main()
