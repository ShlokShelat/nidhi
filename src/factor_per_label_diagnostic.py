"""
factor_per_label_diagnostic.py

Finds which of the 24 factor labels are dragging Subtask-2 down.

Why this matters more than it looks: S2 is MACRO-F1 over 24 labels, so every
label contributes 1/24 = 4.2% of S2 regardless of how rare it is. One rare
label sitting at F1 0.13 instead of 0.50 costs ~0.015 S2 = ~0.0046 composite --
more than the entire risk-side symbolic campaign delivered. Our S2 is 0.6301
against 0.6965 for the leader, a 0.0664 gap worth 0.0199 composite, and it is
entirely plausible that most of it sits in a handful of rare labels rather
than being spread evenly.

On the superseded pre-NLI model (runs/factors_cv_final OOF, macro-F1 0.4401)
the pattern was unambiguous: the six worst labels were all RARE and all
massively OVER-predicted --

    meaning in life                 gold  35   predicted 303   F1 0.130
    sexual orientation issues       gold   8   predicted  24   F1 0.125
    physical health/characteristic  gold  59   predicted 176   F1 0.264
    traumatic experience            gold  47   predicted 124   F1 0.304

That is a per-label threshold calibration problem, not a modelling problem:
for a rare label, a threshold tuned to maximise anything other than that
label's own F1 will flood it with false positives and crater macro-F1.

This script re-measures the same profile for whichever factor prediction file
you give it, so the DEPLOYED 2-seed NLI ensemble can be checked rather than
assumed to share the old model's weaknesses.

Produce deployed-model val predictions first (GPU):
  python src/predict_factors_nli.py --test artifacts/val_split.csv \
      --nli-run runs/factors_nli,runs/factors_nli_s2 \
      --out artifacts/nli_factors_val.csv

Then (CPU):
  python src/factor_per_label_diagnostic.py --pred artifacts/nli_factors_val.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors


def to_matrix(rows, tax):
    idx = {t: i for i, t in enumerate(tax)}
    m = np.zeros((len(rows), len(tax)), dtype=int)
    for r, labs in enumerate(rows):
        for l in labs:
            l = l.strip().lower()
            if l in idx:
                m[r, idx[l]] = 1
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="artifacts/val_split.csv")
    ap.add_argument("--gold-col", default="factors")
    ap.add_argument("--pred", required=True, help="CSV with row_id,factors")
    ap.add_argument("--pred-col", default="factors")
    ap.add_argument("--target", type=float, default=0.50,
                    help="hypothetical F1 for headroom arithmetic")
    args = ap.parse_args()

    gold_df = pd.read_csv(args.gold)
    pred_df = pd.read_csv(args.pred, keep_default_na=False)
    df = gold_df[["row_id", args.gold_col]].merge(
        pred_df[["row_id", args.pred_col]], on="row_id",
        how="inner", suffixes=("_gold", "_pred"))
    gc = args.gold_col + ("_gold" if args.gold_col == args.pred_col else "")
    pc = args.pred_col + ("_pred" if args.gold_col == args.pred_col else "")

    tax = FACTOR_TAXONOMY
    G = to_matrix([parse_factors(x) for x in df[gc]], tax)
    P = to_matrix([parse_factors(x) for x in df[pc]], tax)

    f1 = f1_score(G, P, average=None, zero_division=0)
    pr = precision_score(G, P, average=None, zero_division=0)
    rc = recall_score(G, P, average=None, zero_division=0)

    print(f"posts scored: {len(df)}")
    print(f"MACRO-F1 (Subtask 2 metric): {f1.mean():.4f}\n")
    print(f"{'factor':<44}{'gold':>6}{'pred':>6}{'ratio':>7}"
          f"{'P':>7}{'R':>7}{'F1':>7}{'S2 cost':>9}")
    for i in np.argsort(f1):
        g, p = int(G[:, i].sum()), int(P[:, i].sum())
        ratio = p / g if g else float("inf")
        # what this label costs S2 relative to the target
        cost = max(0.0, args.target - f1[i]) / len(tax)
        print(f"{tax[i]:<44}{g:>6}{p:>6}{ratio:>7.1f}"
              f"{pr[i]:>7.3f}{rc[i]:>7.3f}{f1[i]:>7.3f}{cost:>9.4f}")

    worst = np.argsort(f1)[:6]
    gain = sum(max(0.0, args.target - f1[i]) for i in worst) / len(tax)
    print(f"\nlabels below F1 {args.target}: {int((f1 < args.target).sum())} of {len(tax)}")
    print(f"if the 6 worst labels reached F1 {args.target}: "
          f"S2 {f1.mean():.4f} -> {f1.mean() + gain:.4f} "
          f"({gain:+.4f}) = {0.3 * gain:+.4f} composite")

    over = [(tax[i], int(P[:, i].sum()) / max(int(G[:, i].sum()), 1), f1[i])
            for i in range(len(tax))
            if G[:, i].sum() and P[:, i].sum() > 2 * G[:, i].sum()]
    if over:
        print(f"\nOVER-PREDICTED labels (>2x gold count) -- raise their thresholds:")
        for name, r, s in sorted(over, key=lambda x: -x[1]):
            print(f"    {name:<44} {r:.1f}x gold, F1 {s:.3f}")


if __name__ == "__main__":
    main()
