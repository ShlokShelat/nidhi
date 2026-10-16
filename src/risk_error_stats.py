"""
risk_error_stats.py

What exactly is the risk chain getting wrong, per class?

Scored on `artifacts/champion_oof.csv`, which carries the DEPLOYED chain's
out-of-fold predictions (ensemble + friend + NLI), its pre-rule predictions,
and gold, for 1590 posts.

WHY PER-CLASS AND NOT JUST wF1. The metric is SUPPORT-WEIGHTED F1, so a class
contributes in proportion to how often it appears in gold. An error on
`indicator` (37% of posts) costs far more than the same error on `attempt`
(7%). Knowing which class is weak is not enough -- what matters is which class
is weak AND heavy. This prints the loss each class contributes, which is what
should decide where effort goes.

It also splits the confusion by DIRECTION. A gold-behavior post predicted
ideation is charged twice: once against behavior's recall and once against
ideation's precision. Those double-charged cells are the expensive ones.

Usage:
  python src/risk_error_stats.py
  python src/risk_error_stats.py --oof artifacts/champion_oof.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="artifacts/champion_oof.csv")
    ap.add_argument("--pred-col", default="pred_risk_level")
    args = ap.parse_args()

    d = pd.read_csv(args.oof, keep_default_na=False)
    for c in (args.pred_col, "gold"):
        if c not in d.columns:
            raise SystemExit(f"{args.oof} has no '{c}' (columns {list(d.columns)})")
    y = np.array([normalize_label(x) for x in d["gold"]])
    p = np.array([normalize_label(x) for x in d[args.pred_col]])
    n = len(y)

    wf1 = risk_weighted_f1(list(y), list(p))
    acc = float((y == p).mean())
    print(f"{n} out-of-fold posts   weighted-F1 {wf1:.4f}   accuracy {acc:.4f}")
    print("(out-of-fold predictions come from fold models on a different post "
          "set than test, so this level is not directly comparable to a test "
          "score)")

    print(f"\n{'class':<12}{'gold':>7}{'pred':>7}{'TP':>6}{'prec':>8}"
          f"{'rec':>8}{'F1':>8}{'weight':>9}{'loss':>9}")
    print("-" * 72)
    tot_loss = 0.0
    for c in RISK_LEVELS:
        g = (y == c)
        q = (p == c)
        tp = int((g & q).sum())
        prec = tp / max(int(q.sum()), 1)
        rec = tp / max(int(g.sum()), 1)
        f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
        w = g.mean()
        loss = w * (1 - f1)
        tot_loss += loss
        print(f"{c:<12}{int(g.sum()):>7}{int(q.sum()):>7}{tp:>6}{prec:>8.3f}"
              f"{rec:>8.3f}{f1:>8.3f}{w:>9.3f}{loss:>9.4f}")
    print(f"{'TOTAL':<12}{n:>7}{n:>7}{'':>6}{'':>8}{'':>8}"
          f"{wf1:>8.3f}{1.0:>9.3f}{tot_loss:>9.4f}")
    print("\n'loss' = share of gold x (1 - F1). It is what each class costs the")
    print("weighted score, and it is where effort should go -- not to whichever")
    print("class has the lowest F1.")

    print(f"\n{'CONFUSION  (rows = gold, cols = predicted)':<44}")
    print(f"{'':<12}" + "".join(f"{c[:9]:>11}" for c in RISK_LEVELS))
    print("-" * 56)
    for a in RISK_LEVELS:
        row = "".join(f"{int(((y == a) & (p == b)).sum()):>11}"
                      for b in RISK_LEVELS)
        print(f"{a:<12}{row}")

    errs = [(a, b, int(((y == a) & (p == b)).sum()))
            for a in RISK_LEVELS for b in RISK_LEVELS if a != b]
    errs.sort(key=lambda t: -t[2])
    n_err = sum(e[2] for e in errs)
    print(f"\n{n_err} errors ({100 * n_err / n:.1f}%). Most expensive pairs:")
    print(f"{'gold -> predicted':<34}{'n':>6}{'% of errors':>13}")
    print("-" * 53)
    for a, b, k in errs[:8]:
        if k:
            print(f"{a + ' -> ' + b:<34}{k:>6}{100 * k / n_err:>12.1f}%")

    # how many errors touch each class in either direction
    print(f"\nerrors touching each class (gold side or predicted side):")
    for c in RISK_LEVELS:
        k = int((((y == c) | (p == c)) & (y != p)).sum())
        print(f"  {c:<12}{k:>5}  ({100 * k / n_err:.0f}% of all errors)")

    # confidence of the errors -- are they near-misses or confident?
    pc = [f"prob_{c}" for c in RISK_LEVELS]
    if all(c in d.columns for c in pc):
        P = d[pc].to_numpy(float)
        top = P.max(1)
        bad = y != p
        print(f"\nconfidence: correct median {np.median(top[~bad]):.3f}   "
              f"errors median {np.median(top[bad]):.3f}")
        print(f"  errors above 0.90 confidence: {int((top[bad] > 0.9).sum())} "
              f"of {int(bad.sum())} "
              f"({100 * (top[bad] > 0.9).mean():.0f}%)")
        print("  Confident errors cannot be reached by any decision rule or")
        print("  threshold -- only by a model that reads the post differently.")

    if "pre_rule_pred" in d.columns:
        pre = np.array([normalize_label(x) for x in d["pre_rule_pred"]])
        moved = int((pre != p).sum())
        if moved:
            good = int(((pre != y) & (p == y)).sum())
            bad_ = int(((pre == y) & (p != y)).sum())
            print(f"\nthe 2 symbolic rules moved {moved} OOF posts: "
                  f"{good} fixed, {bad_} broken, net {good - bad_:+d}")


if __name__ == "__main__":
    main()
