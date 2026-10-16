"""
risk_ensemble_weight_sweep_v2.py

Coarse, hand-picked weight sweep over the risk ensemble members: 8
combinations rather than a full grid. A small, fixed candidate set is far
less likely to overfit a single validation split than a fine simplex search,
because the number of ways to fit noise grows with the search space.

Reports not just the score but WHAT the reweighting changes: transition
counts between risk levels, so a small delta can be traced to the specific
posts that moved.

Usage:
  python src/risk_ensemble_weight_sweep_v2.py \
      --val artifacts/val_split.csv \
      --val-probs runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1

PROBCOLS = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]

HAND_PICKED_WEIGHTS = [
    (0.33, 0.33, 0.34),
    (0.40, 0.30, 0.30),
    (0.50, 0.25, 0.25),
    (0.50, 0.30, 0.20),
    (0.60, 0.20, 0.20),
    (0.20, 0.40, 0.40),
    (0.30, 0.40, 0.30),
    (0.30, 0.30, 0.40),
]


def load_val_probs(path, row_ids):
    df = pd.read_csv(path).set_index("row_id")
    return df.loc[row_ids, PROBCOLS].values.astype(float)


def predict_for(probs_list, weights):
    combined = sum(w * p for w, p in zip(weights, probs_list))
    return [RISK_LEVELS[i] for i in combined.argmax(1)]


def transition_summary(old_pred, new_pred):
    """Returns a Counter of (old_label -> new_label) for changed predictions."""
    c = Counter()
    for o, n in zip(old_pred, new_pred):
        if o != n:
            c[(o, n)] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-probs", required=True,
                    help="comma-sep val_predictions.csv paths, one per risk model")
    args = ap.parse_args()

    val_df = pd.read_csv(args.val)
    row_ids = val_df["row_id"].tolist()
    gold = [normalize_label(x) for x in val_df["risk_level"]]

    prob_paths = [p.strip() for p in args.val_probs.split(",")]
    probs_list = [load_val_probs(p, row_ids) for p in prob_paths]

    equal_w = (0.33, 0.33, 0.34)
    equal_pred = predict_for(probs_list, equal_w)
    equal_f1 = risk_weighted_f1(gold, equal_pred)
    print(f"EQUAL WEIGHTS {equal_w} -> val weighted-F1 = {equal_f1:.4f}  (baseline)\n")
    print("=" * 78)

    results = []
    for w in HAND_PICKED_WEIGHTS:
        pred = predict_for(probs_list, w)
        f1 = risk_weighted_f1(gold, pred)
        trans = transition_summary(equal_pred, pred)
        n_changed = sum(trans.values())
        results.append((w, f1, n_changed, trans))

        print(f"Weights: {w}")
        print(f"  Val weighted-F1: {f1:.4f}  (delta {f1-equal_f1:+.4f})")
        print(f"  Posts changed: {n_changed}")
        if trans:
            for (o, n), cnt in sorted(trans.items(), key=lambda x: -x[1]):
                print(f"    {o:<10} -> {n:<10}: {cnt}")
        print()

    results.sort(key=lambda r: -r[1])
    print("=" * 78)
    print("RANKED SUMMARY")
    print(f"{'weights':<24}{'val_wF1':>10}{'delta':>9}{'posts_chg':>11}")
    for w, f1, n_changed, _ in results:
        print(f"{str(w):<24}{f1:>10.4f}{f1-equal_f1:>+9.4f}{n_changed:>11}")

    print("\n" + "=" * 78)
    best_w, best_f1, best_n, best_trans = results[0]
    if best_f1 <= equal_f1:
        print("No hand-picked combination beats equal weighting. Keep 1/3-1/3-1/3.")
    elif best_n < 10:
        print(f"Best candidate {best_w} only changes {best_n} posts — "
              "too few for the delta to be distinguishable from split noise. "
              "Keep equal weighting.")
    else:
        near_best = [r for r in results if r[1] >= best_f1 - 0.002]
        if len(near_best) > 1:
            # prefer the one closest to equal weighting among near-ties
            near_best.sort(key=lambda r: sum(abs(x - 1/3) for x in r[0]))
            chosen = near_best[0]
            print(f"Multiple combinations tie within 0.002 — choosing simplest "
                  f"(closest to equal): {chosen[0]}")
        else:
            chosen = results[0]
            print(f"Clear best candidate: {chosen[0]}")
        print(f"  val weighted-F1: {equal_f1:.4f} -> {chosen[1]:.4f} ({chosen[1]-equal_f1:+.4f})")
        print(f"  posts changed: {chosen[2]}")
        print("\nCAVEAT: validation-split result only. Confirm on held-out "
              "data before adopting.")


if __name__ == "__main__":
    main()
