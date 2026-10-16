"""
risk_disagreement_analysis.py

Instead of asking "what does the final prediction say", asks "how much
did the 3 risk models agree before averaging". Per team's proposal:
2/3 agreement (uncertainty) and 3/3 agreement (confidence) are
fundamentally different situations the ensemble currently treats
almost the same.

Reports:
  - overall agreement distribution (3/3, 2/1, all-different)
  - accuracy conditioned on agreement level (does 3/3 agreement mean
    the ensemble is actually more likely to be right?)
  - which individual model, when it's the lone dissenter in a 2/1
    split, is more often RIGHT vs WRONG (is one model adding signal
    or just noise?)
  - where disagreements concentrate (adjacent-class vs far-apart)

Usage:
  python src/risk_disagreement_analysis.py \
      --val artifacts/val_split.csv \
      --val-probs runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
      --names risk_v1,risk_v1_s14,risk_v1_s15
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS, normalize_label

PROBCOLS = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]


def load_val_pred(path, row_ids):
    df = pd.read_csv(path).set_index("row_id")
    probs = df.loc[row_ids, PROBCOLS].values.astype(float)
    preds = [RISK_LEVELS[i] for i in probs.argmax(1)]
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-probs", required=True)
    ap.add_argument("--names", required=True, help="comma-sep display names, same order as --val-probs")
    args = ap.parse_args()

    val_df = pd.read_csv(args.val)
    row_ids = val_df["row_id"].tolist()
    gold = [normalize_label(x) for x in val_df["risk_level"]]

    prob_paths = [p.strip() for p in args.val_probs.split(",")]
    names = [n.strip() for n in args.names.split(",")]
    preds_per_model = [load_val_pred(p, row_ids) for p in prob_paths]  # [model][post]

    n = len(row_ids)
    agreement_counts = Counter()
    acc_by_agreement = {"3/3": [0, 0], "2/1": [0, 0], "3-way-split": [0, 0]}  # [correct, total]
    lone_dissenter_right = Counter()  # model_name -> times it was the lone dissenter AND correct
    lone_dissenter_wrong = Counter()
    disagree_pairs = Counter()

    order = ["indicator", "ideation", "behavior", "attempt"]

    for i in range(n):
        votes = [preds_per_model[m][i] for m in range(len(names))]
        g = gold[i]
        vote_counts = Counter(votes)
        unique = len(vote_counts)

        if unique == 1:
            level = "3/3"
            majority_pred = votes[0]
            agreement_counts["3/3"] += 1
        elif unique == 2:
            level = "2/1"
            agreement_counts["2/1"] += 1
            majority_label, majority_n = vote_counts.most_common(1)[0]
            majority_pred = majority_label
            # find the lone dissenter
            for m, v in enumerate(votes):
                if vote_counts[v] == 1:
                    if v == g:
                        lone_dissenter_right[names[m]] += 1
                    else:
                        lone_dissenter_wrong[names[m]] += 1
            # track WHICH labels disagreed (majority vs dissenter)
            for m, v in enumerate(votes):
                if v != majority_label:
                    disagree_pairs[(majority_label, v)] += 1
        else:
            level = "3-way-split"
            agreement_counts["3-way-split"] += 1
            majority_pred = votes[0]  # arbitrary — genuinely no majority

        correct = int(majority_pred == g)
        acc_by_agreement[level][0] += correct
        acc_by_agreement[level][1] += 1

    print("=" * 70)
    print("AGREEMENT DISTRIBUTION")
    print("=" * 70)
    for level in ["3/3", "2/1", "3-way-split"]:
        cnt = agreement_counts[level]
        correct, total = acc_by_agreement[level]
        acc = correct / total if total else 0.0
        print(f"  {level:<14} {cnt:>4} posts ({100*cnt/n:.1f}%)   "
              f"majority-vote accuracy = {acc:.4f} ({correct}/{total})")

    print(f"\n{'=' * 70}")
    print("LONE DISSENTER: when a model is the ODD ONE OUT in a 2/1 split,")
    print("is it usually right (adding signal) or wrong (adding noise)?")
    print("=" * 70)
    for name in names:
        right = lone_dissenter_right[name]
        wrong = lone_dissenter_wrong[name]
        total = right + wrong
        if total:
            print(f"  {name:<20} right={right:>3}  wrong={wrong:>3}  "
                  f"({100*right/total:.1f}% right when dissenting)")
        else:
            print(f"  {name:<20} never the lone dissenter")

    print(f"\n{'=' * 70}")
    print("WHERE DISAGREEMENTS CONCENTRATE (majority_label -> dissenter_label)")
    print("=" * 70)
    for (maj, diss), cnt in disagree_pairs.most_common(10):
        try:
            adjacency = abs(order.index(maj) - order.index(diss))
        except ValueError:
            adjacency = "?"
        tag = "ADJACENT" if adjacency == 1 else ("SAME" if adjacency == 0 else "FAR")
        print(f"  {maj:<10} <-> {diss:<10}: {cnt:>3}  [{tag}]")


if __name__ == "__main__":
    main()
