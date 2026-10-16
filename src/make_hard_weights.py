"""
make_hard_weights.py

Build per-post training weights that emphasise the risk errors that are most
plausibly learnable.

THE IDEA. Train, find what the model gets wrong, train again with those examples
weighted up. Standard hard-example mining.

THE TRAP IT AVOIDS. Finding the mistakes on val_split would destroy the only
held-out set available -- the model would look excellent on it and fail on
test, the classic validation-overfit failure mode. Every error here comes
from OUT-OF-FOLD predictions: each post scored by a model that never trained on
it.

THE REFINEMENT THAT MATTERS. Not every error is learnable. Some are bad labels,
and training harder on those memorises noise. They can be separated using a
second model with a completely different architecture:

    ours wrong AND friend right   -> LEARNABLE. Some model gets it, so the
                                     signal is in the text.
    ours wrong AND friend wrong   -> SUSPECT. Two unrelated architectures
                                     failing together, and the organizers report
                                     inter-annotator Kappa of 0.79, so these are
                                     plausibly ambiguous or mislabelled.

Upweighting the first group is hard-example mining. Upweighting the second is
memorising annotation noise. This script separates them and lets you weight each
independently -- --noise-weight defaults to 1.0 (untouched), so the first arm
tests ONE thing.

WHY RISK AND NOT FACTORS. For risk the errors are demonstrably learnable: gold
evidence spans classify the confident ideation<->behavior errors at 0.692
against a 0.538 majority baseline, which is why rationale supervision is worth
targeting here. For factors, ours and the friend are both wrong 3.74x above
chance, pointing at label ambiguity rather than model failure.

Usage:
  python src/make_hard_weights.py --hard-weight 3.0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label  # noqa: E402

PCOLS = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="v2_rebuild/runs/risk_ctrl_f{f}",
                    help="fold dirs holding oof_predictions.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--friend", default="artifacts/friend_oof_export.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--train", default="artifacts/train_split.csv",
                    help="only these posts get a weight; the rest are unused")
    ap.add_argument("--hard-weight", type=float, default=3.0,
                    help="weight for errors the friend gets RIGHT (learnable)")
    ap.add_argument("--noise-weight", type=float, default=1.0,
                    help="weight for errors BOTH models miss. 1.0 leaves them "
                         "alone; below 1.0 actively de-emphasises suspected "
                         "label noise. Keep at 1.0 for the first arm.")
    ap.add_argument("--out", default="artifacts/risk_hard_weights.csv")
    args = ap.parse_args()

    # ---- this arm's out-of-fold predictions ---------------------------------
    rows = []
    for f in range(args.k):
        p = os.path.join(args.ours.format(f=f), "oof_predictions.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        rows.append(pd.read_csv(p))
    ours = pd.concat(rows, ignore_index=True)
    ours["row_id"] = ours["row_id"].astype(str)
    miss = [c for c in PCOLS if c not in ours.columns]
    if miss:
        raise SystemExit(f"this arm's OOF lacks {miss}; columns are "
                         f"{list(ours.columns)}")
    ours["ours_pred"] = [RISK_LEVELS[i] for i in
                         ours[PCOLS].to_numpy().argmax(1)]

    # ---- the friend's --------------------------------------------------------
    fr = pd.read_csv(args.friend, keep_default_na=False)
    fr["row_id"] = fr["row_id"].astype(str)
    fcols = [c for c in fr.columns
             if c.lower() in [r.lower() for r in RISK_LEVELS]]
    if len(fcols) != 4:
        raise SystemExit(f"friend export risk columns: {fcols}")
    fr["friend_pred"] = [fcols[i].lower()
                         for i in fr[fcols].to_numpy().argmax(1)]

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab["gold"] = [normalize_label(x) for x in lab["risk_level"]]

    d = (lab[["row_id", "gold"]]
         .merge(ours[["row_id", "ours_pred"]], on="row_id", how="inner")
         .merge(fr[["row_id", "friend_pred"]], on="row_id", how="inner"))
    print(f"{len(d)} posts with gold + this arm's OOF + friend OOF")

    ours_wrong = d.ours_pred != d.gold
    friend_wrong = d.friend_pred != d.gold
    learnable = ours_wrong & ~friend_wrong
    suspect = ours_wrong & friend_wrong

    print(f"\n  this arm is wrong on     {int(ours_wrong.sum()):>5}")
    print(f"    of which LEARNABLE     {int(learnable.sum()):>5}   "
          f"(friend gets them right)")
    print(f"    of which SUSPECT       {int(suspect.sum()):>5}   "
          f"(both models miss)")
    exp = float(ours_wrong.sum()) * float(friend_wrong.sum()) / len(d)
    print(f"    independence would predict {exp:.0f} both-wrong; "
          f"overlap ratio {int(suspect.sum())/max(exp,1e-9):.2f}")

    print(f"\n{'gold':<12}{'n':>6}{'missed':>9}{'learnable':>11}"
          f"{'suspect':>9}")
    print("-" * 48)
    for c in RISK_LEVELS:
        m = d.gold == c
        print(f"{c:<12}{int(m.sum()):>6}{int((m & ours_wrong).sum()):>9}"
              f"{int((m & learnable).sum()):>11}{int((m & suspect).sum()):>9}")

    # ---- weights, restricted to the posts actually trained on ---------------
    w = pd.Series(1.0, index=d.index)
    w[learnable] = args.hard_weight
    w[suspect] = args.noise_weight
    d["weight"] = w

    tr = pd.read_csv(args.train, keep_default_na=False)
    tr["row_id"] = tr["row_id"].astype(str)
    out = tr[["row_id"]].merge(d[["row_id", "weight"]], on="row_id", how="left")
    n_default = int(out.weight.isna().sum())
    out["weight"] = out.weight.fillna(1.0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(f"  {len(out)} training posts   "
          f"{int((out.weight > 1.0).sum())} upweighted to "
          f"{args.hard_weight}   {n_default} had no OOF prediction "
          f"(left at 1.0)")
    print(f"  mean weight {out.weight.mean():.3f}   "
          f"effective sample size {out.weight.sum():.0f} of {len(out)}")
    print("\nA mean weight far above 1 means the upweighting dominates and the")
    print("model will chase hard cases at the expense of the easy majority.")
    print("hard_weight 2-3 is the usual safe range; above 5 tends to overfit")
    print("the mined set.")


if __name__ == "__main__":
    main()
