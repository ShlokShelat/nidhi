"""
friend_ablation.py

Answers the only question that matters about the friend model: does it add
INDEPENDENT information, or does it dilute stronger models?

Standalone F1 is the wrong test. A weaker model earns its place in an ensemble
by making uncorrelated errors. So this measures, per post:

    UNIQUE FIX   -- our models are wrong, friend is right, and including
                    friend flips the blend to correct
    UNIQUE BREAK -- our models are right, friend is wrong, and including
                    friend flips the blend to incorrect

Net = fixes - breaks is the friend model's actual contribution.

Scope note: the friend enters the champion at
    base = 0.5 * mean(our seeds) + 0.5 * friend
and the risk-NLI blend then applies 0.30 * NLI + 0.70 * base on top. This
script measures the effect at the BASE level, where the friend actually acts.
The NLI blend afterwards will damp whatever is measured here, so treat these
numbers as an upper bound on the friend's influence, not the final delta.

Everything is also reported per author-disjoint half, because a contribution
that only appears in one half is the failure mode this campaign keeps hitting.

Usage:
  python src/friend_ablation.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1
from rule_ablation_v2 import author_split

PC = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]
FC = ["Indicator", "Ideation", "Behavior", "Attempt"]
OURS = ["risk_v1", "risk_v1_s14", "risk_v1_s15"]
FULL = ["risk_full_s13", "risk_full_s14", "risk_full_s15"]


def load(runs, row_ids):
    out = {}
    for r in runs:
        p = f"runs/{r}/val_predictions.csv"
        if os.path.exists(p):
            d = pd.read_csv(p).set_index("row_id")
            out[r] = d.loc[row_ids, PC].values.astype(float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--include-full", action="store_true",
                    help="also report variants using the full-data models. "
                         "NOTE: those saw val in training, so any number "
                         "involving them is memorisation, not evaluation.")
    args = ap.parse_args()

    val = pd.read_csv(args.val)
    rid = val["row_id"].tolist()
    gold = [normalize_label(x) for x in val["risk_level"]]
    ours = load(OURS, rid)
    equal = np.mean(list(ours.values()), axis=0)

    fr = pd.read_csv("artifacts/friend_risk_probs_val.csv").set_index("row_id")
    friend = np.zeros((len(rid), 4))
    missing = []
    for i, r in enumerate(rid):
        if r in fr.index:
            p = fr.loc[r, FC].values.astype(float)
        else:
            p = equal[i]
            missing.append(i)
        friend[i] = p / p.sum()
    print(f"val posts {len(rid)}   friend predictions missing for "
          f"{len(missing)} (they fall back to our average, so the friend "
          f"cannot affect those posts at all)")

    def lab(P):
        return [RISK_LEVELS[int(i)] for i in P.argmax(1)]

    without = lab(equal)
    with_ = lab(0.5 * equal + 0.5 * friend)
    fr_only = lab(friend)

    print(f"\n{'configuration':<34}{'weighted-F1':>13}")
    for name, pred in (("our 3 seeds only", without),
                       ("0.5*ours + 0.5*friend (DEPLOYED)", with_),
                       ("friend alone", fr_only)):
        print(f"{name:<34}{risk_weighted_f1(gold, pred):>13.4f}")

    fixes, breaks = [], []
    for i in range(len(gold)):
        if without[i] == gold[i] and with_[i] != gold[i]:
            breaks.append(i)
        elif without[i] != gold[i] and with_[i] == gold[i]:
            fixes.append(i)
    print(f"\nFRIEND'S ACTUAL CONTRIBUTION (base level)")
    print(f"  unique FIXES  : {len(fixes)}")
    print(f"  unique BREAKS : {len(breaks)}")
    print(f"  NET           : {len(fixes) - len(breaks):+d}")

    fr_right_we_wrong = sum(1 for i in range(len(gold))
                            if fr_only[i] == gold[i] and without[i] != gold[i])
    fr_wrong_we_right = sum(1 for i in range(len(gold))
                            if fr_only[i] != gold[i] and without[i] == gold[i])
    print(f"\n  posts friend gets right that we get wrong : {fr_right_we_wrong}"
          f"   <- the information it could add")
    print(f"  posts we get right that friend gets wrong : {fr_wrong_we_right}"
          f"   <- the risk it brings")
    dis = sum(1 for i in range(len(gold)) if fr_only[i] != without[i])
    print(f"  disagreement with our ensemble            : {dis} "
          f"({dis/len(gold):.1%})")

    if breaks:
        print(f"\n  break transitions (correct -> wrong):")
        for (a, b), k in Counter((without[i], with_[i]) for i in breaks).most_common():
            print(f"    {a:<10} -> {b:<10} {k}")
    if fixes:
        print(f"  fix transitions (wrong -> correct):")
        for (a, b), k in Counter((without[i], with_[i]) for i in fixes).most_common():
            print(f"    {a:<10} -> {b:<10} {k}")

    half = author_split(val)[0].values
    print(f"\nPER AUTHOR-DISJOINT HALF")
    print(f"{'half':<8}{'ours only':>11}{'+friend':>10}{'delta':>9}"
          f"{'fixes':>7}{'breaks':>8}")
    for h in ("A", "B"):
        idx = [i for i in range(len(gold)) if half[i] == h]
        g = [gold[i] for i in idx]
        f_wo = risk_weighted_f1(g, [without[i] for i in idx])
        f_w = risk_weighted_f1(g, [with_[i] for i in idx])
        nf = sum(1 for i in idx if without[i] != gold[i] and with_[i] == gold[i])
        nb = sum(1 for i in idx if without[i] == gold[i] and with_[i] != gold[i])
        print(f"{h:<8}{f_wo:>11.4f}{f_w:>10.4f}{f_w - f_wo:>+9.4f}"
              f"{nf:>7}{nb:>8}")

    # weight sweep -- diagnostic only, NOT a tuning recommendation
    print(f"\nWEIGHT SWEEP (diagnostic -- picking a weight here would be "
          f"exactly the\nval-tuning that has failed 7/7 times in this campaign)")
    for w in np.round(np.arange(0.0, 1.01, 0.1), 1):
        pred = lab((1 - w) * equal + w * friend)
        print(f"  w_friend={w:.1f}  wF1 {risk_weighted_f1(gold, pred):.4f}")

    if args.include_full:
        full = load(FULL, rid)
        if full:
            six = np.mean(list(ours.values()) + list(full.values()), axis=0)
            print(f"\n*** CONTAMINATED -- full-data models trained on val ***")
            print(f"  6 seeds only          "
                  f"{risk_weighted_f1(gold, lab(six)):.4f}")
            print(f"  0.5*6seeds+0.5*friend "
                  f"{risk_weighted_f1(gold, lab(0.5*six + 0.5*friend)):.4f}")
            print(f"  (both meaningless as evaluation -- shown only to confirm "
                  f"the memorisation effect)")


if __name__ == "__main__":
    main()
