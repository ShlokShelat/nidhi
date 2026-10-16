"""
risk_argmax_audit.py

Does the recorded risk label agree with its own probabilities?

WHY THIS EXISTS. Reading the confident `behavior -> ideation` misses turned up
rows like:

    P01645  pred ideation   P(behavior) = 0.94   ("Planning ... call an Uber
                                                   ... end it ... before this year")
    P02257  pred ideation   P(behavior) = 0.89   ("purchase the instrument of my
                                                   demise ... written notes, made
                                                   a video, requested the day off")

In each case the LARGEST class probability is `behavior` while the stored label
says `ideation`. That is not a modelling error -- the model got it right and
something downstream recorded the wrong class. Possible causes, in order of
likelihood:

  1. the two symbolic rules overwrote the argmax (legitimate, but then
     `pre_rule_pred` should still match the probabilities)
  2. probabilities and labels come from different stages of the chain and were
     joined without being re-derived
  3. a column ordering mismatch between the prob columns and RISK_LEVELS

Case 1 is a design choice worth auditing; cases 2 and 3 are bugs that would cost
real points, because every post where argmax != label is a decision made by
something other than the model.

This counts them, splits pre-rule from post-rule, checks whether the mismatches
are systematically one direction (which would indicate an ordering bug), and
reports what fixing them would do to weighted-F1.

Usage:
  python src/risk_argmax_audit.py
  python src/risk_argmax_audit.py --oof artifacts/champion_oof.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402

PC = [f"prob_{c}" for c in RISK_LEVELS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="artifacts/champion_oof.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.oof, keep_default_na=False)
    miss = [c for c in PC if c not in d.columns]
    if miss:
        raise SystemExit(f"{args.oof} lacks {miss}")

    P = d[PC].to_numpy(float)
    am = np.array([RISK_LEVELS[i] for i in P.argmax(1)])
    pred = np.array([normalize_label(x) for x in d["pred_risk_level"]])
    gold = np.array([normalize_label(x) for x in d["gold"]])
    n = len(d)

    print(f"{n} OOF rows   prob columns: {PC}")
    print(f"row sums: mean {P.sum(1).mean():.4f}  min {P.sum(1).min():.4f}  "
          f"max {P.sum(1).max():.4f}   (should be ~1.0)")

    bad = am != pred
    print(f"\nargmax(prob) != pred_risk_level on {int(bad.sum())} of {n} rows "
          f"({100 * bad.mean():.1f}%)")

    if "pre_rule_pred" in d.columns:
        pre = np.array([normalize_label(x) for x in d["pre_rule_pred"]])
        bad_pre = am != pre
        moved = pre != pred
        print(f"argmax(prob) != pre_rule_pred  on {int(bad_pre.sum())} rows "
              f"({100 * bad_pre.mean():.1f}%)")
        print(f"the rules moved {int(moved.sum())} rows")
        print(f"  mismatches EXPLAINED by the rules: "
              f"{int((bad & moved).sum())}")
        print(f"  mismatches NOT explained by the rules: "
              f"{int((bad & ~moved).sum())}  <- these are the concerning ones")
        unexplained = bad & ~moved
    else:
        unexplained = bad

    if unexplained.any():
        print(f"\nDIRECTION of the {int(unexplained.sum())} unexplained "
              f"mismatches (argmax -> recorded):")
        c = Counter(zip(am[unexplained], pred[unexplained]))
        for (a, b), k in sorted(c.items(), key=lambda t: -t[1])[:10]:
            print(f"  {a:<10} -> {b:<10} {k:>5}")
        print("\nA single dominant direction suggests a COLUMN ORDERING bug")
        print("(prob columns not aligned to RISK_LEVELS). A spread suggests")
        print("a join or staging problem instead.")

        # what would obeying the model's own probabilities be worth?
        base = risk_weighted_f1(list(gold), list(pred))
        fixed = pred.copy()
        fixed[unexplained] = am[unexplained]
        alt = risk_weighted_f1(list(gold), list(fixed))
        right_now = int((pred[unexplained] == gold[unexplained]).sum())
        right_if = int((am[unexplained] == gold[unexplained]).sum())
        print(f"\non those {int(unexplained.sum())} rows: recorded label is "
              f"right {right_now} times, argmax would be right {right_if}")
        print(f"weighted-F1 as recorded {base:.4f} -> obeying argmax "
              f"{alt:.4f}   delta {alt - base:+.4f}")
        print(f"  (composite impact {0.4 * (alt - base):+.4f})")
    else:
        print("\nEvery row matches its own argmax once the rules are accounted")
        print("for. The label pipeline is consistent and there is no bug here --")
        print("the confident behavior->ideation misses are genuine model errors.")

    # sanity: is prob_behavior really the behavior column?
    print("\nSANITY CHECK -- mean probability per column, by GOLD class")
    print(f"{'gold':<12}" + "".join(f"{c.replace('prob_',''):>12}" for c in PC))
    print("-" * 60)
    for g in RISK_LEVELS:
        m = gold == g
        if m.sum():
            print(f"{g:<12}" + "".join(f"{P[m, j].mean():>12.3f}"
                                       for j in range(len(PC))))
    print("\nThe diagonal should dominate each row. If it does not, the prob")
    print("columns are not aligned to the class names and every downstream")
    print("blend has been mixing the wrong distributions.")


if __name__ == "__main__":
    main()
