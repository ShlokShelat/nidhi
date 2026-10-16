"""
risk_candidate_check.py

Pre-submission sanity checks for a candidate risk prediction file when NO
honest validation score exists for it.

This is the situation with the full-data models (risk_full_s13/s14/s15). They
were trained on artifacts/full_train.csv, which contains all 372 val posts, so
their runs/*/val_predictions.csv score 0.9733 / 0.9648 / 0.9517 -- pure
memorisation. Those files must never be used to judge them, and this script
refuses to compute any val metric to remove the temptation.

What is checkable without labels:

  * how much the candidate actually changes
  * whether the changes are directionally lopsided (a model that has drifted
    usually dumps mass into one class)
  * whether the resulting class distribution still tracks the labeled prior
  * how much the new models agree with each other and with the incumbent

None of this proves the candidate is better. It only catches a candidate that
is obviously broken, which is the most that can be established here. The
decision to submit rests on the prior argument (29% more training data), not
on these numbers.

Usage:
  python src/risk_candidate_check.py \
      --candidate artifacts/risk_blend_6model_rules.csv \
      --champion  artifacts/risk_blend_scheduledfix.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, parse_factors

MAX_SANE_DISAGREEMENT = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--champion", default="artifacts/risk_blend_scheduledfix.csv")
    ap.add_argument("--labeled", default="data/labeled.csv")
    ap.add_argument("--col", default="pred_risk_level")
    args = ap.parse_args()

    a = pd.read_csv(args.champion)
    b = pd.read_csv(args.candidate)
    if set(a["row_id"]) != set(b["row_id"]):
        raise SystemExit("row_id sets differ")
    m = a[["row_id", args.col]].merge(b[["row_id", args.col]], on="row_id",
                                      suffixes=("_champ", "_cand"))
    champ = [normalize_label(x) for x in m[args.col + "_champ"]]
    cand = [normalize_label(x) for x in m[args.col + "_cand"]]
    n = len(m)

    changed = [(r, c, d) for r, c, d in zip(m["row_id"], champ, cand) if c != d]
    frac = len(changed) / n
    print(f"posts               : {n}")
    print(f"predictions changed : {len(changed)}  ({frac:.1%})")

    lab = pd.read_csv(args.labeled)
    prior = Counter(normalize_label(x) for x in lab["suicide risk"])
    npl = sum(prior.values())

    print(f"\n{'class':<12}{'champion':>10}{'candidate':>11}{'delta':>8}"
          f"{'prior_exp':>11}{'cand/prior':>12}")
    cc, cd = Counter(champ), Counter(cand)
    for k in RISK_LEVELS:
        exp = prior[k] / npl * n
        ratio = cd[k] / exp if exp else float("nan")
        print(f"{k:<12}{cc[k]:>10}{cd[k]:>11}{cd[k] - cc[k]:>+8}"
              f"{exp:>11.1f}{ratio:>12.2f}")

    print(f"\nchange transitions (champion -> candidate):")
    tr = Counter((c, d) for _, c, d in changed)
    for (c, d), k in tr.most_common():
        direction = ("escalate" if RISK_LEVELS.index(d) > RISK_LEVELS.index(c)
                     else "de-escalate")
        print(f"  {c:<10} -> {d:<10} {k:>4}   {direction}")
    up = sum(k for (c, d), k in tr.items()
             if RISK_LEVELS.index(d) > RISK_LEVELS.index(c))
    down = len(changed) - up
    print(f"  escalations {up}, de-escalations {down}")

    print(f"\n{'=' * 72}\nVERDICT")
    problems = []
    if frac > MAX_SANE_DISAGREEMENT:
        problems.append(f"disagreement {frac:.1%} exceeds "
                        f"{MAX_SANE_DISAGREEMENT:.0%} -- candidate is a very "
                        f"different model, not a refinement")
    # Only meaningful once there are enough changes for direction to carry
    # information; a 1-post change is trivially "all one direction".
    if len(changed) >= 8 and (up == 0 or down == 0):
        problems.append("all changes go the same direction -- looks like a "
                        "systematic shift, not case-by-case improvement")
    for k in RISK_LEVELS:
        exp = prior[k] / npl * n
        if exp >= 10 and (cd[k] < 0.5 * exp or cd[k] > 2.0 * exp):
            problems.append(f"class '{k}' predicted {cd[k]} vs {exp:.0f} "
                            f"expected from the labeled prior")
    if not len(changed):
        problems.append("candidate is identical to the champion -- nothing to test")

    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
        print("\n  Investigate before submitting.")
    else:
        print("  No pathology detected. This does NOT mean the candidate is")
        print("  better -- there is no honest validation score for it. Submit")
        print("  only on the strength of the more-training-data argument, and")
        print("  keep it a single-variable change so the result is readable.")


if __name__ == "__main__":
    main()
