"""
risk_arm_transitions.py

WHERE did an arm's predictions move, and was the move good or bad?

WHY THIS EXISTS. The gaur arm lost weighted-F1 overall, a scalar that alone
would suggest the definitions lane is closed. The per-class table says
something completely different:

    class        base F1   gaur F1
    indicator      0.872     0.879
    ideation       0.766     0.772
    behavior       0.749     0.707
    attempt        0.726     0.542   <- the entire loss

Two classes IMPROVED. One collapsed and dragged the mean under. A scalar delta
cannot tell you that, and closing a lane on a scalar delta is how a correct idea
gets thrown away for a fixable packaging error.

THE HYPOTHESIS THIS TESTS. The gaur `behavior` definition contains "harms OR HAS
HARMED themselves ... scars" -- past-tense self-injury, which is exactly what
`attempt` posts describe (risk_attempt_boundary.py measured self-harm words in
20.3% of attempt spans). Widening behavior with no upper bound should pull gold
ATTEMPT posts down into behavior. If that is what happened, the flow shows up as
a single dominant off-diagonal cell and the fix is an exclusion clause, not
abandoning the definitions.

WHAT IT PRINTS. For each gold class, a base-prediction x arm-prediction crosstab
with the moves scored: RESCUED (wrong -> right), BROKEN (right -> wrong), and
churn. Then the net per-class ledger, and the posts behind the largest broken
cell so the mechanism is readable rather than inferred.

Usage:
  python src/risk_arm_transitions.py \
      --base artifacts/nligaur/oof_base.csv --arm artifacts/nligaur/oof_gaur.csv
  python src/risk_arm_transitions.py --show 15
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--arm", default="artifacts/nligaur/oof_gaur.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    b = pd.read_csv(args.base, keep_default_na=False)
    g = pd.read_csv(args.arm, keep_default_na=False)
    for d in (b, g):
        d["row_id"] = d["row_id"].astype(str)
    m = b[["row_id", "gold", "pred_risk_level"]].merge(
        g[["row_id", "pred_risk_level"]], on="row_id", suffixes=("_b", "_a"))
    m["gold"] = [normalize_label(x) for x in m["gold"]]
    m["pb"] = [normalize_label(x) for x in m["pred_risk_level_b"]]
    m["pa"] = [normalize_label(x) for x in m["pred_risk_level_a"]]
    print(f"{len(m)} posts common to both arms")
    print(f"weighted-F1   base {risk_weighted_f1(list(m.gold), list(m.pb)):.4f}"
          f"   arm {risk_weighted_f1(list(m.gold), list(m.pa)):.4f}")

    moved = m[m.pb != m.pa]
    resc = moved[(moved.pb != moved.gold) & (moved.pa == moved.gold)]
    brok = moved[(moved.pb == moved.gold) & (moved.pa != moved.gold)]
    churn = len(moved) - len(resc) - len(brok)
    print(f"\n{len(moved)} predictions moved: {len(resc)} RESCUED, "
          f"{len(brok)} BROKEN, {churn} wrong->wrong")

    print("\n" + "=" * 70)
    print("NET LEDGER BY GOLD CLASS  (what the arm bought and what it cost)")
    print("=" * 70)
    print(f"{'gold class':<12}{'posts':>7}{'rescued':>9}{'broken':>8}{'net':>7}")
    print("-" * 45)
    for c in RISK_LEVELS:
        n = int((m.gold == c).sum())
        r = int((resc.gold == c).sum())
        k = int((brok.gold == c).sum())
        print(f"{c:<12}{n:>7}{r:>9}{k:>8}{r - k:>+7}")

    print("\n" + "=" * 70)
    print("WHERE THE BROKEN POSTS WENT  (base pred -> arm pred, gold class)")
    print("=" * 70)
    cnt = Counter(zip(brok.gold, brok.pb, brok.pa))
    print(f"{'gold':<12}{'was':<12}{'became':<12}{'n':>5}")
    print("-" * 41)
    for (gd, pb, pa), k in sorted(cnt.items(), key=lambda t: -t[1]):
        print(f"{gd:<12}{pb:<12}{pa:<12}{k:>5}")
    print("\nOne dominant cell = a definition reaching into a neighbouring")
    print("class. The fix for that is an EXCLUSION clause, not abandoning the")
    print("definitions. A spread = the arm is simply noisier.")

    print("\n" + "=" * 70)
    print("AND WHAT IT RESCUED  (base pred -> arm pred, gold class)")
    print("=" * 70)
    cnt2 = Counter(zip(resc.gold, resc.pb, resc.pa))
    print(f"{'gold':<12}{'was':<12}{'became':<12}{'n':>5}")
    print("-" * 41)
    for (gd, pb, pa), k in sorted(cnt2.items(), key=lambda t: -t[1]):
        print(f"{gd:<12}{pb:<12}{pa:<12}{k:>5}")

    # read the posts behind the biggest breakage
    if cnt and os.path.exists(args.labeled):
        (gd, pb, pa), _ = max(cnt.items(), key=lambda t: t[1])
        lab = pd.read_csv(args.labeled, keep_default_na=False)
        lab["row_id"] = lab["row_id"].astype(str)
        lab = lab.set_index("row_id")
        sub = brok[(brok.gold == gd) & (brok.pb == pb) & (brok.pa == pa)]
        print("\n" + "=" * 70)
        print(f"THE LARGEST BREAKAGE: gold {gd}, {pb} -> {pa} "
              f"({len(sub)} posts, showing {min(args.show, len(sub))})")
        print("=" * 70)
        for rid in sub["row_id"].head(args.show):
            if rid not in lab.index:
                continue
            ev = str(lab.loc[rid, "evidence"])
            body = " ".join(str(lab.loc[rid, "post"]).split())
            print(f"\n--- {rid}")
            print(f"    SPAN: {ev}")
            print(f"    POST: {body[:200]}{'...' if len(body) > 200 else ''}")
        print("\nIf these posts describe an act ALREADY CARRIED OUT while the")
        print("arm's `behavior` text says 'harms OR HAS HARMED themselves',")
        print("the definition is the cause and it is a two-line fix.")


if __name__ == "__main__":
    main()
