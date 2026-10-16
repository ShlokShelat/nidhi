"""
risk_evidence_coupling.py

How much of the EVIDENCE score is decided by the RISK model?

WHY THE TWO HEADS ARE NOT INDEPENDENT. The span taggers never see the risk
label -- they read the raw post. But the pipeline couples them through one
line in evidence_vote_test.py:

    if risk[r] == "indicator":
        kept = []

together with one property of the gold data: a post's gold evidence is empty
EXACTLY when its gold risk level is `indicator`. Put those together and every
post falls into one of four buckets, three of which are decided before a
tagger runs:

    pred=ind, gold=ind    both empty            -> evidence F1 = 1.0  (free win)
    pred=ind, gold!=ind   nothing emitted       -> evidence F1 = 0.0  (total loss)
    pred!=ind, gold=ind   >=1 span emitted      -> evidence F1 = 0.0  (total loss)
    pred!=ind, gold!=ind  the taggers decide    -> evidence F1 = s_i

Only the fourth bucket is a span-quality problem. In the other three the
evidence score is a restatement of the risk label.

WHAT THIS IS FOR. A risk fix is worth its 0.4 composite weight on the risk
term alone. Indicator-boundary errors additionally destroy evidence at weight
0.3, so their true leverage is larger, and this quantifies by how much.

Note the asymmetry this exposes: errors ACROSS the indicator boundary cost
twice, while errors among ideation/behavior/attempt cost the risk term only and
leave evidence untouched.

HOW THE ARITHMETIC WORKS. Buckets A, B and C are exact from the risk confusion
alone. Bucket D needs the taggers' mean per-post F1, recovered from the pooled
evidence macro passed on the command line:

    evidence_macro = (A * 1.0 + 0 + 0 + sum_D s_i) / N
    =>  mean(s_D)  = (evidence_macro * N - A) / |D|

Pass --evidence-macro with the OOF number for the shipped evidence config.

Usage:
  python src/risk_evidence_coupling.py --evidence-macro 0.7821
  python src/risk_evidence_coupling.py --risk artifacts/champion_oof.csv \
      --evidence-macro 0.7821
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, parse_evidence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk", default="artifacts/champion_oof.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--evidence-macro", type=float, required=True,
                    help="pooled OOF evidence macro-F1 for the shipped config")
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    d = pd.read_csv(args.risk, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    d = d[d["row_id"].isin(lab.index)].copy()
    d["g"] = [normalize_label(x) for x in d["gold"]]
    d["p"] = [normalize_label(x) for x in d["pred_risk_level"]]

    # verify the premise rather than assume it
    n_ev = np.array([len(parse_evidence(lab.loc[r, "evidence"]))
                     for r in d["row_id"]])
    gi = (d["g"] == "indicator").to_numpy()
    print(f"{len(d)} OOF posts")
    print("\nPREMISE CHECK -- is gold evidence empty exactly on indicator?")
    print(f"  gold indicator     : {int(gi.sum())}  of which empty evidence "
          f"{int((n_ev[gi] == 0).sum())}")
    print(f"  gold non-indicator : {int((~gi).sum())}  of which empty evidence "
          f"{int((n_ev[~gi] == 0).sum())}")
    if (n_ev[gi] != 0).any() or (n_ev[~gi] == 0).any():
        print("  -> NOT a clean identity; the buckets below are approximate")
    else:
        print("  -> clean identity, the decomposition below is exact")

    pi = (d["p"] == "indicator").to_numpy()
    N = len(d)
    A = int((pi & gi).sum())
    B = int((pi & ~gi).sum())
    C = int((~pi & gi).sum())
    D = int((~pi & ~gi).sum())

    sD = (args.evidence_macro * N - A) / max(D, 1)

    print("\n" + "=" * 66)
    print("WHERE THE EVIDENCE SCORE COMES FROM")
    print("=" * 66)
    print(f"{'bucket':<34}{'posts':>7}{'ev F1 each':>12}{'total':>10}")
    print("-" * 66)
    print(f"{'A  pred=ind gold=ind  (free win)':<34}{A:>7}{1.0:>12.3f}{A:>10.1f}")
    print(f"{'B  pred=ind gold!=ind (killed)':<34}{B:>7}{0.0:>12.3f}{0.0:>10.1f}")
    print(f"{'C  pred!=ind gold=ind (killed)':<34}{C:>7}{0.0:>12.3f}{0.0:>10.1f}")
    print(f"{'D  neither is ind     (taggers)':<34}{D:>7}{sD:>12.3f}"
          f"{sD * D:>10.1f}")
    print("-" * 66)
    print(f"{'evidence macro':<34}{N:>7}{'':>12}{args.evidence_macro:>10.4f}")

    print(f"\n{100 * (A + B + C) / N:.1f}% of posts have their evidence score "
          f"decided by the risk label alone.")
    print(f"Only the {D} posts in bucket D ({100 * D / N:.1f}%) are a "
          f"span-quality problem.")

    # what a perfect indicator boundary would be worth
    gain_ev = (C + B * sD) / N
    print("\n" + "=" * 66)
    print("IF THE INDICATOR BOUNDARY WERE PERFECT")
    print("=" * 66)
    print(f"  {C} posts move 0.0 -> 1.0          +{C / N:.4f}")
    print(f"  {B} posts move 0.0 -> {sD:.3f} (taggers)  +{B * sD / N:.4f}")
    print(f"  evidence macro {args.evidence_macro:.4f} -> "
          f"{args.evidence_macro + gain_ev:.4f}   ({gain_ev:+.4f})")
    print(f"  composite via the 0.3 evidence weight       "
          f"{0.3 * gain_ev:+.4f}")
    print("  ...and that is BEFORE the 0.4-weighted risk gain from the same "
          "fix.")

    print("\n" + "=" * 66)
    print("VALUE PER RISK ERROR, BY BOUNDARY")
    print("=" * 66)
    n_cross = B + C
    n_within = int(((d["g"] != d["p"]) & ~pi & ~gi).sum())
    print(f"  {n_cross} errors cross the indicator boundary -> cost risk AND "
          f"evidence")
    print(f"  {n_within} errors are among ideation/behavior/attempt -> cost "
          f"risk only")
    print(f"\n  one recovered C-post is worth {0.3 / N:+.5f} composite in "
          f"evidence alone")
    print(f"  one recovered B-post is worth {0.3 * sD / N:+.5f} composite in "
          f"evidence alone")
    print("\nErrors within ideation/behavior/attempt are the SECOND kind.")
    print("The indicator boundary is the one that pays twice.")


if __name__ == "__main__":
    main()
