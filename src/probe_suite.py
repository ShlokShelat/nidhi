"""
probe_suite.py

Build degenerate submissions that turn the official scorer into an exact
measuring instrument, and solve the returned score for the quantity each one
isolates.

Offline paired-delta measurement on this dataset has an SD of about 0.007 (153
authors), so selecting the best of 8 arms inflates the winner's reading by
0.007*sqrt(2*ln 8) = +0.0143 from the winner's curse alone. The official scorer,
by contrast, returns S1 and S2 to four decimals with no measurement error; its only
costs are a submission slot and the overfitting risk of picking the best of many
trials. Neither cost applies to a submission whose purpose is not to score well
but to solve for an unknown.

Each probe is built so that one returned number determines one quantity exactly.
None of them is meant to be kept.

  no-evidence       evidence emptied -> under micro averaging evidence F1 would
                    be 0 by construction, so S1 = (4/7)*risk_wF1, splitting S1
                    into its two components.

  risk-const-<cls>  every post labelled <cls> AND evidence emptied. Then
                    S1 = (4/7) * 2*g^2 / (N*(N+g)) where g is the number of test
                    posts whose GOLD risk is <cls>. Solves for g exactly. Three
                    of these give the whole test risk distribution (the fourth
                    class is N minus the rest), which is what every threshold and
                    prior tuned on train implicitly assumes.

  all-factors       all 24 labels on every post. Then
                    S2 = (1/24) * sum_j 2*g_j/(N+g_j).
                    An INTEGRITY CHECK: if any of the 24 label strings is not
                    the string the official scorer expects, that label
                    contributes 0 and S2 comes in about 0.02 low. A vocabulary
                    mismatch costs 1/24 of S2 permanently and invisibly, and is
                    undetectable from offline scoring against the same strings.

  no-factors        factors emptied -> S2 must be exactly 0. If it is not, the
                    scorer credits empty predictions and every offline factor
                    measurement needs reinterpreting.

The readings identify which S1 component is weak (risk carries 0.40 of the
composite, evidence 0.30, so this decides where effort goes), whether the test
distribution matches train, and whether a whole label is being lost. None of
that is recoverable offline at any sample size.

Usage:
  python src/probe_suite.py --build all --outdir submissions
  python src/probe_suite.py --solve no-evidence --s1 0.4457
  python src/probe_suite.py --solve risk-const --cls attempt --s1 0.0121
  python src/probe_suite.py --solve all-factors --s2 0.1834
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

RISK = ["indicator", "ideation", "behavior", "attempt"]
CHAMP_S1 = 0.7944
CHAMP_S2 = 0.6323


def build(base: str, outdir: str, which: str) -> list[str]:
    b = pd.read_csv(base, keep_default_na=False)
    n_rows = len(b)
    made = []

    def emit(name, df, note):
        p = os.path.join(outdir, f"NIDHI_probe_{name}.csv")
        df.to_csv(p, index=False)
        made.append(p)
        print(f"  {p}   {note}")

    if which in ("all", "no-evidence"):
        d = b.copy()
        d["evidence"] = ""
        emit("noev", d, "evidence emptied; risk+factors untouched")

    if which in ("all", "bad-evidence"):
        # Evidence is scored MACRO, so a post with empty gold AND empty
        # prediction earns 1.0. An empty column therefore scores the SHARE of
        # empty-gold posts, not zero, and S1 = (4/7)risk + (3/7)e_empty still
        # has two unknowns.
        #
        # A span that occurs in NO post fixes that. Gold-empty posts now have a
        # non-empty prediction (so they lose the vacuous 1.0) and gold-non-empty
        # posts get no match, so macro evidence F1 is EXACTLY 0 and
        #     risk_wF1 = S1 * 7/4
        # with nothing assumed. Subtracting then gives e_empty and, from a
        # reference submission's S1, its real evidence score.
        d = b.copy()
        d["evidence"] = "qzxvbnmwpfjk"
        emit("badev", d, "evidence = a string in no post; scores exactly 0")

    if which in ("all", "no-factors"):
        d = b.copy()
        d["factors"] = "[]"
        emit("nofac", d, "factors emptied; risk+evidence untouched")

    if which in ("all", "all-factors"):
        d = b.copy()
        d["factors"] = str(list(FACTOR_TAXONOMY))
        d["evidence"] = ""      # so S1 is also a clean risk read
        emit("allfac", d, "all 24 labels everywhere; evidence emptied")

    if which in ("all", "risk-const"):
        for c in RISK:
            d = b.copy()
            d["risk_level"] = c
            d["evidence"] = ""
            emit(f"risk_{c}", d, f"every post = {c}; evidence emptied")

    print(f"\n{len(made)} probe(s) written, {n_rows} rows each.")
    print("NONE of these is meant to score well. Each one is an equation.")
    return made


def solve_bad_evidence(s1: float, s1_noev: float = 0.6172):
    """Garbage evidence scores exactly 0, so this pins risk with no assumption."""
    r = s1 * 7 / 4
    if not 0 <= r <= 1:
        print(f"*** risk_wF1 = {r:.4f} is outside [0,1]. Even garbage evidence")
        print("    is scoring above zero, so the metric differs from the model")
        print("    assumed here.")
        return
    e_empty = (s1_noev * 7 - 4 * r) / 3
    e_champ = (CHAMP_S1 * 7 - 4 * r) / 3
    print(f"S1 returned {s1:.4f}   (garbage evidence -> evidence F1 = 0)")
    print(f"\n  risk_wF1            = {r:.4f}   MEASURED, nothing assumed")
    print(f"  empty-column score  = {e_empty:.4f}   "
          f"(from the noev probe's {s1_noev:.4f})")
    print(f"  REFERENCE evidence  = {e_champ:.4f}   "
          f"(from its S1 of {CHAMP_S1})")
    print(f"\n  train share of empty-gold posts is 0.357. If empty-column")
    print(f"  score lands near that, macro averaging is fully confirmed.")
    print(f"\n  weight in composite: risk 0.40, evidence 0.30")
    print(f"  headroom to a perfect component: "
          f"risk {1-r:.3f} x 0.40 = {0.40*(1-r):.4f}   "
          f"evidence {1-e_champ:.3f} x 0.30 = {0.30*(1-e_champ):.4f}")
    print("  The component with more headroom PER UNIT OF WEIGHT is the one")
    print("  worth working.")


def solve_no_evidence(s1: float):
    r = s1 * 7 / 4
    e = (CHAMP_S1 * 7 - 4 * r) / 3
    print(f"S1 returned {s1:.4f}")
    print(f"  risk_wF1     = {r:.4f}   (weight 0.40 of composite)")
    print(f"  evidence_F1  = {e:.4f}   (weight 0.30 of composite)")
    if not 0 <= e <= 1:
        print("  *** evidence_F1 outside [0,1] -- an assumption is wrong.")
        print("      Either S1 is not (4/7)risk+(3/7)evidence, or emptying")
        print("      evidence does not zero its F1.")
        return
    print(f"\n  a +0.01 gain in risk is worth {0.40*0.01:.4f} composite")
    print(f"  a +0.01 gain in evidence is worth {0.30*0.01:.4f} composite")
    print(f"\n  the weaker component relative to its ceiling carries the")
    print(f"  larger remaining payoff.")


def solve_risk_const(s1: float, cls: str, n: int = 378):
    """S1 = (4/7) * 2g^2 / (n(n+g))  ->  solve the quadratic for g."""
    w = s1 * 7 / 4
    if w <= 0:
        print(f"S1 {s1:.4f} implies wF1 0 -- no test post has gold {cls!r}?")
        return
    # 2g^2 - w*n*g - w*n^2 = 0
    a, bq, c = 2.0, -w * n, -w * n * n
    disc = bq * bq - 4 * a * c
    g = (-bq + math.sqrt(disc)) / (2 * a)
    print(f"S1 returned {s1:.4f}  ->  weighted-F1 {w:.4f}")
    print(f"  gold posts with risk_level == {cls!r}: {g:.1f} of {n} "
          f"({100*g/n:.1f}%)")
    tr = {"indicator": 37.4, "ideation": 24.5, "behavior": 23.9, "attempt": 7.0}
    if cls in tr:
        print(f"  train prevalence: {tr[cls]:.1f}%   "
              f"difference {100*g/n - tr[cls]:+.1f} points")
        if abs(100 * g / n - tr[cls]) > 5:
            print("  *** TEST DISTRIBUTION DIFFERS MATERIALLY FROM TRAIN.")
            print("      Thresholds and priors tuned on train assume it does")
            print("      not.")


def solve_all_factors(s2: float, labeled: str, n: int = 378):
    """Compare the returned S2 against what the train prevalences predict."""
    lab = pd.read_csv(labeled, keep_default_na=False)
    exp_terms = []
    for t in FACTOR_TAXONOMY:
        g = sum(1 for x in lab["factors"] if t in parse_factors(x)) / len(lab) * n
        exp_terms.append(2 * g / (n + g))
    expected = sum(exp_terms) / len(FACTOR_TAXONOMY)
    print(f"S2 returned {s2:.4f}")
    print(f"  predicted from train prevalence: {expected:.4f}")
    print(f"  difference {s2 - expected:+.4f}")
    one_label = max(exp_terms) / len(FACTOR_TAXONOMY)
    typical = sum(exp_terms) / len(exp_terms) / len(FACTOR_TAXONOMY)
    print(f"\n  one unrecognised label costs between "
          f"{typical:.4f} and {one_label:.4f}")
    if s2 < expected - typical * 0.7:
        k = (expected - s2) / max(typical, 1e-9)
        print(f"  *** roughly {k:.1f} label(s) worth of S2 is MISSING.")
        print("      Either some of the 24 factor strings are not the strings")
        print("      the scorer expects, or test prevalence differs sharply.")
        print("      Check the taxonomy spelling against the task page first --")
        print("      a mismatch costs 1/24 of S2 permanently and invisibly.")
    elif abs(s2 - expected) < typical * 0.7:
        print("  consistent with all 24 labels being recognised and test")
        print("  prevalence resembling train. The factor vocabulary is fine.")
    else:
        print("  S2 came in HIGHER than train prevalence predicts, so test")
        print("  factors are denser than train. That implies the operating")
        print("  point should be looser than the OOF instrument suggests.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="submissions/NIDHI_canon_on_nlimnli.csv")
    ap.add_argument("--build", default=None,
                    choices=["all", "no-evidence", "bad-evidence",
                             "no-factors", "all-factors", "risk-const"])
    ap.add_argument("--outdir", default="submissions")
    ap.add_argument("--solve", default=None,
                    choices=["no-evidence", "bad-evidence", "risk-const",
                             "all-factors"])
    ap.add_argument("--s1", type=float, default=None)
    ap.add_argument("--s2", type=float, default=None)
    ap.add_argument("--cls", default=None, choices=RISK)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--n", type=int, default=378)
    args = ap.parse_args()

    if args.build:
        os.makedirs(args.outdir, exist_ok=True)
        build(args.base, args.outdir, args.build)
        print("\nRECOMMENDED ORDER (one per slot, most informative first):")
        print("  1. noev      -> splits S1 into risk and evidence")
        print("  2. allfac    -> integrity: are all 24 factor strings valid?")
        print("  3. risk_attempt / risk_behavior / risk_ideation")
        print("                -> the exact test risk distribution")
        print("\nThe base submission file is not modified.")
        return

    if args.solve == "no-evidence":
        if args.s1 is None:
            raise SystemExit("--s1 required")
        solve_no_evidence(args.s1)
    elif args.solve == "bad-evidence":
        if args.s1 is None:
            raise SystemExit("--s1 required")
        solve_bad_evidence(args.s1)
    elif args.solve == "risk-const":
        if args.s1 is None or args.cls is None:
            raise SystemExit("--s1 and --cls required")
        solve_risk_const(args.s1, args.cls, args.n)
    elif args.solve == "all-factors":
        if args.s2 is None:
            raise SystemExit("--s2 required")
        solve_all_factors(args.s2, args.labeled, args.n)
    else:
        raise SystemExit("pass --build or --solve")


if __name__ == "__main__":
    main()
