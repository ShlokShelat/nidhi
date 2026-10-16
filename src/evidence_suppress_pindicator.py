#!/usr/bin/env python
"""Additionally suppress evidence on high-p_indicator posts. CPU, seconds.

THE MECHANIC (measured, not assumed). Evidence phrase-F1 is MACRO over posts
and a post with empty gold AND empty prediction scores a free 1.0. Measured on
the val decode:

    emitting a span on a gold-EMPTY post costs exactly            1.000
    suppressing a post that HAD gold costs, on the real 18 cases  0.130

It is only 0.130 because the ambiguity that fools the risk model also silences
the taggers -- those posts were scoring badly anyway. With that 7.7:1
asymmetry the break-even suppression point is

    P(gold empty) > q/(1+q) = 0.115,   NOT 0.5

so an argmax decision rule STRUCTURALLY OVER-EMITS. The shipped decode
suppresses only when the argmax risk label is `indicator`; this adds a second
gate on the blended indicator probability.

THE HONEST CAVEAT. The 0.45 threshold was FITTED on val, so this particular
operating point is not guaranteed to generalize -- the cost asymmetry above is
solid, but a fitted threshold on top of it is not.

The blended probability reproduces the deployed risk chain's first stage:
    M = 0.70 * ensembled_risk.csv + 0.30 * nli_risk_test_probs.csv
Risk labels and factors are carried through untouched -- only `evidence` moves,
so S1 must move ONLY through its evidence term and S2 must be unchanged.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, parse_evidence  # noqa: E402


def pick(df, level):
    """Column for one risk level, tolerating p_Indicator vs prob_indicator."""
    for c in df.columns:
        if c == "row_id":
            continue
        s = c.lower().replace("prob_", "").replace("p_", "").strip()
        if s == level:
            return c
    raise SystemExit(f"no column for '{level}' in {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="submissions/NIDHI_ff11.csv")
    ap.add_argument("--ens", default="artifacts/ensembled_risk.csv")
    ap.add_argument("--nli", default="artifacts/nli_risk_test_probs.csv")
    ap.add_argument("--w-ens", type=float, default=0.70)
    ap.add_argument("--t", type=float, default=0.45)
    ap.add_argument("--max-fire", type=int, default=25,
                    help="refuse if the rule fires on more rows than this. On "
                         "val it fired on 6 of 372; a much larger count on "
                         "test means the blended probability is not on the "
                         "same scale and the threshold does not transfer.")
    ap.add_argument("--out", default="submissions/NIDHI_psupp.csv")
    args = ap.parse_args()

    base = pd.read_csv(args.base, keep_default_na=False)
    base["row_id"] = base["row_id"].astype(str)
    ens = pd.read_csv(args.ens, keep_default_na=False)
    ens["row_id"] = ens["row_id"].astype(str)
    nli = pd.read_csv(args.nli, keep_default_na=False)
    nli["row_id"] = nli["row_id"].astype(str)

    ec, nc = pick(ens, "indicator"), pick(nli, "indicator")
    em = dict(zip(ens["row_id"], ens[ec].astype(float)))
    nm = dict(zip(nli["row_id"], nli[nc].astype(float)))
    missing = [r for r in base["row_id"] if r not in em or r not in nm]
    if missing:
        raise SystemExit(f"{len(missing)} base rows lack a probability; "
                         f"first: {missing[:3]}")

    p = np.array([args.w_ens * em[r] + (1 - args.w_ens) * nm[r]
                  for r in base["row_id"]])
    lab = [normalize_label(x) for x in base["risk_level"]]
    has_ev = [len(parse_evidence(x)) > 0 for x in base["evidence"]]

    fire = [(p[i] >= args.t) and (lab[i] != "indicator") and has_ev[i]
            for i in range(len(base))]
    n_fire = int(sum(fire))

    print(f"{len(base)} rows   blended p_indicator: "
          f"mean {p.mean():.3f}  median {np.median(p):.3f}")
    print(f"already suppressed (argmax indicator): "
          f"{sum(1 for l in lab if l == 'indicator')}")
    print(f"rule fires on {n_fire} additional rows at t={args.t}")
    print(f"  their p_indicator: "
          f"{', '.join(f'{p[i]:.2f}' for i in range(len(base)) if fire[i])}")

    if n_fire == 0:
        raise SystemExit("fires on 0 rows -- nothing to ship.")
    if n_fire > args.max_fire:
        raise SystemExit(
            f"REFUSING: fires on {n_fire} rows, above --max-fire "
            f"{args.max_fire}. On val it fired on 6 of 372. A much larger "
            f"count means the blended probability is not on the val scale, so "
            f"the fitted 0.45 does not transfer and this is no longer the "
            f"change that was measured.")

    out = base.copy()
    ev = list(out["evidence"])
    for i in range(len(out)):
        if fire[i]:
            ev[i] = "none"
    out["evidence"] = ev

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    nr = int((base["risk_level"].astype(str) != out["risk_level"].astype(str)).sum())
    nf = int((base["factors"].astype(str) != out["factors"].astype(str)).sum())
    ne = int((base["evidence"].astype(str) != out["evidence"].astype(str)).sum())
    print(f"\nwrote {args.out}")
    print(f"  risk_level differs {nr}   factors differs {nf}   "
          f"evidence differs {ne}")
    if nr or nf:
        raise SystemExit("REFUSING: risk or factors moved. Only evidence may "
                         "change.")
    spans_before = sum(len(parse_evidence(x)) for x in base["evidence"])
    spans_after = sum(len(parse_evidence(x)) for x in out["evidence"])
    print(f"  spans {spans_before} -> {spans_after} "
          f"({spans_before - spans_after} removed)")
    print(f"  empty-evidence rows "
          f"{sum(1 for x in base['evidence'] if not parse_evidence(x))} -> "
          f"{sum(1 for x in out['evidence'] if not parse_evidence(x))}")
    print("\nS2 MUST return exactly 0.6908 (factors untouched).")
    print("S1 moves only through its evidence term; risk is byte-identical.")


if __name__ == "__main__":
    main()
