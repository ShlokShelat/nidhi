"""
splice_factor_columns.py

Combine a specialist's factor columns with the champion's for the rest.

WHY. Held-out F1 splits by polarity -- the five protective factors average
0.504 against 0.651 for the nineteen risk ones, and the polar pairs are worse
(`social support` 0.533 vs `poor social support` 0.817). The organizers' own
paper (arXiv 2507.10008) learns risk and protective factors with SEPARATE
parameters, on the stated grounds that they "operate independently rather than
being the reverse of risk factors". A specialist trained with
FACTOR_SUBSET=protective never has to switch polarity mid-task.

Each side keeps ITS OWN thresholds. That matters: the specialist's probability
distribution is not the champion's, and reusing one model's cutoffs on another's
scores is precisely the mistake that has cost score before. Nothing is refitted
here -- every threshold comes from the run that produced the probabilities it is
applied to.

The specialist's npz has zeros in the columns it never scored (score_all skips
non-subset factors under FACTOR_SUBSET), so the splice is verified rather than
assumed: a specialist column that is all-zero, or a champion column being
overwritten by an untrained one, aborts.

Usage:
  python src/splice_factor_columns.py \
      --base artifacts/factor_runs/nli_ensemble_mnli_test.npz \
      --specialist artifacts/factor_runs/nli_ensemble_prot_test.npz \
      --out artifacts/nli_ensemble_factors_prot.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY  # noqa: E402

N = len(FACTOR_TAXONOMY)
PROTECTIVE = ["social support", "coping strategy", "psychological capital",
              "sense of responsibility", "meaning in life"]


def load(p):
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    z = np.load(p, allow_pickle=True)
    return (np.asarray(z["probs"], dtype=float),
            np.asarray(z["thresholds"], dtype=float),
            [str(x) for x in z["row_id"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="the champion's test npz -- supplies every column not "
                         "taken from the specialist")
    ap.add_argument("--specialist", required=True)
    ap.add_argument("--factors", default=",".join(PROTECTIVE),
                    help="which columns come from the specialist")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare",
                    default="submissions/NIDHI_canon_on_nlimnli.csv")
    args = ap.parse_args()

    Pb, Tb, rb = load(args.base)
    Ps, Ts, rs = load(args.specialist)
    if rb != rs:
        order = {r: i for i, r in enumerate(rs)}
        if set(rb) != set(rs):
            raise SystemExit("base and specialist score different posts")
        idx = [order[r] for r in rb]
        Ps = Ps[idx]
        print("NOTE realigned the specialist to the base's row order")
    n = len(rb)

    want = [f.strip() for f in args.factors.split(",") if f.strip()]
    bad = [f for f in want if f not in FACTOR_TAXONOMY]
    if bad:
        raise SystemExit(f"not in the taxonomy: {bad}")
    cols = [FACTOR_TAXONOMY.index(f) for f in want]

    # the specialist must actually have scored these, and only these
    dead = [FACTOR_TAXONOMY[j] for j in cols if not np.any(Ps[:, j] > 0)]
    if dead:
        raise SystemExit(
            f"specialist produced all-zero probabilities for {dead} -- it did "
            f"not score them. Was FACTOR_SUBSET set correctly at inference?")
    stray = [FACTOR_TAXONOMY[j] for j in range(N)
             if j not in cols and np.any(Ps[:, j] > 0)]
    if stray:
        print(f"NOTE specialist also scored {stray}; those columns are being "
              f"IGNORED and taken from the base instead.")

    on_b = (Pb >= Tb).astype(int)
    P, T = Pb.copy(), Tb.copy()
    for j in cols:
        P[:, j] = Ps[:, j]
        T[j] = Ts[j]          # each side keeps its OWN cutoff
    on = (P >= T).astype(int)

    print(f"{n} posts   splicing {len(cols)} of {N} columns from the specialist")
    print(f"\n{'factor':<34}{'base th':>9}{'spec th':>9}{'base n':>8}"
          f"{'new n':>7}{'delta':>7}")
    print("-" * 76)
    for j in cols:
        print(f"{FACTOR_TAXONOMY[j]:<34}{Tb[j]:>9.2f}{Ts[j]:>9.2f}"
              f"{int(on_b[:, j].sum()):>8}{int(on[:, j].sum()):>7}"
              f"{int(on[:, j].sum()) - int(on_b[:, j].sum()):>+7}")

    zero = [FACTOR_TAXONOMY[j] for j in range(N) if on[:, j].sum() == 0]
    if zero:
        raise SystemExit(f"REFUSING: {zero} predicted zero times "
                         f"(each costs 1/24 of S2 outright)")

    out = pd.DataFrame({
        "row_id": rb,
        "factors": [str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
                    for i in range(n)],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(f"  {on.sum(1).mean():.2f} factors/post   "
          f"base was {on_b.sum(1).mean():.2f}   gold 2.92")

    if os.path.exists(args.compare):
        ch = pd.read_csv(args.compare, keep_default_na=False).set_index("row_id")
        ch_changed = sum(1 for r, f in zip(out.row_id, out.factors)
                         if str(ch.loc[r, "factors"]) != f)
        print(f"  changes the champion on {ch_changed} of {n} posts "
              f"({100*ch_changed/n:.1f}%)")
        if ch_changed < 40:
            print("  UNDER 40 -- below the resolution of a 378-post test set.")


if __name__ == "__main__":
    main()
