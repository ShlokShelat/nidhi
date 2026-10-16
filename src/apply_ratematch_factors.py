"""
apply_ratematch_factors.py

Turn factor probabilities into a factor column using PREVALENCE RATE-MATCHING
instead of fitted thresholds.

WHY THIS EXISTS. Training on all labelled posts leaves no held-out set on which
to fit per-label thresholds. Rate-matching removes the need for one: for each
label it keeps the top-k predictions, where k is what the label's known training
prevalence implies for the number of posts being scored. It fits nothing, reads
no test label, and has zero parameters to transfer -- so it cannot fail the way
a threshold or weight fitted on one distribution and applied to another does.

WHAT IT DOES NOT DO. It forces the predicted count per label to match train
prevalence exactly. If test prevalence genuinely differs from train prevalence,
that mismatch is a cost this rule pays by construction.

Usage:
  python src/apply_ratematch_factors.py \
      --probs artifacts/factor_runs/nli_ensemble_mnliall_test.npz \
      --out artifacts/nli_ensemble_factors_mnliall.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True,
                    help="npz from predict_factors_nli.py --dump-probs")
    ap.add_argument("--labeled", default="artifacts/full_train.csv",
                    help="source of prevalence; NEVER the scored posts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default="submissions/NIDHI_canon_on_nlimnli.csv",
                    help="reference submission, for a change count")
    args = ap.parse_args()

    z = np.load(args.probs, allow_pickle=True)
    P = np.asarray(z["probs"], dtype=float)
    rid = [str(x) for x in z["row_id"]]
    n = len(rid)
    if P.shape != (n, N):
        raise SystemExit(f"probs shape {P.shape}, expected ({n}, {N})")

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    prev = np.array([sum(1 for x in lab["factors"] if t in parse_factors(x))
                     / len(lab) for t in FACTOR_TAXONOMY])

    print(f"{n} posts   prevalence from {args.labeled} ({len(lab)} posts)")
    if "thresholds" in z:
        th_in = np.asarray(z["thresholds"], dtype=float)
        print(f"IGNORING the npz's own thresholds (min {th_in.min():.2f} "
              f"max {th_in.max():.2f}) -- for an all-data model they were "
              f"fitted in-sample.")

    on = np.zeros((n, N), dtype=int)
    ths = np.zeros(N)
    for j in range(N):
        k = int(round(prev[j] * n))
        # never let a label go silent: a label predicted zero times scores F1=0
        # outright and costs 1/24 of S2, which no gain elsewhere recovers
        k = min(max(k, 1), n)
        order = np.argsort(-P[:, j], kind="stable")
        on[order[:k], j] = 1
        ths[j] = float(P[order[k - 1], j])

    print(f"\n{'factor':<44}{'prev%':>7}{'k':>5}{'cutoff':>8}")
    print("-" * 66)
    for j, t in enumerate(FACTOR_TAXONOMY):
        print(f"{t:<44}{100*prev[j]:>7.1f}{int(on[:, j].sum()):>5}{ths[j]:>8.3f}")

    dead = [FACTOR_TAXONOMY[j] for j in range(N) if on[:, j].sum() == 0]
    if dead:
        raise SystemExit(f"REFUSING: {dead} predicted zero times")

    out = pd.DataFrame({
        "row_id": rid,
        "factors": [str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
                    for i in range(n)],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    empty = int((on.sum(1) == 0).sum())
    print(f"\nwrote {args.out}")
    print(f"  {on.sum(1).mean():.2f} factors/post   {empty} posts with none")

    if os.path.exists(args.compare):
        ch = pd.read_csv(args.compare, keep_default_na=False).set_index("row_id")
        changed = sum(1 for r, f in zip(out.row_id, out.factors)
                      if str(ch.loc[r, "factors"]) != f)
        print(f"  changes the reference factor column on {changed} of {n} "
              f"posts ({100*changed/n:.1f}%)")
        if changed < 40:
            print("  UNDER 40 -- below the resolution of a 378-post test set.")


if __name__ == "__main__":
    main()
