"""
build_hybrid_defs.py

Keep the published definition where it already works; use the derived one only
where it does not.

WHY A HYBRID. src/derive_definitions.py rewrote all 24 definitions, including
labels the published wording already handles well. `hopelessness` sits at
AP 0.890 / F1 0.813 -- one of the strongest -- and the derivation narrows it
sharply ("not merely despair, self-hatred, or fleeting thoughts of death").
That narrowing might be right, or it might cost a label that was fine. The
all-derived run cannot tell those apart, because a gain on `meaning in life`
and a loss on `hopelessness` net out in the macro.

THE SELECTION PROBLEM, AND HOW IT IS AVOIDED. Picking which labels to swap
based on their measured F1 means choosing on the same 1635 posts we then score
-- 24 binary decisions fitted to the evaluation set. Milder than 24 continuous
thresholds fitted on 372 posts, but the same species of error, and that species
has cost this campaign three submissions.

So the swap list is CROSS-FITTED. Fold f's list is decided from the incumbent's
per-label F1 on the OTHER FOUR FOLDS. Fold f's own posts never influence which
definitions fold f uses. The lists therefore differ slightly between folds,
exactly as the thresholds and the derived definitions already do.

The cut-off is a single number (--cut), not 24. Labels below it get the derived
definition; labels at or above it keep the published one.

Usage:
  python src/build_hybrid_defs.py --cut 0.65 \
      --derived artifacts/derived_defs --out artifacts/hybrid_defs
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from scoring import FACTOR_TAXONOMY, parse_factors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--oof", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--derived", default="artifacts/derived_defs")
    ap.add_argument("--out", default="artifacts/hybrid_defs")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cut", type=float, default=0.65,
                    help="labels whose incumbent F1 is below this get the "
                         "derived definition")
    args = ap.parse_args()

    os.environ.pop("FACTOR_DEFS_JSON", None)
    from train_factors_nli import FACTOR_DEFINITIONS

    lab = pd.read_csv(args.labeled).set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

    # incumbent OOF predictions, kept fold-separated so F1 can be computed
    # while EXCLUDING one fold at a time
    fold_of, rids, PR, TH = [], [], [], []
    for f in range(args.k):
        p = args.oof.format(f=f)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = np.load(p, allow_pickle=True)
        n = len(d["row_id"])
        rids.extend(list(d["row_id"]))
        PR.append(d["probs"])
        TH.append(np.tile(d["thresholds"], (n, 1)))
        fold_of.extend([f] * n)
    PR, TH, fold_of = np.concatenate(PR), np.concatenate(TH), np.array(fold_of)
    P = (PR >= TH).astype(int)

    Y = np.zeros(P.shape, dtype=int)
    for i, r in enumerate(rids):
        for x in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1

    def f1(j, mask):
        tp = int((P[mask, j] & Y[mask, j]).sum())
        if tp == 0:
            return 0.0
        fp = int((P[mask, j] & (1 - Y[mask, j])).sum())
        fn = int(((1 - P[mask, j]) & Y[mask, j]).sum())
        return 2 * tp / (2 * tp + fp + fn)

    os.makedirs(args.out, exist_ok=True)
    print(f"cut = {args.cut}   (below -> derived, at/above -> published)\n")
    all_lists = {}
    for f in range(args.k):
        other = fold_of != f          # <- fold f's own posts excluded
        derived = json.load(open(os.path.join(args.derived, f"defs_f{f}.json")))
        out, swapped = {}, []
        for t in FACTOR_TAXONOMY:
            s = f1(idx[t], other)
            if s < args.cut:
                out[t] = derived[t]
                swapped.append(t)
            else:
                out[t] = FACTOR_DEFINITIONS[t]
        json.dump(out, open(os.path.join(args.out, f"defs_f{f}.json"), "w"),
                  indent=2)
        all_lists[f] = set(swapped)
        print(f"fold {f}: {len(swapped)}/24 swapped   "
              f"(decided on the other {args.k - 1} folds)")

    print(f"\n{'factor':<44}{'F1 (all folds)':>15}  swapped in folds")
    print("-" * 82)
    allm = np.ones(len(P), bool)
    for t in FACTOR_TAXONOMY:
        s = f1(idx[t], allm)
        fs = [str(f) for f in range(args.k) if t in all_lists[f]]
        tag = ",".join(fs) if fs else "-- kept published --"
        print(f"{t:<44}{s:>15.3f}  {tag}")

    stable = sum(1 for t in FACTOR_TAXONOMY
                 if len({t in all_lists[f] for f in range(args.k)}) == 1)
    print(f"\n{stable}/24 labels get the same treatment in all five folds.")
    print("Labels that flip between folds sit near the cut and their F1 is")
    print("unstable -- expect little from them either way.")
    print(f"\nwrote {args.out}/defs_f*.json")
    print("\nRun with:")
    print(f"  sbatch --export=ALL,DEFS={args.out},TAG=hyb "
          f"scripts/factors_derived_defs.sh")


if __name__ == "__main__":
    main()
