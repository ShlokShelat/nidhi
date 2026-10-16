"""
factor_operating_point.py

Where is the macro-F1 optimum as a function of how many factors we predict?

WHY. Across three different models, predicted factor count and score move
together monotonically -- more factors tracks a higher S2 -- which suggests
the shipped cutoffs may be too strict. But those three models are three
DIFFERENT MODELS, so count is completely confounded with model identity.
"More factors is better" and "the champion is simply the best model" fit
that data equally well.

This disentangles them on held-out data, for free. It holds ONE model fixed and
sweeps only its operating point: every deployed threshold is multiplied by a
constant c, and macro-F1 is measured across the 1635 OOF posts. Nothing is
retrained.

The instrument already hints at the answer without ever testing above the
shipped point:
    3.37 factors/post (deployed thresholds)   0.6203
    3.18              (refit)                 0.6168
    2.92              (ratematch)             0.6027
Monotone in the same direction -- but 3.37 was simply the loosest policy tried.
If the peak sits ABOVE 3.44 this supports loosening; if 3.44 is already past the
peak, the earlier readings were about model quality and loosening will
lose.

THE SCALE IS CROSS-FITTED. c is chosen on one author half and scored on the
other, so the reported optimum is not the in-sample best. That matters: picking
c on the same posts you score it on would manufacture a peak wherever the noise
happened to be.

Usage:
  python src/factor_operating_point.py --arm mnli
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_oof_instrument import (author_halves, gold_matrix,  # noqa: E402
                                   macro_f1, per_label_f1)
from scoring import FACTOR_TAXONOMY  # noqa: E402

N = len(FACTOR_TAXONOMY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mnli")
    ap.add_argument("--dir", default="artifacts/factor_oof")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    PR, TH, rid = [], [], []
    for f in range(args.k):
        p = os.path.join(args.dir, f"{args.arm}_f{f}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        z = np.load(p, allow_pickle=True)
        pr = np.asarray(z["probs"], dtype=float)
        PR.append(pr)
        TH.append(np.tile(np.asarray(z["thresholds"], dtype=float),
                          (len(pr), 1)))
        rid += list(z["row_id"])
    P = np.concatenate(PR, axis=0)
    T = np.concatenate(TH, axis=0)     # per-post copy of its fold's thresholds

    lab = pd.read_csv(args.labeled, keep_default_na=False).set_index("row_id")
    ev = lab.loc[rid].reset_index()
    Y = gold_matrix(ev)
    auth = ev["anon_user_id"].to_numpy()
    half = author_halves(auth)
    A, B = half == "A", half == "B"

    base = macro_f1((P >= T).astype(np.int8), Y)
    base_n = float((P >= T).sum()) / len(Y)
    print(f"arm {args.arm}   {len(Y)} OOF posts   {pd.unique(auth).size} authors")
    print(f"shipped operating point: {base_n:.2f} factors/post   "
          f"macro {base:.4f}")
    print(f"gold: {Y.sum(1).mean():.2f} factors/post")
    print("\nc < 1 loosens every cutoff proportionally and predicts MORE.\n")

    grid = np.round(np.arange(0.55, 1.51, 0.05), 2)
    print(f"{'c':>6}{'fac/post':>10}{'macro':>9}{'d vs shipped':>14}")
    print("-" * 40)
    rows = []
    for c in grid:
        pred = (P >= T * c).astype(np.int8)
        m = macro_f1(pred, Y)
        fp = float(pred.sum()) / len(Y)
        rows.append((c, fp, m))
        mark = "   <- shipped" if abs(c - 1.0) < 1e-9 else ""
        star = "  *" if m > base else ""
        print(f"{c:>6.2f}{fp:>10.2f}{m:>9.4f}{m - base:>+14.4f}{mark}{star}")

    best = max(rows, key=lambda r: r[2])
    print(f"\nin-sample best: c={best[0]:.2f} at {best[1]:.2f} factors/post "
          f"({best[2]:.4f}, {best[2]-base:+.4f})")

    # ---- cross-fitted: pick c on one half, score the other ------------------
    def pick(rows_mask):
        return max(grid, key=lambda c:
                   macro_f1((P[rows_mask] >= T[rows_mask] * c).astype(np.int8),
                            Y[rows_mask]))

    cA, cB = pick(B), pick(A)     # cA is chosen on B, applied to A
    X = np.zeros_like(Y)
    X[A] = (P[A] >= T[A] * cA).astype(np.int8)
    X[B] = (P[B] >= T[B] * cB).astype(np.int8)
    xf = macro_f1(X, Y)
    print(f"cross-fitted: c={cA:.2f} on half A, {cB:.2f} on half B   "
          f"macro {xf:.4f} ({xf - base:+.4f})   "
          f"{float(X.sum())/len(Y):.2f} factors/post")

    # ---- paired author bootstrap on the cross-fitted variant ----------------
    uniq = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uniq}
    rng = np.random.default_rng(args.seed)
    d = np.empty(args.boot)
    B0 = (P >= T).astype(np.int8)
    for i in range(args.boot):
        p_ = rng.choice(len(uniq), size=len(uniq), replace=True)
        ix = np.concatenate([by[uniq[k]] for k in p_])
        yb = Y[ix]
        d[i] = macro_f1(X[ix], yb) - macro_f1(B0[ix], yb)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"paired delta {xf - base:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"win {100*float((d > 0).mean()):.1f}%")

    print("\n" + "=" * 62)
    if cA < 1.0 and cB < 1.0:
        print("BOTH halves independently chose to LOOSEN. That supports the")
        print("earlier trend and makes c40 a reasoned bet rather than a")
        print("coin flip. The size of the gain still has to clear the ~0.0157")
        print("null band to be worth much.")
    elif cA > 1.0 and cB > 1.0:
        print("BOTH halves chose to TIGHTEN. The shipped point is already past")
        print("the peak, so the earlier readings were about model")
        print("quality, not operating point -- c40 will likely LOSE.")
        print("DO NOT UPLOAD IT.")
    elif cA == 1.0 and cB == 1.0:
        print("BOTH halves chose EXACTLY the shipped point. The cutoffs are")
        print("already at the held-out optimum, so moving them in either")
        print("direction should lose. DO NOT UPLOAD c40.")
    else:
        print(f"The halves disagree (c={cA:.2f} vs {cB:.2f}), so the optimum is")
        print("not resolvable here and c40 stays a coin flip. Uploading it buys")
        print("information either way, but do not expect a gain.")
    print("=" * 62)
    print("\nNOTE this sweep moves the operating point of ONE model on data it")
    print("never trained on. Unlike the earlier readings, count is")
    print("NOT confounded with model identity here -- which is exactly the")
    print("question c40 was built to answer.")


if __name__ == "__main__":
    main()
