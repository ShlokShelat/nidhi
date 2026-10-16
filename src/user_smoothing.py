"""
user_smoothing.py

The factor model scores every post in isolation. It should not.

THE SIGNAL. Measured on all 1635 labelled posts (153 users, 10.7 posts each),
P(a post carries label L | ANOTHER post by the same user carries L) against the
marginal P(L):

    low socio-economic status      3.3% ->  20.4%   6.2x
    sexual orientation             0.5% ->   2.6%   5.3x
    cognitive deficits             2.0% ->   9.7%   4.8x
    exposure to others' suicide    0.9% ->   4.0%   4.7x
    substance use                  2.0% ->   9.5%   4.7x
    poor school performance        1.0% ->   3.0%   3.1x
    meaning in life                2.8% ->   7.1%   2.6x

Nineteen of 24 labels have lift above 1.3, and the lift is LARGEST on the rare
labels -- which is exactly where macro-F1 concentrates, since every label
contributes 1/24 regardless of prevalence. A factor like `low socio-economic
status` is substantially a property of the PERSON, not of the individual post.

And the test set is built the same way: 378 posts over 36 users, 10.5 posts
each, every user with more than one post. The information is sitting there.

WHAT THIS DOES. For each post and label, blend the post's own probability with
the mean probability over that user's OTHER posts:

    p_adj = (1 - w) * p_post + w * mean(p over the same user's other posts)

LEAVE-ONE-OUT is essential: including the post itself would just shrink every
probability toward a value it already dominates and would leak at test time in
a way that cannot be reproduced honestly. Excluding it makes the second term a
genuine outside signal.

w = 0.5 is the equal-weight, ZERO-FITTED-PARAMETER choice (the sweep below is
printed for information; picking a fitted value from it would add one
parameter, and fitted parameters of this kind have generalized poorly from
val to test).

HOW IT IS JUDGED. Thresholds are re-fitted for each w -- smoothing changes the
probability distribution, so reusing the old cut-offs would measure the wrong
thing. Cross-fitted over author-disjoint halves, which is safe here because a
user lives entirely within one half, so the leave-one-out mean never crosses
the split.

Usage:
  python src/user_smoothing.py --runs factors_nli_mnli factors_nli_mnli_s2
  python src/user_smoothing.py --runs factors_nli factors_nli_s2
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

N = len(FACTOR_TAXONOMY)
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def f1_at(pred, y):
    tp = int((pred & y).sum())
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + int((pred & (1 - y)).sum())
                     + int(((1 - pred) & y).sum()))


def macro(P, Y):
    return float(np.mean([f1_at(P[:, j], Y[:, j]) for j in range(N)]))


def per_label(P, Y):
    return [f1_at(P[:, j], Y[:, j]) for j in range(N)]


def fit_th(PR, Y, rows):
    th = np.full(N, 0.5)
    for j in range(N):
        if Y[rows, j].sum() == 0:
            continue
        best, bt = -1.0, 0.5
        for t in GRID:
            v = f1_at((PR[rows, j] >= t).astype(int), Y[rows, j])
            if v > best:
                best, bt = v, t
        th[j] = bt
    return th


def author_halves(auth):
    s = pd.Series(auth)
    c = s.value_counts()
    half, size = {}, {"A": 0, "B": 0}
    for a in sorted(c.index, key=lambda x: (-c[x], str(x))):
        p = "A" if size["A"] <= size["B"] else "B"
        half[a] = p
        size[p] += c[a]
    return s.map(half).to_numpy()


def loo_user_mean(P, users):
    """Mean probability over each user's OTHER posts, per label."""
    out = np.zeros_like(P)
    u = pd.Series(users)
    for uid in u.unique():
        m = (u == uid).to_numpy()
        k = int(m.sum())
        if k <= 1:
            out[m] = P[m]          # nothing to borrow from
            continue
        tot = P[m].sum(0)
        out[m] = (tot - P[m]) / (k - 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["factors_nli_mnli", "factors_nli_mnli_s2"])
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    VP, rid = [], None
    for r in args.runs:
        d = np.load(f"{args.dir}/{r}_val.npz", allow_pickle=True)
        VP.append(d["probs"])
        rid = list(d["row_id"])
    P = np.mean(VP, axis=0)

    va = va.set_index("row_id").loc[rid].reset_index()
    if "anon_user_id" not in va.columns:
        raise SystemExit("val split has no anon_user_id column")
    users = va["anon_user_id"].to_numpy()

    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(rid), N), dtype=int)
    for i, cell in enumerate(va["factors"]):
        for f in parse_factors(cell):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1

    nu = len(set(users))
    print(f"{len(rid)} val posts, {nu} users, {len(rid)/nu:.1f} posts/user")
    print(f"users with >1 post: "
          f"{int((pd.Series(users).value_counts() > 1).sum())} of {nu}\n")

    U = loo_user_mean(P, users)
    half = author_halves(users)
    A, B = half == "A", half == "B"

    print("=" * 78)
    print("USER SMOOTHING, thresholds re-fitted per w and cross-fitted by author")
    print("=" * 78)
    print(f"{'w':>6}{'macro':>10}{'d vs w=0':>11}{'halfA':>9}{'halfB':>9}"
          f"{'/post':>8}")
    print("-" * 78)
    base = None
    store = {}
    for w in args.weights:
        Q = (1 - w) * P + w * U
        Px = np.zeros_like(Y)
        Px[A] = (Q[A] >= fit_th(Q, Y, B)).astype(int)
        Px[B] = (Q[B] >= fit_th(Q, Y, A)).astype(int)
        mA, mB = macro(Px[A], Y[A]), macro(Px[B], Y[B])
        m = macro(Px, Y)
        if base is None:
            base = m
        store[w] = (m, per_label(Px, Y))
        mark = "   <- zero-parameter choice" if abs(w - 0.5) < 1e-9 else ""
        print(f"{w:>6.2f}{m:>10.4f}{m-base:>+11.4f}{mA:>9.4f}{mB:>9.4f}"
              f"{Px.sum(1).mean():>8.2f}{mark}")

    print("\nw=0 is the current architecture: every post scored in isolation.")
    print("Judge w=0.50 only. Picking the best w here fits one parameter, and")
    print("fitted parameters of this kind have generalized poorly to test.")

    if 0.5 in store and 0.0 in store:
        print("\n" + "=" * 78)
        print("PER LABEL, w=0 -> w=0.5   (movers only, sorted by rarity)")
        print("=" * 78)
        a, b = store[0.0][1], store[0.5][1]
        n = Y.sum(0)
        print(f"{'factor':<44}{'n':>4}{'w=0':>8}{'w=0.5':>8}{'delta':>8}")
        print("-" * 74)
        rare = com = 0.0
        for j in sorted(range(N), key=lambda x: n[x]):
            d = b[j] - a[j]
            if n[j] <= 20:
                rare += d
            else:
                com += d
            if abs(d) > 0.01:
                print(f"{FACTOR_TAXONOMY[j]:<44}{int(n[j]):>4}"
                      f"{a[j]:>8.3f}{b[j]:>8.3f}{d:>+8.3f}")
        print(f"\n  total delta on RARE labels (n <= 20 in val) {rare:>+7.3f}")
        print(f"  total delta on COMMON labels                {com:>+7.3f}")
        print("\n  The measured lift is largest on the rare labels, so that is")
        print("  where a gain should appear. A gain concentrated on the common")
        print("  labels would not match the mechanism and should be distrusted.")


if __name__ == "__main__":
    main()
