"""
factor_dupcluster.py

Average factor scores WITHIN near-duplicate post clusters before decoding.

THE MEASURED PREMISE. Annotators label near-identical text near-identically:
on train's 240 near-duplicate pairs (jaccard >= 0.5 on token sets) gold is
94.2% risk-consistent and 0.949 factor-cell-consistent, and 86% have identical
evidence. The champion is NOT: it is factor-inconsistent on 30 of 39 such pairs
in test. Averaging the bag's scores within a cluster before the quantile decode
forces the consistency the annotators show, and fits nothing.

WHY THIS IS NOT THE AUTHOR-SMOOTHING LANE THAT DIED. Pooling scores across ALL
of an author's posts was measured monotonically harmful as the pooling weight
increased, with the fraction of positive splits collapsing toward zero,
because model errors are author-correlated, so pooling amplifies them.
Near-duplicate clusters are a far narrower relation -- the SAME TEXT, not the
same person -- and a proxy measurement put the restricted version on the
positive side. Same signal, opposite sign, because the restriction is what
makes it valid.

WHAT THIS DOES. Cluster posts by token-set Jaccard above a threshold, average
the combined probability matrix within each cluster, decode with the SAME
fire-rate quantile rule, and score. Judged on the 1635 out-of-fold posts with
the shipped bag's own probabilities, paired over author halves -- not on a
proxy model.

FALSIFIER, FIXED BEFORE LOOKING. Must show a positive macro gain on the OOF
posts with BOTH author halves positive, at a threshold chosen a priori (0.5),
not swept. A threshold sweep here would be a fitted parameter, and fitted
parameters have consistently transferred only partially in this project.

Usage:
  python src/factor_dupcluster.py --arms mnli dmnli cmine prot d4 qnli mnli7 lnli
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_bag_ship import combine, f1_at  # noqa: E402
from scoring import FACTOR_TAXONOMY, normalize_span, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def clusters(posts, thr):
    """Greedy token-set-Jaccard clustering. Deterministic: posts are visited in
    order and joined to the first cluster whose representative they match."""
    toks = [set(normalize_span(p).split()) for p in posts]
    cl, of = [], [-1] * len(posts)
    for i, t in enumerate(toks):
        if not t:
            continue
        for ci, c in enumerate(cl):
            j = toks[c[0]]
            inter = len(t & j)
            if inter and inter / len(t | j) >= thr:
                c.append(i)
                of[i] = ci
                break
        else:
            of[i] = len(cl)
            cl.append([i])
    return [c for c in cl if len(c) > 1], of


def macro(P, Y):
    return float(np.mean([f1_at(P[:, j], Y[:, j]) for j in range(N)]))


def decode(S, Y, half):
    """Fire-rate quantile decode, thresholds cross-fitted over author halves."""
    X = np.zeros_like(Y)
    A, B = half == "A", half == "B"
    for src, dst in ((B, A), (A, B)):
        for j in range(N):
            best, bt = -1.0, 0.5
            for t in GRID:
                v = f1_at((S[src, j] >= t).astype(int), Y[src, j])
                if v > best:
                    best, bt = v, t
            X[dst, j] = (S[dst, j] >= bt).astype(int)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["mnli", "dmnli", "cmine", "prot", "d4", "qnli",
                             "mnli7", "lnli"])
    ap.add_argument("--rule", default="logit")
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    O, rid = [], None
    for a in args.arms:
        pr, ids = [], []
        for f in range(args.k):
            z = np.load(f"{args.dir}/{a}_cv_oof_f{f}.npz", allow_pickle=True)
            pr.append(z["probs"].astype(float))
            ids.extend(str(x) for x in z["row_id"])
        pr = np.concatenate(pr)
        if rid is None:
            rid = ids
        elif ids != rid:
            o = {k: i for i, k in enumerate(ids)}
            pr = pr[[o[k] for k in rid]]
        O.append(pr)
    S = combine(np.stack(O), args.rule, args.arms)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id").loc[rid]
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(rid), N), dtype=int)
    for i, c in enumerate(lab["factors"]):
        for x in parse_factors(c):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1
    auth = lab["anon_user_id"].to_numpy()

    from evidence_vote_oof import author_halves
    half = author_halves(auth)

    cl, _ = clusters([str(p) for p in lab["post"]], args.thr)
    n_in = sum(len(c) for c in cl)
    print(f"{len(rid)} OOF posts, {pd.unique(auth).size} authors")
    print(f"{len(cl)} near-duplicate clusters at jaccard>={args.thr}, "
          f"covering {n_in} posts ({100 * n_in / len(rid):.1f}%)")
    if not cl:
        raise SystemExit("no clusters -- nothing to average, lane is inert")

    # how inconsistent are we on these, and how consistent is gold?
    gc = pc = tot = 0
    base_pred = decode(S, Y, half)
    for c in cl:
        for a in range(len(c)):
            for b in range(a + 1, len(c)):
                tot += 1
                gc += int((Y[c[a]] == Y[c[b]]).all())
                pc += int((base_pred[c[a]] == base_pred[c[b]]).all())
    print(f"within-cluster pairs {tot}: gold identical on {gc} "
          f"({100 * gc / max(tot,1):.0f}%), ours identical on {pc} "
          f"({100 * pc / max(tot,1):.0f}%)")

    Sa = S.copy()
    for c in cl:
        Sa[c] = S[c].mean(axis=0)
    pred_a = decode(Sa, Y, half)

    b, a_ = macro(base_pred, Y), macro(pred_a, Y)
    print(f"\n{'arm':<28}{'macroF1':>10}{'delta':>10}{'composite':>12}")
    print("-" * 60)
    print(f"{'shipped decode':<28}{b:>10.4f}{0.0:>+10.4f}{0.0:>+12.4f}")
    print(f"{'+ dup-cluster averaging':<28}{a_:>10.4f}{a_ - b:>+10.4f}"
          f"{0.3 * (a_ - b):>+12.4f}")
    print(f"posts whose labels changed: "
          f"{int((base_pred != pred_a).any(axis=1).sum())}")

    uniq = pd.unique(auth)
    by = {x: np.flatnonzero(auth == x) for x in uniq}
    rng = np.random.default_rng(0)
    d = np.empty(args.boot)
    for i in range(args.boot):
        pk = rng.choice(len(uniq), len(uniq), replace=True)
        ix = np.concatenate([by[uniq[j]] for j in pk])
        d[i] = macro(pred_a[ix], Y[ix]) - macro(base_pred[ix], Y[ix])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\npaired author bootstrap: {a_ - b:+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  win {100 * (d > 0).mean():.1f}%")

    hA = half == "A"
    dA = macro(pred_a[hA], Y[hA]) - macro(base_pred[hA], Y[hA])
    dB = macro(pred_a[~hA], Y[~hA]) - macro(base_pred[~hA], Y[~hA])
    print(f"half A {dA:+.4f}   half B {dB:+.4f}")

    ok = (a_ - b) >= 0.004 and dA > 0 and dB > 0
    print("\n" + "=" * 60)
    print("PASSES -- worth one submission." if ok else
          "FAILS the pre-stated bar (positive gain, both halves). Lane closes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
