#!/usr/bin/env python
"""WEIGHTED arm pooling for factors.

The factor column pools its arms with `logit_mean`, an UNWEIGHTED mean of
logits, even though the arms differ in solo quality. This script compares that
default against two weighting schemes.

THREE SCHEMES, in increasing risk of overfitting:

  equal     the unweighted logit_mean                    0 params
  quality   w proportional to each arm's OWN solo macro  0 FITTED params --
            the weights come from the arms, not from the scored data
  fitted    coordinate ascent on the FIT authors         one param per arm

`quality` is the informative middle case. It cannot overfit the evaluation set
because nothing is tuned on it, yet it still drops the equal-weight
assumption. `quality` winning while `fitted` does not would mean there is real
signal in arm quality but too few authors to estimate one free weight per arm.

PROTOCOL. Weights AND thresholds are both fitted on FIT authors and scored on
HELD-OUT authors over many splits. Judge by WIN RATE and per-split sd rather
than the CI of the mean: with this few authors the per-split spread is large,
so a mean can look significant while individual splits disagree. A factor macro
delta becomes composite by x0.3, and transfer to held-out authors is not
one-to-one, so treat the SIGN as the finding rather than the magnitude.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_threshold_decode import (  # noqa: E402
    N, best_threshold_and_f1, cells, load_oof)

SHIPPED = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli",
           "ltail", "friendfac", "cbnsynapi"]


def wlogit(S, w):
    """Weighted logit-mean. w is (n_arms,), non-negative, renormalised."""
    X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
    L = np.log(X / (1 - X))                      # (arms, posts, labels)
    w = np.asarray(w, dtype=float)
    w = w / max(w.sum(), 1e-9)
    return 1.0 / (1.0 + np.exp(-np.tensordot(w, L, axes=(0, 0))))


def fit_thresholds(P, Y):
    th = np.zeros(N)
    for j in range(N):
        th[j], _ = best_threshold_and_f1(P[:, j], Y[:, j])
    return th


def macro_at(P, Y, th):
    Pb = (P >= th[None, :])
    tp = (Pb & (Y == 1)).sum(0)
    den = 2 * tp + (Pb & (Y == 0)).sum(0) + (~Pb & (Y == 1)).sum(0)
    return float(np.mean(np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=SHIPPED)
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--rounds", type=int, default=2,
                    help="coordinate-ascent sweeps for the `fitted` scheme")
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    S = np.asarray(S, dtype=float)                # (arms, posts, labels)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], len(rid))
    acol = "anon_user_id" if "anon_user_id" in lab.columns else "author"
    authors = np.array([str(lab.loc[r, acol]) for r in rid])
    ua = np.array(sorted(set(authors)))
    print(f"{len(rid)} OOF posts / {len(ua)} authors / {len(args.arms)} arms")

    # ---- solo quality per arm (used by the `quality` scheme) ---------------
    solo = np.zeros(len(args.arms))
    for a in range(len(args.arms)):
        th = fit_thresholds(S[a], Y)
        solo[a] = macro_at(S[a], Y, th)
    print("\nsolo macro per arm (this is what `quality` weights by):")
    for a, nm in enumerate(args.arms):
        print(f"  {nm:<12} {solo[a]:.4f}")
    # centre on the weakest so differences, not levels, drive the weights
    qw = np.clip(solo - solo.min() + 1e-3, 1e-3, None)

    rng = np.random.default_rng(args.seed)
    res = {"equal": [], "quality": [], "fitted": []}
    picked = []

    for _ in range(args.splits):
        perm = rng.permutation(len(ua))
        fa = set(ua[perm[:len(ua) // 2]])
        fm = np.array([a in fa for a in authors])
        if fm.all() or not fm.any():
            continue
        Sf, Yf = S[:, fm], Y[fm]
        Sh, Yh = S[:, ~fm], Y[~fm]

        def score(w):
            Pf = wlogit(Sf, w)
            th = fit_thresholds(Pf, Yf)          # thresholds fitted on FIT too
            return macro_at(wlogit(Sh, w), Yh, th), th

        eq = np.ones(len(args.arms))
        res["equal"].append(score(eq)[0])
        res["quality"].append(score(qw)[0])

        # coordinate ascent on FIT only
        w = np.ones(len(args.arms))
        for _r in range(args.rounds):
            for a in range(len(args.arms)):
                best, bv = w[a], -1.0
                for cand in (0.0, 0.5, 1.0, 1.5, 2.0):
                    w2 = w.copy()
                    w2[a] = cand
                    if w2.sum() <= 0:
                        continue
                    Pf = wlogit(Sf, w2)
                    v = macro_at(Pf, Yf, fit_thresholds(Pf, Yf))
                    if v > bv:
                        bv, best = v, cand
                w[a] = best
        res["fitted"].append(score(w)[0])
        picked.append(w.copy())

    e = np.array(res["equal"])
    print("\n" + "=" * 68)
    print("HELD-OUT macro (weights AND thresholds fitted on FIT authors)")
    print("=" * 68)
    print(f"{'scheme':<10}{'macro':>9}{'d vs equal':>12}{'win':>7}"
          f"{'sd':>9}{'d comp':>9}")
    for k in ("equal", "quality", "fitted"):
        v = np.array(res[k])
        d = v - e
        w_ = 100 * float((d > 0).mean()) if k != "equal" else float("nan")
        print(f"{k:<10}{v.mean():>9.4f}{d.mean():>+12.4f}"
              f"{w_:>6.0f}%{d.std(ddof=1):>9.4f}{0.3*d.mean():>+9.4f}")

    if picked:
        W = np.mean(picked, axis=0)
        print("\nmean fitted weight per arm (1.0 = equal):")
        for a, nm in enumerate(args.arms):
            print(f"  {nm:<12} {W[a]:.2f}")

    print("\n" + "=" * 68)
    best = max(("quality", "fitted"),
               key=lambda k: (np.array(res[k]) - e).mean())
    d = np.array(res[best]) - e
    w_ = 100 * float((d > 0).mean())
    if d.mean() > 0 and w_ >= 90:
        print(f"{best} CLEARS: {d.mean():+.4f} macro at {w_:.0f}% win "
              f"= {0.3*d.mean():+.4f} composite.")
        print("Before building with these weights, verify that the equal-")
        print("weight rebuild still reproduces the reference column, so a")
        print("rebuild bug cannot be mistaken for the weighting effect.")
    else:
        print(f"Best is {best} at {d.mean():+.4f} / {w_:.0f}% win -- below the")
        print("90% win-rate bar, so equal weighting is not the limitation.")
    print("=" * 68)


if __name__ == "__main__":
    main()
