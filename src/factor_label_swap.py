#!/usr/bin/env python
"""Apply a purpose-built factor arm ONLY to the labels it was built for.

THE PROBLEM THIS SOLVES. Adding a 12th arm to an 11-arm logit mean gives it
1/12 weight on ALL 24 labels. `aspprot` is built to target the PROTECTIVE
labels (its hypotheses come from Wang et al., the source of the taxonomy);
spreading that targeted signal across 24 labels dilutes it on the labels it
was designed for while perturbing the 19 risk factors it was never meant to
touch.

WHY THIS IS NOT FREE PER-LABEL POOLING. `factor_perlabel_pool` searches over
which arms to use for each of 24 labels -- 24 selections fitted on the data,
which is enough degrees of freedom to overfit the evaluation. Here the label
set is PRE-SPECIFIED from the arm's design: protective labels score materially
below risk factors, the arm's hypotheses were written for exactly those labels,
and the choice is fixed before scoring. That is one a-priori decision, not 24
fitted ones.

FOUR PRE-SPECIFIED VARIANTS, all scored the same way:
  base    the current pool, unchanged
  add12   the new arm as a 12th member on every label (the standard move)
  swap    on the target labels ONLY, use the new arm alone
  boost-w on the target labels ONLY, count the new arm w extra times in the
          logit mean (w=1 is add12 restricted to those labels)

Thresholds are refit on FIT authors for every variant, and all variants are
scored on the same HELD-OUT authors. Judge by WIN RATE and per-split sd: the
CI of the mean across overlapping splits understates the spread of a single
draw. A factor macro delta enters the composite score with weight 0.3.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_threshold_decode import (  # noqa: E402
    N, best_threshold_and_f1, cells, load_oof)
from scoring import FACTOR_TAXONOMY  # noqa: E402

SHIPPED = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli",
           "ltail", "friendfac", "cbnsynapi"]
PROTECTIVE = ["social support", "coping strategy", "psychological capital",
              "sense of responsibility", "meaning in life"]
IDX = {t.strip().lower(): i for i, t in enumerate(FACTOR_TAXONOMY)}


def lmean(S, w=None):
    X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
    L = np.log(X / (1 - X))
    if w is None:
        m = L.mean(0)
    else:
        w = np.asarray(w, float)
        m = np.tensordot(w / w.sum(), L, axes=(0, 0))
    return 1.0 / (1.0 + np.exp(-m))


def fit_th(P, Y):
    th = np.zeros(N)
    for j in range(N):
        th[j], _ = best_threshold_and_f1(P[:, j], Y[:, j])
    return th


def macro_at(P, Y, th):
    Pb = P >= th[None, :]
    tp = (Pb & (Y == 1)).sum(0)
    den = 2 * tp + (Pb & (Y == 0)).sum(0) + (~Pb & (Y == 1)).sum(0)
    return float(np.mean(np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))))


def per_label_f1(P, Y, th):
    Pb = P >= th[None, :]
    tp = (Pb & (Y == 1)).sum(0)
    den = 2 * tp + (Pb & (Y == 0)).sum(0) + (~Pb & (Y == 1)).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=SHIPPED)
    ap.add_argument("--new", required=True,
                    help="the purpose-built arm, e.g. aspprot")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="target labels; default = the 4 protective labels "
                         "whose hypotheses the new arm rewrites (coping "
                         "strategy is excluded, its hypothesis is unchanged)")
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    targets = args.labels or ["meaning in life", "psychological capital",
                              "sense of responsibility", "social support"]
    tj = []
    for t in targets:
        j = IDX.get(t.strip().lower())
        if j is None:
            raise SystemExit(f"'{t}' not in the taxonomy")
        tj.append(j)
    print(f"target labels ({len(tj)}): {targets}")

    rid, S = load_oof(args.dir, list(args.arms) + [args.new], args.k)
    S = np.asarray(S, float)
    Sbase, Snew = S[:-1], S[-1]
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], len(rid))
    acol = "anon_user_id" if "anon_user_id" in lab.columns else "author"
    authors = np.array([str(lab.loc[r, acol]) for r in rid])
    ua = np.array(sorted(set(authors)))
    print(f"{len(rid)} OOF posts / {len(ua)} authors / "
          f"{len(args.arms)} shipped arms + {args.new}")

    Pbase = lmean(Sbase)
    Padd = lmean(S)

    def variant(name):
        if name == "base":
            return Pbase
        if name == "add12":
            return Padd
        P = Pbase.copy()
        if name == "swap":
            P[:, tj] = Snew[:, tj]
            return P
        if name.startswith("boost"):
            w = float(name.split("-")[1])
            n = len(args.arms)
            ww = np.concatenate([np.ones(n), [w]])
            Pb = lmean(S, ww)
            P[:, tj] = Pb[:, tj]
            return P
        raise ValueError(name)

    names = ["base", "add12", "swap", "boost-1", "boost-3", "boost-6"]
    PV = {n: variant(n) for n in names}

    # ---- solo read on the target labels, whole OOF -------------------------
    thb = fit_th(Pbase, Y)
    fb = per_label_f1(Pbase, Y, thb)
    thn = fit_th(Snew, Y)
    fn_ = per_label_f1(Snew, Y, thn)
    print(f"\nTARGET-LABEL F1, whole OOF (in-sample thresholds -- direction "
          f"only)")
    print(f"  {'label':<28}{'pool':>8}{args.new:>12}{'delta':>9}"
          f"{'gold n':>8}")
    for t, j in zip(targets, tj):
        print(f"  {t:<28}{fb[j]:>8.3f}{fn_[j]:>12.3f}"
              f"{fn_[j]-fb[j]:>+9.3f}{int(Y[:, j].sum()):>8}")
    print("  If the new arm is not clearly better on these labels SOLO, no")
    print("  combination rule will rescue it -- stop here.")

    # ---- cross-fitted comparison -------------------------------------------
    rng = np.random.default_rng(args.seed)
    res = {n: [] for n in names}
    for _ in range(args.splits):
        perm = rng.permutation(len(ua))
        fa = set(ua[perm[:len(ua) // 2]])
        fm = np.array([a in fa for a in authors])
        if fm.all() or not fm.any():
            continue
        for n in names:
            P = PV[n]
            th = fit_th(P[fm], Y[fm])
            res[n].append(macro_at(P[~fm], Y[~fm], th))

    e = np.array(res["base"])
    print(f"\n{'variant':<10}{'macro':>9}{'d vs base':>11}{'win':>7}"
          f"{'sd':>9}{'d comp':>9}")
    for n in names:
        v = np.array(res[n])
        d = v - e
        w = 100 * float((d > 0).mean()) if n != "base" else float("nan")
        print(f"{n:<10}{v.mean():>9.4f}{d.mean():>+11.4f}{w:>6.0f}%"
              f"{d.std(ddof=1):>9.4f}{0.3*d.mean():>+9.4f}")

    best = max([n for n in names if n != "base"],
               key=lambda n: (np.array(res[n]) - e).mean())
    d = np.array(res[best]) - e
    w = 100 * float((d > 0).mean())
    print("\n" + "=" * 70)
    print(f"BEST: {best}  {d.mean():+.4f} macro at {w:.0f}% win "
          f"= {0.3*d.mean():+.4f} composite")
    if w >= 90 and d.mean() > 0:
        print("CLEARS. Rebuild with factor_threshold_stabilise.py --use tfit.")
        print("Verify the UNCHANGED pool still reproduces the current factor")
        print("column cell-for-cell FIRST, then apply the swap.")
        if best != "add12":
            print("\nThe winner is a TARGETED variant, not add12: the arm")
            print("helps on the labels it was built for and hurts elsewhere,")
            print("which is the dilution effect this script tests for.")
    else:
        print(f"Does NOT clear 90%. If `swap` beats `add12` but neither clears,")
        print("the arm is directionally right and too weak -- more seeds of it")
        print("would help more than another combination rule.")
    print("=" * 70)


if __name__ == "__main__":
    main()
