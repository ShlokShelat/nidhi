"""
factor_perlabel_pool.py

THE METRIC AVERAGES OVER LABELS. THE POOL DOES NOT VARY BY LABEL.

Factors are scored by MACRO-F1 over 24 labels -- every label worth 1/24
regardless of how rare it is. But BAG8 pools the same eight arms for all 24,
and per-label AP shows that is leaving signal on the table:

    label                   pos   best arm         BAG8 best
    poor school perf         16   synsty  0.839    qnli   0.824
    substance use            33   fenum   0.772    mnli7  0.719
    cognitive deficits       33   dv2xl   0.634    mnli7  0.619
    meaning in life          45   paircm  0.241    lnli   0.230
    low socio-economic       54   paircm  0.875    prot   0.864

Non-BAG8 arms win on 5 of the 8 rarest labels. `fenum` in particular is LAST on
every common label (hopelessness 0.846 vs lnli 0.914) and FIRST on the rarest
(substance use 0.772 vs 0.719) -- the exact signature of `--full-enum`, which
gives each label 1635 training pairs instead of 4x its positive count, so rare
labels get ~16x more signal and common ones relatively less.

**A pool that is uniform across labels cannot exploit that. A per-label pool
can, and the metric is built to reward it.**

THE OVERFITTING HAZARD IS REAL AND IS THE POINT OF THE DESIGN. Choosing among
29 arms independently for each of 24 labels is enormous freedom on 1635 posts.
So three strategies are compared, in increasing order of freedom, all nested --
composition chosen on FIT authors, thresholds refitted on FIT authors, scored
on HELD-OUT authors:

  bag8        the incumbent, no freedom
  all29       one global decision: pool everything. One bit of freedom.
  perlabel    per label, BAG8 or BAG8+one extra arm. 24 x 22 choices.
  perlabel-k  per label, the top-k arms by FIT-half AP. k is global.

If `all29` wins, take it -- it cannot overfit. If only `perlabel` wins and
`all29` does not, that is the search fitting the sample, which is what killed
the factor threshold shrinkage and the greedy membership sweeps today.

SHIPPABILITY. An arm without `{arm}_cv_test.npz` cannot be deployed. Those are
reported separately and the run is repeated with shippable arms only, so the
headline number is one we could actually submit.

Usage:
  python src/factor_perlabel_pool.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_structure import (BAG8, best_thr, cells,  # noqa: E402
                              load_oof, logit_mean)
from scoring import FACTOR_TAXONOMY  # noqa: E402

N = len(FACTOR_TAXONOMY)


def per_label_f1(Pb, Y):
    tp = (Pb & Y).sum(0)
    den = 2 * tp + (Pb & ~Y).sum(0) + (~Pb & Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def ap(p, y):
    if y.sum() == 0:
        return 0.0
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum()
                 / ys.sum())


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--k", type=int, default=5)
    a_.add_argument("--topk", type=int, default=5)
    a_.add_argument("--splits", type=int, default=100)
    a_.add_argument("--seed", type=int, default=13)
    a_.add_argument("--shippable-only", action="store_true")
    args = a_.parse_args()

    have = sorted({os.path.basename(p).replace("_cv_oof_f0.npz", "")
                   for p in glob.glob(f"{args.dir}/*_cv_oof_f0.npz")})
    P_by, ship = {}, {}
    for a in have:
        if len(glob.glob(f"{args.dir}/{a}_cv_oof_f*.npz")) < args.k:
            continue
        try:
            rid, S = load_oof(args.dir, [a], args.k)
        except Exception:
            continue
        P_by[a] = logit_mean(S)
        ship[a] = os.path.exists(f"{args.dir}/{a}_cv_test.npz")
    arms = [a for a in P_by if (ship[a] or not args.shippable_only)]
    core = [a for a in BAG8 if a in P_by]
    n = len(rid)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n).astype(bool)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])
    ua = pd.unique(auth)
    sup = Y.sum(0)
    print(f"OOF {n} posts / {len(ua)} authors")
    print(f"  {len(arms)} arms usable"
          + ("  (shippable only)" if args.shippable_only else "")
          + f", BAG8 = {len(core)}")
    noship = [a for a in P_by if not ship[a]]
    if not args.shippable_only:
        print(f"  NOT shippable (no _cv_test.npz): {', '.join(noship)}")

    cache = {}

    def pool(members):
        key = tuple(sorted(members))
        if key not in cache:
            cache[key] = logit_mean(
                np.stack([P_by[m] for m in members], axis=0))
        return cache[key]

    Pc = pool(core)
    Pa = pool(arms)

    rng = np.random.default_rng(args.seed)
    subs = []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        k = max(int(len(o) * 0.2), 2)
        ho = set(o[:k])
        hm = np.array([a in ho for a in auth])
        fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
        if len(fi) > 50 and len(hi) > 20:
            subs.append((fi, hi))

    def score_matrix(Pmat, fi, hi):
        """per-label held-out F1 with thresholds refitted on FIT"""
        t = np.array([best_thr(Pmat[fi, j], Y[fi, j].astype(int)) or 0.5
                      for j in range(N)])
        return per_label_f1(Pmat[hi] >= t, Y[hi])

    res = {"bag8": [], "all": [], "perlabel": [], "perlabel-k": []}
    picks = Counter()
    for fi, hi in subs:
        res["bag8"].append(score_matrix(Pc, fi, hi).mean())
        res["all"].append(score_matrix(Pa, fi, hi).mean())

        # per-label: BAG8 or BAG8 + one extra, chosen on FIT
        f_lab = np.empty(N)
        for j in range(N):
            tj = best_thr(Pc[fi, j], Y[fi, j].astype(int)) or 0.5
            bestv = per_label_f1((Pc[fi, j] >= tj)[:, None],
                                 Y[fi, j][:, None])[0]
            bestP, bestn = Pc, "bag8"
            for a in arms:
                if a in core:
                    continue
                Pm = pool(core + [a])
                t2 = best_thr(Pm[fi, j], Y[fi, j].astype(int)) or 0.5
                v = per_label_f1((Pm[fi, j] >= t2)[:, None],
                                 Y[fi, j][:, None])[0]
                if v > bestv:
                    bestv, bestP, bestn = v, Pm, a
            picks[bestn] += 1
            t = best_thr(bestP[fi, j], Y[fi, j].astype(int)) or 0.5
            f_lab[j] = per_label_f1((bestP[hi, j] >= t)[:, None],
                                    Y[hi, j][:, None])[0]
        res["perlabel"].append(f_lab.mean())

        # per-label top-k by FIT-half AP -- less freedom, no per-label search
        f_lab2 = np.empty(N)
        for j in range(N):
            aps = sorted(arms, key=lambda a: -ap(P_by[a][fi, j],
                                                 Y[fi, j]))[:args.topk]
            Pm = pool(aps)
            t = best_thr(Pm[fi, j], Y[fi, j].astype(int)) or 0.5
            f_lab2[j] = per_label_f1((Pm[hi, j] >= t)[:, None],
                                     Y[hi, j][:, None])[0]
        res["perlabel-k"].append(f_lab2.mean())

    base = np.array(res["bag8"])
    print(f"\n  BAG8 held-out macro {base.mean():.4f}\n")
    print(f"  {'strategy':<14}{'macro':>9}{'d':>9}{'d comp':>9}{'win':>6}"
          f"{'LOSS':>6}{'sd':>9}")
    for k in ("all", "perlabel", "perlabel-k"):
        v = np.array(res[k])
        d = v - base
        print(f"  {k:<14}{v.mean():>9.4f}{d.mean():>+9.4f}"
              f"{0.3*d.mean():>+9.4f}{np.mean(d > 0):>5.0%}"
              f"{np.mean(d < 0):>6.0%}{0.3*d.std():>9.4f}")

    print("\n  per-label picks (how often each extra arm was chosen):")
    for a, c in picks.most_common(10):
        tag = "" if a == "bag8" or ship.get(a) else "   NOT SHIPPABLE"
        print(f"    {a:<12}{c:>6}{tag}")
    print("\n  `all` has ONE bit of freedom and cannot overfit -- if it wins,")
    print("  take it. If only `perlabel` wins, the per-label search is fitting")
    print("  the sample, which is what killed threshold shrinkage and the")
    print("  greedy membership sweeps.")
    print("  Re-run with --shippable-only for the number we could submit.")


if __name__ == "__main__":
    main()
