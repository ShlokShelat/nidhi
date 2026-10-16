"""
factor_member_loo.py

LEAVE-ONE-OUT MEMBERSHIP AUDIT FOR THE FACTOR ENSEMBLE.

The logit-mean pool gives every member equal weight, so a member that is
redundant with the rest contributes noise without contributing signal, and a
member that is simply weaker than the others drags the mean toward its own
errors. Additions to a saturated pool therefore stop paying, and the remaining
direction on the membership axis is REMOVAL.

WHY A POOL ACCUMULATES DEAD WEIGHT. Members are justified against the pool that
existed when they were added. Once a stronger arm arrives and covers the same
labels, the older member is no longer carrying independent information, but
nothing removes it automatically. `ltail`, for example, is best-in-pool on
`cognitive deficits` and second on `hopelessness`, which makes whatever
previously covered those labels a removal candidate. This also means an earlier
leave-one-out audit does not carry over: adding a member changes which of the
others are redundant.

TWO TRAPS THIS AVOIDS.

  * THRESHOLDS ARE REFIT PER SUBSET. Dropping an arm shifts the logit mean, so
    scoring the smaller pool at the full pool's thresholds measures the
    threshold mismatch rather than the member. Every subset gets its own
    `best_thr` fitted on the FIT authors only.
  * JUDGED BY WIN RATE AND PER-SPLIT SD, never the CI of the mean. The splits
    overlap heavily, so the CI of the mean shrinks with the number of splits
    and says nothing about the spread of a single draw.

It also runs NESTED greedy elimination -- membership chosen on FIT authors,
scored on HELD-OUT ones -- because a drop that only wins when chosen with
hindsight is the search fitting its sample rather than a property of the pool.

Usage:
  python src/factor_member_loo.py
  python src/factor_member_loo.py --pool mnli,dmnli,cmine,prot,d4,qnli,mnli7,lnli,ltail
"""

from __future__ import annotations

import argparse
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


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--pool", default=",".join(list(BAG8) + ["ltail"]),
                    help="the deployed pool; default is BAG8 + ltail")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--k", type=int, default=5)
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    pool = [a.strip() for a in args.pool.split(",") if a.strip()]
    # Per-arm probability matrices, then pooled by logit-mean over arms. This
    # is the exact idiom factor_perlabel_pool.py uses, rather than
    # concatenating raw load_oof output, so there is no assumption about the
    # leading dimension load_oof returns.
    P_by = {}
    for a in pool:
        try:
            rid, S = load_oof(args.dir, [a], args.k)
        except Exception as e:
            raise SystemExit(f"cannot load arm {a}: {e}")
        P_by[a] = logit_mean(S)
    n = len(rid)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n).astype(bool)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])
    ua = pd.unique(auth)
    ship = {a: os.path.exists(f"{args.dir}/{a}_cv_test.npz") for a in pool}
    print(f"OOF {n} posts / {len(ua)} authors")
    print(f"  pool {len(pool)}: {', '.join(pool)}")
    miss = [a for a in pool if not ship[a]]
    if miss:
        print(f"  WARNING no _cv_test.npz for: {', '.join(miss)} -- a pool "
              f"that cannot be rebuilt on test is not a shippable finding")

    cache = {}

    def P_of(members):
        key = tuple(sorted(members))
        if key not in cache:
            cache[key] = logit_mean(np.stack(
                [P_by[m] for m in members], axis=0))
        return cache[key]

    rng = np.random.default_rng(args.seed)
    subs = []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        ho = set(o[:max(int(len(o) * 0.2), 2)])
        hm = np.array([a in ho for a in auth])
        fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
        if len(fi) > 50 and len(hi) > 20:
            subs.append((fi, hi))

    def score(members, fi, hi):
        """Thresholds refit on FIT for THIS subset, scored on HELD-OUT."""
        P = P_of(members)
        t = np.array([best_thr(P[fi, j], Y[fi, j].astype(int)) or 0.5
                      for j in range(N)])
        return float(np.mean(per_label_f1(P[hi] >= t, Y[hi])))

    base = np.array([score(pool, fi, hi) for fi, hi in subs])
    hdr = "=" * 74
    print(f"\n  full pool held-out macro {base.mean():.4f}\n")
    print(hdr)
    print("1. DROP ONE -- thresholds refit per subset")
    print(hdr)
    print(f"  {'drop':<12}{'macro':>9}{'d':>9}{'d comp':>9}"
          f"{'win':>6}{'LOSS':>6}{'sd':>9}")
    best = (None, -9.0, 0.0)
    for a in pool:
        rest = [m for m in pool if m != a]
        v = np.array([score(rest, fi, hi) for fi, hi in subs])
        d = v - base
        w = float(np.mean(d > 0))
        print(f"  {a:<12}{v.mean():>9.4f}{d.mean():>+9.4f}"
              f"{0.3 * d.mean():>+9.4f}{w:>5.0%}{np.mean(d < 0):>6.0%}"
              f"{d.std():>9.4f}")
        if d.mean() > best[1]:
            best = (a, float(d.mean()), w)

    print("\n" + hdr)
    print("2. NESTED GREEDY ELIMINATION -- chosen on FIT, scored on HELD-OUT")
    print(hdr)
    kept_n, deltas, dropped = [], [], Counter()
    for fi, hi in subs:
        inner = fi[: max(len(fi) // 4, 10)]
        outer = fi[max(len(fi) // 4, 10):]
        cur = list(pool)
        while len(cur) > 3:
            b0 = score(cur, outer, inner)
            cand = max(((m, score([x for x in cur if x != m], outer, inner))
                        for m in cur), key=lambda t: t[1])
            if cand[1] <= b0:
                break
            cur.remove(cand[0])
            dropped[cand[0]] += 1
        kept_n.append(len(cur))
        deltas.append(score(cur, fi, hi) - score(pool, fi, hi))
    d = np.array(deltas)
    print(f"  d macro {d.mean():+.4f}   d comp {0.3 * d.mean():+.4f}   "
          f"win {np.mean(d > 0):.0%}   LOSS {np.mean(d < 0):.0%}   "
          f"sd {d.std():.4f}")
    print(f"  members kept {np.mean(kept_n):.1f} +- {np.std(kept_n):.1f} "
          f"of {len(pool)}")
    print(f"  dropped how often: "
          + ", ".join(f"{a} {c}" for a, c in dropped.most_common(6)))

    print("\n" + hdr)
    print("VERDICT")
    print(hdr)
    print(f"  BEST DROP: {best[0]}   {0.3 * best[1]:+.4f} composite at "
          f"{best[2]:.0%} win")
    print(f"  A drop is only a finding at >=90% win with a small per-split sd.")
    print(f"  A member dropped in almost EVERY greedy split is stable; one")
    print(f"  dropped in about half is the search fitting its sample.")
    print(f"\n  If nothing clears it, the membership axis is CLOSED in both")
    print(f"  directions and factors are finished at this pool.")


if __name__ == "__main__":
    main()
