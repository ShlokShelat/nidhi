"""
factor_member_forward.py

FORWARD SELECTION OVER FACTOR ENSEMBLE MEMBERS.

`artifacts/factor_runs/` holds many arms with 5-fold OOF, a subset of which
also have a `_cv_test.npz` and can therefore be deployed. BAG8 is the current
pool. This script asks whether any remaining arm belongs in it, using nested
forward selection against that pool: candidates are chosen on FIT authors and
scored on HELD-OUT authors, so a candidate that only looks good on the data
that selected it is exposed.

WHY THE DECODE MUST BE REFITTED. `tfit` is 24 per-label thresholds fitted to
the pool's probability distribution. Adding an arm shifts that distribution, so
reusing the incumbent thresholds would measure "stale thresholds", not "extra
arm". They are refitted on FIT authors for every subset.

SHIPPABILITY GATE. An arm without `_cv_test.npz` has no test-time artefact and
cannot be deployed even if it helps, so those arms are reported separately and
never selected. Selecting one would produce a result that cannot be reproduced
at inference time.

BAR: >=90% win with a real LOSS rate, stable across nested splits. A held-out
delta measured here attenuates on unseen authors, so a delta that only just
clears zero is not enough to earn a slot.

Usage:
  python src/factor_member_forward.py
  python src/factor_member_forward.py --rounds 2
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


def macro_f1(Pb, Y):
    tp = (Pb & Y).sum(0)
    den = 2 * tp + (Pb & ~Y).sum(0) + (~Pb & Y).sum(0)
    return float(np.mean(np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--k", type=int, default=5)
    a_.add_argument("--splits", type=int, default=120)
    a_.add_argument("--rounds", type=int, default=1)
    a_.add_argument("--extra", default=None,
                    help="comma-separated arms to add ALL AT ONCE as a "
                         "pre-specified set (not a search)")
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    have = sorted({os.path.basename(p).replace("_cv_oof_f0.npz", "")
                   for p in glob.glob(f"{args.dir}/*_cv_oof_f0.npz")})
    S_by, shippable = {}, {}
    for a in have:
        if len(glob.glob(f"{args.dir}/{a}_cv_oof_f*.npz")) < args.k:
            continue
        try:
            _, S = load_oof(args.dir, [a], args.k)
        except Exception:
            continue
        S_by[a] = S
        shippable[a] = os.path.exists(f"{args.dir}/{a}_cv_test.npz")
    core = [a for a in BAG8 if a in S_by]
    cands = [a for a in S_by if a not in core and shippable[a]]
    noship = [a for a in S_by if a not in core and not shippable[a]]
    if len(core) < len(BAG8):
        print(f"  note: only {len(core)}/{len(BAG8)} BAG8 arms loaded")

    rid, _ = load_oof(args.dir, core, args.k)
    n = len(rid)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n).astype(bool)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])
    ua = pd.unique(auth)
    print(f"OOF {n} posts / {len(ua)} authors")
    print(f"  pool {len(core)}: {', '.join(core)}")
    print(f"  shippable candidates {len(cands)}: {', '.join(cands)}")
    print(f"  NOT shippable (no _cv_test.npz), excluded: "
          f"{', '.join(noship) if noship else 'none'}")

    cache = {}

    def pool(members):
        key = tuple(sorted(members))
        if key not in cache:
            cache[key] = logit_mean(
                np.concatenate([S_by[m] for m in members], axis=0))
        return cache[key]

    def thr(P, ix):
        return np.array([best_thr(P[ix, j], Y[ix, j].astype(int)) or 0.5
                         for j in range(N)])

    def scored(members, fi, hi):
        P = pool(members)
        return macro_f1(P[hi] >= thr(P, fi), Y[hi])

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

    base = np.array([scored(core, fi, hi) for fi, hi in subs])
    print(f"\n  BAG8 held-out macro {base.mean():.4f}\n")
    print("=" * 74)
    print("1. ADD ONE -- thresholds refitted per subset, cross-fitted")
    print("=" * 74)
    print(f"  {'add':<12}{'macro':>9}{'d':>9}{'d comp':>9}{'win':>6}"
          f"{'LOSS':>6}{'sd':>9}")
    res = []
    for c in cands:
        v = np.array([scored(core + [c], fi, hi) for fi, hi in subs])
        d = v - base
        print(f"  {c:<12}{v.mean():>9.4f}{d.mean():>+9.4f}"
              f"{0.3*d.mean():>+9.4f}{np.mean(d > 0):>5.0%}"
              f"{np.mean(d < 0):>6.0%}{0.3*d.std():>9.4f}")
        res.append((float(np.mean(d > 0)), 0.3 * float(d.mean()), c))

    print("\n" + "=" * 74)
    print("2. NESTED FORWARD SELECTION -- chosen on FIT, scored on HELD-OUT")
    print("=" * 74)
    ds, added = [], []
    for fi, hi in subs:
        inner = fi[: max(len(fi) // 4, 10)]
        outer = fi[max(len(fi) // 4, 10):]
        mem = list(core)
        cur = scored(mem, outer, inner)
        for _ in range(args.rounds):
            best, bc = cur, None
            for c in cands:
                v = scored(mem + [c], outer, inner)
                if v > best:
                    best, bc = v, c
            if bc is None:
                break
            mem.append(bc)
            added.append(bc)
            cur = best
        ds.append(scored(mem, fi, hi) - scored(core, fi, hi))
    d = np.array(ds)
    print(f"  d macro {d.mean():+.4f}   d comp {0.3*d.mean():+.4f}"
          f"   win {np.mean(d > 0):.0%}   LOSS {np.mean(d < 0):.0%}")
    print("  added how often: " + ", ".join(
        f"{k} {v}" for k, v in Counter(added).most_common(6)))

    if args.extra:
        # A PRE-SPECIFIED set, not a search. Greedy selection is myopic: several
        # arms each with a small individual gain and an unconvincing win rate
        # may help together while no single one clears the bar and no greedy
        # step finds the combination. Naming
        # the set in advance means there is nothing to overfit -- but it is
        # only honest BEFORE looking, so this is fixed at the four arms with
        # positive means in section 1.
        ex = [x.strip() for x in args.extra.split(",")
              if x.strip() in S_by and x.strip() not in core]
        if ex:
            v = np.array([scored(core + ex, fi, hi) for fi, hi in subs])
            d = v - base
            print("\n" + "=" * 74)
            print(f"3. PRE-SPECIFIED SET: BAG8 + {', '.join(ex)}")
            print("=" * 74)
            print(f"  macro {v.mean():.4f}   d {d.mean():+.4f}   "
                  f"d comp {0.3*d.mean():+.4f}   win {np.mean(d > 0):.0%}   "
                  f"LOSS {np.mean(d < 0):.0%}   sd {0.3*d.std():.4f}")
            print("  If this beats every single addition, the arms are")
            print("  complementary and greedy search was too myopic to see it.")

    if res:
        res.sort(reverse=True)
        w, dc, c = res[0]
        print(f"\n  BEST ADD: {c}   {dc:+.4f} composite at {w:.0%} win")
        ok = w >= 0.90 and dc > 0
        print(f"  {'CLEARS' if ok else 'does NOT clear'} the bar.")
        print("  Cross-check with section 2: an arm added in most splits is a")
        print("  finding; one added in half is the search fitting its sample.")


if __name__ == "__main__":
    main()
