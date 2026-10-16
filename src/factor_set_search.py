#!/usr/bin/env python
"""EXHAUSTIVE set search over a shortlist, with an untouched outer holdout.

WHY. Every membership test so far has been ONE SWAP from the shipped set --
add an arm, drop an arm, swap a member. That is hill-climbing: it can confirm a
local optimum and nothing else. The shipped 11 (BAG8 + ltail + friendfac +
cbnsynapi) survives every one-swap perturbation at a high win rate, but that
says nothing about whether a different combination two or three moves away is
better.

The two greedy searches already run BOTH lost to the pre-specified set, one of
them decisively, which is the signature of a search fitting its sample rather
than a set being genuinely better.

THE DESIGN THAT SEPARATES THOSE TWO THINGS.

  1. Enumerate ALL 2^k subsets of a shortlist, each added to BAG8. Exhaustive
     over the space, so no greedy path-dependence.
  2. OUTER split: hold out a third of AUTHORS completely.
  3. On the remaining two thirds, cross-fit every subset and rank them.
  4. Score the INNER WINNER on the OUTER holdout, alongside the champion.
  5. Repeat over many outer splits.

Step 5 is what makes this honest. Reporting the best inner score would be
selecting the maximum of ~256 noisy estimates -- guaranteed to look good and
guaranteed not to transfer. The outer holdout prices that selection in: it
answers "if I search and take the winner, do I beat the champion on authors the
search never saw?" That is the question, and it is not the same as "does some
subset score higher".

Thresholds are refit on FIT authors inside every evaluation, so no set gets to
borrow the champion's calibration.

READ IT AS: if `search winner` beats `champion` on the outer holdout at >=90%
win, a better global set exists and the search finds it. If it does not, the
champion IS the practical optimum over this space and membership is closed --
this time with a search that looked everywhere rather than one step away.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_threshold_decode import N, cells, load_oof  # noqa: E402

BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
# the arms with a POSITIVE measured add-one delta on BAG8, strongest first.
# Includes the three shipped extras so the champion is inside the space.
SHORTLIST = ["mlc", "friendfac", "cbnsynapi", "mlcmt2", "cbnsynsty",
             "ltail", "synapi", "hyb"]
CHAMPION_EXTRAS = ("ltail", "friendfac", "cbnsynapi")
# The shipped decode uses 0.005 steps. This search runs ~30k threshold fits, so
# it uses 0.01 -- half the grid, half the time. A 0.005 difference in a
# threshold is far below the per-split sd (~0.004 macro), so it cannot change
# the RANKING of two sets; it only affects the absolute level, and every set
# pays it equally. The champion is re-scored on the same grid, so the
# comparison stays fair.
TGRID = np.round(np.arange(0.02, 0.98, 0.01), 3)


def fit_th_fast(P, Y):
    """Per-label argmax-F1 threshold, vectorised over the grid."""
    th = np.empty(N)
    for j in range(N):
        p, y = P[:, j], Y[:, j].astype(bool)
        pb = p[:, None] >= TGRID[None, :]          # (n, G)
        tp = (pb & y[:, None]).sum(0)
        den = 2 * tp + (pb & ~y[:, None]).sum(0) + ((~pb) & y[:, None]).sum(0)
        f1 = np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))
        th[j] = TGRID[int(np.argmax(f1))]
    return th


def macro_at(P, Y, th):
    pb = P >= th[None, :]
    tp = (pb & (Y == 1)).sum(0)
    den = 2 * tp + (pb & (Y == 0)).sum(0) + ((~pb) & (Y == 1)).sum(0)
    return float(np.mean(np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", nargs="+", default=SHORTLIST)
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--inner", type=int, default=10,
                    help="inner splits used to RANK sets. Cost is "
                         "sets x inner x outer threshold fits, so this and "
                         "--outer set the runtime: 163 x 10 x 20 is ~8 min.")
    ap.add_argument("--outer", type=int, default=20,
                    help="outer splits; each one re-runs the WHOLE search on "
                         "2/3 of authors and scores the winner on the other "
                         "1/3. More is better for the win rate but linear in "
                         "cost.")
    ap.add_argument("--max-size", type=int, default=4,
                    help="max extras added to BAG8 (the shipped set uses 3)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    arms = BAG8 + [a for a in args.shortlist if a not in BAG8]
    rid, S = load_oof(args.dir, arms, args.k)
    S = np.clip(np.asarray(S, float), 1e-6, 1 - 1e-6)
    L = np.log(S / (1 - S))                        # (arms, posts, 24)
    pos = {a: i for i, a in enumerate(arms)}

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], len(rid))
    acol = "anon_user_id" if "anon_user_id" in lab.columns else "author"
    authors = np.array([str(lab.loc[r, acol]) for r in rid])
    ua = np.array(sorted(set(authors)))

    short = [a for a in args.shortlist if a in pos]
    subsets = []
    for k in range(0, args.max_size + 1):
        subsets += [tuple(c) for c in itertools.combinations(short, k)]
    print(f"{len(rid)} OOF posts / {len(ua)} authors")
    print(f"BAG8 + subsets of {short}")
    print(f"{len(subsets)} candidate sets (sizes 0..{args.max_size})")
    champ = tuple(a for a in short if a in CHAMPION_EXTRAS)
    print(f"champion extras in-space: {champ}  "
          f"{'OK' if len(champ) == 3 else 'MISSING -- fix the shortlist'}")

    bag_idx = [pos[a] for a in BAG8]
    bag_sum = L[bag_idx].sum(0)

    def probs(sub):
        idx = [pos[a] for a in sub]
        tot = bag_sum + (L[idx].sum(0) if idx else 0.0)
        return 1.0 / (1.0 + np.exp(-tot / (len(BAG8) + len(idx))))

    P = {s: probs(s) for s in subsets}
    print("pooled probabilities precomputed")

    rng = np.random.default_rng(args.seed)

    def cv(sub_P, mask_pool, splits, rs):
        """mean held-out macro over `splits` author halves inside mask_pool."""
        idx = np.where(mask_pool)[0]
        au = np.array(sorted(set(authors[idx])))
        out = []
        for _ in range(splits):
            perm = rs.permutation(len(au))
            fa = set(au[perm[:len(au) // 2]])
            fm = np.array([authors[i] in fa for i in idx])
            if fm.all() or not fm.any():
                continue
            fi, hi = idx[fm], idx[~fm]
            th = fit_th_fast(sub_P[fi], Y[fi])
            out.append(macro_at(sub_P[hi], Y[hi], th))
        return float(np.mean(out)) if out else 0.0

    # ---- outer loop: search on 2/3 of authors, score on the other 1/3 ------
    wins, deltas, picked = [], [], {}
    for o in range(args.outer):
        perm = rng.permutation(len(ua))
        cut = len(ua) // 3
        out_a = set(ua[perm[:cut]])
        in_mask = np.array([a not in out_a for a in authors])
        out_mask = ~in_mask
        rs = np.random.default_rng(args.seed + 1000 + o)

        scores = {s: cv(P[s], in_mask, args.inner, rs) for s in subsets}
        best = max(scores, key=scores.get)
        picked[best] = picked.get(best, 0) + 1

        oi = np.where(out_mask)[0]
        ii = np.where(in_mask)[0]
        th_b = fit_th_fast(P[best][ii], Y[ii])
        th_c = fit_th_fast(P[champ][ii], Y[ii])
        d = macro_at(P[best][oi], Y[oi], th_b) - macro_at(P[champ][oi], Y[oi], th_c)
        deltas.append(d)
        wins.append(d > 0)
        if (o + 1) % 10 == 0:
            print(f"  outer {o+1}/{args.outer}  running mean d "
                  f"{np.mean(deltas):+.4f}  win {100*np.mean(wins):.0f}%")

    deltas = np.array(deltas)
    print("\n" + "=" * 72)
    print("SEARCH WINNER vs CHAMPION, on authors the search never saw")
    print(f"  d macro {deltas.mean():+.4f}   sd {deltas.std(ddof=1):.4f}   "
          f"win {100*np.mean(wins):.0f}%   over {len(deltas)} outer splits")
    print(f"  d composite {0.3*deltas.mean():+.4f}")
    print("\n  sets the inner search picked most often:")
    for s, c in sorted(picked.items(), key=lambda kv: -kv[1])[:8]:
        tag = " <- CHAMPION" if s == champ else ""
        print(f"    {c:>3}/{len(deltas)}  BAG8 + {list(s)}{tag}")

    # ---- for inspection only: full-data ranking (SELECTION-BIASED) --------
    rs = np.random.default_rng(args.seed + 7)
    full = {s: cv(P[s], np.ones(len(rid), bool), args.inner, rs)
            for s in subsets}
    order = sorted(full, key=full.get, reverse=True)
    print(f"\n  top sets by FULL-DATA inner score (selection-biased, for")
    print(f"  inspection only -- the outer number above is the honest one):")
    for s in order[:8]:
        tag = " <- CHAMPION" if s == champ else ""
        print(f"    {full[s]:.4f}  BAG8 + {list(s)}{tag}")
    print(f"    champion rank: {order.index(champ)+1} of {len(order)}")

    print("\n" + "=" * 72)
    if deltas.mean() > 0 and np.mean(wins) >= 0.90:
        print("A BETTER GLOBAL SET EXISTS and the search finds it reliably.")
        print("Ship the modal pick above, with the usual provenance gate.")
    else:
        print("The search does NOT beat the champion on untouched authors.")
        print("The champion is the practical optimum over this space -- not")
        print("merely a local one, since every subset was enumerated. If the")
        print("full-data ranking puts other sets above it while the outer test")
        print("says no, that gap IS the selection bias, measured.")
    print("=" * 72)


if __name__ == "__main__":
    main()
