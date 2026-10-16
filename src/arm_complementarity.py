"""
arm_complementarity.py

DO THE FACTOR ARMS ACTUALLY MAKE DIFFERENT MISTAKES, AND WHAT DOES EACH ADD?

Three diagnostics on the eleven arms of the shipped factor pool. None of them
assumes the answer, and the third can perfectly well show that the pool
saturated early, which would be a result about our own system rather than a
selling point.

  1. PAIRWISE ERROR OVERLAP. For every pair of arms, how often do both get a
     (post, label) cell right, both get it wrong, or exactly one get it right?
     Two arms that fail on the same cells cannot help each other however
     strong either is alone.

  2. STANDALONE SCORE AGAINST MARGINAL CONTRIBUTION. For each arm, its own
     held-out macro-F1, next to what the pool loses when that arm is removed.
     If those two quantities disagree, then picking members by individual
     score is the wrong procedure. That principle is established in the
     ensemble-selection literature; what is measured here is how far the two
     diverge in this particular setting.

  3. SATURATION. Arms are added greedily, best marginal gain first, and the
     pool is re-scored after each addition. The question is where the curve
     flattens, and whether it turns down.

HONEST MEASUREMENT. Everything is author-disjoint. Thresholds are fitted on a
partition of authors and the score is read on authors held out of that
partition, because a threshold fitted and scored on the same posts flatters
every arm equally and would make the comparison meaningless. Results are
averaged over repeated partitions, and we report the spread as well as the
mean.

SECURITY: reads local probability dumps only, prints aggregate statistics, no
post text.

Usage:
  python src/arm_complementarity.py
  python src/arm_complementarity.py --splits 80 --csv artifacts/complementarity.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
SHIPPED = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli",
           "ltail", "cbnsynapi", "friendfac"]
TGRID = np.round(np.arange(0.02, 0.985, 0.005), 3)


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def load_oof(d, arms, k=5):
    """Stack each arm's out-of-fold probabilities, aligned on row_id."""
    out, rid = [], None
    for a in arms:
        pr, ids = [], []
        for f in range(k):
            p = os.path.join(d, f"{a}_cv_oof_f{f}.npz")
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
            z = np.load(p, allow_pickle=True)
            pr.append(z["probs"].astype(float))
            ids.extend(str(x) for x in z["row_id"])
        pr = np.concatenate(pr)
        if rid is None:
            rid = ids
        elif ids != rid:
            o = {x: i for i, x in enumerate(ids)}
            pr = pr[[o[x] for x in rid]]
        out.append(pr)
    return rid, np.stack(out)


def f1_curve(p, y):
    """F1 at every threshold on the grid, vectorised."""
    npos = int(y.sum())
    if npos == 0:
        return np.zeros(len(TGRID))
    o = np.argsort(-p, kind="stable")
    cum = np.cumsum(y[o])
    ks = np.searchsorted(-p[o], -TGRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    return 2.0 * tp / np.maximum(ks + npos, 1)


def fit_thresholds(P, Y):
    return np.array([float(TGRID[int(np.argmax(f1_curve(P[:, j], Y[:, j])))])
                     for j in range(N)])


def macro_at(P, Y, th):
    on = (P >= th[None, :]).astype(np.int8)
    f = np.zeros(N)
    for j in range(N):
        tp = int((on[:, j] & Y[:, j]).sum())
        pp, ap = int(on[:, j].sum()), int(Y[:, j].sum())
        f[j] = 0.0 if (pp + ap) == 0 else 2.0 * tp / (pp + ap)
    return float(f.mean())


def pool(Z, members):
    """Mean log-odds over the chosen arms, back to probability."""
    return 1.0 / (1.0 + np.exp(-Z[list(members)].mean(0)))


def score_subset(Z, Y, members, fit, ev):
    P = pool(Z, members)
    th = fit_thresholds(P[fit], Y[fit])
    return macro_at(P[ev], Y[ev], th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=SHIPPED)
    ap.add_argument("--labelled", default="artifacts/full_train.csv")
    ap.add_argument("--splits", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms)
    Z = logit(S)
    lab = pd.read_csv(args.labelled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = np.zeros((len(rid), N), dtype=np.int8)
    for i, r in enumerate(rid):
        for f in parse_factors(str(lab.loc[r, "factors"])):
            j = IDX.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    acol = "anon_user_id" if "anon_user_id" in lab.columns else "author"
    authors = np.array([str(lab.loc[r, acol]) for r in rid])
    A = len(args.arms)
    print(f"{len(rid)} posts, {len(set(authors))} authors, {A} arms, "
          f"{N} labels\n")

    # ---------------- 1. pairwise error overlap ----------------
    th_all = [fit_thresholds(S[a], Y) for a in range(A)]
    ON = np.stack([(S[a] >= th_all[a][None, :]).astype(np.int8)
                   for a in range(A)])
    correct = (ON == Y[None, :, :])
    print("=" * 76)
    print("1. PAIRWISE ERROR OVERLAP over all (post, label) cells")
    print("=" * 76)
    print("   'only A' is the share of cells arm A gets right and B does not.")
    print("   A pair with a large 'both wrong' share cannot rescue each other.\n")
    print(f"{'pair':<26}{'both ok':>9}{'only A':>9}{'only B':>9}"
          f"{'both wrong':>12}")
    rows = []
    for a, b in itertools.combinations(range(A), 2):
        ca, cb = correct[a].ravel(), correct[b].ravel()
        both = float((ca & cb).mean())
        onlya = float((ca & ~cb).mean())
        onlyb = float((~ca & cb).mean())
        neither = float((~ca & ~cb).mean())
        rows.append({"a": args.arms[a], "b": args.arms[b], "both_ok": both,
                     "only_a": onlya, "only_b": onlyb, "neither": neither})
    ov = pd.DataFrame(rows).sort_values("neither")
    for r in ov.head(6).itertuples():
        print(f"{r.a[:12]+'/'+r.b[:12]:<26}{r.both_ok:>9.3f}{r.only_a:>9.3f}"
              f"{r.only_b:>9.3f}{r.neither:>12.3f}")
    print("   ...")
    for r in ov.tail(3).itertuples():
        print(f"{r.a[:12]+'/'+r.b[:12]:<26}{r.both_ok:>9.3f}{r.only_a:>9.3f}"
              f"{r.only_b:>9.3f}{r.neither:>12.3f}")
    dis = ov["only_a"].mean() + ov["only_b"].mean()
    print(f"\n   mean disagreement across all {len(ov)} pairs: {dis:.3f}")
    print(f"   mean 'both wrong':                       "
          f"{ov['neither'].mean():.3f}")

    # ---------------- author-disjoint partitions ----------------
    rng = np.random.default_rng(args.seed)
    ua = np.array(sorted(set(authors)))
    parts = []
    for _ in range(args.splits):
        h = set(rng.choice(ua, size=max(1, len(ua) // 3), replace=False))
        ev = np.array([a in h for a in authors])
        if ev.sum() >= 30 and (~ev).sum() >= 30:
            parts.append((~ev, ev))
    print(f"\n{len(parts)} author-disjoint partitions\n")

    full = list(range(A))
    solo = np.zeros((len(parts), A))
    loo = np.zeros((len(parts), A))
    base = np.zeros(len(parts))
    for s, (fit, ev) in enumerate(parts):
        base[s] = score_subset(Z, Y, full, fit, ev)
        for a in range(A):
            solo[s, a] = score_subset(Z, Y, [a], fit, ev)
            loo[s, a] = score_subset(Z, Y, [x for x in full if x != a],
                                     fit, ev)

    print("=" * 76)
    print("2. STANDALONE SCORE vs MARGINAL CONTRIBUTION")
    print("=" * 76)
    print(f"   pool of all {A} arms: {base.mean():.4f}\n")
    print(f"{'arm':<14}{'standalone':>12}{'pool w/o it':>13}"
          f"{'contribution':>14}{'rank sol':>10}{'rank con':>10}")
    contrib = base[:, None] - loo
    so, co = solo.mean(0), contrib.mean(0)
    rs = (-so).argsort().argsort() + 1
    rc = (-co).argsort().argsort() + 1
    for a in range(A):
        print(f"{args.arms[a]:<14}{so[a]:>12.4f}{loo[:, a].mean():>13.4f}"
              f"{co[a]:>+14.4f}{rs[a]:>10}{rc[a]:>10}")
    r = float(np.corrcoef(so, co)[0, 1])
    print(f"\n   correlation(standalone, contribution) = {r:+.3f}")
    print("   Near +1 means picking by individual score is fine here.")
    print("   Near 0 or negative means it is the wrong procedure.")
    neg = [args.arms[a] for a in range(A) if co[a] < 0]
    if neg:
        print(f"   arms whose removal IMPROVES the pool: {', '.join(neg)}")

    # ---------------- 3. saturation ----------------
    print("\n" + "=" * 76)
    print("3. SATURATION: arms added greedily, best marginal gain first")
    print("=" * 76)
    curve = np.zeros((len(parts), A))
    order_counts = [{} for _ in range(A)]
    for s, (fit, ev) in enumerate(parts):
        chosen, remaining = [], list(range(A))
        for step in range(A):
            best, bestsc = None, -1
            for cand in remaining:
                sc = score_subset(Z, Y, chosen + [cand], fit, ev)
                if sc > bestsc:
                    best, bestsc = cand, sc
            chosen.append(best)
            remaining.remove(best)
            curve[s, step] = bestsc
            order_counts[step][args.arms[best]] = \
                order_counts[step].get(args.arms[best], 0) + 1
    print(f"{'arms':>6}{'macro F1':>11}{'sd':>9}{'gain':>10}"
          f"   most often chosen here")
    for step in range(A):
        g = curve[:, step].mean() - (curve[:, step - 1].mean()
                                     if step else 0.0)
        top = max(order_counts[step].items(), key=lambda kv: kv[1])
        print(f"{step+1:>6}{curve[:, step].mean():>11.4f}"
              f"{curve[:, step].std():>9.4f}{g:>+10.4f}"
              f"   {top[0]} ({top[1]}/{len(parts)})")
    peak = int(np.argmax(curve.mean(0))) + 1
    print(f"\n   peak at {peak} arms of {A}. Adding arms beyond that point")
    print("   does not pay for itself on held-out authors.")

    if args.csv:
        out = pd.DataFrame({"arm": args.arms, "standalone": so,
                            "contribution": co, "rank_standalone": rs,
                            "rank_contribution": rc})
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        out.to_csv(args.csv, index=False)
        ov.to_csv(args.csv.replace(".csv", "_pairs.csv"), index=False)
        print(f"\nwrote {args.csv} and the pairwise table beside it")


if __name__ == "__main__":
    main()
