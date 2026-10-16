"""
factor_bag9_tboth.py

RE-TEST THE ABSORPTION LAW UNDER THE tboth DECODE, SINCE IT WAS ORIGINALLY
MEASURED UNDER A DECODE THAT IS NO LONGER IN USE -- AND BECAUSE DIVERSITY,
NOT QUALITY, IS WHAT THE POOL WANTS.

Two measured facts force this:

1. factor_arm_weight.py tried to up-weight the strong arms and
   down-weight the weak ones. EVERY variant tested lost, monotonically as the
   weighting grew sharper, and the arm ranking was STABLE across splits (d4
   consistently weakest, dmnli consistently near the top). So the
   weakest-ranking arms are earning their place through decorrelation, not
   through solo quality. `d4` has the worst solo AP of the eight and removing
   it costs macro-F1.

2. The fixed per-arm dilution tax that closed bag9 was measured at CHAMPION
   RATES -- the quantile decode. That is the same decode that scored `defs`
   positive when its true value under tfit is negative, which invalidated a
   broad set of arm tests measured under it. The tax has never been
   re-measured under a threshold decode.

If diversity is what pays, the arm to add is the most DIFFERENT one, not the
best one. artifacts/factor_runs holds ~20 arms with complete 5-fold OOF that
are not in bag8, several of them cross-architecture (rmnli = RoBERTa,
bmnli = BART, dv2xl / dv1mnli = other DeBERTa generations, hyb, aspv2, fenum).

WHAT IS REPORTED, per candidate:
  add      bag8 + candidate, 9 arms equally weighted
  swap     candidate REPLACES d4, the stable weakest member
  rho      mean correlation of the candidate's logits with the bag8 pool's,
           averaged over labels. LOW rho is the diversity this predicts pays.

THIS IS A SCREEN, NOT A VERDICT. Roughly 20 candidates at 40 splits means one
false winner is EXPECTED at a 5% threshold. Nothing here ships. Survivors --
win rate >= 65% AND a CI excluding zero -- get re-run at --splits 200, and only
then judged. Report the screen honestly including how many were tried.

Usage (cluster, CPU, ~10 min):
  python src/factor_bag9_tboth.py --splits 40
  python src/factor_bag9_tboth.py --splits 200 --only rmnli bmnli
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
TGRID = np.round(np.arange(0.02, 0.985, 0.005), 3)


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_arm(d, tag, k, rid=None):
    pr, ids = [], []
    for f in range(k):
        p = f"{d}/{tag}_cv_oof_f{f}.npz"
        if not os.path.exists(p):
            return None, None
        z = np.load(p, allow_pickle=True)
        pr.append(z["probs"].astype(float))
        ids.extend(str(x) for x in z["row_id"])
    pr = np.concatenate(pr)
    if rid is not None:
        if ids != rid:
            o = {x: i for i, x in enumerate(ids)}
            if any(r not in o for r in rid):
                return None, None
            pr = pr[[o[r] for r in rid]]
        return pr, rid
    return pr, ids


def cells(col, n):
    M = np.zeros((n, N), dtype=np.int8)
    for i, c in enumerate(col):
        for x in set(parse_factors(str(c))):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


def f1_curve(p, y):
    npos = int(y.sum())
    if npos == 0:
        return np.zeros(len(TGRID))
    o = np.argsort(-p, kind="stable")
    ys, ps = y[o], p[o]
    cum = np.cumsum(ys)
    ks = np.searchsorted(-ps, -TGRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    return 2.0 * tp / np.maximum(ks + npos, 1)


def thr_bag(p, y, rng, B, w):
    out, n = [], len(p)
    for _ in range(B):
        ix = rng.integers(0, n, n)
        if y[ix].sum() < 1:
            continue
        c = f1_curve(p[ix], y[ix])
        if w > 1:
            c = np.convolve(c, np.ones(w) / w, mode="same")
        out.append(float(TGRID[int(np.argmax(c))]))
    return float(np.mean(out)) if out else 0.5


def f1pl(P, Y):
    P = P.astype(np.int64)
    Y = Y.astype(np.int64)
    tp = (P * Y).sum(0)
    den = 2 * tp + (P * (1 - Y)).sum(0) + ((1 - P) * Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def score(ZS, Y, F, E, seed0, B, sw):
    ZF = ZS[:, F].mean(0)
    ZE = ZS[:, E].mean(0)
    PF, PE = sig(ZF), sig(ZE)
    th = np.array([
        thr_bag(PF[:, j], Y[F, j], np.random.default_rng(seed0 + j), B, sw)
        for j in range(N)])
    return f1pl((PE >= th[None, :]).astype(np.int8), Y[E]).mean()


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--k", type=int, default=5)
    a_.add_argument("--splits", type=int, default=40)
    a_.add_argument("--tbag", type=int, default=40)
    a_.add_argument("--smooth", type=int, default=9)
    a_.add_argument("--seed", type=int, default=13)
    a_.add_argument("--swap-out", default="d4",
                    help="the stable weakest bag8 member to replace")
    a_.add_argument("--only", nargs="*", default=None)
    args = a_.parse_args()

    base8, rid = [], None
    for t in BAG8:
        pr, rid = load_arm(args.dir, t, args.k, rid)
        if pr is None:
            raise SystemExit(f"bag8 member {t} incomplete")
        base8.append(pr)
    Z8 = logit(np.stack(base8))
    n = len(rid)

    tags = sorted({os.path.basename(p)[: -len("_cv_oof_f0.npz")]
                   for p in glob.glob(f"{args.dir}/*_cv_oof_f0.npz")})
    cands = [t for t in tags if t not in BAG8]
    if args.only:
        cands = [t for t in cands if t in args.only]

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray([str(x) for x in lab.loc[rid, "anon_user_id"]])
    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    eval_frac = 378.0 / (378.0 + n)

    pool8 = Z8.mean(0)
    loaded = {}
    for t in cands:
        pr, _ = load_arm(args.dir, t, args.k, rid)
        if pr is None or pr.shape != (n, N):
            continue
        loaded[t] = logit(pr)
    print(f"OOF {n} posts / {len(uauth)} authors")
    print(f"{len(loaded)} complete candidate arms outside bag8, "
          f"{args.splits} author partitions, eval {eval_frac:.1%}")
    print(f"SCREEN ONLY -- with {len(loaded)} candidates, ~1 false winner at "
          f"5% is EXPECTED.\n")
    if not loaded:
        raise SystemExit("no complete candidate arms found")

    swi = BAG8.index(args.swap_out)
    rho = {}
    for t, Zc in loaded.items():
        rho[t] = float(np.mean([
            np.corrcoef(Zc[:, j], pool8[:, j])[0, 1] for j in range(N)]))

    keys = [("base", None)] + [(m, t) for t in loaded for m in ("add", "swap")]
    res = {k: [] for k in keys}
    rng = np.random.default_rng(args.seed)
    brng = np.random.default_rng(args.seed + 1)

    for _ in range(args.splits):
        order = list(uauth)
        rng.shuffle(order)
        ev, got = [], 0
        for a in order:
            if got >= eval_frac * n:
                break
            ev.append(a)
            got += len(by[a])
        E = np.concatenate([by[a] for a in ev])
        F = np.setdiff1d(np.arange(n), E)
        s0 = int(brng.integers(1 << 31))

        res[("base", None)].append(
            score(Z8, Y, F, E, s0, args.tbag, args.smooth))
        for t, Zc in loaded.items():
            res[("add", t)].append(score(
                np.concatenate([Z8, Zc[None]]), Y, F, E, s0,
                args.tbag, args.smooth))
            Zs = Z8.copy()
            Zs[swi] = Zc
            res[("swap", t)].append(
                score(Zs, Y, F, E, s0, args.tbag, args.smooth))

    base = np.array(res[("base", None)])
    print("=" * 96)
    print(f"BAG8 = {base.mean():.4f}   (this split sample; the level moves "
          f"+-0.005 between samples,")
    print("                            so ONLY the paired deltas below mean "
          "anything)")
    print("=" * 96)
    print(f"{'candidate':<12}{'rho':>7}{'add d':>9}{'95% CI':>21}{'win':>6}"
          f"{'swap d':>9}{'95% CI':>21}{'win':>6}")
    rows = []
    for t in loaded:
        r = {}
        for m in ("add", "swap"):
            d = np.array(res[(m, t)]) - base
            se = d.std(ddof=1) / np.sqrt(len(d))
            r[m] = (d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se,
                    (d > 0).mean())
        rows.append((t, r))
    rows.sort(key=lambda x: -max(x[1]["add"][0], x[1]["swap"][0]))
    for t, r in rows:
        a, s = r["add"], r["swap"]
        print(f"{t:<12}{rho[t]:>7.3f}{a[0]:>+9.4f}"
              f"{f'[{a[1]:+.4f},{a[2]:+.4f}]':>21}{a[3]:>6.0%}"
              f"{s[0]:>+9.4f}{f'[{s[1]:+.4f},{s[2]:+.4f}]':>21}{s[3]:>6.0%}")

    surv = [(t, m) for t, r in rows for m in ("add", "swap")
            if r[m][3] >= 0.65 and r[m][1] > 0]
    print(f"\n  survivors (win >= 65% AND CI excludes zero): "
          f"{surv if surv else 'NONE'}")
    if surv:
        names = " ".join(sorted({t for t, _ in surv}))
        print(f"  CONFIRM before believing any of it:")
        print(f"    python src/factor_bag9_tboth.py --splits 200 "
              f"--only {names}")
    else:
        print("  The absorption law survives the decode change: no arm outside")
        print("  bag8 earns a place under tboth either. bag8 is saturated.")

    print("\n  rho = mean correlation with the bag8 pool. If the surviving")
    print("  candidates are the LOW-rho ones, diversity is what pays and the")
    print("  Qwen-72B arm (a 72B decoder against 8 encoders) is the best")
    print("  remaining bet. If rho does not predict the delta, it is not.")


if __name__ == "__main__":
    main()
