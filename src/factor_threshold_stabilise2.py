"""
factor_threshold_stabilise2.py

PUSH THE ONE AXIS THAT WORKED.

`tboth` (bagged + smoothed thresholds) beat the shipped `tfit` with the CI
clear of zero, and the divergence pattern confirmed the mechanism:
identical thresholds on `hopelessness` (745 golds), wildly different on
`poor school performance` (16 golds, where tfit's argmax hit 0.970). The loss is
ESTIMATION NOISE in 24 numbers, not missing information -- a wide range of arm
additions, replacements, and an LLM fine-tune have all failed to add any.

So the question is how much further the noise can be squeezed. Four extensions,
all still ZERO new parameters -- the same 24 thresholds, estimated better:

  tpred    PROPER bagging. This averages the THRESHOLDS; textbook bagging
           averages the PREDICTIONS. For a step function those are different
           objects: mean(1[p >= t_b]) != 1[p >= mean(t_b)]. Fit B threshold
           sets, predict with each, majority-vote. Costs nothing extra -- it
           reuses the very same B fits that tbag already computes.

  tauth    AUTHOR-level bootstrap inside the bag. The resample here draws
           POSTS, but the data is clustered by author and the EVALUATION
           resamples authors.
           A post-level bootstrap understates the true variance, so it
           under-shrinks. Bagging should resample the unit it is defending
           against.

  tapred   both of the above together.

  and a sweep of B and the smoothing window, which were guesses (B=40, w=9) and
  have never been tuned. If tboth is not converged in B, that is free.

ALSO REPORTED: FLIP RATE per label -- the share of eval posts whose prediction
changes across the B threshold sets. That is a direct read of how unstable each
label's operating point is, and it should track the gold count. Labels with a
high flip rate are where any remaining gain lives.

Judged exactly as before: cross-fit at the real 1635-fit / 378-eval ratio on
author partitions, scored against `tboth` (the new incumbent, NOT tfit). Report
both the mean and the WIN RATE -- a positive mean with a win rate near 50%
would still lose.

Usage:
  python src/factor_threshold_stabilise2.py
  python src/factor_threshold_stabilise2.py --splits 300 --sweep
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
PROTECTIVE = ["social support", "coping strategy", "psychological capital",
              "sense of responsibility", "meaning in life"]
PROT_J = [IDX[t] for t in PROTECTIVE]
TGRID = np.round(np.arange(0.02, 0.985, 0.005), 3)


def load_oof(d, arms, k):
    O, rid = [], None
    for a in arms:
        pr, ids = [], []
        for f in range(k):
            p = f"{d}/{a}_cv_oof_f{f}.npz"
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
        O.append(pr)
    return rid, np.stack(O)


def logit_mean(S):
    X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))


def cells(col, n):
    M = np.zeros((n, N), dtype=np.int8)
    for i, cell in enumerate(col):
        for x in set(parse_factors(str(cell))):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


def f1pl(P, Y):
    P = P.astype(np.int64)
    Y = Y.astype(np.int64)
    tp = (P * Y).sum(0)
    den = 2 * tp + (P * (1 - Y)).sum(0) + ((1 - P) * Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def f1_curve(p, y):
    npos = int(y.sum())
    if npos == 0:
        return np.zeros(len(TGRID))
    o = np.argsort(-p, kind="stable")
    cum = np.cumsum(y[o])
    ks = np.searchsorted(-p[o], -TGRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    return 2.0 * tp / np.maximum(ks + npos, 1)


def thr(p, y, w):
    c = f1_curve(p, y)
    if w > 1:
        c = np.convolve(c, np.ones(w) / w, mode="same")
    return float(TGRID[int(np.argmax(c))])


def boot_thresholds(PF, YF, rng, B, w, authF=None):
    """B threshold vectors from B resamples. Author-level if authF is given --
    the evaluation resamples authors, so the bag should too."""
    n = len(PF)
    if authF is not None:
        ua = pd.unique(authF)
        by = {a: np.flatnonzero(authF == a) for a in ua}
    out = np.zeros((B, N))
    for b in range(B):
        if authF is not None:
            pick = rng.choice(len(ua), len(ua), replace=True)
            ix = np.concatenate([by[ua[i]] for i in pick])
        else:
            ix = rng.integers(0, n, n)
        yy = YF[ix]
        pp = PF[ix]
        for j in range(N):
            out[b, j] = thr(pp[:, j], yy[:, j], w) if yy[:, j].sum() >= 1 \
                else 0.5
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=150)
    ap.add_argument("--tbag", type=int, default=40)
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--sweep", action="store_true",
                    help="also sweep B and the smoothing window")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    P = logit_mean(S)
    n = len(rid)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])
    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    eval_frac = 378.0 / (378.0 + n)

    print(f"OOF {n} posts / {len(uauth)} authors   {args.splits} partitions")
    print(f"B={args.tbag}  window={args.smooth}   baseline is TBOTH, the new "
          f"incumbent\n")

    names = ["tfit", "tboth", "tpred", "tauth", "tapred"]
    res = {k: [] for k in names}
    prot = {k: [] for k in names}
    flips = np.zeros(N)
    nsp = 0
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
        PF, YF, PE, YE = P[F], Y[F], P[E], Y[E]

        tp_ = np.array([thr(PF[:, j], YF[:, j], 1) for j in range(N)])
        Bp = boot_thresholds(PF, YF, brng, args.tbag, args.smooth)
        Ba = boot_thresholds(PF, YF, brng, args.tbag, args.smooth, auth[F])

        # votes: mean over B of 1[p >= t_b]  -- proper bagging
        vp = np.stack([(PE >= Bp[b][None, :]) for b in range(args.tbag)]
                      ).mean(0)
        va = np.stack([(PE >= Ba[b][None, :]) for b in range(args.tbag)]
                      ).mean(0)
        flips += np.minimum(vp, 1 - vp).mean(0)
        nsp += 1

        preds = {
            "tfit": (PE >= tp_[None, :]),
            "tboth": (PE >= Bp.mean(0)[None, :]),
            "tpred": (vp >= 0.5),
            "tauth": (PE >= Ba.mean(0)[None, :]),
            "tapred": (va >= 0.5),
        }
        for k in names:
            f = f1pl(preds[k].astype(np.int8), YE)
            res[k].append(f.mean())
            prot[k].append(f[PROT_J].mean())

    ref = np.array(res["tboth"])
    print("=" * 92)
    print("HELD-OUT macro-F1 -- vs TBOTH (the new incumbent)")
    print("=" * 92)
    print(f"{'estimator':<10}{'macro':>9}{'protective':>12}{'d vs tboth':>12}"
          f"{'95% CI':>20}{'win':>6}")
    for k in names:
        v = np.array(res[k])
        if k == "tboth":
            print(f"{k:<10}{v.mean():>9.4f}{np.mean(prot[k]):>12.4f}"
                  f"{'--':>12}{'--':>20}{'--':>6}")
            continue
        d = v - ref
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{k:<10}{v.mean():>9.4f}{np.mean(prot[k]):>12.4f}"
              f"{d.mean():>+12.4f}"
              f"{f'[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]':>20}"
              f"{(d > 0).mean():>6.0%}")

    print("\n" + "=" * 92)
    print("FLIP RATE -- share of eval posts whose call changes across the B "
          "threshold sets")
    print("=" * 92)
    flips /= max(nsp, 1)
    print(f"{'label':<44}{'gold n':>8}{'flip rate':>11}")
    for j in np.argsort(-flips):
        star = "* " if FACTOR_TAXONOMY[j] in PROTECTIVE else "  "
        print(f"{star}{FACTOR_TAXONOMY[j]:<42}{int(Y[:, j].sum()):>8}"
              f"{flips[j]:>10.1%}")
    print("\n  A high flip rate means the operating point is genuinely "
          "unstable for that")
    print("  label. It should track the gold count -- if a COMMON label flips "
          "a lot, its")
    print("  probability distribution is flat near the threshold and no "
          "estimator fixes it.")

    if args.sweep:
        print("\n" + "=" * 92)
        print("SWEEP B AND WINDOW -- both were guesses, never tuned")
        print("=" * 92)
        rng2 = np.random.default_rng(args.seed + 5)
        brng2 = np.random.default_rng(args.seed + 6)
        splits = []
        for _ in range(max(40, args.splits // 3)):
            order = list(uauth)
            rng2.shuffle(order)
            ev, got = [], 0
            for a in order:
                if got >= eval_frac * n:
                    break
                ev.append(a)
                got += len(by[a])
            E = np.concatenate([by[a] for a in ev])
            splits.append((E, np.setdiff1d(np.arange(n), E)))
        print(f"{'B':>5}{'window':>8}{'macro':>10}")
        for B in (20, 40, 100):
            for w in (1, 9, 21):
                sc = []
                for E, F in splits:
                    Bt = boot_thresholds(P[F], Y[F], brng2, B, w)
                    sc.append(f1pl((P[E] >= Bt.mean(0)[None, :]).astype(np.int8),
                                   Y[E]).mean())
                print(f"{B:>5}{w:>8}{np.mean(sc):>10.4f}")

    print("\n" + "=" * 92)
    print("HOW TO DECIDE")
    print("=" * 92)
    print("Beat TBOTH, not tfit -- tboth is what we are about to submit.")
    print("Require a positive mean AND a win rate that is not a coin flip.")
    print("If nothing beats it, the variance-reduction lane is spent and the")
    print("remaining 0.02 to the ceiling is irreducible with this pool.")


if __name__ == "__main__":
    main()
