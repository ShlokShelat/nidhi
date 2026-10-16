"""
factor_arm_weight.py

Tests GLOBAL weighting of the factor ensemble arms against the equal-weight
logit mean.

The arms differ substantially in ranking quality (see factor_incumbent_id.py),
yet the logit mean gives every one of them the same vote. This script asks
whether a single ordering over arms, shared by all 24 labels, improves the
held-out decode.

The weighting is deliberately global rather than per label. Per-label arm
selection makes 24 independent discrete choices, each fitted on a handful of
positives, which is enough freedom to fit noise; one ordering shared across
labels is estimated from every positive in the set.

    equal      the control, reproducing the unweighted decode exactly.
    dropk      discard the k weakest arms by FIT-side mean AP, k = 1..4.
               One small discrete parameter, and a direct test of whether the
               weak arms are harmful or are useful diversity.
    softmax    w_a proportional to exp((AP_a - mean AP) / T). Large T is equal
               weight, small T is winner-take-all. One continuous parameter,
               and it degrades gracefully instead of discarding an arm.
    perlabel   the same softmax with AP measured per label, included so the
               global-vs-per-label comparison is explicit.

Arm AP is re-estimated on the FIT authors inside every split and never on the
eval posts, so a ranking that is lucky on some rows cannot be both chosen and
rewarded on those same rows.

Read the WIN RATE alongside the mean: with paired splits, a positive mean at a
win rate near 50% comes from a few large splits rather than a consistent
effect, and will not reproduce.

Usage (CPU):
  python src/factor_arm_weight.py --splits 80
  python src/factor_arm_weight.py --build-test artifacts/factors_armw_test.csv \
      --use dropk --k-drop 2
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


def resolve_test(d, a):
    for c in (f"{d}/{a}_cv_test.npz", f"{d}/{a}_test.npz",
              f"{d}/nli_ensemble_{a}_test.npz"):
        if os.path.exists(c):
            return c
    return None


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def cells(col, n):
    M = np.zeros((n, N), dtype=np.int8)
    for i, c in enumerate(col):
        for x in set(parse_factors(str(c))):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


def ap(p, y):
    s = y.sum()
    if s == 0:
        return np.nan
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / s)


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


def thr_smooth(p, y, w=9):
    c = f1_curve(p, y)
    if w > 1:
        c = np.convolve(c, np.ones(w) / w, mode="same")
    return float(TGRID[int(np.argmax(c))])


def thr_bag(p, y, rng, B, w):
    out, n = [], len(p)
    for _ in range(B):
        ix = rng.integers(0, n, n)
        if y[ix].sum() < 1:
            continue
        out.append(thr_smooth(p[ix], y[ix], w))
    return float(np.mean(out)) if out else 0.5


def f1pl(P, Y):
    P = P.astype(np.int64)
    Y = Y.astype(np.int64)
    tp = (P * Y).sum(0)
    den = 2 * tp + (P * (1 - Y)).sum(0) + ((1 - P) * Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def arm_ap(Z, Y, per_label):
    """mean AP per arm (A,) or per arm-and-label (A,N), on the rows given."""
    A = Z.shape[0]
    M = np.full((A, N), np.nan)
    for a in range(A):
        for j in range(N):
            M[a, j] = ap(Z[a, :, j], Y[:, j])
    return M if per_label else np.nanmean(M, axis=1)


def weights(kind, apv, par):
    """apv is (A,) or (A,N). Returns weights of the same shape, summing to 1
    over the arm axis."""
    if kind == "equal":
        w = np.ones_like(apv)
    elif kind == "dropk":
        w = np.ones_like(apv)
        if apv.ndim == 1:
            w[np.argsort(apv)[:int(par)]] = 0.0
        else:
            for j in range(apv.shape[1]):
                w[np.argsort(apv[:, j])[:int(par)], j] = 0.0
    else:  # softmax
        c = apv - np.nanmean(apv, axis=0, keepdims=True)
        w = np.exp(np.clip(c / max(par, 1e-6), -50, 50))
    w = np.nan_to_num(w, nan=0.0)
    s = w.sum(axis=0, keepdims=True)
    return w / np.where(s == 0, 1, s)


def pool(Z, w):
    """Z (A,n,N), w (A,) or (A,N) -> (n,N) probabilities."""
    ww = w[:, None, :] if w.ndim == 2 else w[:, None, None]
    return sig((Z * ww).sum(axis=0))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--arms", nargs="+", default=BAG8)
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--k", type=int, default=5)
    a_.add_argument("--splits", type=int, default=80)
    a_.add_argument("--tbag", type=int, default=40)
    a_.add_argument("--smooth", type=int, default=9)
    a_.add_argument("--seed", type=int, default=13)
    a_.add_argument("--build-test", default=None)
    a_.add_argument("--use", default="dropk",
                    choices=["dropk", "softmax", "perlabel"])
    a_.add_argument("--k-drop", type=int, default=2)
    a_.add_argument("--temp", type=float, default=0.02)
    a_.add_argument("--shipped", default="submissions/NIDHI_tboth.csv")
    args = a_.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    Z = logit(S)
    n = len(rid)
    A = len(args.arms)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray([str(x) for x in lab.loc[rid, "anon_user_id"]])
    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    eval_frac = 378.0 / (378.0 + n)

    print(f"OOF {n} posts / {len(uauth)} authors   {A} arms {args.arms}")
    print(f"{args.splits} author partitions, eval {eval_frac:.1%}   "
          f"threshold bagging B={args.tbag} w={args.smooth}")
    print("arm AP is re-estimated on the FIT authors inside every split\n")

    full = arm_ap(Z, Y, False)
    print("full-OOF mean AP per arm (context only -- never used to choose):")
    for i in np.argsort(-full):
        print(f"   {args.arms[i]:<8}{full[i]:>8.4f}")

    variants = [("equal", 0.0)] + \
               [("dropk", k) for k in (1, 2, 3, 4)] + \
               [("softmax", t) for t in (0.05, 0.02, 0.01)] + \
               [("perlabel", 0.02)]
    res = {v: [] for v in variants}
    dropped = {t: 0 for t in args.arms}
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
        seed0 = int(brng.integers(1 << 31))

        g = arm_ap(Z[:, F], Y[F], False)
        pl = None
        for i in np.argsort(g)[:2]:
            dropped[args.arms[i]] += 1

        for kind, par in variants:
            if kind == "perlabel":
                if pl is None:
                    pl = arm_ap(Z[:, F], Y[F], True)
                w = weights("softmax", pl, par)
            else:
                w = weights(kind, g, par)
            PF, PE = pool(Z[:, F], w), pool(Z[:, E], w)
            th = np.array([
                thr_bag(PF[:, j], Y[F, j], np.random.default_rng(seed0 + j),
                        args.tbag, args.smooth) for j in range(N)])
            res[(kind, par)].append(
                f1pl((PE >= th[None, :]).astype(np.int8), Y[E]).mean())

    base = np.array(res[("equal", 0.0)])
    print("\n" + "=" * 88)
    print("GLOBAL ARM WEIGHTING, BAGGED THRESHOLDS -- held-out macro-F1")
    print("=" * 88)
    print(f"{'variant':<12}{'param':>8}{'macro':>9}{'delta':>10}"
          f"{'95% CI':>21}{'win':>6}")
    for kind, par in variants:
        v = np.array(res[(kind, par)])
        if kind == "equal":
            print(f"{'equal':<12}{'--':>8}{v.mean():>9.4f}{'--':>10}"
                  f"{'-- (baseline)':>21}{'--':>6}")
            continue
        d = v - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        nm = f"{kind}" + (f" k={int(par)}" if kind == "dropk" else "")
        print(f"{nm:<12}{par:>8.3g}{v.mean():>9.4f}{d.mean():>+10.4f}"
              f"{f'[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]':>21}"
              f"{(d > 0).mean():>6.0%}")

    bk = max((v for v in variants if v[0] != "equal"),
             key=lambda v: np.mean(res[v]))
    d = np.array(res[bk]) - base
    print(f"\n  best {bk[0]} {bk[1]:g}: {d.mean():+.4f} macro  "
          f"win {(d > 0).mean():.0%}  composite {0.3*d.mean():+.4f}")
    print("  A POSITIVE MEAN WITH A WIN RATE NEAR 50% IS NOISE -- it comes")
    print("  from a few large splits, not a consistent effect. Both have to")
    print("  move together before a variant counts.")
    print("\n  how often each arm was in the two weakest by fit AP:")
    for t, c in sorted(dropped.items(), key=lambda x: -x[1]):
        print(f"   {t:<8}{c:>5} / {args.splits}  ({c/args.splits:.0%})")
    print("  a stable ranking means dropk is choosing a real ordering; if the")
    print("  counts are spread evenly the ranking is noise and so is dropk.")

    if args.build_test:
        T, tid = [], None
        for t in args.arms:
            p = resolve_test(args.dir, t)
            if p is None:
                raise SystemExit(f"no test dump for {t}")
            z = np.load(p, allow_pickle=True)
            T.append(z["probs"].astype(float))
            tid = [str(x) for x in z["row_id"]]
        Zt = logit(np.stack(T))

        ship = pd.read_csv(args.shipped, keep_default_na=False)
        shipM = cells(ship["factors"], len(ship))
        order = {r: i for i, r in enumerate(ship["row_id"].astype(str))}
        perm = [order[r] for r in tid]

        we = weights("equal", full, 0.0)
        P0 = pool(Z, we)
        t0 = np.array([thr_bag(P0[:, j], Y[:, j], np.random.default_rng(999+j),
                               args.tbag, args.smooth) for j in range(N)])
        repro = int(((pool(Zt, we) >= t0[None, :]).astype(np.int8)
                     != shipM[perm]).sum())
        print(f"\nPROVENANCE: equal-weight reconstruction vs {args.shipped} -- "
              f"{repro} of {len(tid)*N} cells differ")
        if repro:
            raise SystemExit("REFUSING to write -- reconstruction mismatch.")

        if args.use == "perlabel":
            w = weights("softmax", arm_ap(Z, Y, True), args.temp)
        elif args.use == "dropk":
            w = weights("dropk", full, args.k_drop)
        else:
            w = weights("softmax", full, args.temp)
        Pf = pool(Z, w)
        th = np.array([thr_bag(Pf[:, j], Y[:, j], np.random.default_rng(999+j),
                               args.tbag, args.smooth) for j in range(N)])
        on = (pool(Zt, w) >= th[None, :]).astype(np.int8)

        out = ship.copy()
        col = list(out["factors"])
        moved = 0
        for i, row in enumerate(tid):
            labs = str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
            kx = order[row]
            if str(col[kx]) != labs:
                moved += 1
            col[kx] = labs
        out["factors"] = col
        os.makedirs(os.path.dirname(args.build_test) or ".", exist_ok=True)
        out.to_csv(args.build_test, index=False)
        print(f"{args.use}: {on.sum()/len(tid):.2f} factors/post "
              f"(shipped {shipM.sum()/len(ship):.2f}, gold 2.92)")
        print(f"cells changed {int((on != shipM[perm]).sum())}   "
              f"posts changed {moved}")
        print(f"wrote {args.build_test}")
        print("Only the factors column changes; every other column is copied "
              "through unmodified.")


if __name__ == "__main__":
    main()
