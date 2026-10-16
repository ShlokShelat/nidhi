"""
factor_antonym.py

Measures, and optionally corrects, the coupling between protective factors and
the risk factors that are their surface-level polarity opposites.

The PFA paper (arXiv 2507.10008) the corpus follows defines 19 risk and 5
protective factors. Four pairs read as plain antonyms:

    RISK      poor social support        PROTECTIVE   social support
    RISK      hopelessness               PROTECTIVE   meaning in life
    RISK      hopelessness/low self-est  PROTECTIVE   psychological capital
    RISK      emotion dysregulation      PROTECTIVE   coping strategy

Lexical overlap with opposite polarity is a well-documented NLI failure mode,
so an entailment arm can be expected to treat these pairs as mutually
exclusive. The annotation does not: in this corpus the gold correlation of each
pair is POSITIVE, because a single post can describe both a support that is
absent and a support that is present. Section 1 of the report prints the gold
and predicted correlation side by side so the direction and size of any
mismatch is visible rather than assumed.

THE CORRECTION, ONE GLOBAL PARAMETER, SIGN LEFT TO THE DATA:

    z_A' = z_A - beta * mean(z_coupled)

beta > 0 subtracts, treating the pair as antonyms. beta < 0 adds, treating them
as co-occurring. beta = 0 reproduces the unadjusted decode bit for bit, which
is the built-in control on the harness. Thresholds are refit on the adjusted
scores inside every fold, because changing the score changes where the cut
belongs.

This is per-label stacking restricted to five labels and a single coefficient.
Full 24x24 stacking (factor_structure.py) fits hundreds of coefficients from
the same posts, which at this sample size fits noise; one coefficient aimed at
labels with a measured mis-coupling is the smallest form of the idea that can
still be estimated.

Evaluation is cross-fit on AUTHOR partitions at the deployment fit/eval ratio,
with thresholds refit inside every fold, and reports paired deltas and a win
rate over the splits rather than a single split's number.

Usage:
  python src/factor_antonym.py
  python src/factor_antonym.py --build-test artifacts/factors_anti_test.csv --beta 0.1
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
TGRID = np.round(np.arange(0.02, 0.985, 0.005), 3)

# target -> the risk factors whose language is its polarity opposite.
# Symmetric entries are included because the confusion runs both ways: an
# encoder that cannot tell "support" from "no support" mislabels both.
ANT = {
    "social support":          ["poor social support"],
    "poor social support":     ["social support"],
    "meaning in life":         ["hopelessness"],
    "psychological capital":   ["hopelessness", "low self-esteem"],
    "coping strategy":         ["emotion dysregulation"],
}


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
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def logit_mean(S):
    return 1.0 / (1.0 + np.exp(-np.mean(logit(S), axis=0)))


def cells(col, n):
    M = np.zeros((n, N), dtype=np.int8)
    for i, cell in enumerate(col):
        for x in set(parse_factors(str(cell))):
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


def thr_smooth(p, y, w=9):
    c = f1_curve(p, y)
    if w > 1:
        c = np.convolve(c, np.ones(w) / w, mode="same")
    return float(TGRID[int(np.argmax(c))])


def thr_bag(p, y, rng, B, w):
    """Threshold as the mean argmax over B bootstraps of a smoothed F1 curve.

    Bagging the argmax stops the cut landing on a single lucky point of a
    jagged, low-count F1 curve.
    """
    n = len(p)
    out = []
    for _ in range(B):
        ix = rng.integers(0, n, n)
        yy = y[ix]
        if yy.sum() < 1:
            continue
        out.append(thr_smooth(p[ix], yy, w))
    return float(np.mean(out)) if out else 0.5


def f1pl(P, Y):
    P = P.astype(np.int64)
    Y = Y.astype(np.int64)
    tp = (P * Y).sum(0)
    den = 2 * tp + (P * (1 - Y)).sum(0) + ((1 - P) * Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def ap(p, y):
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    cum = np.cumsum(ys)
    prec = cum / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def adjust(Z, beta):
    """z_A' = z_A - beta * mean(z of A's antagonists). beta=0 is a no-op."""
    if beta == 0:
        return Z
    Z2 = Z.copy()
    for tgt, srcs in ANT.items():
        j = IDX.get(tgt)
        js = [IDX[s] for s in srcs if s in IDX]
        if j is None or not js:
            continue
        Z2[:, j] = Z[:, j] - beta * Z[:, js].mean(axis=1)
    return Z2


TOUCHED = sorted({IDX[t] for t in ANT if t in IDX})


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dir", default="artifacts/factor_runs")
    ap_.add_argument("--arms", nargs="+", default=BAG8)
    ap_.add_argument("--labeled", default="artifacts/full_train.csv")
    ap_.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap_.add_argument("--k", type=int, default=5)
    ap_.add_argument("--splits", type=int, default=150)
    ap_.add_argument("--tbag", type=int, default=40)
    ap_.add_argument("--smooth", type=int, default=9)
    ap_.add_argument("--seed", type=int, default=13)
    ap_.add_argument("--betas", nargs="+", type=float,
                     default=[-0.35, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20])
    ap_.add_argument("--build-test", default=None)
    ap_.add_argument("--beta", type=float, default=0.10,
                     help="beta to use for --build-test")
    ap_.add_argument("--shipped", default="artifacts/factors_tboth_test.csv",
                     help="the column the new one is diffed against")
    args = ap_.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    P = logit_mean(S)
    Z = logit(P)
    n = len(rid)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])
    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    eval_frac = 378.0 / (378.0 + n)

    print(f"OOF {n} posts / {len(uauth)} authors   arms {args.arms}")

    # ---------------- 1. does the confusion exist? --------------------------
    print("\n" + "=" * 100)
    print("1. ANTONYM COUPLING -- gold correlation vs predicted correlation")
    print("=" * 100)
    print(f"{'A (target)':<26}{'B (antonym)':<24}{'nA':>5}{'nB':>5}"
          f"{'gold r':>9}{'pred r':>9}{'pA|B-only':>11}{'pA|neither':>12}{'lift':>8}")
    flagged = []
    for tgt, srcs in ANT.items():
        for s in srcs:
            a, b = IDX[tgt], IDX[s]
            ya, yb = Y[:, a].astype(float), Y[:, b].astype(float)
            gr = float(np.corrcoef(ya, yb)[0, 1])
            pr = float(np.corrcoef(Z[:, a], Z[:, b])[0, 1])
            m1 = (yb == 1) & (ya == 0)
            m0 = (yb == 0) & (ya == 0)
            v1 = float(P[m1, a].mean()) if m1.any() else float("nan")
            v0 = float(P[m0, a].mean()) if m0.any() else float("nan")
            lift = v1 / v0 if v0 else float("nan")
            bad = (pr < gr - 0.05)
            flagged.append(bad)
            print(f"{tgt:<26}{s:<24}{int(ya.sum()):>5}{int(yb.sum()):>5}"
                  f"{gr:>+9.3f}{pr:>+9.3f}{v1:>11.3f}{v0:>12.3f}"
                  f"{lift:>7.2f}x{'  <-- UNDER-COUPLED' if bad else ''}")
    print("\n  A POSITIVE gold r means the pair attracts in the annotation and")
    print("  is not behaving as an antonym pair. pred r well BELOW gold r means")
    print("  the scores impose an independence (or opposition) the annotation")
    print("  denies, which is what a negative beta corrects.")

    # ---------------- 2. does discrimination collapse on B posts? -----------
    print("\n" + "=" * 100)
    print("2. RANKING QUALITY ON THE CONTESTED SUBSET (AP for A)")
    print("=" * 100)
    print(f"{'A (target)':<26}{'B (antonym)':<24}{'AP all':>9}"
          f"{'AP | B=1':>10}{'AP | B=0':>10}{'drop':>9}")
    for tgt, srcs in ANT.items():
        for s in srcs:
            a, b = IDX[tgt], IDX[s]
            m1, m0 = Y[:, b] == 1, Y[:, b] == 0
            A, A1, A0 = (ap(P[:, a], Y[:, a]),
                         ap(P[m1, a], Y[m1, a]),
                         ap(P[m0, a], Y[m0, a]))
            print(f"{tgt:<26}{s:<24}{A:>9.3f}{A1:>10.3f}{A0:>10.3f}"
                  f"{A1 - A0:>+9.3f}")
    print("\n  a large negative drop means A is much harder to rank exactly where")
    print("  its antonym is present -- which is where the subtraction can help.")

    # ---------------- 3. cross-fit the fix ----------------------------------
    print("\n" + "=" * 100)
    print(f"3. ANTAGONIST SUBTRACTION -- {args.splits} author partitions, "
          f"eval {eval_frac:.1%}, bagged thresholds refit per fold")
    print("=" * 100)

    betas = list(args.betas)
    if 0.0 not in betas:
        betas.insert(0, 0.0)
    res = {b: [] for b in betas}
    resP = {b: [] for b in betas}
    rng = np.random.default_rng(args.seed)
    brng = np.random.default_rng(args.seed + 1)
    PJ = [IDX[t] for t in PROTECTIVE]

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

        # baseline thresholds on the UNADJUSTED scores, all 24 labels
        seed0 = brng.integers(1 << 31)
        base_t = np.array([
            thr_bag(P[F, j], Y[F, j], np.random.default_rng(seed0 + j),
                    args.tbag, args.smooth) for j in range(N)])

        for b in betas:
            if b == 0.0:
                Pe, th = P[E], base_t
            else:
                Za, Ze = adjust(Z[F], b), adjust(Z[E], b)
                Pa = 1.0 / (1.0 + np.exp(-Za))
                Pe = 1.0 / (1.0 + np.exp(-Ze))
                th = base_t.copy()
                # only the touched labels changed, so only they need refitting
                for j in TOUCHED:
                    th[j] = thr_bag(Pa[:, j], Y[F, j],
                                    np.random.default_rng(seed0 + j),
                                    args.tbag, args.smooth)
            pred = (Pe >= th[None, :]).astype(np.int8)
            f = f1pl(pred, Y[E])
            res[b].append(f.mean())
            resP[b].append(f[PJ].mean())

    base = np.array(res[0.0])
    print(f"{'beta':>7}{'macro24':>10}{'delta':>10}{'95% CI':>21}{'win':>6}"
          f"{'protective5':>13}{'d prot':>9}")
    for b in betas:
        v = np.array(res[b])
        vp = np.array(resP[b])
        if b == 0.0:
            print(f"{b:>7.2f}{v.mean():>10.4f}{'--':>10}{'-- (baseline)':>21}"
                  f"{'--':>6}{vp.mean():>13.4f}{'--':>9}")
            continue
        d = v - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{b:>7.2f}{v.mean():>10.4f}{d.mean():>+10.4f}"
              f"{f'[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]':>21}"
              f"{(d > 0).mean():>6.0%}{vp.mean():>13.4f}"
              f"{vp.mean()-np.array(resP[0.0]).mean():>+9.4f}")

    bb = max((b for b in betas if b != 0.0), key=lambda b: np.mean(res[b]))
    d = np.array(res[bb]) - base
    print(f"\n  best beta {bb}: {d.mean():+.4f} macro, win {(d > 0).mean():.0%}, "
          f"composite {0.3*d.mean():+.4f}")
    print("  beta=0 is the unadjusted decode and serves as the harness control.")

    # ---------------- 4. build ----------------------------------------------
    if args.build_test:
        T, tid = [], None
        for a in args.arms:
            p = resolve_test(args.dir, a)
            if p is None:
                raise SystemExit(f"no test dump for {a}")
            z = np.load(p, allow_pickle=True)
            T.append(z["probs"].astype(float))
            tid = [str(x) for x in z["row_id"]]
        Pt = logit_mean(np.stack(T))
        Zt = logit(Pt)

        ship = pd.read_csv(args.shipped, keep_default_na=False)
        shipM = cells(ship["factors"], len(ship))
        order = {r: i for i, r in enumerate(ship["row_id"].astype(str))}
        perm = [order[r] for r in tid]

        seed0 = 999
        t0 = np.array([thr_bag(P[:, j], Y[:, j],
                               np.random.default_rng(seed0 + j),
                               args.tbag, args.smooth) for j in range(N)])
        repro = int(((Pt >= t0[None, :]).astype(np.int8) != shipM[perm]).sum())
        print(f"\nPROVENANCE: baseline reconstruction vs {args.shipped} -- "
              f"{repro} of {len(tid)*N} cells differ")
        if repro:
            raise SystemExit(
                "REFUSING to write -- the reconstruction does not match the "
                "shipped column, so the new file would mix a rebuild bug with "
                "the intended change.")

        Pa_full = 1.0 / (1.0 + np.exp(-adjust(Z, args.beta)))
        Pt_a = 1.0 / (1.0 + np.exp(-adjust(Zt, args.beta)))
        th = t0.copy()
        for j in TOUCHED:
            th[j] = thr_bag(Pa_full[:, j], Y[:, j],
                            np.random.default_rng(seed0 + j),
                            args.tbag, args.smooth)
        on = (Pt_a >= th[None, :]).astype(np.int8)

        out = ship.copy()
        col = list(out["factors"])
        moved = 0
        for i, row in enumerate(tid):
            labs = str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
            k = order[row]
            if str(col[k]) != labs:
                moved += 1
            col[k] = labs
        out["factors"] = col
        os.makedirs(os.path.dirname(args.build_test) or ".", exist_ok=True)
        out.to_csv(args.build_test, index=False)
        print(f"beta {args.beta}: {on.sum()/len(tid):.2f} factors/post "
              f"(shipped {shipM.sum()/len(ship):.2f}, gold 2.92)")
        print(f"cells changed {int((on != shipM[perm]).sum())}   "
              f"posts changed {moved}")
        print(f"wrote {args.build_test}")
        print("Only the factors column changes; every other column is copied "
              "through unmodified.")


if __name__ == "__main__":
    main()
