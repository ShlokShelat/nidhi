"""
factor_author_tboth.py

AUTHOR POOLING WAS CLOSED UNDER A DECODE THAT STRUCTURALLY FORBIDS IT.

factor_author_pool.py tested the blend under a QUANTILE decode, which fires
exactly k = rate x n posts per label. The decode has since moved to a
THRESHOLD (factor_threshold_decode.py), and then to tboth.

That is not a bookkeeping detail, it is the whole mechanism:

  * pooling pulls every post by one author toward their shared mean. It is a
    COUNT-CHANGING intervention -- a trait carrier should get the label on all
    ten of their posts, a non-carrier on none.
  * a quantile decode fires a FIXED total count. Every post pooling promotes
    must demote another. The intervention's entire effect is cancelled by
    construction, and what survives is re-ranking noise.
  * a threshold has no such constraint. All ten posts can clear it.

The gold structure says the effect being suppressed is large. Leave-one-out,
using no text whatsoever, only the author's OTHER posts:

    label                       base   P(y|author has it elsewhere)   lift
    low socio-economic status  0.033              0.204              36.0x
    substance use              0.020              0.095              11.1x
    cognitive deficits         0.020              0.097              10.0x
    social support             0.069              0.157               7.0x
    exposure to others suicide 0.009              0.040               6.8x
    hopelessness               0.456              0.482               4.6x

Every one of the 24 labels lifts by at least 2.1x, and the lift is LARGEST on
the rare labels -- which are exactly the labels dragging a macro average down.

DEPLOYABLE, NOT MERELY DIAGNOSTIC. The test csv carries anon_user_id: 378 posts
over 36 authors, 7 to 28 posts each, median 9. Train and test authors are
DISJOINT (verified, 0 overlap), so pooling a test post uses only other TEST
posts by that author. No gold is consulted anywhere -- this is transductive,
not leaky. The cross-fit below mirrors that exactly: eval-author means are
computed from eval posts only, never from fit posts.

VARIANTS
  w=0        tboth itself. MUST reproduce 0.6476 or the harness is broken.
  wglobal    one w for all 24 labels. 1 fitted parameter at most, and the
             fixed-w rows (0.15/0.25/0.40) cost ZERO.
  wtrait     w only on the labels that are person-attributes rather than
             events, chosen A PRIORI from the taxonomy's semantics, not from
             any score. 0 fitted parameters.

Blending is in LOGIT space, matching how the 8 arms are already pooled.

Usage (cluster, CPU):
  python src/factor_author_tboth.py --splits 120
  python src/factor_author_tboth.py --build-test artifacts/factors_apool_test.csv --w 0.25
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

# Person-attributes: things that do not change between a person's posts.
# Chosen from what the label MEANS, before seeing any score, so this costs
# zero fitted parameters. Everything else is treated as an event.
TRAITS = [
    "sexual orientation related issues",
    "dysfunctional family",
    "physical health/characteristic",
    "low socio-economic status",
    "poor school performance",
    "substance use",
    "cognitive deficits",
    "prior self-harm or suicidal thought/attempt",
    "traumatic experience",
    "interpersonal violence",
    "exposure to others' suicide",
    "mental health issues",
]


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


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit_mean(S):
    return sig(np.mean(logit(S), axis=0))


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
    out = []
    n = len(p)
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


def blend(Z, auth, wvec):
    """z' = (1-w_j) z + w_j * (author mean of z). Groups come ONLY from the
    rows passed in, which is what makes the cross-fit deployment-faithful."""
    if not np.any(wvec):
        return Z
    Z2 = Z.copy()
    for a in np.unique(auth):
        m = auth == a
        mu = Z[m].mean(axis=0, keepdims=True)
        Z2[m] = (1 - wvec)[None, :] * Z[m] + wvec[None, :] * mu
    return Z2


def fit_thr(P, Y, seed0, B, sw):
    return np.array([
        thr_bag(P[:, j], Y[:, j], np.random.default_rng(seed0 + j), B, sw)
        for j in range(N)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=120)
    ap.add_argument("--tbag", type=int, default=40)
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ws", nargs="+", type=float,
                    default=[0.0, 0.15, 0.25, 0.40, 0.60])
    ap.add_argument("--build-test", default=None)
    ap.add_argument("--w", type=float, default=0.25)
    ap.add_argument("--trait-only", action="store_true",
                    help="use the a-priori TRAITS set for --build-test")
    ap.add_argument("--shipped", default="submissions/NIDHI_tboth.csv")
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    P = logit_mean(S)
    Z = logit(P)
    n = len(rid)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray([str(x) for x in lab.loc[rid, "anon_user_id"]])
    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    eval_frac = 378.0 / (378.0 + n)
    TJ = np.array([IDX[t] for t in TRAITS if t in IDX])

    print(f"OOF {n} posts / {len(uauth)} authors "
          f"({n/len(uauth):.1f} per author)   arms {args.arms}")
    print(f"{args.splits} author partitions, eval {eval_frac:.1%}   "
          f"tboth B={args.tbag} w={args.smooth}")
    print(f"a-priori TRAITS: {len(TJ)} of {N} labels\n")

    ws = list(args.ws)
    if 0.0 not in ws:
        ws.insert(0, 0.0)
    names = [("wglobal", w) for w in ws] + \
            [("wtrait", w) for w in ws if w > 0]
    res = {k: [] for k in names}
    perlab = {k: [] for k in names}
    dens = {k: [] for k in names}

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

        for kind, w in names:
            wv = np.zeros(N)
            if kind == "wglobal":
                wv[:] = w
            else:
                wv[TJ] = w
            ZF = blend(Z[F], auth[F], wv)
            ZE = blend(Z[E], auth[E], wv)
            th = fit_thr(sig(ZF), Y[F], seed0, args.tbag, args.smooth)
            pred = (sig(ZE) >= th[None, :]).astype(np.int8)
            f = f1pl(pred, Y[E])
            res[(kind, w)].append(f.mean())
            perlab[(kind, w)].append(f)
            dens[(kind, w)].append(pred.sum() / len(E))

    base = np.array(res[("wglobal", 0.0)])
    print("=" * 92)
    print("AUTHOR POOLING UNDER tboth -- held-out macro-F1")
    print("=" * 92)
    print(f"{'variant':<12}{'w':>6}{'macro':>9}{'delta':>10}{'95% CI':>21}"
          f"{'win':>6}{'fac/post':>10}")
    for kind, w in names:
        v = np.array(res[(kind, w)])
        if kind == "wglobal" and w == 0.0:
            print(f"{'tboth':<12}{0.0:>6.2f}{v.mean():>9.4f}{'--':>10}"
                  f"{'-- (control)':>21}{'--':>6}"
                  f"{np.mean(dens[(kind, w)]):>10.2f}")
            continue
        d = v - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{kind:<12}{w:>6.2f}{v.mean():>9.4f}{d.mean():>+10.4f}"
              f"{f'[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]':>21}"
              f"{(d > 0).mean():>6.0%}{np.mean(dens[(kind, w)]):>10.2f}")

    bk = max((k for k in names if k != ("wglobal", 0.0)),
             key=lambda k: np.mean(res[k]))
    d = np.array(res[bk]) - base
    print(f"\n  best {bk[0]} w={bk[1]}: {d.mean():+.4f} macro  "
          f"win {(d > 0).mean():.0%}  composite {0.3*d.mean():+.4f}")
    print("  w=0.00 IS tboth. If it does not land on 0.6476 the harness is "
          "wrong, not the idea.")

    A0 = np.array(perlab[("wglobal", 0.0)]).mean(0)
    A1 = np.array(perlab[bk]).mean(0)
    print(f"\n{'label':<44}{'gold n':>7}{'tboth':>8}{'pooled':>8}{'delta':>9}")
    for j in np.argsort(-(A1 - A0)):
        star = "* " if FACTOR_TAXONOMY[j] in PROTECTIVE else \
               ("T " if j in TJ else "  ")
        print(f"{star}{FACTOR_TAXONOMY[j]:<42}{int(Y[:, j].sum()):>7}"
              f"{A0[j]:>8.3f}{A1[j]:>8.3f}{A1[j]-A0[j]:>+9.3f}")
    print("\n  'T' marks the a-priori trait labels. If the gains concentrate "
          "there the\n  mechanism is real; if they are scattered it is noise.")

    if args.build_test:
        T, tid = [], None
        for a in args.arms:
            p = resolve_test(args.dir, a)
            if p is None:
                raise SystemExit(f"no test dump for {a}")
            z = np.load(p, allow_pickle=True)
            T.append(z["probs"].astype(float))
            tid = [str(x) for x in z["row_id"]]
        Zt = logit(logit_mean(np.stack(T)))

        te = pd.read_csv(args.test, keep_default_na=False)
        te["row_id"] = te["row_id"].astype(str)
        amap = dict(zip(te["row_id"], te["anon_user_id"].astype(str)))
        tauth = np.array([amap[r] for r in tid])
        if len(set(tauth) & set(uauth)):
            raise SystemExit("train/test authors overlap -- pooling would leak")

        ship = pd.read_csv(args.shipped, keep_default_na=False)
        shipM = cells(ship["factors"], len(ship))
        order = {r: i for i, r in enumerate(ship["row_id"].astype(str))}
        perm = [order[r] for r in tid]

        t0 = fit_thr(P, Y, 999, args.tbag, args.smooth)
        repro = int(((sig(Zt) >= t0[None, :]).astype(np.int8)
                     != shipM[perm]).sum())
        print(f"\nPROVENANCE: tboth reconstruction vs {args.shipped} -- "
              f"{repro} of {len(tid)*N} cells differ")
        if repro:
            raise SystemExit(
                "REFUSING to write -- reconstruction does not match the "
                "shipped column, so the new file would mix a rebuild bug with "
                "the intended change.")

        wv = np.zeros(N)
        if args.trait_only:
            wv[TJ] = args.w
        else:
            wv[:] = args.w
        thb = fit_thr(sig(blend(Z, auth, wv)), Y, 999, args.tbag, args.smooth)
        on = (sig(blend(Zt, tauth, wv)) >= thb[None, :]).astype(np.int8)

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
        print(f"w={args.w} trait_only={args.trait_only}: "
              f"{on.sum()/len(tid):.2f} factors/post "
              f"(shipped {shipM.sum()/len(ship):.2f}, gold 2.92)")
        print(f"cells changed {int((on != shipM[perm]).sum())}   "
              f"posts changed {moved}")
        print(f"wrote {args.build_test}")
        print("Only the factors column changes -- S1 must return 0.8061.")


if __name__ == "__main__":
    main()
