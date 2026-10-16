"""
factor_threshold_refine.py

PUSH THE LANE THAT IS ACTUALLY PAYING.

Established: a THRESHOLD decode beats a QUANTILE decode out-of-sample.

    shrunk (quant, prevalence)   beats champ
    qfit   (quant, fitted rates) beats champ, by less  -- fitting RATES hurts
    tfit   (THRESH, fitted)      the largest gain       -- fitting THRESHOLDS helps
    thalf  (THRESH, F1*/2)       close behind tfit

`tfit` recovers roughly half of the in-sample ceiling, so the rest of the
decode gap is still on the table. This tests four ways to close more of it.

THE OBVIOUS ONE FIRST. `shrunk` beat `gold` because shrinking a noisy estimate
toward a stable prior wins. Nobody had done that to the THRESHOLDS. `tfit` fits
24 of them independently and `sexual orientation` has 8 positives to fit from.

  tshr    shrink each threshold halfway to the support-weighted global median
  tsupp   shrink by 1/sqrt(support) -- rare labels, whose thresholds are the
          noisiest, get pulled hardest. Principled rather than a flat knob.
  ecount  a genuinely different decode: E[count] = sum_j p_j, so predict
          n_p = round(sum_j p_j) factors for post p and take its top n_p.
          This is the only rule here that decides PER POST rather than per
          label -- gold runs 0:128, 1:429, 2:347, 3:233 ... and nothing we
          ship models that distribution at all. Probabilities are normalised
          by each label's fit-set mean first, or a systematically high-scoring
          label would win every post.
  platt   per-label Platt scaling, then ONE global threshold. If calibration
          makes labels comparable, 24 thresholds collapse into 1.

EVERYTHING IS COMPARED AGAINST tfit, which is what ships tomorrow. Beating
`champ` is not the bar any more.

Every rule is estimated on FIT authors and scored on ~378 held-out posts, the
real 1635/378 ratio, so any fitted quantity pays its own estimation cost inside
the measurement.

Usage:
  python src/factor_threshold_refine.py
  python src/factor_threshold_refine.py --build-test artifacts/factors_X_test.csv --use tsupp
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


def resolve_test(d, a):
    for c in (f"{d}/{a}_cv_test.npz", f"{d}/{a}_test.npz",
              f"{d}/nli_ensemble_{a}_test.npz"):
        if os.path.exists(c):
            return c
    return None


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


def best_threshold(p, y):
    npos = int(y.sum())
    if npos == 0:
        return 0.5
    o = np.argsort(-p, kind="stable")
    ps, ys = p[o], y[o]
    cum = np.cumsum(ys)
    ks = np.searchsorted(-ps, -TGRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    f1 = 2.0 * tp / np.maximum(ks + npos, 1)
    return float(TGRID[int(np.argmax(f1))])


def platt_fit(p, y):
    """1-D logistic on the logit. Returns (a, b) for sigmoid(a*z+b)."""
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    if y.sum() == 0 or y.sum() == len(y):
        return 1.0, 0.0
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
    lr.fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def platt_apply(p, ab):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1.0 / (1.0 + np.exp(-(ab[:, 0][None, :] * z + ab[:, 1][None, :])))


def ecount_pred(P, mu, scale):
    """Per-post: n = round(scale * sum_j p_j), take the top n by p/mu."""
    n = len(P)
    tot = np.clip(np.round(scale * P.sum(1)).astype(int), 0, N)
    R = P / np.maximum(mu, 1e-6)[None, :]
    on = np.zeros((n, N), dtype=np.int8)
    for i in range(n):
        k = tot[i]
        if k <= 0:
            continue
        on[i, np.argpartition(-R[i], k - 1)[:k]] = 1
    return on


def fit_rules(PF, YF):
    th = np.array([best_threshold(PF[:, j], YF[:, j]) for j in range(N)])
    sup = YF.sum(0)
    gmed = float(np.average(th, weights=np.maximum(sup, 1)))
    # 1/sqrt(support): a label with 8 positives is pulled far harder than one
    # with 745, which is where the threshold noise actually lives
    lam = np.clip(1.0 / np.sqrt(np.maximum(sup, 1)) * 4.0, 0.0, 0.9)
    ab = np.array([platt_fit(PF[:, j], YF[:, j]) for j in range(N)])
    PC = platt_apply(PF, ab)
    gth = max(TGRID, key=lambda t: f1pl((PC >= t).astype(np.int8), YF).mean())
    mu = PF.mean(0)
    scale = YF.sum(1).mean() / max(PF.sum(1).mean(), 1e-6)
    return {
        "tfit": ("t", th),
        "tshr": ("t", 0.5 * th + 0.5 * gmed),
        "tsupp": ("t", (1 - lam) * th + lam * gmed),
        "ecount": ("e", (mu, scale)),
        "platt": ("p", (ab, gth)),
    }


def apply_rule(kind, par, P):
    if kind == "t":
        return (P >= par[None, :]).astype(np.int8)
    if kind == "e":
        return ecount_pred(P, par[0], par[1])
    return (platt_apply(P, par[0]) >= par[1]).astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=150)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--build-test", default=None)
    ap.add_argument("--use", default="tsupp",
                    choices=["tfit", "tshr", "tsupp", "ecount", "platt"])
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

    ch = pd.read_csv(args.champion, keep_default_na=False)
    champ = cells(ch["factors"], len(ch))
    eval_frac = len(ch) / (len(ch) + n)

    print(f"OOF {n} posts / {len(uauth)} authors   {args.splits} partitions, "
          f"eval {eval_frac:.1%}")
    print("all rules estimated on FIT, scored on HELD-OUT, compared to tfit\n")

    names = ["tfit", "tshr", "tsupp", "ecount", "platt"]
    res = {k: [] for k in names}
    dens = {k: [] for k in names}
    zero = {k: [] for k in names}
    prot = {k: [] for k in names}
    rng = np.random.default_rng(args.seed)
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
        rules = fit_rules(P[F], Y[F])
        for k in names:
            pr = apply_rule(*rules[k], P[E])
            f = f1pl(pr, Y[E])
            res[k].append(f.mean())
            prot[k].append(f[PROT_J].mean())
            dens[k].append(pr.sum() / len(E))
            zero[k].append((pr.sum(1) == 0).mean())

    ref = np.array(res["tfit"])
    print("=" * 100)
    print("HELD-OUT macro-F1 -- vs tfit, which ships tomorrow")
    print("=" * 100)
    print(f"{'rule':<8}{'macro':>9}{'protective':>12}{'d vs tfit':>11}"
          f"{'95% CI':>20}{'win':>6}{'fac/post':>10}{'0-fac':>7}")
    for k in names:
        v = np.array(res[k])
        if k == "tfit":
            print(f"{k:<8}{v.mean():>9.4f}{np.mean(prot[k]):>12.4f}"
                  f"{'--':>11}{'--':>20}{'--':>6}"
                  f"{np.mean(dens[k]):>10.2f}{np.mean(zero[k]):>7.1%}")
            continue
        d = v - ref
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{k:<8}{v.mean():>9.4f}{np.mean(prot[k]):>12.4f}"
              f"{d.mean():>+11.4f}"
              f"{f'[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]':>20}"
              f"{(d > 0).mean():>6.0%}"
              f"{np.mean(dens[k]):>10.2f}{np.mean(zero[k]):>7.1%}")
    print("\n  composite = 0.3 x the macro delta.  gold density 2.92 "
          "factors/post, gold 0-fac 7.8%")

    best = max(names, key=lambda k: np.mean(res[k]))
    db = np.array(res[best]) - ref
    print(f"\n  best: {best}   vs tfit {db.mean():+.4f}  "
          f"win {(db > 0).mean():.0%}  composite {0.3 * db.mean():+.4f}")
    if best == "tfit":
        print("  -> nothing beats plain fitted thresholds. The decode lane is "
              "finished; ship tfit")
        print("     and put the remaining effort into better rankings.")

    if args.build_test:
        full = fit_rules(P, Y)
        T, tid = [], None
        for a in args.arms:
            p = resolve_test(args.dir, a)
            if p is None:
                raise SystemExit(f"no test dump for {a}")
            z = np.load(p, allow_pickle=True)
            T.append(z["probs"].astype(float))
            tid = [str(x) for x in z["row_id"]]
        Pt = logit_mean(np.stack(T))
        nT = len(tid)
        order = {r: i for i, r in enumerate(ch["row_id"].astype(str))}
        perm = [order[r] for r in tid]

        cr = champ.sum(0) / len(ch)
        rep = np.zeros((nT, N), dtype=np.int8)
        for j in range(N):
            kk = min(max(int(round(cr[j] * nT)), 1), nT - 1)
            rep[:, j] = (Pt[:, j] >= np.sort(Pt[:, j])[-kk]).astype(np.int8)
        diff = int((rep != champ[perm]).sum())
        print(f"\nPROVENANCE: champ-rate reconstruction -- {diff} of {nT*N} "
              f"cells differ")
        if diff:
            raise SystemExit("REFUSING to write -- reconstruction mismatch.")

        on = apply_rule(*full[args.use], Pt)
        out = ch.copy()
        newcol = list(out["factors"])
        moved = 0
        for i, row in enumerate(tid):
            labs = str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
            kk = order[row]
            if str(newcol[kk]) != labs:
                moved += 1
            newcol[kk] = labs
        out["factors"] = newcol
        os.makedirs(os.path.dirname(args.build_test) or ".", exist_ok=True)
        out.to_csv(args.build_test, index=False)
        print(f"rule {args.use}: {on.sum()/nT:.2f} factors/post   "
              f"cells changed {int((on != champ[perm]).sum())}   "
              f"posts {moved}   0-fac {int((on.sum(1) == 0).sum())}/{nT}")
        print(f"wrote {args.build_test}")
        print("Only the factors column changes -- S1 must return 0.8061.")


if __name__ == "__main__":
    main()
