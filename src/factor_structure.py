"""
factor_structure.py

Evaluate factor-decode rules that exploit structure a per-label threshold
cannot see, and optionally build a test factor column from one of them.

Each of the 24 factor labels is scored by an independent sigmoid, so the pooled
arm probabilities carry no notion of which labels co-occur, nor of what kind of
post they came from. The two arms below add exactly that. Both are measured
against `tfit`: one probability threshold per label, each chosen to maximise
that label's F1 on held-out predictions, which is the decode in use.

ARM 1 -- RISK-CONDITIONED THRESHOLDS. Factor prevalence is conditional on risk
level -- prior self-harm is commoner in attempt posts, access to suicide means
in behavior posts -- while a single global threshold per label must average
over all four levels. Fitting and application both use the PREDICTED risk label
and never gold, so the held-out measurement and deployment see the same kind of
input.
    trisk   a separate threshold per (label, risk level)   96 numbers
    toff    one global threshold per label plus a per-risk-level logit
            offset shared across labels                    24 + 4

ARM 2 -- COMPETITION AMONG THE PROTECTIVE LABELS. The five protective labels
(social support, coping strategy, psychological capital, sense of
responsibility, meaning in life) describe heavily overlapping content, and an
annotator assigns whichever one fits best. Independent sigmoids cannot express
that choice -- nothing in the loss ever makes the five compete -- so a post
dense in protective content tends to fire several of them at once.
    prot5   for each protective label, a logistic on ALL FIVE protective
            probabilities, so the five can finally trade off  25 numbers
    stack   for each label, a logistic on [own prob, total probability mass of
            the post], the cheapest form of "how loaded is this post"   48

PARAMETERS ARE PAID INSIDE THE MEASUREMENT. Every coefficient is fitted on the
FIT authors of a split and scored on held-out authors, at the same fit/eval
ratio as deployment, so a 96-parameter rule carries its own estimation cost
inside the number it reports. Fitting once on everything and reporting on that
same data would make the extra parameters look free.

HOW TO READ THE TABLE. The reference row is `tfit`, the decode in use. Judge an
arm by its WIN RATE over splits rather than by the mean difference: the splits
are resampled from one fixed set of authors and therefore overlap heavily, so
the standard error of the mean understates the real uncertainty and an interval
clear of zero can sit alongside a rule that loses on nearly half of the splits.

Usage:
  python src/factor_structure.py
  python src/factor_structure.py --build-test artifacts/factors_X_test.csv --use prot5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (FACTOR_TAXONOMY, RISK_LEVELS, normalize_label,  # noqa: E402
                     parse_factors)

N = len(FACTOR_TAXONOMY)
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
# The deployed factor column pools ELEVEN arms (factor_threshold_stabilise.py:
# BAG8 + ltail + cbnsynapi + friendfac). Anything left on the BAG8 default
# measures a smaller pool than the one that ships, so its numbers describe a
# different model.
SHIPPED11 = BAG8 + ["ltail", "cbnsynapi", "friendfac"]
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


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


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


def best_thr(p, y):
    npos = int(y.sum())
    if npos < 2:
        return None
    o = np.argsort(-p, kind="stable")
    ps, ys = p[o], y[o]
    cum = np.cumsum(ys)
    ks = np.searchsorted(-ps, -TGRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    f1 = 2.0 * tp / np.maximum(ks + npos, 1)
    return float(TGRID[int(np.argmax(f1))])


def fit_logistic(X, y):
    from sklearn.linear_model import LogisticRegression
    if y.sum() < 3 or y.sum() == len(y):
        return None
    m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
    m.fit(X, y)
    return m


def fit_all(PF, YF, rF):
    """Every rule's parameters, from the fit part only."""
    out = {}
    thr = np.array([best_thr(PF[:, j], YF[:, j]) or 0.5 for j in range(N)])
    out["tfit"] = ("thr", thr)

    # per (label, risk level), falling back to the global threshold when a cell
    # is too thin to fit -- attempt has ~114 posts in total
    tr = np.tile(thr, (len(RISK_LEVELS), 1))
    for ri, rl in enumerate(RISK_LEVELS):
        m = rF == rl
        if m.sum() < 40:
            continue
        for j in range(N):
            t = best_thr(PF[m, j], YF[m, j])
            if t is not None:
                tr[ri, j] = t
    out["trisk"] = ("thr_risk", tr)

    # one shared logit offset per risk level, thresholds stay global
    off = np.zeros(len(RISK_LEVELS))
    for ri, rl in enumerate(RISK_LEVELS):
        m = rF == rl
        if m.sum() < 40:
            continue
        best, bo = -1.0, 0.0
        for o in np.arange(-1.5, 1.55, 0.1):
            pr = (1.0 / (1.0 + np.exp(-(lg(PF[m]) + o))) >= thr[None, :])
            f = f1pl(pr.astype(np.int8), YF[m]).mean()
            if f > best:
                best, bo = f, float(o)
        off[ri] = bo
    out["toff"] = ("off", (thr, off))

    # the five protective labels get to compete
    models, pthr = {}, {}
    Xp = lg(PF[:, PROT_J])
    for j in PROT_J:
        m = fit_logistic(Xp, YF[:, j])
        if m is None:
            continue
        models[j] = m
        q = m.predict_proba(Xp)[:, 1]
        t = best_thr(q, YF[:, j])
        if t is not None:
            pthr[j] = t
    out["prot5"] = ("prot5", (models, pthr, thr))

    # own prob + how loaded the post is overall
    smodels, sthr = {}, {}
    tot = PF.sum(1)
    for j in range(N):
        X = np.column_stack([lg(PF[:, j]), tot])
        m = fit_logistic(X, YF[:, j])
        if m is None:
            continue
        smodels[j] = m
        q = m.predict_proba(X)[:, 1]
        t = best_thr(q, YF[:, j])
        if t is not None:
            sthr[j] = t
    out["stack"] = ("stack", (smodels, sthr, thr))
    return out


def apply_rule(kind, par, P, r):
    n = len(P)
    if kind == "thr":
        return (P >= par[None, :]).astype(np.int8)
    if kind == "thr_risk":
        on = np.zeros((n, N), dtype=np.int8)
        for ri, rl in enumerate(RISK_LEVELS):
            m = r == rl
            if m.any():
                on[m] = (P[m] >= par[ri][None, :]).astype(np.int8)
        return on
    if kind == "off":
        thr, off = par
        on = np.zeros((n, N), dtype=np.int8)
        for ri, rl in enumerate(RISK_LEVELS):
            m = r == rl
            if m.any():
                q = 1.0 / (1.0 + np.exp(-(lg(P[m]) + off[ri])))
                on[m] = (q >= thr[None, :]).astype(np.int8)
        return on
    if kind == "prot5":
        models, pthr, thr = par
        on = (P >= thr[None, :]).astype(np.int8)
        Xp = lg(P[:, PROT_J])
        for j, m in models.items():
            if j in pthr:
                on[:, j] = (m.predict_proba(Xp)[:, 1] >= pthr[j]).astype(np.int8)
        return on
    models, sthr, thr = par
    on = (P >= thr[None, :]).astype(np.int8)
    tot = P.sum(1)
    for j, m in models.items():
        if j in sthr:
            X = np.column_stack([lg(P[:, j]), tot])
            on[:, j] = (m.predict_proba(X)[:, 1] >= sthr[j]).astype(np.int8)
    return on


def risk_of(path, rows):
    d = pd.read_csv(path, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    d = d.drop_duplicates("row_id").set_index("row_id")
    col = next((c for c in ("pred_risk_level", "risk_level", "cot")
                if c in d.columns), None)
    if col is not None:
        return np.array([normalize_label(d.loc[x, col]) if x in d.index
                         else RISK_LEVELS[0] for x in rows])
    pc = [f"prob_{c}" for c in RISK_LEVELS]
    if all(c in d.columns for c in pc):
        M = d.loc[[x for x in rows], pc].to_numpy(float)
        return np.array([RISK_LEVELS[i] for i in M.argmax(1)])
    raise SystemExit(f"{path}: no risk label column")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="submissions/NIDHI_factfit.csv")
    ap.add_argument("--risk-oof", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=120)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--build-test", default=None)
    ap.add_argument("--use", default="prot5",
                    choices=["trisk", "toff", "prot5", "stack"])
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

    rk = risk_of(args.risk_oof, rid)
    ch = pd.read_csv(args.champion, keep_default_na=False)
    eval_frac = len(ch) / (len(ch) + n)

    print(f"OOF {n} posts / {len(uauth)} authors   {args.splits} partitions, "
          f"eval {eval_frac:.1%}")
    print(f"predicted-risk mix on OOF: "
          f"{dict(pd.Series(rk).value_counts())}\n")

    names = ["tfit", "trisk", "toff", "prot5", "stack"]
    res = {k: [] for k in names}
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
        rules = fit_all(P[F], Y[F], rk[F])
        for k in names:
            pr = apply_rule(*rules[k], P[E], rk[E])
            f = f1pl(pr, Y[E])
            res[k].append(f.mean())
            prot[k].append(f[PROT_J].mean())

    ref = np.array(res["tfit"])
    print("=" * 96)
    print("HELD-OUT macro-F1 -- vs tfit, which is what ships")
    print("=" * 96)
    print(f"{'rule':<8}{'params':>8}{'macro':>9}{'protective':>12}"
          f"{'d vs tfit':>11}{'95% CI':>20}{'win':>6}")
    npar = {"tfit": 24, "trisk": 96, "toff": 28, "prot5": 24 + 25, "stack": 24 + 48}
    for k in names:
        v = np.array(res[k])
        if k == "tfit":
            print(f"{k:<8}{npar[k]:>8}{v.mean():>9.4f}{np.mean(prot[k]):>12.4f}"
                  f"{'--':>11}{'--':>20}{'--':>6}")
            continue
        d = v - ref
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{k:<8}{npar[k]:>8}{v.mean():>9.4f}{np.mean(prot[k]):>12.4f}"
              f"{d.mean():>+11.4f}"
              f"{f'[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]':>20}"
              f"{(d > 0).mean():>6.0%}")

    best = max(names, key=lambda k: np.mean(res[k]))
    db = np.array(res[best]) - ref
    print(f"\n  best: {best}   vs tfit {db.mean():+.4f}  "
          f"win {(db > 0).mean():.0%}  composite {0.3 * db.mean():+.4f}")
    print("  A rule needs BOTH a positive mean AND a win rate above ~70% to be "
          "worth a slot;")
    print("  `platt` had the first without the second and would have lost.")
    if best == "tfit":
        print("\n  -> structure does not help either. Factors are finished "
              "offline; only a")
        print("     better-trained model can move them now.")

    if args.build_test:
        full = fit_all(P, Y, rk)
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
        rt = risk_of(args.champion, tid)

        cur = cells(ch["factors"], len(ch))
        base = apply_rule(*full["tfit"], Pt, rt)
        diff = int((base != cur[[order[r] for r in tid]]).sum())
        print(f"\nPROVENANCE: tfit reconstruction vs the SHIPPED factfit column "
              f"-- {diff} of {nT*N} cells differ")
        if diff > 0.01 * nT * N:
            raise SystemExit("REFUSING to write -- reconstruction mismatch.")

        on = apply_rule(*full[args.use], Pt, rt)
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
        out.to_csv(args.build_test, index=False)
        print(f"rule {args.use}: {on.sum()/nT:.2f} factors/post   "
              f"posts changed {moved}   wrote {args.build_test}")
        print("Only the factors column changes -- S1 must return 0.8061.")


if __name__ == "__main__":
    main()
