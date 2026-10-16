"""
factor_qwen_bag9.py

CLOSE QWEN ON CROSS-FITTED F1, NOT ON AP -- USING ONLY THE FOLDS WE ALREADY HAVE.

Qwen-72B loses to the pool on mean AP on one fold and roughly ties on another.
That looked decisive at first glance. It is not, and this repo already
documents why:

    arm_robustness.py:20 -- "dmnli's own run log reports mean average
    precision against the incumbent's -- it RANKS WORSE -- yet it still
    gains under cross-fitted thresholds."

AP measures solo ranking quality. Ensemble contribution is a different
quantity, and factor_arm_weight.py measured that the two come apart badly
here: `d4` has the WORST solo AP of the eight bag members and removing it
costs macro-F1. So an arm that ranks worse alone can still pay in the pool.

NO NEW GPU IS NEEDED. facqnli writes runs/factors_{TAG}_f{f}/val_probs.npz
(scripts/factors_qwen_nli.sh:63,109). Folds 0 and 3 exist = ~653 posts. This
restricts bag8 to exactly those rows and cross-fits there.

    bag8            the 8 arms on these rows
    bag8+qwen       9 arms, equal weight
    swap d4->qwen   qwen replaces the stable weakest member

REDUCED POWER, STATED UP FRONT. ~653 posts instead of 1635 and ~60 authors
instead of 153, so the confidence intervals will be roughly 1.6x wider than
the full-OOF tests. This CANNOT confirm a small positive; it can only tell us
whether the lane is worth 36 more GPU-hours for the remaining three folds.

  clearly negative, CI excluding zero    -> closed, do not run folds 1/2/4
  positive with a decent win rate        -> run them, then judge on 1635 posts
  flat with a CI spanning zero           -> underpowered, and the AP evidence
                                            already points down. Closed on
                                            cost, not on proof.

Usage (cluster, CPU, ~2 min):
  python src/factor_qwen_bag9.py
  python src/factor_qwen_bag9.py --tag q72
"""

from __future__ import annotations

import argparse
import glob
import os
import re
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


def ap(p, y):
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum()
                 / ys.sum())


def read_npz(path):
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    pk = next((k for k in ("probs", "prob", "p", "y_prob") if k in keys), None)
    rk = next((k for k in ("row_id", "row_ids", "rid", "ids") if k in keys),
              None)
    if pk is None or rk is None:
        raise SystemExit(f"{path}: unexpected keys {keys} "
                         f"(need a probs array and a row_id array)")
    return z[pk].astype(float), [str(x) for x in z[rk]]


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--runs", default="runs")
    a_.add_argument("--tag", default=None,
                    help="Qwen TAG; auto-discovered if omitted")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--tbag", type=int, default=40)
    a_.add_argument("--smooth", type=int, default=9)
    a_.add_argument("--seed", type=int, default=13)
    a_.add_argument("--swap-out", default="d4")
    a_.add_argument("--decode", default="tfit", choices=["tfit", "tboth"],
                    help="tfit is the CHAMPION's decode. tboth is a "
                         "different decode configuration that is not "
                         "shipped, so an arm judged under tboth is judged "
                         "against a system that is not deployed.")
    args = a_.parse_args()

    pats = sorted(glob.glob(f"{args.runs}/factors_*_f*/val_probs.npz"))
    if not pats:
        raise SystemExit(f"no {args.runs}/factors_*_f*/val_probs.npz found")
    found = {}
    for p in pats:
        m = re.search(r"factors_(.+)_f(\d+)[/\\]val_probs\.npz$",
                      p.replace("\\", "/"))
        if m:
            found.setdefault(m.group(1), {})[int(m.group(2))] = p
    print("Qwen-style dumps discovered:")
    for t, d in sorted(found.items()):
        print(f"   tag {t:<12} folds {sorted(d)}")
    tag = args.tag or max(found, key=lambda t: len(found[t]))
    if tag not in found:
        raise SystemExit(f"tag {tag!r} not among {sorted(found)}")
    folds = sorted(found[tag])
    print(f"\nusing tag {tag!r}, folds {folds}\n")

    QP, qrid = [], []
    for f in folds:
        pr, ids = read_npz(found[tag][f])
        QP.append(pr)
        qrid.extend(ids)
    QP = np.concatenate(QP)
    if QP.shape[1] != N:
        raise SystemExit(f"qwen probs have {QP.shape[1]} columns, expected {N}")

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rid = [r for r in qrid if r in lab.index]
    keepq = [i for i, r in enumerate(qrid) if r in lab.index]
    QP = QP[keepq]
    n = len(rid)

    Z8 = []
    for t in BAG8:
        pr, ids = [], []
        for f in range(5):
            p = f"{args.dir}/{t}_cv_oof_f{f}.npz"
            if not os.path.exists(p):
                continue
            a, b = read_npz(p)
            pr.append(a)
            ids.extend(b)
        pr = np.concatenate(pr)
        o = {x: i for i, x in enumerate(ids)}
        miss = [r for r in rid if r not in o]
        if miss:
            raise SystemExit(f"{t} missing {len(miss)} of the qwen rows")
        Z8.append(logit(pr[[o[r] for r in rid]]))
    Z8 = np.stack(Z8)
    ZQ = logit(QP)

    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray([str(lab.loc[r, "anon_user_id"]) for r in rid])
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    keep = [j for j in range(N) if Y[:, j].sum() > 0]

    pool8 = sig(Z8.mean(0))
    print(f"{n} posts / {len(ua)} authors, {len(keep)} labels with gold")
    print(f"  mean AP  POOL8 {np.nanmean([ap(pool8[:, j], Y[:, j]) for j in keep]):.4f}"
          f"   qwen {np.nanmean([ap(QP[:, j], Y[:, j]) for j in keep]):.4f}")
    rho = float(np.mean([np.corrcoef(ZQ[:, j], Z8.mean(0)[:, j])[0, 1]
                         for j in keep]))
    print(f"  rho(qwen, pool8) = {rho:.3f}   "
          f"(the 20 arms tested today spanned 0.725-0.958)\n")

    swi = BAG8.index(args.swap_out)
    Z9 = np.concatenate([Z8, ZQ[None]])
    Zs = Z8.copy()
    Zs[swi] = ZQ
    variants = {"bag8": Z8, "bag8+qwen": Z9,
                f"swap {args.swap_out}->qwen": Zs}

    frac = 378.0 / (378.0 + 1635.0)
    res = {k: [] for k in variants}
    rng = np.random.default_rng(args.seed)
    brng = np.random.default_rng(args.seed + 1)
    for _ in range(args.splits):
        order = list(ua)
        rng.shuffle(order)
        ev, got = [], 0
        for a in order:
            if got >= frac * n:
                break
            ev.append(a)
            got += len(by[a])
        E = np.concatenate([by[a] for a in ev])
        F = np.setdiff1d(np.arange(n), E)
        s0 = int(brng.integers(1 << 31))
        for k, Z in variants.items():
            PF, PE = sig(Z[:, F].mean(0)), sig(Z[:, E].mean(0))
            if args.decode == "tfit":
                # tfit is what the CHAMPION ships. tboth is a different,
                # undeployed decode configuration, so judging a 9th arm
                # under tboth would describe a system that is not run.
                th = np.array([
                    float(TGRID[int(np.argmax(f1_curve(PF[:, j], Y[F, j])))])
                    for j in range(N)])
            else:
                th = np.array([
                    thr_bag(PF[:, j], Y[F, j], np.random.default_rng(s0 + j),
                            args.tbag, args.smooth) for j in range(N)])
            res[k].append(f1pl((PE >= th[None, :]).astype(np.int8),
                               Y[E]).mean())

    base = np.array(res["bag8"])
    print("=" * 82)
    print(f"CROSS-FITTED MACRO-F1 on {n} posts, {args.splits} author "
          f"partitions")
    print("=" * 82)
    print(f"{'variant':<22}{'macro':>9}{'delta':>10}{'95% CI':>21}{'win':>6}")
    for k in variants:
        v = np.array(res[k])
        if k == "bag8":
            print(f"{k:<22}{v.mean():>9.4f}{'--':>10}{'-- (control)':>21}"
                  f"{'--':>6}")
            continue
        d = v - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{k:<22}{v.mean():>9.4f}{d.mean():>+10.4f}"
              f"{f'[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]':>21}"
              f"{(d > 0).mean():>6.0%}")

    print(f"\n  Power is REDUCED -- {n} posts and {len(ua)} authors against")
    print("  1635/153 in the full tests, so CIs are ~1.6x wider. A flat result")
    print("  here does not prove absence; it means the lane is not worth 36")
    print("  more GPU-hours given the AP evidence already points down.")
    print("  A clearly NEGATIVE result with a CI excluding zero DOES close it.")


if __name__ == "__main__":
    main()
