"""
factor_incumbent_id.py

WHAT IS "THE INCUMBENT" IN THE QWEN-72B LOG?

The Qwen-72B training log reports it beating an incumbent's mean AP on fold 3
by an unusually large margin. The size of that claim depends entirely on what
the incumbent IS, and arm_robustness.py defines it as a single file:

    arms = {"INCUMBENT": "artifacts/fac_oof_f{f}.npz"}

which is NOT one of the artifacts/factor_runs/*_cv_oof_f*.npz bag8 arms. So it
is one of two things and they mean opposite things:

  a single DeBERTa arm   Qwen-72B beating one arm by that margin is
                         unremarkable. Eight arms pooled already beat any one
                         of them, so the real bar is far higher and this
                         proves little.

  the pooled bag8 output A SOLO model beating the entire 8-arm ensemble by
                         that margin on the same rows would be the strongest
                         factor result measured so far, and it would mean
                         equal-weight membership as a 9th arm is the wrong
                         way to use it.

This prints, on the SAME fold-3 rows and the SAME labels, the mean AP of every
bag8 arm alone, the logit-mean pool of all eight, and fac_oof itself. Whichever
one fac_oof matches is what the incumbent is. No inference required.

Usage (cluster, CPU, seconds):
  python src/factor_incumbent_id.py --fold 3
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


def cells(col, n):
    M = np.zeros((n, N), dtype=np.int8)
    for i, c in enumerate(col):
        for x in set(parse_factors(str(c))):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


def ap(p, y):
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum()
                 / ys.sum())


def logit_mean(S):
    X = np.clip(np.asarray(S, float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--dir", default="artifacts/factor_runs")
    a.add_argument("--inc", default="artifacts/fac_oof_f{f}.npz")
    a.add_argument("--arms", nargs="+", default=BAG8)
    a.add_argument("--qwen", default="q72nli",
                   help="tag of the Qwen-72B arm if its dump exists")
    a.add_argument("--fold", type=int, default=3)
    a.add_argument("--labeled", default="artifacts/full_train.csv")
    args = a.parse_args()

    f = args.fold
    ip = args.inc.format(f=f)
    if not os.path.exists(ip):
        raise SystemExit(f"missing {ip}")
    z = np.load(ip, allow_pickle=True)
    inc = z["probs"].astype(float)
    rid = [str(x) for x in z["row_id"]]
    n = len(rid)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    keep = [j for j in range(N) if Y[:, j].sum() > 0]
    print(f"fold {f}: {n} posts, {len(keep)} of {N} labels have gold here "
          f"(the rest score 0 for everyone and are dropped)\n")

    got = {}
    for t in args.arms + [args.qwen]:
        p = f"{args.dir}/{t}_cv_oof_f{f}.npz"
        if not os.path.exists(p):
            if t == args.qwen:
                continue
            raise SystemExit(f"missing {p}")
        zz = np.load(p, allow_pickle=True)
        pr = zz["probs"].astype(float)
        ids = [str(x) for x in zz["row_id"]]
        if ids != rid:
            o = {x: i for i, x in enumerate(ids)}
            miss = [r for r in rid if r not in o]
            if miss:
                raise SystemExit(f"{t} is missing {len(miss)} of fold {f}")
            pr = pr[[o[r] for r in rid]]
        got[t] = pr

    def m(P):
        return float(np.nanmean([ap(P[:, j], Y[:, j]) for j in keep]))

    pool = logit_mean(np.stack([got[t] for t in args.arms]))
    rows = [(t, m(got[t])) for t in args.arms]
    rows.sort(key=lambda r: -r[1])

    print("=" * 66)
    print(f"MEAN AP ON THE SAME FOLD-{f} ROWS AND LABELS")
    print("=" * 66)
    for t, v in rows:
        print(f"  {t:<22}{v:>9.4f}")
    print(f"  {'-'*22}{'-'*9}")
    pv = m(pool)
    print(f"  {'POOL of 8 (logit)':<22}{pv:>9.4f}")
    iv = m(inc)
    print(f"  {'fac_oof (INCUMBENT)':<22}{iv:>9.4f}   <- what the log calls "
          f"the incumbent")
    if args.qwen in got:
        print(f"  {'Qwen-72B':<22}{m(got[args.qwen]):>9.4f}")

    best_arm = rows[0]
    da, dp = abs(iv - best_arm[1]), abs(iv - pv)
    print()
    if dp < da:
        print(f"  fac_oof matches the POOL ({pv:.4f}, gap {dp:.4f}) more "
              f"closely than the best\n  single arm {best_arm[0]} "
              f"({best_arm[1]:.4f}, gap {da:.4f}).")
        print("  => the incumbent IS the 8-arm ensemble. A solo Qwen-72B "
              "beating it is a\n     major result: equal-weight 9th-arm "
              "membership would waste it, and the\n     right test is a "
              "WEIGHT SWEEP of Qwen against the pool.")
    else:
        print(f"  fac_oof matches a SINGLE ARM ({best_arm[0]} "
              f"{best_arm[1]:.4f}, gap {da:.4f}) more\n  closely than the "
              f"pool ({pv:.4f}, gap {dp:.4f}).")
        print(f"  => the headline gain is measured against ONE arm, not "
              f"the ensemble.\n     The real bar is the pool at {pv:.4f}, and "
              f"Qwen must clear THAT.")

    # PER-LABEL pool AP -- paste the job log's "AP qwen" column against this
    # to see whether Qwen beats THE POOL anywhere, rather than beating the
    # legacy fac_oof baseline that the log actually compares against.
    print("\n" + "=" * 78)
    print(f"PER-LABEL, FOLD {f} -- compare the POOL column against the job "
          f"log's 'AP qwen'")
    print("=" * 78)
    print(f"{'factor':<44}{'n':>5}{'POOL':>9}{'inc':>9}{'pool-inc':>10}")
    for j in keep:
        print(f"{FACTOR_TAXONOMY[j]:<44}{int(Y[:, j].sum()):>5}"
              f"{ap(pool[:, j], Y[:, j]):>9.3f}{ap(inc[:, j], Y[:, j]):>9.3f}"
              f"{ap(pool[:, j], Y[:, j]) - ap(inc[:, j], Y[:, j]):>+10.3f}")
    print("\n  The log's per-label deltas are vs 'inc'. A label where the POOL")
    print("  also beats inc by as much as Qwen does is NOT a routing target --")
    print("  Qwen only earns a label by beating the POOL column above.")

    # the headline is dominated by labels with almost no gold. Split it.
    print("\n" + "=" * 66)
    print("HOW MUCH OF ANY GAP IS CARRIED BY LABELS WITH ALMOST NO GOLD")
    print("=" * 66)
    for cut in (20, 10):
        big = [j for j in keep if Y[:, j].sum() >= cut]
        sml = [j for j in keep if Y[:, j].sum() < cut]

        def mm(P, js):
            return float(np.nanmean([ap(P[:, j], Y[:, j]) for j in js]))
        print(f"  n>={cut:<3} ({len(big):>2} labels)   pool {mm(pool, big):.4f}"
              f"   inc {mm(inc, big):.4f}" +
              (f"   qwen {mm(got[args.qwen], big):.4f}"
               if args.qwen in got else ""))
        print(f"  n< {cut:<3} ({len(sml):>2} labels)   pool {mm(pool, sml):.4f}"
              f"   inc {mm(inc, sml):.4f}" +
              (f"   qwen {mm(got[args.qwen], sml):.4f}"
               if args.qwen in got else ""))
    print("\n  AP on 2-4 positives moves by whole tenths when one post shifts.")
    print("  A headline carried by the n<10 rows is not yet a result.")


if __name__ == "__main__":
    main()
