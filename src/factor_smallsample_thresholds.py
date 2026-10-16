"""
factor_smallsample_thresholds.py

Fits factor thresholds to expected F1 on a test-sized draw rather than to F1 on
the pooled OOF sample.

WHY THE TWO OBJECTIVES DIFFER. F1 is nonlinear in the counts, so

    argmax_t  F1(t; full OOF)   !=   argmax_t  E[ F1(t; 378-post draw) ]

and the gap grows as the label gets rarer, because the tails behave
differently:

    a label with ~2 gold positives on 378 posts, precision ~0.5
      predict 0  ->  F1 = 0            ALWAYS. a guaranteed zero.
      predict 1  ->  F1 = 1 or 0       coin flip
      predict 2  ->  2*1/(2+1) = 0.67 when one lands, and one lands more often

Predicting nothing on a rare label is a certain zero; predicting a few is a
positive-expectation lottery. Over the full OOF sample that asymmetry averages
away and the point estimate is fine. On a test-sized draw it dominates, and a
third of the taxonomy has under 60 OOF positives. Each label is 1/24 of the
factor macro = 1.25% of composite.

This is not a prevalence or quantile decode: it does not force the fire RATE
toward a training estimate. It changes the OBJECTIVE the threshold is fitted to
-- from "F1 on the sample in hand" to "expected F1 on a draw the size of the
scored one" -- and leaves the fire rate wherever that objective puts it.

WHAT IT DOES
  * resample author-disjoint subsets matching the test geometry (36 authors,
    ~378 posts; the public test set is 36 authors with zero overlap with
    training)
  * for each label, choose the threshold maximising E[F1] over those draws
    instead of F1 on the pooled sample
  * compare against the point-estimate threshold cross-fitted by author, so
    the new objective is fitted on FIT authors and scored on HELD-OUT authors
    exactly like the old one

The prediction is specific and falsifiable: gains should concentrate on
LOW-SUPPORT labels and be ~zero on high-support ones such as `hopelessness`.
A movement on a high-support label means the effect is not the claimed one.

Usage:
  python src/factor_smallsample_thresholds.py
  python src/factor_smallsample_thresholds.py --draws 400
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_structure import (SHIPPED11, best_thr, cells,  # noqa: E402
                              load_oof, logit_mean)
from scoring import FACTOR_TAXONOMY  # noqa: E402

N = len(FACTOR_TAXONOMY)
TGRID = np.round(np.arange(0.02, 0.985, 0.01), 3)


def f1_counts(tp, npred, npos):
    den = npred + npos
    return 0.0 if den == 0 else 2.0 * tp / den


def macro_f1(Pb, Y):
    tp = (Pb & Y).sum(0)
    den = 2 * tp + (Pb & ~Y).sum(0) + (~Pb & Y).sum(0)
    return float(np.mean(np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))))


def expected_f1_thresholds(P, Y, auth, draws, n_auth, rng):
    """Per label, the threshold maximising E[F1] over author-disjoint draws of
    the TEST's geometry -- rather than F1 on the pooled sample."""
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    idxs = []
    for _ in range(draws):
        pick = rng.choice(ua, size=min(n_auth, len(ua)), replace=False)
        idxs.append(np.concatenate([by[a] for a in pick]))
    out = np.empty(N)
    for j in range(N):
        pj, yj = P[:, j], Y[:, j]
        best, bt = -1.0, 0.5
        for t in TGRID:
            pb = pj >= t
            tot = 0.0
            for ix in idxs:
                a, b = pb[ix], yj[ix]
                tot += f1_counts(int((a & b).sum()), int(a.sum()),
                                 int(b.sum()))
            v = tot / len(idxs)
            if v > best:
                best, bt = v, t
        out[j] = bt
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--arms", nargs="+", default=SHIPPED11)
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--k", type=int, default=5)
    a_.add_argument("--draws", type=int, default=200)
    a_.add_argument("--test-authors", type=int, default=36)
    a_.add_argument("--splits", type=int, default=120)
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    P = logit_mean(S)
    n = len(rid)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n).astype(bool)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])
    ua = pd.unique(auth)
    sup = Y.sum(0)
    print(f"OOF {n} posts / {len(ua)} authors   "
          f"test geometry: {args.test_authors} authors, ~378 posts")
    print(f"  labels with <60 positives: {int((sup < 60).sum())} of {N}")

    rng = np.random.default_rng(args.seed)
    print("\n" + "=" * 78)
    print("1. HOW FAR DO THE TWO OBJECTIVES DISAGREE?")
    print("=" * 78)
    t_point = np.array([best_thr(P[:, j], Y[:, j].astype(int)) or 0.5
                        for j in range(N)])
    t_exp = expected_f1_thresholds(P, Y, auth, args.draws,
                                   args.test_authors, rng)
    order = np.argsort(sup)
    print(f"  {'label':<38}{'pos':>5}{'tfit t':>8}{'E[F1] t':>9}"
          f"{'shift':>8}{'fire%':>8}")
    for j in order:
        fr = float((P[:, j] >= t_exp[j]).mean())
        print(f"  {FACTOR_TAXONOMY[j][:37]:<38}{int(sup[j]):>5}"
              f"{t_point[j]:>8.2f}{t_exp[j]:>9.2f}"
              f"{t_exp[j]-t_point[j]:>+8.2f}{100*fr:>8.1f}")
    lo, hi = sup < 60, sup >= 60
    print(f"\n  mean shift, LOW support (<60):  "
          f"{np.mean(t_exp[lo] - t_point[lo]):+.3f}")
    print(f"  mean shift, HIGH support:       "
          f"{np.mean(t_exp[hi] - t_point[hi]):+.3f}")
    print("  The prediction is a NEGATIVE shift concentrated on low support --")
    print("  fire more where a zero is otherwise guaranteed. A shift on")
    print("  high-support labels would mean this is not the claimed effect.")

    print("\n" + "=" * 78)
    print("2. CROSS-FITTED -- both objectives fitted on FIT authors, scored on")
    print("   HELD-OUT authors of the TEST's size")
    print("=" * 78)
    rng2 = np.random.default_rng(args.seed + 1)
    d_all, d_lo, d_hi = [], [], []
    for _ in range(args.splits):
        o = list(ua)
        rng2.shuffle(o)
        k = max(int(len(o) * 0.25), 2)
        ho = set(o[:k])
        hm = np.array([a in ho for a in auth])
        fi, hi_ = np.flatnonzero(~hm), np.flatnonzero(hm)
        if len(fi) < 60 or len(hi_) < 20:
            continue
        tp_ = np.array([best_thr(P[fi, j], Y[fi, j].astype(int)) or 0.5
                        for j in range(N)])
        te_ = expected_f1_thresholds(
            P[fi], Y[fi], auth[fi], max(args.draws // 4, 40),
            args.test_authors, rng2)
        a1 = macro_f1(P[hi_] >= tp_, Y[hi_])
        a2 = macro_f1(P[hi_] >= te_, Y[hi_])
        d_all.append(a2 - a1)
        # per-support-band, to test the prediction rather than just the total
        for band, acc in ((lo, d_lo), (hi, d_hi)):
            t1 = np.where(band, tp_, tp_)
            t2 = np.where(band, te_, tp_)
            acc.append(macro_f1(P[hi_] >= t2, Y[hi_])
                       - macro_f1(P[hi_] >= t1, Y[hi_]))
    for nm, d in (("all labels", d_all), ("low-support only", d_lo),
                  ("high-support only", d_hi)):
        d = np.array(d)
        print(f"  {nm:<20}{d.mean():>+10.4f} macro   "
              f"{0.3*d.mean():>+8.4f} comp   win {np.mean(d > 0):>4.0%}   "
              f"LOSS {np.mean(d < 0):>4.0%}   sd {0.3*d.std():.4f}")
    print("\n  `low-support only` applies the new threshold to rare labels and")
    print("  keeps the point-estimate threshold everywhere else -- if the")
    print("  effect is real that row should carry it, and it is also the")
    print("  narrowest change that can realise it.")


if __name__ == "__main__":
    main()
