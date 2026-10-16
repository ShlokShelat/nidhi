"""
factor_threshold_shrink.py

tfit FITS 24 INDEPENDENT THRESHOLDS. ON A LABEL WITH 8 POSITIVES THAT IS NOISE.

`factor_label_anatomy.py` measured, cross-fitted by author, that the gap
between tfit's threshold and the best possible threshold tracks how much the
threshold WANDERS across splits, which tracks support:

    label                        sup    thr        threshold gap
    sexual orientation             8    0.54+-0.18      0.121
    exposure to others' suicide   14    0.44+-0.22      0.183
    poor school performance       16    0.83+-0.24      0.103
    substance use                 33    0.70+-0.17      0.120
    ...
    hopelessness                 745    0.83+-0.03      0.016

A threshold with sd 0.22 is not a property of the label; it is a property of
whichever authors landed in the fit half. Total oracle headroom is
substantial, and it is concentrated in the low-support labels.

THE FIX IS THE STANDARD ONE FOR A NOISY PER-GROUP ESTIMATE: shrink each label's
threshold toward a pooled value, by how little evidence it has.

    t_j' = w_j * t_j + (1 - w_j) * t_pool,      w_j = n_j / (n_j + k)

`hopelessness` (745 positives) keeps its own threshold; `sexual orientation`
(8) is pulled almost entirely to the pool. ONE parameter, k, and the mechanism
is decided in advance rather than discovered in the data -- which is the
distinction that separated `route_attempt` from every rule that lost
([[rules-shelf-closed-provenance]]).

FOUR VARIANTS, all cross-fitted -- thresholds AND k chosen on FIT authors,
scored on HELD-OUT authors:

  tfit        the incumbent, 24 free thresholds
  shrink      the formula above, k swept on the fit half only
  floor       tfit, except labels with fewer than m fit positives take the
              pooled threshold outright. The crude version; if it matches
              `shrink`, prefer it -- it has no continuous parameter.
  prevalence  threshold set so the fire rate equals the label's FIT prevalence.
              Zero threshold parameters at all. [[factor-decode-calibration-bug]]
              measured a gain from a shrunk-prevalence decode, and this is the
              per-label version of that idea.

HONEST EXPECTATION. The oracle headroom above is an upper bound that itself exploits
noise -- choosing the best threshold ON the held-out data flatters rare labels
most. Expect to recover a fraction. Judge by WIN RATE over splits and by the
per-label breakdown, never by the mean alone ([[ci-of-the-mean-error]]).

Usage:
  python src/factor_threshold_shrink.py
  python src/factor_threshold_shrink.py --splits 300
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_structure import (BAG8, best_thr, cells,  # noqa: E402
                              load_oof, logit_mean)
from scoring import FACTOR_TAXONOMY  # noqa: E402

N = len(FACTOR_TAXONOMY)
KGRID = [0, 5, 10, 20, 40, 80, 160, 320]
MGRID = [0, 10, 20, 40, 80]


def macro_f1(Pb, Y):
    tp = (Pb & Y).sum(0)
    den = 2 * tp + (Pb & ~Y).sum(0) + (~Pb & Y).sum(0)
    return float(np.mean(np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--arms", nargs="+", default=BAG8)
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--k", type=int, default=5)
    a_.add_argument("--splits", type=int, default=200)
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
    print(f"OOF {n} posts / {len(ua)} authors, {N} labels, "
          f"{args.splits} author-disjoint splits\n")

    def thr_vec(idx):
        return np.array([best_thr(P[idx, j], Y[idx, j].astype(int))
                         or 0.5 for j in range(N)])

    def prev_thr(fi, hi):
        """Threshold placed so the HELD-OUT fire rate matches FIT prevalence.
        Uses only the fit labels and the held-out probability RANKING, never a
        held-out label -- so it is honest despite touching held-out scores."""
        t = np.empty(N)
        for j in range(N):
            rate = float(Y[fi, j].mean())
            k = max(int(round(rate * len(hi))), 0)
            s = np.sort(P[hi, j])[::-1]
            t[j] = 1.1 if k == 0 else (s[k - 1] if k <= len(s) else 0.0)
        return t

    rng = np.random.default_rng(args.seed)
    names = ["tfit", "shrink", "floor", "prevalence"]
    per = {k: [] for k in names}
    chosen = {"k": [], "m": []}
    perlab = {k: np.zeros((0, N)) for k in names}
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        kk = max(int(len(o) * 0.2), 2)
        ho = set(o[:kk])
        hm = np.array([a in ho for a in auth])
        fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
        if len(fi) < 50 or len(hi) < 20:
            continue
        # split the FIT half again, so k and m are chosen without ever
        # touching held-out data
        io = list(pd.unique(auth[fi]))
        rng.shuffle(io)
        inner = set(io[:max(int(len(io) * 0.25), 2)])
        im = np.array([a in inner for a in auth[fi]])
        a_i, b_i = fi[~im], fi[im]
        t_a = thr_vec(a_i)
        npos_a = Y[a_i].sum(0).astype(float)
        pool_a = float(np.median(t_a))

        def sc(t, idx):
            return macro_f1(P[idx] >= t, Y[idx])

        bk = max(KGRID, key=lambda k_: sc(
            (npos_a / (npos_a + k_)) * t_a
            + (1 - npos_a / (npos_a + k_)) * pool_a, b_i))
        bm = max(MGRID, key=lambda m_: sc(
            np.where(npos_a < m_, pool_a, t_a), b_i))
        chosen["k"].append(bk)
        chosen["m"].append(bm)

        t_f = thr_vec(fi)
        npos = Y[fi].sum(0).astype(float)
        pool = float(np.median(t_f))
        w = npos / (npos + bk)
        cand = {
            "tfit": t_f,
            "shrink": w * t_f + (1 - w) * pool,
            "floor": np.where(npos < bm, pool, t_f),
            "prevalence": prev_thr(fi, hi),
        }
        for k_, t in cand.items():
            Pb = P[hi] >= t
            per[k_].append(macro_f1(Pb, Y[hi]))
            tp = (Pb & Y[hi]).sum(0)
            den = 2 * tp + (Pb & ~Y[hi]).sum(0) + (~Pb & Y[hi]).sum(0)
            f = np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))
            perlab[k_] = np.vstack([perlab[k_], f])

    base = np.array(per["tfit"])
    print(f"  chosen k {np.mean(chosen['k']):.0f}+-{np.std(chosen['k']):.0f}"
          f"    chosen m {np.mean(chosen['m']):.0f}+-"
          f"{np.std(chosen['m']):.0f}   (inner-split, never held-out)\n")
    print(f"  {'decode':<14}{'macro':>9}{'d macro':>10}{'d comp':>10}"
          f"{'win':>6}{'LOSS':>6}{'sd':>9}")
    for k_ in names:
        v = np.array(per[k_])
        d = v - base
        print(f"  {k_:<14}{v.mean():>9.4f}{d.mean():>+10.4f}"
              f"{0.3*d.mean():>+10.4f}{np.mean(d > 0):>5.0%}"
              f"{np.mean(d < 0):>6.0%}{0.3*d.std():>9.4f}")

    best = max((k_ for k_ in names if k_ != "tfit"),
               key=lambda k_: np.mean(per[k_]))
    print(f"\n  PER-LABEL, {best} vs tfit -- where the change actually lands")
    print(f"  {'label':<38}{'sup':>5}{'tfit':>8}{best:>9}{'delta':>9}")
    d = perlab[best].mean(0) - perlab["tfit"].mean(0)
    order = np.argsort(-np.abs(d))
    for j in order[:10]:
        print(f"  {FACTOR_TAXONOMY[j][:37]:<38}{int(Y[:, j].sum()):>5}"
              f"{perlab['tfit'].mean(0)[j]:>8.3f}"
              f"{perlab[best].mean(0)[j]:>9.3f}{d[j]:>+9.3f}")
    print("\n  If the gains sit on the LOW-SUPPORT labels the mechanism is")
    print("  confirmed: those thresholds were noise and pooling fixed them.")
    print("  If a high-support label LOSES badly, the pool is dragging a")
    print("  well-estimated threshold and `floor` should beat `shrink`.")
    print("  A stable chosen k across splits means k is a property of the")
    print("  problem; a wandering k means it is being fitted per sample and")
    print("  the gain will not transfer.")


if __name__ == "__main__":
    main()
