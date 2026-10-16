"""
factor_label_anatomy.py

WHICH OF THE 24 LABELS IS THE MACRO-F1 ACTUALLY LOSING, AND IS IT THE
THRESHOLD OR THE RANKING?

Factors carry 0.3 and score 0.6774 against risk's 0.8047, so this is where the
points are. But `factors_macro_f1` averages over LABELS, not posts: a label
with 2 gold positives counts exactly as much as `hopelessness` with 180. A rare
label that never fires scores 0 and costs **1/24 = 4.2% of the factor score by
itself** -- four such labels are worth more than the entire risk gap.

`factor_regime_gap.py` just closed the OOF/test shrinkage lane (median SD ratio
0.962), so the decode is applying its thresholds to the distribution it was
fitted on. That rules out the [[factor-decode-calibration-bug]] mechanism as a
GLOBAL explanation -- but it also showed `poor social support` at SD ratio 0.84
firing 7.3pp less on test, which a median hides and a macro metric does not.

THE DECOMPOSITION THAT MATTERS, per label, all cross-fitted by author:

  tfit F1        what ships (thresholds fitted on FIT authors, scored on
                 HELD-OUT authors, so honest)
  oracle-thr F1  the best threshold chosen ON THE HELD-OUT DATA. Cheating, and
                 that is the point: it is the ceiling reachable WITHOUT a
                 better model.
  AP             average precision -- ranking quality, threshold-free.

Read the two gaps:
  tfit -> oracle-thr   THRESHOLD headroom. Free: no new model, just a better
                       calibration. If this is large the label is mis-decoded.
  oracle-thr -> 1.0    RANKING headroom. Needs a better model; no threshold
                       can reach it.

This is the same split that closed risk: [[risk-architecture-audit]] found risk
errors were confidently wrong, so decoding could not fix them, and every decode
experiment afterwards failed exactly as predicted. Doing that test per FACTOR
label, before building anything, is the cheap version of the lesson.

Output is sorted by MACRO HEADROOM -- (1 - F1)/24 -- so the top row is the
single label whose repair buys the most, not the label with the worst F1.

Usage:
  python src/factor_label_anatomy.py
  python src/factor_label_anatomy.py --splits 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_structure import (BAG8, cells, best_thr,  # noqa: E402
                              load_oof, logit_mean)
from scoring import FACTOR_TAXONOMY  # noqa: E402

N = len(FACTOR_TAXONOMY)
# The five protective factors, from the organisers' taxonomy. Marked `*` in
# factor_regime_gap.py's output. [[protective-factor-gap]]: these score 0.504
# against the risk factors' 0.651.
PROTECTIVE = {"social support", "coping strategy", "psychological capital",
              "sense of responsibility", "meaning in life"}


def ap_score(p, y):
    """Average precision -- ranking quality, independent of any threshold."""
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    cum = np.cumsum(ys)
    prec = cum / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--arms", nargs="+", default=BAG8)
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--k", type=int, default=5)
    a_.add_argument("--splits", type=int, default=120)
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    P = logit_mean(S)
    n = len(rid)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    auth = np.asarray(lab.loc[rid, "anon_user_id"])
    ua = pd.unique(auth)
    print(f"OOF {n} posts / {len(ua)} authors, {N} labels, "
          f"{args.splits} author-disjoint splits\n")

    # cross-fitted: threshold from FIT authors, metrics on HELD-OUT authors
    acc = {k: [[] for _ in range(N)] for k in
           ("tp", "fp", "fn", "otp", "ofp", "ofn", "thr")}
    rng = np.random.default_rng(args.seed)
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        k = max(int(len(o) * 0.2), 2)
        ho = set(o[:k])
        hm = np.array([a in ho for a in auth])
        fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
        if len(fi) < 50 or len(hi) < 20:
            continue
        for j in range(N):
            t = best_thr(P[fi, j], Y[fi, j])
            t = 0.5 if t is None else t
            pr, y = P[hi, j] >= t, Y[hi, j].astype(bool)
            acc["tp"][j].append(int((pr & y).sum()))
            acc["fp"][j].append(int((pr & ~y).sum()))
            acc["fn"][j].append(int((~pr & y).sum()))
            acc["thr"][j].append(t)
            ot = best_thr(P[hi, j], Y[hi, j])          # oracle, on held-out
            opr = P[hi, j] >= (0.5 if ot is None else ot)
            acc["otp"][j].append(int((opr & y).sum()))
            acc["ofp"][j].append(int((opr & ~y).sum()))
            acc["ofn"][j].append(int((~opr & y).sum()))

    def f1(tp, fp, fn):
        s = 2 * sum(tp) + sum(fp) + sum(fn)
        return 0.0 if s == 0 else 2 * sum(tp) / s

    rowsout = []
    for j, name in enumerate(FACTOR_TAXONOMY):
        tp, fp, fn = acc["tp"][j], acc["fp"][j], acc["fn"][j]
        f = f1(tp, fp, fn)
        of = f1(acc["otp"][j], acc["ofp"][j], acc["ofn"][j])
        pr = sum(tp) / max(sum(tp) + sum(fp), 1)
        rc = sum(tp) / max(sum(tp) + sum(fn), 1)
        rowsout.append({
            "j": j, "name": name, "sup": int(Y[:, j].sum()),
            "rate": Y[:, j].mean(), "f1": f, "orc": of, "prec": pr,
            "rec": rc, "ap": ap_score(P[:, j], Y[:, j]),
            "thr": float(np.mean(acc["thr"][j])),
            "thr_sd": float(np.std(acc["thr"][j])),
            "prot": name.strip().lower() in PROTECTIVE})
    macro = float(np.mean([r["f1"] for r in rowsout]))
    omac = float(np.mean([r["orc"] for r in rowsout]))
    print(f"held-out MACRO-F1 {macro:.4f}   with oracle thresholds "
          f"{omac:.4f}   (+{omac-macro:.4f} free)\n")

    rowsout.sort(key=lambda r: -(1 - r["f1"]))
    print(f"  {'label':<38}{'sup':>5}{'F1':>7}{'orcF1':>7}{'thrgap':>8}"
          f"{'prec':>6}{'rec':>6}{'AP':>6}{'thr':>13}{'macro cost':>11}")
    for r in rowsout:
        tag = "*" if r["prot"] else " "
        tcol = "%.2f+-%.2f" % (r["thr"], r["thr_sd"])   # py3.10 f-string quotes
        print(f" {tag}{r['name'][:37]:<38}{r['sup']:>5}{r['f1']:>7.3f}"
              f"{r['orc']:>7.3f}{r['orc']-r['f1']:>+8.3f}{r['prec']:>6.2f}"
              f"{r['rec']:>6.2f}{r['ap']:>6.2f}{tcol:>13}"
              f"{(1-r['f1'])/N:>11.4f}")

    pr_ = [r for r in rowsout if r["prot"]]
    rk_ = [r for r in rowsout if not r["prot"]]
    print(f"\n  protective ({len(pr_)}) macro {np.mean([r['f1'] for r in pr_]):.4f}"
          f"   risk ({len(rk_)}) macro {np.mean([r['f1'] for r in rk_]):.4f}")

    print("\n" + "=" * 100)
    print("WHERE THE HEADROOM IS -- top 8 by macro cost")
    print("=" * 100)
    thr_gain = sum(max(r["orc"] - r["f1"], 0) for r in rowsout) / N
    print(f"  ALL-LABEL threshold headroom (free, no new model): "
          f"{thr_gain:+.4f} macro = {0.3*thr_gain:+.4f} composite")
    print(f"  {'label':<38}{'costs macro':>12}{'of which threshold':>20}"
          f"{'of which ranking':>18}")
    for r in rowsout[:8]:
        t = max(r["orc"] - r["f1"], 0) / N
        print(f"  {r['name'][:37]:<38}{(1-r['f1'])/N:>12.4f}{t:>20.4f}"
              f"{(1-r['orc'])/N:>18.4f}")
    print("\n  THRESHOLD headroom is free -- a better calibration reaches it.")
    print("  RANKING headroom needs a better model and no decode can touch it.")
    print("  A label with high AP but low F1 is mis-thresholded; a label with")
    print("  low AP is genuinely not learned, and prior analysis already found")
    print("  protective is a DISCRIMINATION failure -- expect")
    print("  the starred rows to sit in the ranking column, not the threshold")
    print("  one. If they sit in the threshold column instead, that is new and")
    print("  it is free.")
    print("\n  thr +- sd: a threshold that wanders across splits is fitted to")
    print("  each sample, not a property of the label -- those are the rare")
    print("  labels where tfit is guessing and a shrunk/global threshold may")
    print("  beat it.")


if __name__ == "__main__":
    main()
