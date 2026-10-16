"""
truncation_probe.py

FREE falsifier for the sequence-length lane, on existing OOF dumps. CPU only.

THE FINDING IT TESTS. The factor encoders read max_len 320 (~205 words of
post after the hypothesis); the LLM arms read 288 (~180 words). Only 6.4% of
train posts exceed that horizon -- but those posts average 7.8 unique gold
factors against 2.6 for the rest, so they carry 17% of ALL positive
(post,label) cells. The concentration lands on exactly the labels that set
the macro score: meaning in life 33% of positives on truncated posts, social
support 33%, traumatic experience 33%, low socio-economic status 33%, sense
of responsibility 29%, exposure to others' suicide 36% -- against 11% for
hopelessness and 9% for suicide means. (Counted from word lengths in
artifacts/full_train.csv.)

WHAT THIS SCRIPT DECIDES. If the models were reading the whole post, their
per-label F1 on truncated posts would match their F1 on short posts with the
same gold density. If truncation is costing recall, F1 on the truncated
subset will sit clearly below the matched-density short subset, and the gap
will be largest for the LLM arms (shortest window). Pre-stated:

    RECALL GAP on truncated posts >= 0.10 on the weak labels  -> retrain at
        512 is justified (order mnli first, judge with arm_robustness.py)
    gap < 0.05 -> the tails are redundant with the heads; lane CLOSED.

Usage:
  python src/truncation_probe.py --arms mnli dmnli cmine prot d4 qnli mnli7 lnli
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
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def logit_mean(S):
    X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=205,
                    help="words of post the encoder actually reads at "
                         "max_len 320 (~205); use 180 for the LLM window")
    args = ap.parse_args()

    O, rid = [], None
    for a in args.arms:
        pr, ids = [], []
        for f in range(args.k):
            z = np.load(f"{args.dir}/{a}_cv_oof_f{f}.npz", allow_pickle=True)
            pr.append(z["probs"].astype(float))
            ids.extend(str(x) for x in z["row_id"])
        pr = np.concatenate(pr)
        if rid is None:
            rid = ids
        elif ids != rid:
            o = {x: i for i, x in enumerate(ids)}
            pr = pr[[o[x] for x in rid]]
        O.append(pr)
    P = logit_mean(np.stack(O))

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(rid), N), dtype=int)
    for i, r in enumerate(rid):
        for x in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1
    words = lab.loc[rid, "post"].astype(str).str.split().str.len().to_numpy()
    long = words > args.horizon
    print(f"{len(rid)} OOF posts; {int(long.sum())} exceed the "
          f"{args.horizon}-word horizon "
          f"({Y[long].sum()} of {Y.sum()} positive cells)")

    # per-label threshold fitted on ALL posts (as deployed), then P/R/F1
    # separately on the two subsets
    print(f"\n{'label':<44}{'npos_L':>7}{'recL':>7}{'recS':>7}"
          f"{'f1L':>7}{'f1S':>7}")
    print("-" * 79)
    gaps = []
    for j in range(N):
        best, bt = -1.0, 0.5
        for t in GRID:
            pred = (P[:, j] >= t).astype(int)
            tp = int((pred & Y[:, j]).sum())
            if tp == 0:
                continue
            v = 2 * tp / (2 * tp + int((pred & (1 - Y[:, j])).sum())
                          + int(((1 - pred) & Y[:, j]).sum()))
            if v > best:
                best, bt = v, t
        pred = (P[:, j] >= bt).astype(int)

        def prf(mask):
            tp = int((pred[mask] & Y[mask, j]).sum())
            fp = int((pred[mask] & (1 - Y[mask, j])).sum())
            fn = int(((1 - pred[mask]) & Y[mask, j]).sum())
            rec = tp / (tp + fn) if tp + fn else float("nan")
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
            return rec, f1

        rl, fl = prf(long)
        rs, fs = prf(~long)
        npl = int(Y[long, j].sum())
        if npl >= 5 and not np.isnan(rl):
            gaps.append((rs - rl, npl))
        print(f"{FACTOR_TAXONOMY[j]:<44}{npl:>7}{rl:>7.2f}{rs:>7.2f}"
              f"{fl:>7.2f}{fs:>7.2f}")

    w = np.array([g[1] for g in gaps], dtype=float)
    g = np.array([g[0] for g in gaps])
    print(f"\nmean recall gap (short - long), labels with >=5 long positives: "
          f"{g.mean():+.3f}   positive-weighted {np.average(g, weights=w):+.3f}")
    print("\n>= +0.10 -> truncation is costing recall where the positives "
          "are; retrain mnli at max_len 512 and judge over 50 splits.")
    print("< +0.05  -> the unread tails are redundant; CLOSE the lane.")


if __name__ == "__main__":
    main()
