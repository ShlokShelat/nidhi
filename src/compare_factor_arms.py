"""
compare_factor_arms.py

Re-scores every factor arm on EQUAL FOOTING.

WHY THIS IS NEEDED. A threshold fitted on the same posts it is then scored on
inflates macro-F1, and the inflation grows as the fitting set shrinks. So an
arm scored with one threshold set fitted on all 1635 OOF posts is not
comparable to an arm scored with five per-fold sets each tuned on the 330
posts of its own fold: the difference between them is partly threshold
optimism, not model quality. That bias is the same order of magnitude as the
differences between arms, so it can reverse the sign of a comparison.

WHAT THIS DOES. Every arm, including the incumbent, scored under three
threshold regimes. Compare arms WITHIN a column, never across:

  per-fold    each fold thresholded on its own held-out posts. Optimistic for
              everyone, which is fine as long as everyone gets it. This is the
              regime the incumbent was always scored in.
  pooled      one set fitted on all 1635. Mildly optimistic, uniformly.
  xfit-half   fitted on one author half, applied to the other. The only
              genuinely honest column, and the lowest.

An arm that wins in all three is real. One that wins in only one is an artefact
of that regime.

Usage:
  python src/compare_factor_arms.py
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
from scoring import FACTOR_TAXONOMY, parse_factors

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
PROT = {"social support", "coping strategy", "psychological capital",
        "sense of responsibility", "meaning in life"}


def macro(P, Y):
    out = []
    for j in range(Y.shape[1]):
        tp = int((P[:, j] & Y[:, j]).sum())
        out.append(0.0 if tp == 0 else 2 * tp /
                   (2 * tp + int((P[:, j] & (1 - Y[:, j])).sum())
                    + int(((1 - P[:, j]) & Y[:, j]).sum())))
    return float(np.mean(out)), out


def author_halves(auth):
    s = pd.Series(auth)
    counts = s.value_counts()
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    half, size = {}, {"A": 0, "B": 0}
    for a in order:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


def fit_per_label(PR, Y, rows):
    th = np.full(PR.shape[1], 0.5)
    for j in range(PR.shape[1]):
        best, bt = -1.0, 0.5
        for t in GRID:
            p = (PR[rows, j] >= t).astype(int)
            tp = int((p & Y[rows, j]).sum())
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + int((p & (1 - Y[rows, j])).sum())
                           + int(((1 - p) & Y[rows, j]).sum()))
            if f1 > best:
                best, bt = f1, t
        th[j] = bt
    return th


def load(pattern, k=5):
    r, P, T = [], [], []
    for f in range(k):
        p = pattern.format(f=f)
        if not os.path.exists(p):
            return None
        d = np.load(p, allow_pickle=True)
        r.extend(list(d["row_id"]))
        P.append(d["probs"])
        T.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
    return list(r), np.concatenate(P), np.concatenate(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

    arms = {"INCUMBENT": "artifacts/fac_oof_f{f}.npz"}
    for p in sorted(glob.glob(os.path.join(args.dir, "*_cv_oof_f0.npz"))):
        tag = os.path.basename(p)[: -len("_cv_oof_f0.npz")]
        arms[tag] = os.path.join(args.dir, tag + "_cv_oof_f{f}.npz")

    ref = load(arms["INCUMBENT"], args.k)
    if ref is None:
        raise SystemExit("incumbent OOF missing")
    rid0 = ref[0]
    Y = np.zeros(ref[1].shape, dtype=int)
    for i, r in enumerate(rid0):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    half = author_halves([lab.loc[r, "anon_user_id"] for r in rid0])
    A, B = half == "A", half == "B"
    print(f"{len(rid0)} OOF posts   halves {int(A.sum())}/{int(B.sum())}\n")

    rows = []
    per_label_store = {}
    for tag, pat in arms.items():
        got = load(pat, args.k)
        if got is None:
            print(f"  skip {tag}: incomplete folds")
            continue
        r, PR, TH = got
        if r != rid0:
            o = {x: i for i, x in enumerate(r)}
            if set(r) != set(rid0):
                print(f"  skip {tag}: different post set")
                continue
            order = [o[x] for x in rid0]
            PR, TH = PR[order], TH[order]

        m_pf, per_pf = macro((PR >= TH).astype(int), Y)
        m_po, _ = macro((PR >= fit_per_label(PR, Y, np.ones(len(Y), bool))
                         ).astype(int), Y)
        Px = np.zeros_like(Y)
        Px[A] = (PR[A] >= fit_per_label(PR, Y, B)).astype(int)
        Px[B] = (PR[B] >= fit_per_label(PR, Y, A)).astype(int)
        m_xf, _ = macro(Px, Y)
        rows.append(dict(arm=tag, per_fold=m_pf, pooled=m_po, xfit=m_xf))
        per_label_store[tag] = per_pf

    df = pd.DataFrame(rows).set_index("arm")
    base = df.loc["INCUMBENT"]
    print(f"{'arm':<14}{'per-fold':>10}{'pooled':>10}{'xfit-half':>11}"
          f"{'  |':>3}{'d per-fold':>12}{'d pooled':>10}{'d xfit':>9}")
    print("-" * 82)
    for tag, x in df.iterrows():
        d = "" if tag == "INCUMBENT" else (
            f"{x.per_fold - base.per_fold:>+12.4f}"
            f"{x.pooled - base.pooled:>+10.4f}"
            f"{x.xfit - base.xfit:>+9.4f}")
        print(f"{tag:<14}{x.per_fold:>10.4f}{x.pooled:>10.4f}{x.xfit:>11.4f}"
              f"{'  |':>3}{d}")

    print("\nCompare arms WITHIN a column. Reading one arm's per-fold number")
    print("against another arm's pooled number compares two different amounts")
    print("of threshold optimism, not two models.")

    wins = [t for t in df.index if t != "INCUMBENT"
            and df.loc[t, "per_fold"] > base.per_fold
            and df.loc[t, "pooled"] > base.pooled
            and df.loc[t, "xfit"] > base.xfit]
    print(f"\narms ahead in ALL THREE regimes: {wins if wins else 'none'}")

    if wins:
        best = max(wins, key=lambda t: df.loc[t, "xfit"])
        print(f"\nPER-LABEL, INCUMBENT -> {best}  (per-fold regime, movers only)")
        b, n = per_label_store["INCUMBENT"], per_label_store[best]
        for j, t in enumerate(FACTOR_TAXONOMY):
            if abs(n[j] - b[j]) > 0.02:
                print(f"  {t:<44}{b[j]:>7.3f}{n[j]:>8.3f}{n[j]-b[j]:>+8.3f}"
                      f"{'  prot' if t in PROT else ''}")

    df.to_csv("artifacts/factor_arms_comparison.csv")
    print("\nwrote artifacts/factor_arms_comparison.csv")


if __name__ == "__main__":
    main()
