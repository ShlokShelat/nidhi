"""
factor_diagnosis.py

Attribute the factor deficit to DATA, MODEL, or THRESHOLD -- per label.

WHY THIS INSTEAD OF ANOTHER EXPERIMENT. Nine factor ideas have been tried and
all nine failed. Every one assumed a cause without measuring it: consistency
rules assumed the outputs were inconsistent, threshold recalibration assumed
the cut-offs were wrong, synthetic data assumed the rare labels were
data-starved, the protective template assumed the question was malformed. Some
of those assumptions were testable in advance and were not tested.

THE DECOMPOSITION. For each label the model produces a probability. Two things
can go wrong, and they need completely different fixes:

  RANKING   Does the probability put true positives above negatives? Measured
            by AUC and average precision -- no threshold involved. If ranking
            is poor the model does not know, and no cut-off can rescue it.
            Fix = better model or more data.

  DECISION  Given the ranking, is the cut-off in the right place? Measured as
            the gap between F1 at the deployed threshold and F1 at the best
            possible threshold on the same data. Fix = better thresholds.

So the deficit splits three ways:

    1 - F1_deployed  =  (1 - F1_oracle)      how much the RANKING cannot deliver
                      + (F1_oracle - F1_deployed)   what the CUT-OFF is wasting

And the data hypothesis gets its own test: if scarcity is the binding
constraint, AUC should rise with the number of positive examples. If a label
with 700 positives ranks no better than one with 30, the problem is not data.

WHAT THE ANSWER LOOKS LIKE
  high AUC, big oracle gap    -> thresholds. Cheap to fix.
  low AUC, small oracle gap   -> the model cannot separate this label at all.
  AUC rising with n           -> data scarcity is real; more real data helps.
  AUC flat vs n               -> scarcity is NOT the constraint; the label is
                                 intrinsically ambiguous or the architecture
                                 cannot represent it.

Runs on the incumbent's 1635 out-of-fold predictions. CPU, seconds.

Usage:
  python src/factor_diagnosis.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
PROTECTIVE = {"social support", "coping strategy", "psychological capital",
              "sense of responsibility", "meaning in life"}


def f1_at(p, y, t):
    pred = (p >= t).astype(int)
    tp = int((pred & y).sum())
    if tp == 0:
        return 0.0, 0.0, 0.0
    fp = int((pred & (1 - y)).sum())
    fn = int(((1 - pred) & y).sum())
    pr, rc = tp / (tp + fp), tp / (tp + fn)
    return 2 * pr * rc / (pr + rc), pr, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--oof", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")
    rids, PR, TH = [], [], []
    for f in range(args.k):
        d = np.load(args.oof.format(f=f), allow_pickle=True)
        rids.extend(list(d["row_id"]))
        PR.append(d["probs"])
        TH.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
    PR, TH = np.concatenate(PR), np.concatenate(TH)
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros(PR.shape, dtype=int)
    for i, r in enumerate(rids):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    print(f"{len(rids)} out-of-fold posts\n")

    print(f"{'factor':<44}{'n':>5}{'AUC':>7}{'AP':>7}"
          f"{'F1':>7}{'P':>6}{'R':>6}{'oracle':>8}{'thloss':>8}  type")
    print("-" * 106)
    rows = []
    for j, t in enumerate(FACTOR_TAXONOMY):
        y, p = Y[:, j], PR[:, j]
        n = int(y.sum())
        auc = roc_auc_score(y, p) if 0 < n < len(y) else float("nan")
        apr = average_precision_score(y, p) if n else float("nan")
        cur, pr, rc = f1_at(p, y, TH[0, j])
        orc = max(f1_at(p, y, tt)[0] for tt in GRID)
        rows.append((t, n, auc, apr, cur, pr, rc, orc, orc - cur))
        print(f"{t:<44}{n:>5}{auc:>7.3f}{apr:>7.3f}{cur:>7.3f}{pr:>6.2f}"
              f"{rc:>6.2f}{orc:>8.3f}{orc - cur:>8.3f}"
              f"  {'prot' if t in PROTECTIVE else ''}")

    df = pd.DataFrame(rows, columns=[
        "factor", "n", "auc", "ap", "f1", "p", "r", "oracle", "thloss"])

    print("\n" + "=" * 78)
    print("WHERE THE DEFICIT LIVES")
    print("=" * 78)
    cur, orc = df.f1.mean(), df.oracle.mean()
    print(f"  macro-F1 now                       {cur:.4f}")
    print(f"  macro-F1 with perfect thresholds   {orc:.4f}")
    print(f"  ---------------------------------------------")
    print(f"  recoverable by BETTER THRESHOLDS   {orc - cur:.4f}")
    print(f"  unreachable without a BETTER MODEL {1 - orc:.4f}")
    print(f"\n  Note the oracle is an upper bound that cannot be attained --")
    print(f"  it picks each cut-off using the answers. Real threshold gains")
    print(f"  are a fraction of {orc - cur:.4f}.")

    print("\n" + "=" * 78)
    print("IS IT DATA SCARCITY?  (AUC vs number of positives)")
    print("=" * 78)
    ok = df.dropna(subset=["auc"])
    r = np.corrcoef(np.log(ok.n.clip(1)), ok.auc)[0, 1]
    print(f"  correlation(log n, AUC) = {r:+.3f}")
    for lo, hi, name in [(0, 40, "very rare  (<40)"),
                         (40, 130, "rare      (40-130)"),
                         (130, 400, "common   (130-400)"),
                         (400, 10 ** 6, "frequent    (>400)")]:
        s = ok[(ok.n >= lo) & (ok.n < hi)]
        if len(s):
            print(f"  {name:<20} n={len(s):>2} labels   "
                  f"mean AUC {s.auc.mean():.3f}   mean F1 {s.f1.mean():.3f}")
    print("\n  If AUC is flat across these bands, scarcity is NOT the binding")
    print("  constraint and more data will not help. If it climbs, it is.")

    print("\n" + "=" * 78)
    print("PROTECTIVE vs RISK")
    print("=" * 78)
    pm = df.factor.isin(PROTECTIVE)
    print(f"  protective ({int(pm.sum())})  AUC {df[pm].auc.mean():.3f}   "
          f"F1 {df[pm].f1.mean():.3f}   oracle {df[pm].oracle.mean():.3f}")
    print(f"  risk       ({int((~pm).sum())})  AUC {df[~pm].auc.mean():.3f}   "
          f"F1 {df[~pm].f1.mean():.3f}   oracle {df[~pm].oracle.mean():.3f}")
    print("\n  If protective AUC is as good as risk AUC, the phrasing/model is")
    print("  fine and their low F1 is a threshold or prevalence artefact.")
    print("  If protective AUC is genuinely lower, the model cannot read them.")

    print("\n" + "=" * 78)
    print("WORST LABELS BY RECOVERABLE VALUE  (deficit x 1/24)")
    print("=" * 78)
    df["worth"] = (1 - df.f1) / 24
    print(f"{'factor':<44}{'n':>5}{'AUC':>7}{'F1':>7}{'worth':>8}  verdict")
    for _, x in df.sort_values("worth", ascending=False).head(8).iterrows():
        if x.auc > 0.85 and x.thloss > 0.05:
            v = "THRESHOLD -- ranking is good"
        elif x.auc < 0.75:
            v = "MODEL -- cannot separate it"
        elif x.n < 40:
            v = "DATA -- too few positives"
        else:
            v = "hard label, no single cause"
        print(f"{x.factor:<44}{int(x.n):>5}{x.auc:>7.3f}{x.f1:>7.3f}"
              f"{x.worth:>8.4f}  {v}")

    df.to_csv("artifacts/factor_diagnosis.csv", index=False)
    print("\nwrote artifacts/factor_diagnosis.csv")


if __name__ == "__main__":
    main()
