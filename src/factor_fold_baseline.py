"""
factor_fold_baseline.py

What the DEPLOYED 8-arm bag scores on ONE fold's validation posts.

A factor pilot trained on fold 3 prints a val macro-F1 that cannot be compared
to the bag's full-OOF 0.6417 -- different posts, and the per-fold spread is
large. This prints the bag's number on the SAME posts, decoded the same way
(champion rates, 0 fitted parameters), so the pilot has an honest bar.

The number that decides the 5-fold spend is the PROTECTIVE column, not the
overall macro. The fine-tuned model exists to fix protective discrimination; a
model that matches overall but not on those five labels has reproduced the bag,
not improved it.

Usage:
  python src/factor_fold_baseline.py --val artifacts/folds/val_f3.csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    X = np.clip(S, 1e-6, 1 - 1e-6)
    P = 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))
    pos = {r: i for i, r in enumerate(rid)}

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    keep = [r for r in va["row_id"] if r in pos]
    if len(keep) != len(va):
        print(f"WARNING: {len(va) - len(keep)} val rows absent from the OOF "
              f"dumps -- they are excluded")
    sel = [pos[r] for r in keep]
    Pv = P[sel]
    n = len(sel)

    lookup = va.set_index("row_id")["factors"].astype(str)
    Y = np.zeros((n, N), dtype=np.int8)
    for i, r in enumerate(keep):
        for x in set(parse_factors(lookup.loc[r])):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1

    ch = pd.read_csv(args.champion, keep_default_na=False)
    C = np.zeros(N)
    for cell in ch["factors"].astype(str):
        for x in set(parse_factors(cell)):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                C[j] += 1
    rates = C / len(ch)

    pred = np.zeros((n, N), dtype=np.int8)
    for j in range(N):
        k = min(max(int(round(rates[j] * n)), 1), n - 1)
        cut = np.sort(Pv[:, j])[-k]
        pred[:, j] = (Pv[:, j] >= cut).astype(np.int8)

    tp = (pred.astype(int) * Y).sum(0)
    den = 2 * tp + (pred.astype(int) * (1 - Y)).sum(0) + \
        ((1 - pred.astype(int)) * Y).sum(0)
    per = np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))

    print(f"{os.path.basename(args.val)}: {n} posts   arms {args.arms}")
    print(f"decoded at champion rates ({C.sum() / len(ch):.2f} factors/post)\n")
    print(f"{'label':<44}{'gold':>6}{'F1':>8}")
    for j in np.argsort(per):
        star = "* " if FACTOR_TAXONOMY[j] in PROTECTIVE else "  "
        print(f"{star}{FACTOR_TAXONOMY[j]:<42}{int(Y[:, j].sum()):>6}"
              f"{per[j]:>8.3f}")
    print(f"\nTHE BAR for a pilot on this fold")
    print(f"  macro-F1     {per.mean():.4f}")
    print(f"  protective   {per[PROT_J].mean():.4f}   <- the one that matters")
    print(f"  risk factors {per[[j for j in range(N) if j not in PROT_J]].mean():.4f}")
    print("\n(full-OOF reference: macro 0.6417, protective 0.5530)")


if __name__ == "__main__":
    main()
