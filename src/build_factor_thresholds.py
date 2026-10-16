"""
build_factor_thresholds.py

Writes a new factor threshold array, changing ONLY the labels whose
CROSS-FITTED F1 gain is positive, and leaving every other label at its
currently deployed threshold.

Two things this guards against.

1. The aggregate reported by factor_threshold_diagnostic.py sums every label
   it flags THRESHOLD, including ones whose cross-fitted gain is NEGATIVE. On
   the current val split, 3 of the 5 flagged labels get WORSE under honest
   cross-fitted evaluation:

       traumatic experience         cur 0.190  cross 0.439
       low socio-economic status    cur 0.759  cross 0.815
       social support               cur 0.462  cross 0.457
       sense of responsibility      cur 0.600  cross 0.593
       interpersonal violence       cur 0.612  cross 0.565

   Retuning only the two positive-gain labels gives a larger S2 than
   retuning all five. Selectivity is worth more than coverage here.

2. It keeps the change auditable and small. The deployed thresholds come from
   averaging each seed's own thresholds.npy -- values calibrated against each
   seed's individual probability scale, then applied to the SEED-AVERAGED
   probabilities. That mismatch is a genuine internal inconsistency, which is
   the strongest argument for this change. It is NOT a licence to refit all 24
   thresholds on val: those were already tuned on val once, and retuning the
   full set on the same already-tuned data overfits rather than generalizing.

Usage:
  python src/build_factor_thresholds.py \
      --probs artifacts/nli_factors_val_probs.npz \
      --labels "traumatic experience,low socio-economic status" \
      --out artifacts/factor_thresholds_selective.npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors
from rule_ablation_v2 import author_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--gold", default="artifacts/val_split.csv")
    ap.add_argument("--labels", required=True,
                    help="comma-separated factor names to retune; every other "
                         "label keeps its deployed threshold")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.probs, allow_pickle=True)
    probs, ths = d["probs"], d["thresholds"].copy()
    rids = list(d["row_id"])

    g = pd.read_csv(args.gold)
    gmap = dict(zip(g["row_id"], g["factors"]))
    tax = FACTOR_TAXONOMY
    idx = {t: i for i, t in enumerate(tax)}
    Y = np.zeros_like(probs, dtype=int)
    for r, rid in enumerate(rids):
        for l in parse_factors(gmap.get(rid, "")):
            if l.strip().lower() in idx:
                Y[r, idx[l.strip().lower()]] = 1

    half = author_split(g.set_index("row_id").loc[rids].reset_index())[0].values
    A = np.array([h == "A" for h in half])
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)

    targets = [s.strip().lower() for s in args.labels.split(",")]
    for t in targets:
        if t not in idx:
            raise SystemExit(f"unknown factor label: {t!r}")

    print(f"{'label':<40}{'old':>7}{'new':>7}{'curF1':>8}{'crossF1':>9}  note")
    macro_before, macro_after = [], []
    for j, name in enumerate(tax):
        y, p = Y[:, j], probs[:, j]
        cur = f1_score(y, (p >= ths[j]).astype(int), zero_division=0)
        macro_before.append(cur)
        if name not in targets:
            macro_after.append(cur)
            continue

        # honest cross-fitted check, recomputed here so this script cannot
        # ship a threshold the diagnostic never actually validated
        preds = np.zeros_like(y)
        for tune, score in ((A, ~A), (~A, A)):
            best_t, best_f = 0.5, -1.0
            for t_ in grid:
                f = f1_score(y[tune], (p[tune] >= t_).astype(int), zero_division=0)
                if f > best_f:
                    best_t, best_f = t_, f
            preds[score] = (p[score] >= best_t).astype(int)
        cross = f1_score(y, preds, zero_division=0)

        if cross <= cur:
            print(f"{name:<40}{ths[j]:>7.2f}{ths[j]:>7.2f}{cur:>8.3f}{cross:>9.3f}"
                  f"  SKIPPED (cross-fit not better)")
            macro_after.append(cur)
            continue

        best_t, best_f = 0.5, -1.0
        for t_ in grid:
            f = f1_score(y, (p >= t_).astype(int), zero_division=0)
            if f > best_f:
                best_t, best_f = t_, f
        print(f"{name:<40}{ths[j]:>7.2f}{best_t:>7.2f}{cur:>8.3f}{cross:>9.3f}"
              f"  retuned")
        ths[j] = best_t
        # bank the CROSS-FITTED value, not the oracle -- the oracle is what
        # this threshold scores on the data it was chosen from
        macro_after.append(cross)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, ths)
    mb, ma = float(np.mean(macro_before)), float(np.mean(macro_after))
    print(f"\nmacro-F1 {mb:.4f} -> {ma:.4f} (honest, cross-fitted) "
          f"= {0.3 * (ma - mb):+.4f} composite")
    print(f"wrote {args.out}  ({int((ths != d['thresholds']).sum())} of "
          f"{len(tax)} thresholds changed)")


if __name__ == "__main__":
    main()
