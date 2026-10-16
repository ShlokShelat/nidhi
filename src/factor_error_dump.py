"""
factor_error_dump.py

Prints the actual false positives and false negatives for the weak factor
labels, so a human can decide whether the errors share a linguistic or
annotation pattern worth targeting.

Reading, not fixing. Every automated route into these labels is now closed:

  * threshold tuning       cross-fitted negative over the ten costliest labels
  * risk conditioning      P(factor|risk) is flat for every weak label
  * per-label pos_weight   1/5 folds, and worse on all six rare labels
  * ASL                    1/5 folds, catastrophic on the two it targeted
  * keyword rules          the model beats them on every label

The remaining question is whether the model is failing for a recognisable
reason. Default targets are the four labels with enough positives for the
answer to mean anything:

    meaning in life          45 gold   P 0.273  R 0.400
    sense of responsibility  58 gold   P 0.522  R 0.603
    stressful life event    182 gold   P 0.528  R 0.621
    psychological capital   238 gold   P 0.486  R 0.777

Deliberately EXCLUDED: sexual orientation (8 gold) and exposure to others'
suicide (14 gold). Eight examples cannot support a conclusion about a concept,
and any pattern found in them would be noise.

False positives are sorted most-confident-first and false negatives
least-confident-first, so the clearest failures appear at the top.

Usage:
  python src/factor_error_dump.py
  python src/factor_error_dump.py --labels "meaning in life" --n 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors
from factor_label_diagnosis import load_probs, ARMS

DEFAULT = ["meaning in life", "sense of responsibility",
           "stressful life event", "psychological capital"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--labels", default=None,
                    help="comma-separated; default is the four labels with "
                         "enough positives to reason about")
    ap.add_argument("--n", type=int, default=12, help="examples of each kind")
    ap.add_argument("--chars", type=int, default=260)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled)
    gmap = dict(zip(lab["row_id"], lab["factors"]))
    pmap = dict(zip(lab["row_id"], lab["post"]))
    tax = FACTOR_TAXONOMY

    rids, Y, PR, TH, FD = load_probs(ARMS["ctrl"], args.k, gmap, tax)
    if rids is None:
        raise SystemExit("missing artifacts/fac_oof_f*.npz -- run factors_oof_cv.py")
    P = (PR >= TH).astype(int)

    targets = ([s.strip() for s in args.labels.split(",")] if args.labels
               else DEFAULT)
    for t in targets:
        if t not in tax:
            print(f"unknown label {t!r}; skipping")
            continue
        j = tax.index(t)
        y, p, pr = Y[:, j], P[:, j], PR[:, j]
        fp = np.where((p == 1) & (y == 0))[0]
        fn = np.where((p == 0) & (y == 1))[0]
        tp = int(((p == 1) & (y == 1)).sum())

        print("\n" + "=" * 100)
        print(f"{t.upper()}   gold {int(y.sum())}  predicted {int(p.sum())}  "
              f"TP {tp}  FP {len(fp)}  FN {len(fn)}")
        print("=" * 100)

        print(f"\n--- FALSE POSITIVES (model most confident it IS present, "
              f"gold says no) ---")
        for i in fp[np.argsort(-pr[fp])][:args.n]:
            r = rids[i]
            print(f"\n  [{r}] p={pr[i]:.3f}")
            print(f"  post: {str(pmap.get(r, ''))[:args.chars]}")
            print(f"  gold factors: {sorted(set(parse_factors(gmap.get(r, ''))))}")

        print(f"\n--- FALSE NEGATIVES (gold says present, model least "
              f"confident) ---")
        for i in fn[np.argsort(pr[fn])][:args.n]:
            r = rids[i]
            print(f"\n  [{r}] p={pr[i]:.3f}")
            print(f"  post: {str(pmap.get(r, ''))[:args.chars]}")
            print(f"  gold factors: {sorted(set(parse_factors(gmap.get(r, ''))))}")

        # co-occurrence: is the model confusing this label with a neighbour?
        conf = []
        for j2, t2 in enumerate(tax):
            if j2 == j:
                continue
            both_gold = int(((Y[:, j] == 1) & (Y[:, j2] == 1)).sum())
            fp_with = int((Y[fp, j2] == 1).sum()) if len(fp) else 0
            if len(fp) and fp_with / len(fp) > 0.30:
                conf.append((t2, fp_with / len(fp), both_gold / max(int(y.sum()), 1)))
        if conf:
            print(f"\n  labels present in >30% of the FALSE POSITIVES "
                  f"(vs their rate among true golds):")
            for t2, a, b in sorted(conf, key=lambda z: -z[1])[:6]:
                flag = "  <- over-represented" if a > b + 0.15 else ""
                print(f"    {t2:<44} {a:.0%} of FPs   {b:.0%} of golds{flag}")
            print("  A label heavily over-represented among the FPs suggests the")
            print("  model is conflating the two concepts.")


if __name__ == "__main__":
    main()
