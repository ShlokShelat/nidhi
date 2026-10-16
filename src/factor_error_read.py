"""
factor_error_read.py

Read the actual mistakes on the worst factor labels.

WHY THIS AND NOT ANOTHER METRIC. src/factor_diagnosis.py established what the
problem is NOT:

    thresholds      perfect cut-offs would give only a small gain, and real
                    gains are a fraction of that
    data scarcity   AUC is FLAT across prevalence bands (0.932 for <40
                    positives, 0.908 for >400). The model ranks an 8-example
                    label as well as a 745-example one. Correlation between
                    log(n) and AUC is -0.116.

What it IS: a ranking failure visible in average precision, concentrated in
specific labels:

    meaning in life          AP 0.173    F1 0.229    oracle 0.235
    exposure to others'      AP 0.324    F1 0.381    oracle 0.462
    sense of responsibility  AP 0.431    F1 0.484    oracle 0.503
    (against hopelessness    AP 0.890    F1 0.813)

`meaning in life` alone is 1/24 of S2 and almost entirely forfeited. No metric
will explain why. The only way to find out is to read the posts the model gets
wrong and see what the annotators saw that it does not.

TWO KINDS OF ERROR, and they mean different things:

  MISSES        gold says yes, the model scored it LOW. If these read as
                obvious to a human, the model is blind to something learnable.
                If they read as arbitrary, the label is annotated
                inconsistently and no model will find it.

  FALSE ALARMS  gold says no, the model scored it HIGH. If these look like
                they SHOULD be positive, the label boundary is narrower than
                the definition suggests -- which is a labelling convention we
                could encode.

The specific hypothesis worth testing on `meaning in life`: its definition is
"a sense of purpose, goals, or reasons for living", which is PROTECTIVE. But
posts on this forum overwhelmingly describe the ABSENCE of purpose. If
annotators mark the label when purpose is DISCUSSED -- present or absent -- the
model is being trained on two opposite meanings under one name, and AP 0.173 is
exactly what that would look like.

Usage:
  python src/factor_error_read.py
  python src/factor_error_read.py --factor "meaning in life" --n 12
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

DEFAULT = [
    "meaning in life",
    "exposure to others' suicide",
    "sense of responsibility",
    "stressful life event",
    "psychological capital",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--oof", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--factor", default=None,
                    help="one factor; default walks the worst five")
    ap.add_argument("--n", type=int, default=6, help="examples of each kind")
    ap.add_argument("--chars", type=int, default=420)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")
    rids, PR = [], []
    for f in range(args.k):
        d = np.load(args.oof.format(f=f), allow_pickle=True)
        rids.extend(list(d["row_id"]))
        PR.append(d["probs"])
    PR = np.concatenate(PR)
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros(PR.shape, dtype=int)
    for i, r in enumerate(rids):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1

    targets = [args.factor] if args.factor else DEFAULT
    for t in targets:
        if t not in idx:
            print(f"unknown factor {t!r}")
            continue
        j = idx[t]
        y, p = Y[:, j], PR[:, j]
        print("\n" + "=" * 100)
        print(f"{t}   ({int(y.sum())} gold positives of {len(y)} posts)")
        print("=" * 100)

        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        misses = pos[np.argsort(p[pos])][:args.n]
        alarms = neg[np.argsort(-p[neg])][:args.n]
        hits = pos[np.argsort(-p[pos])][:3]

        print(f"\n--- WHAT IT GETS RIGHT (gold yes, highest scored) ---")
        for i in hits:
            other = [FACTOR_TAXONOMY[k] for k in range(24)
                     if Y[i, k] and k != j]
            print(f"\n  [{p[i]:.2f}] {rids[i]}   also: {other[:4]}")
            for line in textwrap.wrap(str(lab.loc[rids[i], "post"])[:args.chars],
                                      92):
                print(f"      {line}")

        print(f"\n--- MISSES (gold YES, model scored LOWEST) ---")
        for i in misses:
            other = [FACTOR_TAXONOMY[k] for k in range(24)
                     if Y[i, k] and k != j]
            print(f"\n  [{p[i]:.2f}] {rids[i]}   also: {other[:4]}")
            for line in textwrap.wrap(str(lab.loc[rids[i], "post"])[:args.chars],
                                      92):
                print(f"      {line}")

        print(f"\n--- FALSE ALARMS (gold NO, model scored HIGHEST) ---")
        for i in alarms:
            other = [FACTOR_TAXONOMY[k] for k in range(24) if Y[i, k]]
            print(f"\n  [{p[i]:.2f}] {rids[i]}   gold: {other[:4]}")
            for line in textwrap.wrap(str(lab.loc[rids[i], "post"])[:args.chars],
                                      92):
                print(f"      {line}")

        print(f"\n--- READ FOR ---")
        print("  * do the MISSES look obvious to you? -> the model is blind to")
        print("    something learnable")
        print("  * do they look arbitrary? -> the label is annotated")
        print("    inconsistently and no model can recover it")
        print("  * do the FALSE ALARMS look like they SHOULD be positive?")
        print("    -> the annotators' boundary is narrower than the published")
        print("    definition, and that boundary is encodable")


if __name__ == "__main__":
    main()
