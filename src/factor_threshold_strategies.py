"""
factor_threshold_strategies.py

Attacks the largest QUANTIFIED defect found in the whole system.

factor_ensemble_audit.py measured per-label threshold tuning on val at
macro-F1 0.6221 in-sample versus 0.5448 cross-fitted -- a 0.077 overfitting
gap. That is not a property of one bad configuration; it is what happens when
24 free parameters are fitted to 186 posts, and the DEPLOYED thresholds were
fitted the same way, so they carry the same disease.

The fix is not a better search over the same 24 parameters. It is fewer
parameters. This compares threshold PARAMETERISATIONS, ordered by degrees of
freedom, every one of them evaluated the same honest way: fit on one
author-disjoint half, score the other, both directions, then macro-F1 over the
combined out-of-fold predictions.

  deployed        24 params, already fitted on this val (shown for reference;
                  its number here is optimistic because it saw all of val)
  per_label       24 params  -- the current approach, refitted honestly
  global          1 param    -- one threshold for every label
  shrunk          2 params   -- per-label tuned, pulled toward the global one
  prevalence      0 params   -- threshold set so the predicted positive count
                                matches the TRAINING prevalence for that label.
                                Uses no val labels at all.
  lipton          1 param    -- Lipton et al. 2014 show the F1-optimal
                                threshold for a calibrated scorer is ~F1*/2;
                                estimate F1* on the fit half, halve it.

If a low-parameter strategy wins cross-fitted, that is a genuine
generalisation gain rather than a val-argmax artifact -- exactly the change
class most likely to transfer to a held-out evaluation.

Usage (CPU, needs the dumped probability matrices):
  python src/factor_threshold_strategies.py \
      --probs artifacts/nli_factors_val_probs.npz
  python src/factor_threshold_strategies.py \
      --probs artifacts/fac_s1_val_probs.npz     # single seed
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors, normalize_label
from rule_ablation_v2 import author_split

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def macro(Y, P):
    return float(f1_score(Y, P, average=None, zero_division=0).mean())


def per_label_tuned(Y, P, rows):
    ths = np.full(P.shape[1], 0.5)
    for j in range(P.shape[1]):
        y, p = Y[rows, j], P[rows, j]
        if y.sum() == 0:
            continue
        best_t, best_f = 0.5, -1.0
        for t in GRID:
            f = f1_score(y, (p >= t).astype(int), zero_division=0)
            if f > best_f:
                best_t, best_f = t, f
        ths[j] = best_t
    return ths


def global_tuned(Y, P, rows):
    best_t, best_f = 0.5, -1.0
    for t in GRID:
        f = macro(Y[rows], (P[rows] >= t).astype(int))
        if f > best_f:
            best_t, best_f = t, f
    return np.full(P.shape[1], best_t)


def shrunk(Y, P, rows):
    per = per_label_tuned(Y, P, rows)
    glob = global_tuned(Y, P, rows)[0]
    best_a, best_f = 0.0, -1.0
    for a in np.round(np.arange(0.0, 1.01, 0.05), 2):
        th = a * per + (1 - a) * glob
        f = macro(Y[rows], (P[rows] >= th).astype(int))
        if f > best_f:
            best_a, best_f = a, f
    return best_a * per + (1 - best_a) * glob


def prevalence_matched(prior, P, rows):
    """0 val parameters: pick the threshold that yields the expected count."""
    ths = np.zeros(P.shape[1])
    n = len(rows)
    for j in range(P.shape[1]):
        target = max(1, int(round(prior[j] * n)))
        col = np.sort(P[rows, j])[::-1]
        ths[j] = col[min(target, len(col)) - 1]
    return ths


def lipton(Y, P, rows):
    """Threshold ~ F1*/2 (Lipton et al. 2014, Thresholding to Maximise F1)."""
    ths = np.full(P.shape[1], 0.5)
    for j in range(P.shape[1]):
        y, p = Y[rows, j], P[rows, j]
        if y.sum() == 0:
            continue
        best_f = max(f1_score(y, (p >= t).astype(int), zero_division=0)
                     for t in GRID)
        ths[j] = max(0.05, min(0.95, best_f / 2.0))
    return ths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--gold", default="artifacts/val_split.csv")
    ap.add_argument("--train", default="artifacts/train_split.csv")
    ap.add_argument("--out-thresholds", default=None,
                    help="write the winning strategy's thresholds, refitted on "
                         "ALL of val, here")
    args = ap.parse_args()

    d = np.load(args.probs, allow_pickle=True)
    P, deployed_th = d["probs"], d["thresholds"]
    rids = list(d["row_id"])

    g = pd.read_csv(args.gold)
    gmap = dict(zip(g["row_id"], g["factors"]))
    tax = FACTOR_TAXONOMY
    idx = {t: i for i, t in enumerate(tax)}
    Y = np.zeros_like(P, dtype=int)
    for r, rid in enumerate(rids):
        for l in parse_factors(gmap.get(rid, "")):
            if l.strip().lower() in idx:
                Y[r, idx[l.strip().lower()]] = 1

    # training prevalence -- external to val, so 'prevalence' uses 0 val params
    tr = pd.read_csv(args.train)
    prior = np.zeros(len(tax))
    for x in tr["factors"]:
        for l in set(parse_factors(x)):
            if l.strip().lower() in idx:
                prior[idx[l.strip().lower()]] += 1
    prior /= max(len(tr), 1)

    half = author_split(g.set_index("row_id").loc[rids].reset_index())[0].values
    A = np.where(half == "A")[0]
    B = np.where(half == "B")[0]
    print(f"val posts {len(rids)}   author halves A={len(A)} B={len(B)}")
    print(f"training prevalence from {len(tr)} posts\n")

    strategies = {
        "per_label (24 params)": lambda rows: per_label_tuned(Y, P, rows),
        "global    (1 param)":   lambda rows: global_tuned(Y, P, rows),
        "shrunk    (2 params)":  lambda rows: shrunk(Y, P, rows),
        "prevalence(0 params)":  lambda rows: prevalence_matched(prior, P, rows),
        "lipton F1/2":           lambda rows: lipton(Y, P, rows),
    }

    print(f"{'strategy':<26}{'in-sample':>11}{'CROSS-FIT':>11}{'overfit gap':>13}")
    dep = macro(Y, (P >= deployed_th).astype(int))
    print(f"{'deployed (reference)':<26}{dep:>11.4f}{'  n/a':>11}"
          f"{'  saw all of val':>13}")

    results = {}
    for name, fit in strategies.items():
        insample = macro(Y, (P >= fit(np.arange(len(Y)))).astype(int))
        pred = np.zeros_like(Y)
        for tune_rows, score_rows in ((A, B), (B, A)):
            th = fit(tune_rows)
            pred[score_rows] = (P[score_rows] >= th).astype(int)
        cf = macro(Y, pred)
        results[name] = cf
        print(f"{name:<26}{insample:>11.4f}{cf:>11.4f}{insample - cf:>13.4f}")

    best = max(results, key=results.get)
    print(f"\n{'=' * 74}")
    print(f"best cross-fitted strategy : {best}  ({results[best]:.4f})")
    print(f"current per-label baseline : {results['per_label (24 params)']:.4f}")
    delta = results[best] - results["per_label (24 params)"]
    print(f"honest improvement         : {delta:+.4f} macro-F1 "
          f"= {0.3 * delta:+.4f} composite")
    print(f"\nvs the DEPLOYED thresholds (which are in-sample here, so this")
    print(f"comparison flatters them): {results[best] - dep:+.4f} macro-F1")

    if args.out_thresholds:
        th = strategies[best](np.arange(len(Y)))
        np.save(args.out_thresholds, th)
        print(f"\nwrote {args.out_thresholds} ({best}, refitted on all of val)")


if __name__ == "__main__":
    main()
