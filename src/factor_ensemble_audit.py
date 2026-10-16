"""
factor_ensemble_audit.py

Tests whether the deployed 2-seed factor ensemble is actually helping, or is
being held back by an internal inconsistency in how its thresholds are formed.

The suspicion, and why it matters:

  predict_factors_nli.py averages the seeds' PROBABILITIES and separately
  averages their THRESHOLDS, then applies one to the other:

      probs = mean(p1, p2)          ths = mean(t1, t2)          on = probs >= ths

  But t1 was tuned against p1's distribution and t2 against p2's. The average
  of two probability vectors is more concentrated than either (variance is
  halved), so a threshold calibrated for the wider individual distribution is
  systematically wrong for the narrower averaged one. Averaging the thresholds
  does not fix that -- it just picks an arbitrary point between two values
  that were answering a different question.

  Two pieces of evidence that this is real, not theoretical:

    * the seeds' saved thresholds disagree by more than 0.20 on 11 of 24
      labels, up to 0.65 (sense of responsibility: 0.10 vs 0.75). Thresholds
      that unstable are not identified by 372 val posts -- the F1-vs-threshold
      curve is flat and argmax is returning noise.
    * the training logs record each seed's own tuned val macro-F1 as 0.6038
      and 0.5782, while the deployed ensemble measures 0.5755 on the same
      val split. Probability averaging that makes an ensemble WORSE than both
      of its members is a calibration bug, not a modelling result.

Four configurations are compared, three of which are internally consistent:

  A  seed1 probs + seed1 thresholds          consistent
  B  seed2 probs + seed2 thresholds          consistent
  C  avg probs   + avg thresholds            DEPLOYED -- inconsistent
  D  avg probs   + thresholds retuned on avg probs, author cross-fitted

D is the fix that keeps the ensemble's benefit. Its number is reported
CROSS-FITTED (tune on one author-disjoint half, score the other, both ways),
never as an oracle, because this campaign's entire transfer record says a
val-argmax number is worthless as a prediction of test behaviour.

Prepare the inputs (GPU, ~2 min each):
  python src/predict_factors_nli.py --test artifacts/val_split.csv \
      --nli-run runs/factors_nli    --out artifacts/_fac_s1_val.csv \
      --dump-probs artifacts/fac_s1_val_probs.npz
  python src/predict_factors_nli.py --test artifacts/val_split.csv \
      --nli-run runs/factors_nli_s2 --out artifacts/_fac_s2_val.csv \
      --dump-probs artifacts/fac_s2_val_probs.npz

Then (CPU):
  python src/factor_ensemble_audit.py \
      --seed-probs artifacts/fac_s1_val_probs.npz,artifacts/fac_s2_val_probs.npz
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

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def macro(Y, P):
    return float(f1_score(Y, P, average=None, zero_division=0).mean())


def tune(Y, probs, rows=None):
    """Per-label argmax-F1 thresholds, optionally on a subset of rows."""
    sel = slice(None) if rows is None else rows
    ths = np.zeros(probs.shape[1])
    for j in range(probs.shape[1]):
        y, p = Y[sel, j], probs[sel, j]
        best_t, best_f = 0.5, -1.0
        for t in GRID:
            f = f1_score(y, (p >= t).astype(int), zero_division=0)
            if f > best_f:
                best_t, best_f = t, f
        ths[j] = best_t
    return ths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-probs", required=True,
                    help="comma-separated per-seed .npz files from --dump-probs")
    ap.add_argument("--gold", default="artifacts/val_split.csv")
    ap.add_argument("--out-thresholds", default=None,
                    help="if given, write config D's thresholds (tuned on ALL "
                         "of val) here for use with predict_factors_nli.py "
                         "--thresholds")
    args = ap.parse_args()

    paths = [p.strip() for p in args.seed_probs.split(",")]
    packs = [np.load(p, allow_pickle=True) for p in paths]
    rids = list(packs[0]["row_id"])
    for pk in packs[1:]:
        if list(pk["row_id"]) != rids:
            raise SystemExit("row_id order differs between seed prob files")

    g = pd.read_csv(args.gold)
    gmap = dict(zip(g["row_id"], g["factors"]))
    tax = FACTOR_TAXONOMY
    idx = {t: i for i, t in enumerate(tax)}
    Y = np.zeros((len(rids), len(tax)), dtype=int)
    for r, rid in enumerate(rids):
        for l in parse_factors(gmap.get(rid, "")):
            if l.strip().lower() in idx:
                Y[r, idx[l.strip().lower()]] = 1

    half = author_split(g.set_index("row_id").loc[rids].reset_index())[0].values
    A = np.where(half == "A")[0]
    B = np.where(half == "B")[0]

    print(f"val posts {len(rids)}   author halves A={len(A)} B={len(B)}\n")
    print(f"{'configuration':<52}{'macro-F1':>10}  note")

    results = {}
    for i, pk in enumerate(packs, 1):
        p, t = pk["probs"], pk["thresholds"]
        s = macro(Y, (p >= t).astype(int))
        results[f"seed{i}"] = s
        print(f"{f'{chr(64+i)}  seed{i} probs + seed{i} thresholds':<52}"
              f"{s:>10.4f}  consistent")

    avg_p = np.mean([pk["probs"] for pk in packs], axis=0)
    avg_t = np.mean([pk["thresholds"] for pk in packs], axis=0)
    dep = macro(Y, (avg_p >= avg_t).astype(int))
    results["deployed"] = dep
    print(f"{'C  avg probs + avg thresholds':<52}{dep:>10.4f}  "
          f"DEPLOYED (inconsistent)")

    # D: retune on the averaged distribution, cross-fitted across author halves
    pred = np.zeros_like(Y)
    for tune_rows, score_rows in ((A, B), (B, A)):
        th = tune(Y, avg_p, rows=tune_rows)
        pred[score_rows] = (avg_p[score_rows] >= th).astype(int)
    cross = macro(Y, pred)
    results["retuned_cross"] = cross
    print(f"{'D  avg probs + retuned thresholds (CROSS-FITTED)':<52}"
          f"{cross:>10.4f}  consistent, honest")

    oracle_th = tune(Y, avg_p)
    oracle = macro(Y, (avg_p >= oracle_th).astype(int))
    print(f"{'   (same, tuned on all of val -- CIRCULAR, ignore)':<52}"
          f"{oracle:>10.4f}  upper bound only")

    print("\n" + "=" * 78)
    best_single = max(results[f"seed{i}"] for i in range(1, len(packs) + 1))
    if dep < best_single:
        print(f"CONFIRMED: the deployed ensemble ({dep:.4f}) scores BELOW the best "
              f"single seed\n({best_single:.4f}). Probability averaging cannot do "
              f"that on its own -- the\nthreshold mismatch is costing "
              f"{best_single - dep:.4f} macro-F1 = "
              f"{0.3 * (best_single - dep):+.4f} composite.")
    else:
        print(f"NOT confirmed: the deployed ensemble ({dep:.4f}) is at least as "
              f"good as every\nsingle seed ({best_single:.4f}). The threshold "
              f"mismatch is not costing measurable\nperformance -- do not "
              f"change it.")

    print(f"\nhonest gain from fixing it (config D vs deployed): "
          f"{cross - dep:+.4f} macro-F1 = {0.3 * (cross - dep):+.4f} composite")
    print(f"gain from simply dropping to the best single seed: "
          f"{best_single - dep:+.4f} macro-F1 = "
          f"{0.3 * (best_single - dep):+.4f} composite")
    print("\nNOTE: picking a seed because it won on val is itself val selection.")
    print("Config D is preferable when close, because it keeps both seeds and")
    print("fixes the actual defect rather than selecting on a noisy comparison.")

    if args.out_thresholds:
        np.save(args.out_thresholds, oracle_th)
        print(f"\nwrote {args.out_thresholds} (config D thresholds, full-val tuned)")


if __name__ == "__main__":
    main()
