"""
Factor threshold-strategy search — CPU-only, validated on gold OOF data.

Macro-F1 weights all 24 labels equally. Deployed thresholds were tuned on the
small 372-post val split, where rare labels have ~no positives, so their
thresholds may be set too conservatively (predicting them ~never -> F1~0
on those labels). Because macro-F1 averages equally, lifting several
near-zero rare labels off the floor can raise S2 more than improving the
already-good common labels.

This script searches threshold STRATEGIES on the gold-labeled OOF
probabilities (honest: these are out-of-fold preds on train, with gold),
and reports which beats the current per-label thresholds. Only if a
strategy wins on this gold data is it worth a submission slot.

Strategies tried:
  A. current saved thresholds (baseline)
  B. global threshold sweep (single t for all labels)
  C. per-label F1-optimal on OOF (re-tune, more data than 372-val)
  D. rare-label recall boost: per-label optimal but cap rare-label
     thresholds at several ceilings so they are actually predicted
  E. blended: F1-optimal but floor common labels, lower rare labels

Run:
  python src/factor_threshold_search.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from scoring import FACTOR_TAXONOMY, factors_macro_f1

N = len(FACTOR_TAXONOMY)


def to_lists(binary):
    return [[FACTOR_TAXONOMY[j] for j in range(N) if row[j]] for row in binary]


def macro_at(probs, gold, ths):
    on = (probs >= ths).astype(int)
    return factors_macro_f1(to_lists(gold), to_lists(on))


def per_label_optimal(probs, gold, grid, cap=None, floor=None):
    ths = np.full(N, 0.5)
    for j in range(N):
        best_t, best_f = 0.5, -1.0
        for t in grid:
            pred = (probs[:, j] >= t).astype(int)
            tp = int(((pred == 1) & (gold[:, j] == 1)).sum())
            fp = int(((pred == 1) & (gold[:, j] == 0)).sum())
            fn = int(((pred == 0) & (gold[:, j] == 1)).sum())
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best_f:
                best_f, best_t = f1, t
        ths[j] = best_t
    if cap is not None:
        rare = [j for j in range(N) if gold[:, j].sum() < 60]
        for j in rare:
            ths[j] = min(ths[j], cap)
    if floor is not None:
        common = [j for j in range(N) if gold[:, j].sum() >= 100]
        for j in common:
            ths[j] = max(ths[j], floor)
    return ths


def main():
    d = np.load("runs/factors_cv_final/oof_probs.npz")
    print("npz keys:", list(d.keys()))
    # infer probs + gold arrays
    probs = d["probs"] if "probs" in d else d[list(d.keys())[0]]
    gold = d["gold"] if "gold" in d else d[list(d.keys())[1]]
    print(f"probs {probs.shape}, gold {gold.shape}\n")

    cur = np.load("runs/factors_cv_final/thresholds.npy")
    grid = np.arange(0.05, 0.90, 0.025)

    base = macro_at(probs, gold, cur)
    print(f"A. current saved thresholds:      {base:.4f}")

    # B global
    bestg, bestgt = -1, 0.5
    for t in grid:
        m = macro_at(probs, gold, np.full(N, t))
        if m > bestg:
            bestg, bestgt = m, t
    print(f"B. best global threshold (t={bestgt:.3f}): {bestg:.4f}")

    # C per-label optimal on OOF
    thsC = per_label_optimal(probs, gold, grid)
    print(f"C. per-label F1-optimal (OOF):    {macro_at(probs, gold, thsC):.4f}")

    # D rare-label recall boost (cap rare thresholds)
    for cap in [0.30, 0.25, 0.20, 0.15]:
        thsD = per_label_optimal(probs, gold, grid, cap=cap)
        print(f"D. per-label + rare cap<={cap:.2f}:   {macro_at(probs, gold, thsD):.4f}")

    # E blended floor+cap
    thsE = per_label_optimal(probs, gold, grid, cap=0.25, floor=0.35)
    print(f"E. rare-cap0.25 + common-floor0.35: {macro_at(probs, gold, thsE):.4f}")

    # Save the best-performing thresholds for test application
    cands = {
        "C_perlabel": thsC,
        "D_cap0.20": per_label_optimal(probs, gold, grid, cap=0.20),
        "D_cap0.25": per_label_optimal(probs, gold, grid, cap=0.25),
        "E_blend": thsE,
    }
    scores = {k: macro_at(probs, gold, v) for k, v in cands.items()}
    best_k = max(scores, key=scores.get)
    print(f"\nBEST strategy: {best_k} at OOF macro {scores[best_k]:.4f} "
          f"(vs current {base:.4f}, delta {scores[best_k]-base:+.4f})")
    np.save("artifacts/thresholds_retuned.npy", cands[best_k])
    print("saved artifacts/thresholds_retuned.npy")
    if scores[best_k] <= base + 0.003:
        print("\n*** gain is marginal/none — NOT worth a submission slot ***")
    else:
        print("\ngain is real on gold OOF — worth a submission to confirm")


if __name__ == "__main__":
    main()
