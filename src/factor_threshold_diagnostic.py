"""
factor_threshold_diagnostic.py

For each of the 24 factor labels, decides WHY it scores badly:

  UNMEASURABLE   too few gold positives here to say anything. On the 372-post
                 val split this is not a corner case -- 'sexual orientation
                 related issues' has 0 gold positives and 'poor school
                 performance' has 1. Their F1 is 0 by construction, they
                 contribute 2 x 0.5/24 = 0.0417 of the naive "headroom"
                 estimate, and NO amount of threshold tuning on val can move
                 them. Treating them as opportunities is a measurement
                 artifact; they need the full labeled set to be assessed.

  THRESHOLD      the ranking is good (high average precision relative to the
                 base rate) but the operating point is wrong. Fixable with no
                 retraining.

  MODEL-LIMITED  average precision is close to the base rate, i.e. the
                 probabilities barely separate positives from negatives.
                 Moving the threshold trades precision for recall and cannot
                 create F1. Retraining or a different formulation is the only
                 route.

The honest number is the CROSS-FITTED F1, not the oracle. Picking the best
threshold on the same data you score is circular -- and doubly so here,
because the deployed thresholds.npy were themselves tuned on val, so val
already flatters the current operating point. This script tunes each
threshold on one author-disjoint half and scores it on the other (both
directions), which is what a real threshold change would actually deliver.

Usage (CPU, after predict_factors_nli.py --dump-probs):
  python src/factor_threshold_diagnostic.py --probs artifacts/nli_factors_val_probs.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors
from rule_ablation_v2 import author_split

MIN_POS = 5          # below this, a label is unmeasurable on this split
AP_LIFT_GOOD = 2.0   # AP >= this x base rate means the ranking carries signal


def best_threshold(y, p, grid):
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    return best_t, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help=".npz from --dump-probs")
    ap.add_argument("--gold", default="artifacts/val_split.csv")
    ap.add_argument("--gold-col", default="factors")
    args = ap.parse_args()

    d = np.load(args.probs, allow_pickle=True)
    probs, ths = d["probs"], d["thresholds"]
    rids = list(d["row_id"])

    g = pd.read_csv(args.gold)
    gmap = dict(zip(g["row_id"], g[args.gold_col]))
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

    rows = []
    for j, name in enumerate(tax):
        y, p = Y[:, j], probs[:, j]
        npos = int(y.sum())
        cur_f1 = f1_score(y, (p >= ths[j]).astype(int), zero_division=0)
        base = npos / len(y) if len(y) else 0.0

        if npos < MIN_POS:
            rows.append(dict(label=name, npos=npos, cur=cur_f1, ap=np.nan,
                             oracle=np.nan, cross=np.nan, thr=ths[j],
                             newthr=np.nan, verdict="UNMEASURABLE"))
            continue

        ap_ = average_precision_score(y, p)
        _, oracle = best_threshold(y, p, grid)

        # cross-fitted: tune on one author half, score on the other
        preds = np.zeros_like(y)
        ok = True
        for tune, score in ((A, ~A), (~A, A)):
            if y[tune].sum() < 2 or y[score].sum() < 1:
                ok = False
                break
            t, _ = best_threshold(y[tune], p[tune], grid)
            preds[score] = (p[score] >= t).astype(int)
        cross = f1_score(y, preds, zero_division=0) if ok else np.nan
        newthr, _ = best_threshold(y, p, grid)

        lift = ap_ / base if base else 0.0
        if lift >= AP_LIFT_GOOD and (oracle - cur_f1) > 0.05:
            verdict = "THRESHOLD"
        elif lift < AP_LIFT_GOOD:
            verdict = "MODEL-LIMITED"
        else:
            verdict = "near-optimal"
        rows.append(dict(label=name, npos=npos, cur=cur_f1, ap=ap_,
                         oracle=oracle, cross=cross, thr=ths[j],
                         newthr=newthr, verdict=verdict))

    df = pd.DataFrame(rows).sort_values("cur")
    print(f"posts {len(Y)}   current MACRO-F1 {np.mean([r['cur'] for r in rows]):.4f}")
    print(f"author halves: A={int(A.sum())} B={int((~A).sum())} (disjoint)\n")
    print(f"{'factor':<44}{'pos':>5}{'curF1':>7}{'AP':>7}{'lift':>6}"
          f"{'oracle':>8}{'cross':>7}{'thr':>6}{'->':>6}  verdict")
    for _, r in df.iterrows():
        fmt = lambda v: "  n/a" if pd.isna(v) else f"{v:.3f}"
        lift = "" if pd.isna(r.ap) else f"{r.ap / (r.npos / len(Y)):.1f}"
        print(f"{r.label:<44}{r.npos:>5}{r.cur:>7.3f}{fmt(r.ap):>7}{lift:>6}"
              f"{fmt(r.oracle):>8}{fmt(r.cross):>7}{r.thr:>6.2f}"
              f"{('' if pd.isna(r.newthr) else f'{r.newthr:.2f}'):>6}  {r.verdict}")

    meas = df[df.verdict != "UNMEASURABLE"]
    thr_fix = meas[meas.verdict == "THRESHOLD"]
    print(f"\n{'=' * 96}")
    print(f"UNMEASURABLE on this split : {int((df.verdict == 'UNMEASURABLE').sum())} "
          f"labels -- {list(df[df.verdict == 'UNMEASURABLE'].label)}")
    print(f"  these cannot be tuned or even scored here; they need the full "
          f"1635-post labeled set (OOF) to get enough positives.")
    print(f"THRESHOLD-fixable          : {len(thr_fix)} labels")
    print(f"MODEL-LIMITED              : {int((meas.verdict == 'MODEL-LIMITED').sum())} labels")

    if len(thr_fix):
        naive = (thr_fix.oracle - thr_fix.cur).sum() / len(tax)
        honest = (thr_fix.cross.fillna(thr_fix.cur) - thr_fix.cur).sum() / len(tax)
        print(f"\nS2 gain if THRESHOLD labels are retuned:")
        print(f"  oracle (CIRCULAR, upper bound only) : {naive:+.4f} S2 "
              f"= {0.3 * naive:+.4f} composite")
        print(f"  cross-fitted (honest estimate)      : {honest:+.4f} S2 "
              f"= {0.3 * honest:+.4f} composite")


if __name__ == "__main__":
    main()
