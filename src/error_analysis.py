"""
Error analysis: turns a validation prediction file into a diagnosis.

Answers, in order of composite-score weight:
  1. Which risk-level confusions cost the most wF1? (0.4 weight)
  2. Is evidence loss coming from precision (bad spans) or recall (missed
     spans), and on which risk classes? (0.3 weight)
  3. Which factor labels are dragging macro F1, and is it a threshold
     problem or a "model never fires" problem? (0.3 weight)

Usage:
  python src/error_analysis.py --val-preds runs/risk_v1/val_predictions.csv \
      [--evidence-preds runs/evidence_v1/val_predictions.csv] \
      [--factor-probs runs/factors_v1/...]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scoring import (
    RISK_LEVELS, normalize_label, parse_evidence, phrase_f1, risk_weighted_f1,
)


def risk_confusion(df: pd.DataFrame) -> None:
    gold = df["risk_level"].map(normalize_label)
    pred = df["pred_risk_level"].map(normalize_label)
    print(f"\nweighted F1: {risk_weighted_f1(gold.tolist(), pred.tolist()):.4f}")
    cm = pd.crosstab(gold, pred, rownames=["gold"], colnames=["pred"])
    cm = cm.reindex(index=RISK_LEVELS, columns=RISK_LEVELS, fill_value=0)
    print("\nconfusion matrix (rows=gold):")
    print(cm.to_string())

    # cost accounting: which single confusion cell, if fixed, buys the most?
    n = len(df)
    print("\ntop confusions by mass (fixing these buys the most wF1):")
    off = [(g, p, cm.loc[g, p]) for g in RISK_LEVELS for p in RISK_LEVELS
           if g != p and cm.loc[g, p] > 0]
    for g, p, c in sorted(off, key=lambda x: -x[2])[:6]:
        print(f"  gold={g:9s} -> pred={p:9s}  {c:4d} posts ({100*c/n:.1f}%)")

    # adjacent vs distant errors: ordinal head should suppress distant ones
    sev = {l: i for i, l in enumerate(RISK_LEVELS)}
    dist = (gold.map(sev) - pred.map(sev)).abs()
    err = dist[dist > 0]
    if len(err):
        print(f"\nof {len(err)} errors: {(err == 1).mean()*100:.0f}% adjacent, "
              f"{(err > 1).mean()*100:.0f}% distant (distant errors indicate "
              f"the ordinal structure is not being exploited)")

    # low-confidence errors -> candidates for LLM-advisor second pass
    prob_cols = [f"prob_{l}" for l in RISK_LEVELS if f"prob_{l}" in df]
    if prob_cols:
        conf = df[prob_cols].max(axis=1)
        wrong = gold.values != pred.values
        print(f"\nmean confidence when correct: {conf[~wrong].mean():.3f}, "
              f"when wrong: {conf[wrong].mean():.3f}")
        thresh = np.percentile(conf, 20)
        frac_err_low = wrong[conf <= thresh].mean()
        print(f"bottom-20% confidence posts contain {frac_err_low*100:.0f}% error "
              f"rate -> size of the review/advisor queue worth building")


def evidence_analysis(df: pd.DataFrame) -> None:
    gold = [parse_evidence(c) for c in df["evidence"]]
    pred = [parse_evidence(c) for c in df["pred_evidence"]]
    res = phrase_f1(gold, pred)
    print(f"\nphrase F1: {res.f1:.4f}  P={res.precision:.4f}  R={res.recall:.4f}")
    if res.precision < res.recall:
        print("-> PRECISION-limited: model proposes spans that don't match "
              "(too long, wrong text, or over-extraction). Tighten decoding.")
    else:
        print("-> RECALL-limited: model misses gold spans. Check truncation "
              "(spans past max_len are unreachable) and B/I label quality.")

    if "risk_level" in df:
        per = pd.DataFrame(res.per_post)
        per["risk"] = df["risk_level"].map(normalize_label).values
        print("\nmean per-post span F1 by risk class:")
        print(per.groupby("risk")["f1"].agg(["mean", "count"]).to_string())

    worst = sorted(res.per_post, key=lambda p: p["f1"])[:5]
    print("\n5 worst posts (indices into val file) for manual inspection:",
          [w["idx"] for w in worst])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-preds", required=True,
                    help="val_predictions.csv from train_risk.py")
    ap.add_argument("--evidence-preds", default=None,
                    help="val_predictions.csv from train_evidence.py")
    args = ap.parse_args()

    df = pd.read_csv(args.val_preds)
    print("=" * 60, "\nRISK LEVEL (weight 0.4)")
    risk_confusion(df)

    if args.evidence_preds:
        edf = pd.read_csv(args.evidence_preds)
        print("\n" + "=" * 60, "\nEVIDENCE (weight 0.3)")
        evidence_analysis(edf)

    print("\n(factors diagnosis: per-label F1 table is printed by "
          "train_factors.py at the end of training; the lowest rows there "
          "are the augmentation to-do list)")


if __name__ == "__main__":
    main()
