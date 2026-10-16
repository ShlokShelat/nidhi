#!/usr/bin/env python
"""Is `meaning in life` failing because the ANNOTATION GUIDELINE sends its
vocabulary to `hopelessness`? Zero GPU, minutes.

WHY THIS EXISTS. `meaning in life` is the worst factor label (F1 ~0.324) and
the five protective factors average ~0.576 against ~0.694 for the 19 risk
factors, so this label sits on the largest deficit in the factor pool.

THE MECHANISM, from the upstream annotation guideline (Li et al. 2022, HKIE
Trans. 29(4), Table A1 -- the source the PFA taxonomy is built from):

    Hopelessness := "Hopelessness, pointlessness, MEANINGLESSNESS, emptiness,
                     feeling stuck/trapped/in a cycle, etc."

So a post saying life is meaningless is gold-POSITIVE for `hopelessness` and
gold-NEGATIVE for `meaning in life`. The hypothesis for meaning-in-life is the
organizers' Table III string -- "Involves cognitive component, motivational
component and affective component" -- which describes a psychometric construct
rather than anything a person writes, so an NLI cross-encoder has little to
match on EXCEPT the token "meaning". That predicts systematic false positives
on the single most common phrasing in the corpus, which is exactly what
destroys F1 on a rare label.

Wang et al. 2022 (Front. Psychiatry 13:984922), the actual source of the 5
protective factors, unpacks it into three entailable propositions:
  cognitive    -- making sense of one's experiences in life
  motivational -- pursuit and attainment of worthwhile goals
  affective    -- satisfaction and happiness accompanying goal achievement

WHAT THIS SCRIPT DECIDES. Retraining an arm with better hypotheses costs ~2 GPU
hours and carries train/serve risk. Do not spend that on a story. Measure:

    of the meaning-in-life FALSE POSITIVES, what fraction carry gold
    `hopelessness`?

  >40%  -> contamination confirmed on the measured data; the retrain is
          de-risked and targeted.
  <40%  -> 0.324 has some other cause. Find it before burning the GPU.

Also reports the same overlap against `sense of responsibility` (Wang shows ML
and SR share a source instrument, the Reasons-for-Living scale, so the overlap
is baked into the taxonomy), and the recall side, showing whether the label
fails by over-firing, under-firing, or both.

Read-only.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_threshold_decode import (  # noqa: E402
    N, best_threshold_and_f1, cells, load_oof, logit_mean)
from scoring import FACTOR_TAXONOMY  # noqa: E402

SHIPPED = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli",
           "ltail", "friendfac", "cbnsynapi"]
IDX = {t.strip().lower(): i for i, t in enumerate(FACTOR_TAXONOMY)}


def col(name):
    j = IDX.get(name.strip().lower())
    if j is None:
        raise SystemExit(f"'{name}' not in the taxonomy: {FACTOR_TAXONOMY}")
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=SHIPPED)
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--target", default="meaning in life")
    ap.add_argument("--suspects", nargs="+",
                    default=["hopelessness", "sense of responsibility",
                             "psychological capital", "coping strategy",
                             "low self-esteem"])
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    P = logit_mean(S)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], len(rid))

    th = np.zeros(N)
    for j in range(N):
        th[j], _ = best_threshold_and_f1(P[:, j], Y[:, j])
    pred = (P >= th[None, :]).astype(int)

    t = col(args.target)
    yt, pt = Y[:, t], pred[:, t]
    tp = int(((pt == 1) & (yt == 1)).sum())
    fp = int(((pt == 1) & (yt == 0)).sum())
    fn = int(((pt == 0) & (yt == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    print(f"{len(rid)} OOF posts, {len(args.arms)} arms, tfit thresholds")
    print(f"\nTARGET: '{args.target}'")
    print(f"  gold positives {int(yt.sum())}   predicted {int(pt.sum())}")
    print(f"  TP {tp}  FP {fp}  FN {fn}")
    print(f"  precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}")
    if fp == 0:
        print("\n  no false positives -- this label fails by RECALL only, so "
              "the contamination story is wrong and a hypothesis rewrite is "
              "not indicated.")
        return

    print(f"\nWHERE THE {fp} FALSE POSITIVES LAND (gold co-labels):")
    print(f"  {'suspect label':<30}{'n':>6}{'% of FP':>10}"
          f"{'base rate':>11}{'lift':>8}")
    rows = []
    for s in args.suspects:
        j = col(s)
        n = int(((pt == 1) & (yt == 0) & (Y[:, j] == 1)).sum())
        share = n / fp
        base = float(Y[:, j].mean())
        rows.append((s, n, share, base, share / max(base, 1e-9)))
    for s, n, share, base, lift in sorted(rows, key=lambda r: -r[2]):
        print(f"  {s:<30}{n:>6}{100*share:>9.1f}%{100*base:>10.1f}%"
              f"{lift:>8.2f}x")

    hop = next(r for r in rows if r[0].strip().lower() == "hopelessness")
    print()
    print("=" * 70)
    print(f"HOPELESSNESS carries {100*hop[2]:.1f}% of the false positives "
          f"(base rate {100*hop[3]:.1f}%, lift {hop[4]:.2f}x)")
    if hop[2] > 0.40:
        print("\nCONTAMINATION CONFIRMED on the measured data. The guideline sends")
        print("'meaninglessness' to hopelessness while the hypothesis is an")
        print("empty construct description, so the arm fires on exactly the")
        print("wrong sentences. The retrain is targeted and de-risked:")
        print("  meaning in life -> making sense of one's experiences /")
        print("                     pursuing worthwhile goals / satisfaction")
        print("                     from achieving something")
        print("  and it must NOT entail on meaninglessness or emptiness.")
    elif hop[4] > 1.5:
        print("\nPARTIAL: enriched over base rate but under the 40% bar. The")
        print("mechanism is real but is not the main cause; check the recall")
        print("side below before committing GPU.")
    else:
        print("\nNOT CONFIRMED. The false positives are not concentrated on")
        print("hopelessness, so the empty-hypothesis story does not explain")
        print("0.324. Do NOT spend the 2 GPU-hours on this rewrite; find the")
        print("actual cause first.")
    print("=" * 70)

    # ---- the recall side ---------------------------------------------------
    print(f"\nTHE {fn} MISSES (gold {args.target}, predicted 0):")
    print(f"  mean pooled prob on missed golds  {P[(pt == 0) & (yt == 1), t].mean():.3f}")
    print(f"  mean pooled prob on true golds    {P[yt == 1, t].mean():.3f}")
    print(f"  threshold                         {th[t]:.3f}")
    for s in args.suspects:
        j = col(s)
        n = int(((pt == 0) & (yt == 1) & (Y[:, j] == 1)).sum())
        print(f"    also gold '{s}': {n}/{fn} ({100*n/max(fn,1):.0f}%)")

    # ---- is the whole protective block worse, or just this label? ----------
    PROT = ["social support", "coping strategy", "psychological capital",
            "sense of responsibility", "meaning in life"]
    pj = [col(p) for p in PROT]
    rj = [j for j in range(N) if j not in pj]

    def f1_of(j):
        a, b = pred[:, j], Y[:, j]
        tp_ = int(((a == 1) & (b == 1)).sum())
        den = 2 * tp_ + int(((a == 1) & (b == 0)).sum()) \
            + int(((a == 0) & (b == 1)).sum())
        return 0.0 if den == 0 else 2 * tp_ / den

    print(f"\nPROTECTIVE vs RISK-FACTOR split (this pool, tfit):")
    print(f"  protective (5)  mean F1 {np.mean([f1_of(j) for j in pj]):.4f}")
    print(f"  risk factors (19) mean F1 {np.mean([f1_of(j) for j in rj]):.4f}")
    print("  per protective label:")
    for p, j in zip(PROT, pj):
        print(f"    {p:<28} F1 {f1_of(j):.3f}   gold n {int(Y[:, j].sum())}")
    print("\nA 0.1 F1 gain on ONE protective label is worth 0.1/24 = +0.0042")
    print("macro = +0.0013 composite. Fixing all five to risk-factor parity")
    print("is worth about +0.0074 composite -- the largest single lever left")
    print("in the factor pool.")


if __name__ == "__main__":
    main()
