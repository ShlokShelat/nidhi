"""
risk_error_anatomy.py

Characterises WHERE a risk model's out-of-fold errors fall, so that effort
goes to defects a fix can actually reach.

The risk classes are ORDINAL (indicator < ideation < behavior < attempt), so
the direction and the size of an error carry information that a flat accuracy
number throws away. The report has five parts:

  1. the full 4x4 confusion, in counts and as a share of each gold class
  2. how much of the error is ADJACENT on the ordinal scale vs a long jump.
     Adjacent errors are a threshold/calibration problem; long jumps are a
     comprehension problem, and the two need different fixes.
  3. the model's CONFIDENCE on its errors, per confusion cell. A cell whose
     errors sit at high probability cannot be reached by any change to the
     decode -- the model is confidently wrong there. Cells with mass near the
     boundary are where a rule or threshold can still act.
  4. what the weighted-F1 ARITHMETIC says eliminating each cell is worth.
     The metric is support-weighted, so the ugliest error rate is not
     necessarily the one that pays.
  5. whether the errors concentrate in particular AUTHORS. If a small number
     of authors carry a large share, the variation is annotation style, which
     no per-post model can fix and which caps the attainable ceiling.

Usage:
  python src/risk_error_anatomy.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
RANK = {c: i for i, c in enumerate(LEVELS)}


def wf1(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    tot = 0.0
    for c in LEVELS:
        g, p = gold == c, pred == c
        tp = int((g & p).sum())
        den = 2 * tp + int((~g & p).sum()) + int((g & ~p).sum())
        tot += (0.0 if den == 0 else 2.0 * tp / den) * g.sum()
    return float(tot / max(len(gold), 1))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    args = a_.parse_args()

    oof = pd.read_csv(args.oof, keep_default_na=False)
    oof["row_id"] = oof["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rows = [r for r in oof["row_id"] if r in lab.index]
    ix = oof.set_index("row_id")
    P = np.stack([ix.loc[rows, f"prob_{c}"].astype(float).to_numpy()
                  for c in LEVELS], axis=1)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    pred = np.array(LEVELS)[P.argmax(1)]
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    top = P.max(1)
    n = len(rows)

    print(f"{n} OOF posts   argmax risk wF1 {wf1(gold, pred):.4f}\n")

    print("=" * 84)
    print("1. CONFUSION -- rows are GOLD, columns are PREDICTED")
    print("=" * 84)
    hdr = "gold v / pred >"
    print(f"{hdr:<16}" + "".join(f"{c:>12}" for c in LEVELS)
          + f"{'n':>7}{'recall':>8}")
    M = np.zeros((4, 4), int)
    for g, p in zip(gold, pred):
        M[RANK[g], RANK[p]] += 1
    for i, c in enumerate(LEVELS):
        tot = M[i].sum()
        print(f"{c:<16}" + "".join(f"{M[i, j]:>12}" for j in range(4))
              + f"{tot:>7}{M[i, i]/max(tot,1):>8.3f}")
    print(f"{'precision':<16}" + "".join(
        f"{M[j, j]/max(M[:, j].sum(),1):>12.3f}" for j in range(4)))

    print("\n" + "=" * 84)
    print("2. ORDINAL DISTANCE -- adjacent slip or long jump?")
    print("=" * 84)
    d = np.array([RANK[p] - RANK[g] for g, p in zip(gold, pred)])
    err = d != 0
    print(f"  errors {int(err.sum())} of {n} ({err.mean():.1%})")
    for k in (-3, -2, -1, 1, 2, 3):
        m = d == k
        if m.sum():
            print(f"    {k:+d} step{'s' if abs(k)>1 else ' '}  "
                  f"{int(m.sum()):>4}  ({m.sum()/max(err.sum(),1):>5.1%} "
                  f"of errors)"
                  + ("   under-estimates severity" if k < 0 else
                     "   over-estimates severity"))
    print(f"\n  adjacent (+-1): {int((np.abs(d) == 1).sum())} "
          f"({(np.abs(d) == 1).sum()/max(err.sum(),1):.1%} of errors)")
    print(f"  net severity bias: {d.mean():+.3f} steps per post "
          f"({'UNDER' if d.mean() < 0 else 'OVER'}-estimating)")
    print("  Adjacent errors are a calibration/threshold problem.")
    print("  Long jumps are a comprehension problem. They need different fixes.")

    print("\n" + "=" * 84)
    print("3. CONFIDENCE ON ERRORS -- what could a decode even reach?")
    print("=" * 84)
    print(f"{'cell':<26}{'n':>6}{'mean top p':>12}{'p>0.7':>8}{'p<0.5':>8}")
    for i, gc in enumerate(LEVELS):
        for j, pc in enumerate(LEVELS):
            if i == j or M[i, j] == 0:
                continue
            m = (gold == gc) & (pred == pc)
            t = top[m]
            print(f"{gc[:9]+' -> '+pc[:9]:<26}{int(m.sum()):>6}"
                  f"{t.mean():>12.3f}{(t > 0.7).mean():>8.0%}"
                  f"{(t < 0.5).mean():>8.0%}")
    m = gold == pred
    print(f"{'(correct)':<26}{int(m.sum()):>6}{top[m].mean():>12.3f}"
          f"{(top[m] > 0.7).mean():>8.0%}{(top[m] < 0.5).mean():>8.0%}")
    print("\n  A cell where most errors sit above p=0.7 is unreachable by any")
    print("  re-decoding -- the model is confidently wrong there. Cells with a")
    print("  high p<0.5 share are where a rule or threshold can still act.")

    print("\n" + "=" * 84)
    print("4. WHAT IS EACH FIX WORTH? (weighted F1 is support-weighted)")
    print("=" * 84)
    base = wf1(gold, pred)
    print(f"  baseline {base:.4f}")
    for i, gc in enumerate(LEVELS):
        for j, pc in enumerate(LEVELS):
            if i == j or M[i, j] < 3:
                continue
            fixed = pred.copy()
            fixed[(gold == gc) & (pred == pc)] = gc
            print(f"  fix ALL {gc[:9]:>9} -> {pc[:9]:<9} "
                  f"({M[i, j]:>3} posts): wF1 {wf1(gold, fixed):.4f}  "
                  f"{wf1(gold, fixed)-base:+.4f}  "
                  f"composite {0.4*(wf1(gold, fixed)-base):+.4f}")
    print("\n  This is the ORACLE value of eliminating each confusion cell.")
    print("  Nothing reaches it, but it ranks where effort could pay.")

    print("\n" + "=" * 84)
    print("5. DO ERRORS CONCENTRATE IN AUTHORS?")
    print("=" * 84)
    df = pd.DataFrame({"a": auth, "e": err.astype(int)})
    g = df.groupby("a")["e"].agg(["mean", "size"])
    g = g[g["size"] >= 5].sort_values("mean", ascending=False)
    print(f"  {len(g)} authors with >=5 posts, error rate "
          f"median {g['mean'].median():.1%}, "
          f"p90 {g['mean'].quantile(0.9):.1%}, max {g['mean'].max():.1%}")
    worst = g.head(8)
    print("  worst authors: " + ", ".join(
        f"{i}({r['mean']:.0%} of {int(r['size'])})"
        for i, r in worst.iterrows()))
    share = (g.head(int(0.1*len(g)))["mean"] *
             g.head(int(0.1*len(g)))["size"]).sum() / max(err.sum(), 1)
    print(f"  the worst 10% of authors carry {share:.0%} of all errors")
    print("  Far above 10% means annotation STYLE varies by author, and no")
    print("  per-post model can fix it -- which would also cap the ceiling.")


if __name__ == "__main__":
    main()
