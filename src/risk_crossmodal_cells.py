"""
risk_crossmodal_cells.py

DOES THE FACTOR MODEL SEE THE PLAN THAT THE RISK MODEL MISSED?

`risk_error_anatomy.py` found the dominant confusion cell:

    behavior -> ideation   89 posts
    indicator -> ideation  79 posts

88.2% of all risk errors are ONE ordinal step, so this is calibration, not
comprehension. And the C-SSRS definitions say precisely what separates the two
classes:

    Ideation  explicit suicidal expression, NO plan
    Behavior  explicit expression AND a plan / self-harm act

So every `behavior -> ideation` error is the risk model FAILING TO SEE A PLAN.
Meanwhile the plan is predicted directly as a factor -- `suicide means (with
access)`, pool AP 0.848 -- from an 8-arm ensemble that is NOT the risk model.
`prior self-harm or suicidal thought/attempt` (AP 0.816) is the other half of
the behavior criterion.

THE QUESTION THIS ANSWERS. On the 89 posts the risk model got wrong, does the
factor model already know? If its `suicide means` probability on those posts
looks like correctly-predicted BEHAVIOR posts, the information exists and is
being thrown away. If it looks like correctly-predicted IDEATION posts, the
plan is genuinely invisible in the text and no cross-modal rule can help.

That distinction has never been checked. `risk_crossmodal_stack.py` added
factor features to a GLOBAL 4-class stack and barely moved the result over
restacking the risk logits alone -- but a global stack spreads 96 parameters
across every class boundary. This asks whether the signal is there in ONE
cell.

IT ALSO SPLITS THE CELL BY CONFIDENCE, because the two halves need different
fixes: errors above p=0.7 are confidently wrong and unreachable by any
decode, errors below p=0.5 are where a rule can act.

Usage:
  python src/risk_crossmodal_cells.py
  python src/risk_crossmodal_cells.py --show 12
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
KEY = ["suicide means (with access)",
       "prior self-harm or suicidal thought/attempt",
       "hopelessness", "emotion dysregulation"]


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--facdir", default="artifacts/factor_runs")
    a_.add_argument("--show", type=int, default=8)
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
    top = P.max(1)
    posts = [str(lab.loc[r, "post"]) for r in rows]

    O, rid = [], None
    for a in BAG8:
        pr, ids = [], []
        for f in range(5):
            z = np.load(f"{args.facdir}/{a}_cv_oof_f{f}.npz", allow_pickle=True)
            pr.append(z["probs"].astype(float))
            ids.extend(str(x) for x in z["row_id"])
        pr = np.concatenate(pr)
        if rid is None:
            rid = ids
        elif ids != rid:
            o = {x: i for i, x in enumerate(ids)}
            pr = pr[[o[x] for x in rid]]
        O.append(pr)
    pool = 1.0 / (1.0 + np.exp(-np.mean(logit(np.stack(O)), 0)))
    o = {x: i for i, x in enumerate(rid)}
    F = pool[[o[r] for r in rows]]
    ki = {k: FACTOR_TAXONOMY.index(k) for k in KEY}

    print(f"{len(rows)} OOF posts. Factor probabilities come from the 8-arm "
          f"pool, which is NOT the risk model.\n")

    groups = {
        "gold behavior, pred behavior (CORRECT)":
            (gold == "behavior") & (pred == "behavior"),
        "gold behavior, pred ideation  (THE CELL)":
            (gold == "behavior") & (pred == "ideation"),
        "gold ideation, pred ideation  (CORRECT)":
            (gold == "ideation") & (pred == "ideation"),
        "gold ideation, pred behavior  (over-called)":
            (gold == "ideation") & (pred == "behavior"),
    }
    print("=" * 96)
    print("1. WHAT THE FACTOR MODEL SAYS, BY CONFUSION CELL")
    print("=" * 96)
    print(f"{'group':<42}{'n':>5}" + "".join(f"{k.split('(')[0][:13]:>15}"
                                             for k in KEY))
    for nm, m in groups.items():
        print(f"{nm:<42}{int(m.sum()):>5}"
              + "".join(f"{F[m, ki[k]].mean():>15.3f}" for k in KEY))
    print("\n  Read the second row against the first and third. If the missed")
    print("  BEHAVIOR posts carry `suicide means` like CORRECT behavior posts,")
    print("  the plan is visible to the factor model and the risk model is")
    print("  discarding it. If they look like correct IDEATION, it is not there.")

    mcell = (gold == "behavior") & (pred == "ideation")
    mok = (gold == "behavior") & (pred == "behavior")
    mid = (gold == "ideation") & (pred == "ideation")
    j = ki["suicide means (with access)"]
    a_, b_, c_ = F[mok, j], F[mcell, j], F[mid, j]
    print(f"\n  `suicide means` separation:")
    print(f"    correct behavior {a_.mean():.3f}   missed behavior "
          f"{b_.mean():.3f}   correct ideation {c_.mean():.3f}")
    if a_.mean() > c_.mean():
        pos = (b_.mean() - c_.mean()) / max(a_.mean() - c_.mean(), 1e-9)
        print(f"    the missed posts sit {pos:.0%} of the way from ideation "
              f"toward behavior")
        print("    >50% means the factor model DOES see the plan -- the signal "
              "is being wasted")
        print("    <20% means it does not, and no cross-modal rule can help")

    print("\n" + "=" * 96)
    print("2. THE CELL SPLIT BY RISK-MODEL CONFIDENCE")
    print("=" * 96)
    for lo, hi, nm in ((0.0, 0.5, "p<0.5   reachable by a rule"),
                       (0.5, 0.7, "0.5-0.7 borderline"),
                       (0.7, 1.01, "p>0.7   confidently wrong")):
        m = mcell & (top >= lo) & (top < hi)
        if not m.sum():
            continue
        print(f"  {nm:<30}{int(m.sum()):>4} posts   "
              f"suicide means {F[m, j].mean():.3f}   "
              f"prior self-harm "
              f"{F[m, ki['prior self-harm or suicidal thought/attempt']].mean():.3f}")
    print("\n  If the factor signal is strong even in the p>0.7 half, a rule")
    print("  could reach errors that re-decoding the risk model cannot.")

    print("\n" + "=" * 96)
    print(f"3. READ THE POSTS -- {args.show} confidently-wrong "
          f"behavior->ideation cases")
    print("=" * 96)
    idx = np.flatnonzero(mcell & (top > 0.7))
    idx = idx[np.argsort(-top[idx])][:args.show]
    for i in idx:
        print(f"\n  [{rows[i]}] p(ideation)={P[i, 1]:.2f} "
              f"p(behavior)={P[i, 2]:.2f}  "
              f"factor suicide-means={F[i, j]:.2f}")
        t = " ".join(posts[i].split())
        print(f"    {t[:300]}")
    print("\n  Look for a shared pattern: is the plan stated obliquely, in the")
    print("  past tense, hypothetically, or about someone else? A pattern here")
    print("  is a feature or a rule; no pattern means annotation noise.")


if __name__ == "__main__":
    main()
