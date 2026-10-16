"""
compare_vs_champion_risk.py

Compares a candidate risk arm against the risk system it would ACTUALLY be
replacing.

WHY THE OPPONENT MATTERS.

  A single-model "control" is not the deployed risk column. The champion's
  risk column is an ENSEMBLE (risk_blend_equal) with a symbolic layer on top,
  which is materially stronger than any one of its members. An arm that beats
  a single control model by a small margin therefore says nothing about
  whether it beats deployment, and adopting it on that basis can lose ground
  even though the control comparison looked positive.

  This script removes that ambiguity by scoring the arm against the champion's
  own predictions on the same posts.

WHY THIS COMPARISON IS FAIR.

  artifacts/val_champion_reconstructed.csv holds the champion pipeline's
  predictions on the 372-post val split. Those models were trained on
  train_split, which EXCLUDES val -- so they are honest held-out predictions.

  An arm's OOF predictions are also honest held-out predictions: each post was
  predicted by a fold model that never saw it.

  Both therefore give unbiased predictions for the SAME 372 posts, scored
  against the same gold, and are directly comparable. The only asymmetry is
  training-set size (champion ~1263 posts, OOF folds ~1305), which slightly
  favours the arm -- so an arm that loses here loses honestly.

READ THE RESULT LIKE THIS.

  arm clearly ahead   -> the arm is genuinely stronger than the deployed risk
                         column on held-out posts.
  arm clearly behind  -> the arm is simply worse than deployment; no further
                         explanation of a downstream loss is needed, and any
                         margin it held over a single-model control was
                         measured against an opponent that does not matter.
  roughly level       -> the arm cannot beat the champion by enough to justify
                         replacing it, though a level arm that DISAGREES a lot
                         may still be useful as an ensemble member.

Usage:
  python src/compare_vs_champion_risk.py \
      --arms 'base=v2_rebuild/runs/risk_llm_Qwen2.5-14B-Instruct_e6_f{f}' \
             'ord=v2_rebuild/runs/risk_llm2_ord_f{f}'
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1


def load_arm(tmpl, k=5):
    """Concatenate whatever folds exist. Arms that skipped fold 0 are fine --
    the comparison is restricted to the intersection later."""
    parts = []
    for f in range(k):
        p = os.path.join(tmpl.replace("{f}", str(f)), "oof_predictions.csv")
        if os.path.exists(p):
            parts.append(pd.read_csv(p)[["row_id", "pred"]])
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True)
    return dict(zip(d["row_id"], [normalize_label(x) for x in d["pred"]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion",
                    default="artifacts/val_champion_reconstructed.csv")
    ap.add_argument("--champion-col", default="champion_final")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="name=path/with/{f} entries")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    ch = pd.read_csv(args.champion)
    gold = {r: normalize_label(g) for r, g in zip(ch["row_id"], ch["gold"])}
    champ = {r: normalize_label(p)
             for r, p in zip(ch["row_id"], ch[args.champion_col])}
    print(f"champion predictions on {len(champ)} val posts "
          f"({args.champion_col})")

    arms = {}
    for spec in args.arms:
        name, tmpl = spec.split("=", 1)
        m = load_arm(tmpl, args.k)
        if m is None:
            print(f"  WARNING: no fold files for arm '{name}' at {tmpl}")
            continue
        arms[name] = m
        print(f"  arm '{name}': {len(m)} OOF predictions")

    if not arms:
        raise SystemExit("no arms loaded")

    # Restrict to posts EVERY arm and the champion cover, so all numbers below
    # are computed on identical rows. Comparing arms on different subsets is
    # the easiest way to manufacture a difference that is not there.
    rows = set(champ) & set(gold)
    for m in arms.values():
        rows &= set(m)
    rows = sorted(rows)
    if not rows:
        raise SystemExit("no overlapping posts between champion val and OOF")
    print(f"\ncomparing on {len(rows)} posts common to the champion and "
          f"all {len(arms)} arm(s)")

    g = [gold[r] for r in rows]
    base = risk_weighted_f1(g, [champ[r] for r in rows])

    print(f"\n{'system':<14}{'weighted-F1':>13}{'vs champion':>13}"
          f"{'agree w/ champ':>16}")
    print("-" * 56)
    print(f"{'CHAMPION':<14}{base:>13.4f}{'--':>13}{'--':>16}")
    for name, m in arms.items():
        p = [m[r] for r in rows]
        f1 = risk_weighted_f1(g, p)
        agree = float(np.mean([m[r] == champ[r] for r in rows]))
        print(f"{name:<14}{f1:>13.4f}{f1 - base:>+13.4f}{100 * agree:>15.1f}%")

    # Where the champion is right and the arm is wrong, and vice versa. A model
    # that is level overall but disagrees a lot is still ensemble material; one
    # that is level and agrees everywhere adds nothing.
    print(f"\n{'arm':<14}{'arm right, champ wrong':>24}"
          f"{'champ right, arm wrong':>24}")
    for name, m in arms.items():
        aw = sum(1 for r in rows if m[r] == gold[r] and champ[r] != gold[r])
        cw = sum(1 for r in rows if champ[r] == gold[r] and m[r] != gold[r])
        print(f"{name:<14}{aw:>24}{cw:>24}")

    print("\n" + "=" * 70)
    print("An arm behind the champion here is simply weaker than the deployed")
    print("risk column, whatever it scored against a single-model control.")
    print("The bar for replacing the risk column is beating the CHAMPION on")
    print("this table -- not beating a control.")
    print("=" * 70)


if __name__ == "__main__":
    main()
