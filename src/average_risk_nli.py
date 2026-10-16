"""
average_risk_nli.py

Average two or more risk-NLI out-of-fold probability files into one, so a
multi-architecture NLI member can be fed to src/champion_oof.py --nli.

WHY. Two NLI checkpoints built on different encoder architectures make
partially independent mistakes, so averaging their probabilities cancels the
uncorrelated component of the error. That is a stronger reason to average than
identical-architecture seed variation, because the errors are less correlated
to begin with. The risk NLI slot carries a single architecture, so it is the
slot where adding a second one has the most to cancel.

WHAT THIS FITS. Nothing. Equal weight per member, no tuning -- so there is no
parameter here that could have been chosen on the evaluation data.

CHECKS. Refuses to run unless every file covers the same row_ids, aligns by
row_id rather than trusting row order, and verifies each row's probabilities
sum to 1 before and after.

Usage:
  python src/average_risk_nli.py \
      --inputs artifacts/nli_risk_mnli_oof_all.csv \
               artifacts/nli_risk_dmnli_oof_all.csv \
      --out    artifacts/nli_risk_mnlipair_oof_all.csv

  python src/champion_oof.py --nli artifacts/nli_risk_mnlipair_oof_all.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

COLS = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]
LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labeled", default="artifacts/full_train.csv",
                    help="optional, only used to report agreement with gold")
    args = ap.parse_args()

    frames = []
    for p in args.inputs:
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = pd.read_csv(p)
        for c in ["row_id"] + COLS:
            if c not in d.columns:
                raise SystemExit(f"{p} lacks column {c!r}; found {list(d.columns)}")
        d = d.set_index("row_id")
        s = d[COLS].sum(1)
        if not np.allclose(s, 1.0, atol=1e-3):
            raise SystemExit(f"{p}: probabilities do not sum to 1 "
                             f"(min {s.min():.4f}, max {s.max():.4f})")
        frames.append(d)
        print(f"  {p}   {len(d)} rows")

    base = frames[0].index
    for p, d in zip(args.inputs[1:], frames[1:]):
        if set(d.index) != set(base):
            raise SystemExit(f"{p} covers a different set of row_ids")

    acc = np.zeros((len(base), len(COLS)), dtype=np.float64)
    for d in frames:
        acc += d.loc[base, COLS].to_numpy(dtype=np.float64)
    P = acc / len(frames)

    out = pd.DataFrame(P, columns=COLS)
    out.insert(0, "row_id", list(base))
    s = out[COLS].sum(1)
    if not np.allclose(s, 1.0, atol=1e-6):
        raise SystemExit("internal error: averaged rows do not sum to 1")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\naveraged {len(frames)} members over {len(base)} posts "
          f"-> {args.out}")

    # How often do the members disagree? Diversity is the whole point -- two
    # members that always agree cannot cancel each other's mistakes, and an
    # ensemble of them will be worth nothing.
    preds = [np.array(LEVELS)[d.loc[base, COLS].to_numpy().argmax(1)]
             for d in frames]
    if len(preds) > 1:
        agree = np.ones(len(base), dtype=bool)
        for q in preds[1:]:
            agree &= (q == preds[0])
        print(f"members agree on {100 * agree.mean():.1f}% of posts "
              f"({int((~agree).sum())} disagreements)")
        print("  Low agreement is GOOD here -- identical members cannot help "
              "each other.\n  Above ~95% and the pair will add little.")

    if os.path.exists(args.labeled):
        lab = pd.read_csv(args.labeled).set_index("row_id")
        if "risk_level" in lab.columns:
            gold = lab.loc[base, "risk_level"].astype(str).str.lower().to_numpy()
            from sklearn.metrics import f1_score
            print()
            for p, q in zip(args.inputs, preds):
                print(f"  {os.path.basename(p):<40} wF1 "
                      f"{f1_score(gold, q, average='weighted'):.4f}")
            avg = np.array(LEVELS)[P.argmax(1)]
            print(f"  {'AVERAGED':<40} wF1 "
                  f"{f1_score(gold, avg, average='weighted'):.4f}")
            print("\n  These are NLI-alone figures over every labelled post. "
                  "What matters is what\n  they do inside the blend -- run "
                  "src/champion_oof.py --nli <out> next.")


if __name__ == "__main__":
    main()
