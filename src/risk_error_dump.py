"""
risk_error_dump.py

Dumps the risk model's errors as readable posts, grouped by confusion cell,
so that the ceiling question is settled by inspection rather than inference.

Aggregate statistics cannot distinguish a fixable defect from label noise:
both show up as a dense confusion cell. Only the posts can. The published
inter-annotator agreement for this label scheme is 0.79 pairwise between four
practicing psychiatrists (0.73 group-wise), so some irreducible share of any
confusion cell is disagreement rather than model error.

WHAT DECIDES IT. If the errors in a cell share a describable pattern -- a
phrasing, a tense, a construction -- that is a fixable defect and the cell is
worth attacking. If they read as posts where a careful annotator could
reasonably have gone either way, the cell is label noise and no model
reaches it.

The distinction is concrete for this task:
  ideation vs behavior   turns on whether a PLAN or SELF-HARM is present. Look
                         for hedged plans ("maybe I'll", "thought about how")
                         and for self-harm mentioned in passing.
  behavior vs attempt    turns on whether an act was CARRIED OUT and is PAST.
                         Tense and completion are the whole question.
  indicator vs ideation  turns on whether suicide is mentioned EXPLICITLY.
                         Metaphor ("done with it all") is the grey zone.

Sorted by CONFIDENCE so the most confident errors come first -- those are the
ones a better model would have to overturn, and if they look ambiguous to a
reader then they are not model failures.

Usage:
  python src/risk_error_dump.py
  python src/risk_error_dump.py --cell ideation behavior --show 15
  python src/risk_error_dump.py --enc artifacts/lora_ql70_oof.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label  # noqa: E402

PC = [f"prob_{c}" for c in RISK_LEVELS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--cell", nargs=2, default=None,
                    metavar=("GOLD", "PRED"),
                    help="dump only this confusion cell in depth")
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--chars", type=int, default=420)
    args = ap.parse_args()

    d = pd.read_csv(args.enc, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    d = d.drop_duplicates("row_id").set_index("row_id")
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    rows = [r for r in d.index if r in lab.index]
    have_probs = all(c in d.columns for c in PC)
    col = next((c for c in ("pred_risk_level", "risk_level")
                if c in d.columns), None)
    if have_probs:
        M = d.loc[rows, PC].to_numpy(float)
        M = M / np.maximum(M.sum(1, keepdims=True), 1e-9)
        pred = [RISK_LEVELS[i] for i in M.argmax(1)]
        conf = M.max(1)
        marg = np.sort(M, 1)[:, -1] - np.sort(M, 1)[:, -2]
    elif col is not None:
        pred = [normalize_label(d.loc[r, col]) for r in rows]
        conf = np.ones(len(rows))
        marg = np.ones(len(rows))
    else:
        raise SystemExit(f"{args.enc}: no probs and no label column")
    gold = [normalize_label(lab.loc[r, "risk_level"]) for r in rows]

    wrong = [i for i in range(len(rows)) if gold[i] != pred[i]]
    print(f"{len(rows)} posts   {len(wrong)} errors "
          f"({len(wrong)/len(rows):.1%})   source {args.enc}\n")

    print("=" * 88)
    print("CONFUSION -- rows = gold, cols = predicted")
    print("=" * 88)
    print(f"{'':<12}" + "".join(f"{c[:9]:>11}" for c in RISK_LEVELS) +
          f"{'recall':>9}")
    cells = Counter((gold[i], pred[i]) for i in range(len(rows)))
    for gc in RISK_LEVELS:
        tot = sum(v for (a, _), v in cells.items() if a == gc)
        line = f"{gc:<12}"
        for pcl in RISK_LEVELS:
            v = cells[(gc, pcl)]
            line += f"{v:>11}" if v else f"{'.':>11}"
        line += f"{cells[(gc, gc)]/max(tot,1):>9.3f}"
        print(line)

    print("\n" + "=" * 88)
    print("ERRORS BY CELL, with mean confidence -- high confidence means the")
    print("model is not hesitating, which is what label noise looks like")
    print("=" * 88)
    print(f"{'gold -> pred':<28}{'n':>5}{'mean conf':>11}{'mean margin':>13}")
    order = sorted(((a, b) for (a, b) in cells if a != b),
                   key=lambda k: -cells[k])
    for a, b in order:
        idx = [i for i in wrong if gold[i] == a and pred[i] == b]
        if not idx:
            continue
        print(f"{a + ' -> ' + b:<28}{len(idx):>5}"
              f"{np.mean([conf[i] for i in idx]):>11.3f}"
              f"{np.mean([marg[i] for i in idx]):>13.3f}")

    todo = [tuple(args.cell)] if args.cell else order[:3]
    posts = lab.loc[rows, "post"].astype(str).to_numpy()
    for a, b in todo:
        idx = [i for i in wrong if gold[i] == a and pred[i] == b]
        if not idx:
            print(f"\n(no errors in {a} -> {b})")
            continue
        idx.sort(key=lambda i: -conf[i])
        print("\n" + "=" * 88)
        print(f"GOLD {a.upper()}  ->  PREDICTED {b.upper()}    "
              f"({len(idx)} posts, most confident first)")
        print("=" * 88)
        for i in idx[: args.show]:
            print(f"\n  [{rows[i]}]  conf {conf[i]:.3f}  margin {marg[i]:.3f}")
            print(f"  {' '.join(posts[i].split())[:args.chars]}")

    print("\n" + "=" * 88)
    print("THE QUESTION TO ANSWER WHILE READING")
    print("=" * 88)
    print("Do the posts in a cell share a DESCRIBABLE pattern -- a phrasing, a")
    print("tense, a construction? Then it is a fixable defect and the cell is")
    print("worth attacking with a targeted hypothesis or rule.")
    print("")
    print("Or would a careful annotator plausibly have gone either way? Then it")
    print("is label noise. Four psychiatrists agreed pairwise at only 0.79 on")
    print("this scheme, so some of what is below must be disagreement rather")
    print("than model error -- the question is how much.")
    print("")
    print("The cells to read hardest are the ones with HIGH mean confidence:")
    print("those are the errors a better model would have to overturn.")


if __name__ == "__main__":
    main()
