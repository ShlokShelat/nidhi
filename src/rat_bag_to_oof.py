"""
rat_bag_to_oof.py

Convert the rationale bag's per-fold val_probs.npz into the oof_predictions.csv
format src/champion_oof.py expects, so the bag can be swapped in as the DeBERTa
member and judged by the deployed recipe.

WHY THE CONVERSION IS NEEDED. train_risk_rationale.py saves
runs/risk_rat_f<k>/val_probs.npz (probs + row_id); champion_oof.py's load_folds
reads runs/<dir>/oof_predictions.csv with columns
prob_indicator/prob_ideation/prob_behavior/prob_attempt. Same content, different
container.

WHY THIS IS THE ONLY TEST THAT COUNTS. The bag's pooled OOF weighted-F1 is
0.7784 over 1635 posts against `risk_ctrl`'s 0.7311 over 1590 -- structurally
the same regime (one fold model per post), so the comparison is meaningful.
But `risk_ctrl` may differ in other hyperparameters, so a favorable reading
on this instrument alone is not sufficient evidence. The number to beat is
the champion blend + rules at **0.7841** on the 1590-post intersection, with
the blend weights FROZEN.

NOTE the stored probabilities come from _combine(), which returns
log(0.5*cat_p + 0.5*ord_p); the trainer already exponentiated them. They are
renormalised here so any float drift cannot bias the argmax.

Usage:
  python src/rat_bag_to_oof.py --tag rat
  python src/champion_oof.py --deberta 'runs/risk_rat_f{f}' \
      --export artifacts/champion_oof_rat.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

PCOLS = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rat")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dir", default="runs/risk_{tag}_f{f}")
    args = ap.parse_args()

    total = 0
    for f in range(args.k):
        d = args.dir.format(tag=args.tag, f=f)
        src = os.path.join(d, "val_probs.npz")
        if not os.path.exists(src):
            raise SystemExit(f"missing {src} -- has fold {f} finished?")
        z = np.load(src, allow_pickle=True)
        P = np.asarray(z["probs"], dtype=float)
        s = P.sum(axis=1, keepdims=True)
        if not np.allclose(s, 1.0, atol=1e-3):
            print(f"  fold {f}: renormalising (row sums "
                  f"{s.min():.4f}..{s.max():.4f})")
        P = P / s
        out = pd.DataFrame(P, columns=PCOLS)
        out.insert(0, "row_id", list(z["row_id"]))
        dst = os.path.join(d, "oof_predictions.csv")
        out.to_csv(dst, index=False)
        total += len(out)
        print(f"  fold {f}: {len(out)} posts -> {dst}")

    print(f"\n{total} posts written across {args.k} folds.")
    print("\nNow judge it through the DEPLOYED recipe -- weights frozen:")
    print(f"  python src/champion_oof.py --deberta 'runs/risk_{args.tag}_f{{f}}' \\")
    print(f"      --export artifacts/champion_oof_{args.tag}.csv")
    print("\nRead the `+rules (BASELINE)` line and compare against 0.7841.")
    print("Both author halves must improve, or it is noise -- a change can")
    print("look good on one instrument and still fail to transfer to the")
    print("official score.")


if __name__ == "__main__":
    main()
