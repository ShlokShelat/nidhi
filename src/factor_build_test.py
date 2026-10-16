"""
factor_build_test.py

Build the TEST factor column from the top-3 diverse factor models.

VALIDATED ON VAL (372 posts, both author-disjoint halves, all arms thresholded
the same way the deployed system was):

    INCUMBENT (factors_nli + factors_nli_s2)   0.6221   A 0.6116  B 0.5896
    mean-prob top3 (d6 + d4 + d3)              0.6672
    mean-prob ALL (10 models)                  0.6554
    best single factors_d6                     0.6510

top3 is both the highest and the best balanced across halves. The
parameter-free alternatives point the same way, so the gain is not an
artefact of picking winners on val.

The underlying finding is simpler than the ensemble: factors_d6 ALONE beats
the shipped two-seed ensemble. It is seed 21, lr 2e-5, 3 epochs,
max_len 384. Factors were not stuck at 0.63 because the task is hard -- the
shipped model was undertuned and nobody had swept it.

TWO CONVENTIONS PRESERVED so this is a single-variable change:

  Thresholds are fitted per label on ALL 372 val posts, which is how the
  deployed thresholds were produced. Cross-fitting them here would change the
  fitting procedure at the same time as the model.

  Every post gets at least one factor. The current submission has ZERO empty
  factor cells, so a post whose probabilities all fall below threshold is given
  its argmax label. Letting posts go empty would change the output convention
  as well as the model, and gold does have 7.8% empty posts -- worth testing,
  but as its own experiment.

Output format matches the submission exactly: a Python-list repr string,
"['mental health issues', 'hopelessness']".

EXPECT A SMALL SUBMISSION MOVE. Offline gains on this kind of val split tend
to transfer to test at only a fraction of their measured value. Applying that
discount here, the macro gain above should be read as a much smaller
composite gain, not taken at face value.

Usage:
  python src/factor_build_test.py
  python src/patch_submission_risk.py \
      --base submissions/NIDHI_evidence_vote8.csv \
      --factors artifacts/factors_top3_test.csv \
      --out submissions/NIDHI_factors_top3.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    pk = next((k for k in ("probs", "p", "y", "scores") if k in d.files), None)
    rk = next((k for k in ("row_id", "row_ids", "ids") if k in d.files), None)
    if pk is None or rk is None:
        raise SystemExit(f"{path}: expected prob/row_id, got {d.files}")
    return list(d[rk]), np.asarray(d[pk], dtype=float)


def stack(names, split, d):
    rid0, mats = None, []
    for n in names:
        rids, PR = load_npz(os.path.join(d, f"{n}_{split}.npz"))
        if rid0 is None:
            rid0 = rids
        elif list(rids) != list(rid0):
            o = {r: i for i, r in enumerate(rids)}
            PR = PR[[o[r] for r in rid0]]
        mats.append(PR)
    return rid0, np.mean(mats, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="factors_d6,factors_d4,factors_d3")
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--out", default="artifacts/factors_top3_test.csv")
    ap.add_argument("--allow-empty", action="store_true",
                    help="let posts have zero factors. The current submission "
                         "has none, so this changes the output convention as "
                         "well as the model -- keep it off for a clean test.")
    args = ap.parse_args()

    names = [n.strip() for n in args.models.split(",")]
    print(f"averaging {len(names)} models: {names}")

    # ---- fit thresholds on val, exactly as the deployed system did --------
    vr, VP = stack(names, "val", args.dir)
    lab = pd.read_csv(args.val).set_index("row_id")
    keep = [i for i, r in enumerate(vr) if r in lab.index]
    vr = [vr[i] for i in keep]
    VP = VP[keep]
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros(VP.shape, dtype=int)
    for i, r in enumerate(vr):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1

    th = np.full(VP.shape[1], 0.5)
    for j in range(VP.shape[1]):
        best, bt = -1.0, 0.5
        for t in GRID:
            pred = (VP[:, j] >= t).astype(int)
            tp = int((pred & Y[:, j]).sum())
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + int((pred & (1 - Y[:, j])).sum())
                           + int(((1 - pred) & Y[:, j]).sum()))
            if f1 > best:
                best, bt = f1, t
        th[j] = bt

    P = (VP >= th).astype(int)
    f1s = []
    for j in range(P.shape[1]):
        tp = int((P[:, j] & Y[:, j]).sum())
        fp = int((P[:, j] & (1 - Y[:, j])).sum())
        fn = int(((1 - P[:, j]) & Y[:, j]).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    print(f"val macro-F1 at these thresholds: {np.mean(f1s):.4f}"
          f"   (incumbent 0.6221)")
    print(f"thresholds: min {th.min():.2f} median {np.median(th):.2f} "
          f"max {th.max():.2f}")

    # ---- apply to test ---------------------------------------------------
    tr, TP = stack(names, "test", args.dir)
    B = (TP >= th).astype(int)

    forced = 0
    rows = []
    for i, r in enumerate(tr):
        sel = [FACTOR_TAXONOMY[j] for j in range(B.shape[1]) if B[i, j]]
        if not sel and not args.allow_empty:
            sel = [FACTOR_TAXONOMY[int(TP[i].argmax())]]
            forced += 1
        rows.append((r, str(sel)))

    out = pd.DataFrame(rows, columns=["row_id", "factors"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(out_path := args.out, index=False)

    n = int(B.sum())
    print(f"\ntest: {n} factor labels over {len(tr)} posts "
          f"({n / len(tr):.2f}/post)")
    print(f"  forced a label on {forced} otherwise-empty posts")
    print(f"wrote {out_path}")
    print("\nPer-label test counts:")
    for j, t in enumerate(FACTOR_TAXONOMY):
        print(f"  {t:<46}{int(B[:, j].sum()):>5}   th={th[j]:.2f}")
    print("\nNext:")
    print("  python src/patch_submission_risk.py \\")
    print("      --base submissions/NIDHI_evidence_vote8.csv \\")
    print(f"      --factors {out_path} \\")
    print("      --out submissions/NIDHI_factors_top3.csv")


if __name__ == "__main__":
    main()
