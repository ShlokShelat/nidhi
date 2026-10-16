"""
Cross-model factor ensemble (cautious, ours-dominant).

Blends our DeBERTa factor probabilities (factors_cv_final) with the
teammate's MentalRoBERTa factor probabilities. Their model is individually
weaker on factors (0.28 vs our 0.43 macro) and systematically more liberal
(high probs on many labels), so a naive 50/50 would flood predictions.
Default --weight-ours 0.7 keeps ours dominant; theirs nudges borderline
calls — which may specifically help tail-label recall, since macro-F1
weights every label equally regardless of prevalence.

Columns are matched BY NAME (their CSV header uses the same 24 label
strings), never by position.

Usage:
  python src/ensemble_factors.py \
      --test data/test.csv \
      --our-factors-run runs/factors_cv_final \
      --friend-probs artifacts/friend_factor_probs.csv \
      --weight-ours 0.7 \
      --out artifacts/ensembled_factors.csv

Output: row_id + factors (Python-list string), consumable by
predict.py --factors-override.
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from scoring import FACTOR_TAXONOMY
from predict import _factor_probs

N = len(FACTOR_TAXONOMY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--our-factors-run", required=True)
    ap.add_argument("--friend-probs", required=True)
    ap.add_argument("--weight-ours", type=float, default=0.7,
                    help="global fallback weight when --per-label-f1 not given")
    ap.add_argument("--our-f1", default=None,
                    help="comma-sep 24 per-label OOF F1 for OUR model (taxonomy order)")
    ap.add_argument("--friend-f1", default=None,
                    help="comma-sep 24 per-label OOF F1 for FRIEND model (taxonomy order)")
    ap.add_argument("--min-support-labels", default=None,
                    help="comma-sep label names with too few positives -> force ours-dominant")
    ap.add_argument("--out", default="artifacts/ensembled_factors.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load(args.test)
    ids = df["row_id"].tolist()

    print("our factor probs ...")
    ours = _factor_probs(args.our_factors_run, df, device)  # (N,24)
    ths = np.load(os.path.join(args.our_factors_run, "thresholds.npy"))

    fp = pd.read_csv(args.friend_probs)
    fp.columns = [c.strip() for c in fp.columns]
    fp = fp.set_index("row_id")
    # match by label NAME
    missing_cols = [f for f in FACTOR_TAXONOMY if f not in fp.columns]
    if missing_cols:
        raise SystemExit(f"friend file missing label columns: {missing_cols}")

    friend = np.zeros_like(ours)
    miss = 0
    for i, rid in enumerate(ids):
        if rid in fp.index:
            friend[i] = fp.loc[rid, list(FACTOR_TAXONOMY)].values.astype(float)
        else:
            friend[i] = ours[i]
            miss += 1
    if miss:
        print(f"WARNING: {miss} row_ids missing from friend file; used ours")

    # per-label weights from OOF F1 if provided, else global fallback
    if args.our_f1 and args.friend_f1:
        of1 = np.array([float(x) for x in args.our_f1.split(",")])
        ff1 = np.array([float(x) for x in args.friend_f1.split(",")])
        assert len(of1) == N and len(ff1) == N, "need 24 F1 values each"
        # weight on OURS = our_f1 / (our_f1 + friend_f1), clamped so neither
        # model is fully ignored (keeps diversity); tiny-support -> ours-dominant
        denom = np.maximum(of1 + ff1, 1e-6)
        w_ours = np.clip(of1 / denom, 0.25, 0.85)
        if args.min_support_labels:
            tiny = set(s.strip() for s in args.min_support_labels.split(",") if s.strip())
            for j, lab in enumerate(FACTOR_TAXONOMY):
                if lab in tiny:
                    w_ours[j] = 0.8   # don't trust friend on labels with tiny OOF support
        print("per-label w_ours (min/mean/max): "
              f"{w_ours.min():.2f}/{w_ours.mean():.2f}/{w_ours.max():.2f}")
    else:
        w_ours = np.full(N, args.weight_ours)
    blended = w_ours[None, :] * ours + (1 - w_ours[None, :]) * friend
    on = blended >= ths

    fac_lists = [[FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]]
                 for i in range(len(ids))]
    out = pd.DataFrame({"row_id": ids,
                        "factors": [str(list(f)) for f in fac_lists]})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    ours_on = ours >= ths
    changed = int((on != ours_on).any(axis=1).sum())
    w_display = (f"per-label(min={w_ours.min():.2f},mean={w_ours.mean():.2f},max={w_ours.max():.2f})"
                if args.our_f1 and args.friend_f1 else f"{args.weight_ours}")
    print(f"blend w_ours={w_display}: mean factors/post "
          f"{on.sum(1).mean():.2f} (ours-alone {ours_on.sum(1).mean():.2f})")
    print(f"changed factor sets on {changed}/{len(ids)} posts vs ours-alone")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
