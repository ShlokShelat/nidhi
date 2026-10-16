"""
pseudo_label_factors.py

Pseudo-label a target set with a teacher that never saw it.

`src/self_train_factors.py` does the same for `artifacts/external_raw.csv`,
which is crawled text and therefore distribution-shifted relative to the
scored data. The target set here is not shifted -- same subreddit, same author
space, same annotation regime -- so transductive self-training on it adds
supervision without importing a new text distribution. This is the approach
used by the 2024 winner of this competition family (arXiv 2410.04501).

This writes `row_id,post,factors` -- exactly the three columns
`train_factors_nli.py --external` consumes (it does
`keep = [c for c in ["row_id","post","factors"] if c in ext.columns]`).

GATING, matching self_train_factors.py:
  --gate        keep a factor only when p >= gate (default 0.85, the
                Lee-2013 / BOND convention and what the existing script uses)
  --cap         at most N pseudo-positives per factor, so one confident label
                cannot flood training
  --min-labels  drop posts that gained nothing; an all-negative pseudo row
                teaches the model to predict nothing

Usage:
  python src/pseudo_label_factors.py --probs runs/x/val_probs.npz \\
      --posts artifacts/folds/val_f0.csv --out artifacts/pseudo/f0.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY  # noqa: E402

N = len(FACTOR_TAXONOMY)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--probs", required=True,
                    help=".npz with `probs` (n x 24) and `row_id`")
    a_.add_argument("--posts", required=True,
                    help="csv with row_id and post for the target set")
    a_.add_argument("--out", required=True)
    a_.add_argument("--gate", type=float, default=0.85)
    a_.add_argument("--cap", type=int, default=60)
    a_.add_argument("--min-labels", type=int, default=1)
    args = a_.parse_args()

    z = np.load(args.probs, allow_pickle=True)
    P = z["probs"].astype(float)
    rid = [str(x) for x in z["row_id"]]
    if P.shape[1] != N:
        raise SystemExit(f"probs have {P.shape[1]} columns, expected {N}")

    d = pd.read_csv(args.posts, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    pm = dict(zip(d["row_id"], d["post"]))
    miss = [r for r in rid if r not in pm]
    if miss:
        raise SystemExit(f"{len(miss)} rows have no post text")

    keep = P >= args.gate
    # per-factor cap: keep the most confident `cap` positives for each label
    for j in range(N):
        idx = np.flatnonzero(keep[:, j])
        if len(idx) > args.cap:
            drop = idx[np.argsort(-P[idx, j])[args.cap:]]
            keep[drop, j] = False

    rows = []
    for i, r in enumerate(rid):
        pos = [FACTOR_TAXONOMY[j] for j in range(N) if keep[i, j]]
        if len(pos) < args.min_labels:
            continue
        rows.append((r, pm[r], str(pos)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows, columns=["row_id", "post", "factors"]).to_csv(
        args.out, index=False)
    per = keep.sum(0)
    print(f"{len(rid)} target posts -> {len(rows)} pseudo-labelled "
          f"(gate {args.gate}, cap {args.cap})")
    print(f"pseudo-positives {int(keep.sum())}, "
          f"{keep.sum()/max(len(rows),1):.2f} per kept post")
    top = np.argsort(-per)[:6]
    print("  most-labelled: " + ", ".join(
        f"{FACTOR_TAXONOMY[j]} {per[j]}" for j in top if per[j]))
    zero = [FACTOR_TAXONOMY[j] for j in range(N) if per[j] == 0]
    if zero:
        print(f"  {len(zero)} factors got NO pseudo-positive "
              f"-- self-training cannot help those labels")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
