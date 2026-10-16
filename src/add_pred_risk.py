"""
Annotate a CSV with a `pred_risk_level` column from the risk model(s).

Used to condition the factor model on PREDICTED (not gold) risk, so the
factor model sees the same signal quality at train and test time.

Usage:
  python src/add_pred_risk.py --in artifacts/train_split.csv \
      --risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
      --out artifacts/train_split_r.csv
  python src/add_pred_risk.py --in data/test.csv \
      --risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
      --out artifacts/test_r.csv

NOTE on train conditioning: using the risk ensemble (trained on the train
split) to predict the train split is mildly optimistic, because the models
have already seen those rows. A strictly honest setup conditions on
out-of-fold risk predictions instead.
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from predict import predict_risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--risk-run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load(args.inp)
    preds, _ = predict_risk(args.risk_run, df, device)
    df["pred_risk_level"] = preds
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    from collections import Counter
    print(f"wrote {args.out} with pred_risk_level: {dict(Counter(preds))}")


if __name__ == "__main__":
    main()
