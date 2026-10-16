"""
Predict risk levels on the test set with the trained definition-grounded
risk-NLI model.

Outputs a CSV (row_id, risk_level) for predict.py --risk-override, and
saves the (N,4) softmax probability matrix so the risk-NLI model can be
ensembled with the other risk models.

Supports comma-separated --run for seed-ensembling multiple risk-NLI seeds
by averaging softmax probabilities.

Usage:
  python src/predict_risk_nli.py --test data/test.csv \
      --run runs/risk_nli --out artifacts/risk_nli.csv
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
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from scoring import RISK_LEVELS
from train_risk_nli import RiskNLIScorer, score_posts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--run", required=True,
                    help="one run dir, or comma-separated dirs to seed-ensemble")
    ap.add_argument("--out", default="artifacts/risk_nli.csv")
    ap.add_argument("--max-len", type=int, default=384)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    df = load(args.test)
    run_dirs = [r.strip() for r in args.run.split(",")]
    all_probs = []
    for rd in run_dirs:
        model_name = open(os.path.join(rd, "tok")).read().strip()
        tok = AutoTokenizer.from_pretrained(model_name)
        model = RiskNLIScorer(model_name).to(device)
        model.load_state_dict(
            torch.load(os.path.join(rd, "model.pt"), map_location=device),
            strict=False)
        model.eval()
        all_probs.append(score_posts(model, df, tok, args.max_len, device, use_bf16))
        print(f"  scored {rd}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    probs = np.mean(all_probs, axis=0)
    pred = [RISK_LEVELS[i] for i in probs.argmax(1)]

    out = pd.DataFrame({
        "row_id": df["row_id"].tolist(),
        "risk_level": [p.capitalize() for p in pred],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    np.save(args.out.replace(".csv", "_probs.npy"), probs)

    from collections import Counter
    dist = Counter(pred)
    print(f"\nwrote {args.out}")
    print(f"  risk distribution: {dict(dist)}")
    print(f"  saved probs -> {args.out.replace('.csv', '_probs.npy')}")


if __name__ == "__main__":
    main()
