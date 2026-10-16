"""
dump_nli_probs.py

Saves the risk-NLI models' probabilities for ANY split (val or test), once.

blend_risk_saved.py recomputes these from the checkpoints on every run and
never writes them out, which forces every downstream experiment onto a GPU.
Dumping them once makes all subsequent ensemble work CPU-only, and it also
freezes the numbers so two experiments compare identical inputs.

Gold labels are optional: on test there are none, so the self-check score is
simply skipped rather than crashing.

Usage:
  # val (prints the self-check weighted-F1)
  python src/dump_nli_probs.py --data artifacts/val_split.csv \
      --out artifacts/nli_risk_val_probs.csv
  # test
  python src/dump_nli_probs.py --data data/test.csv \
      --out artifacts/nli_risk_test_probs.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--nli-run", default="runs/risk_nli,runs/risk_nli_s2")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import blend_risk_saved as B

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")
    use_bf16 = torch.cuda.is_bf16_supported()

    df = B.load(args.data) if hasattr(B, "load") else pd.read_csv(args.data)
    runs = [r.strip() for r in args.nli_run.split(",")]
    print(f"scoring {len(runs)} NLI runs over {len(df)} posts from {args.data}")

    P = np.asarray(B.nli_probs_for(df, runs, args.max_len, device, use_bf16),
                   dtype=float)
    if P.shape != (len(df), 4):
        raise SystemExit(f"unexpected NLI prob shape {P.shape}")

    out = pd.DataFrame({"row_id": df["row_id"].tolist()})
    for j, c in enumerate(RISK_LEVELS):
        out[f"prob_{c}"] = P[:, j]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")

    if "risk_level" in df.columns:
        gold = [normalize_label(x) for x in df["risk_level"]]
        pred = [RISK_LEVELS[i] for i in P.argmax(1)]
        print(f"NLI alone: weighted-F1 = {risk_weighted_f1(gold, pred):.4f}")
    else:
        from collections import Counter
        print("no gold column -- distribution: "
              f"{dict(Counter(RISK_LEVELS[i] for i in P.argmax(1)))}")


if __name__ == "__main__":
    main()
