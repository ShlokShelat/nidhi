"""
dump_nli_val_probs.py

Saves the risk-NLI models' val probabilities to disk, once.

WHY. Testing whether a candidate arm adds to the CHAMPION ensemble -- rather
than to some weaker single-model control -- requires the champion's own val
probabilities. From scripts/rebuild_and_apply_rules.sh the champion risk
column is:

    ensembled = 0.5 x mean(risk_v1, risk_v1_s14, risk_v1_s15)
              + 0.5 x friend
    blended   = 0.70 x ensembled + 0.30 x mean(risk_nli, risk_nli_s2)
    then      urgency + completed_act rules

Four of those six models already have saved val probabilities:
    runs/risk_v1*/val_predictions.csv   (prob_indicator/ideation/behavior/attempt)
    artifacts/friend_risk_probs_val.csv (1590 rows, covers val)

The two NLI models do not -- blend_risk_saved.py recomputes them from the
checkpoints every run and never writes them out. This script does that pass
once and saves the result, so every subsequent ensemble experiment is CPU-only.

It reuses blend_risk_saved.nli_probs_for rather than reimplementing the
scoring. A second copy of that logic would drift from the original, and the
dumped probabilities would then silently stop matching the ones the blend
actually uses.

Usage (GPU, a couple of minutes):
  python src/dump_nli_val_probs.py \
      --val artifacts/val_split.csv \
      --nli-run runs/risk_nli,runs/risk_nli_s2 \
      --out artifacts/nli_risk_val_probs.csv
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
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--nli-run", default="runs/risk_nli,runs/risk_nli_s2")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--out", default="artifacts/nli_risk_val_probs.csv")
    args = ap.parse_args()

    import torch
    import blend_risk_saved as B

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch, not the login node")
    use_bf16 = torch.cuda.is_bf16_supported()

    # Use the blend script's own loader when it has one, so any normalisation
    # it applies to the val frame is applied here identically.
    val_df = B.load(args.val) if hasattr(B, "load") else pd.read_csv(args.val)
    runs = [r.strip() for r in args.nli_run.split(",")]
    print(f"scoring {len(runs)} NLI runs over {len(val_df)} val posts: {runs}")

    P = B.nli_probs_for(val_df, runs, args.max_len, device, use_bf16)
    P = np.asarray(P, dtype=float)
    if P.shape != (len(val_df), 4):
        raise SystemExit(f"unexpected NLI prob shape {P.shape}")

    out = pd.DataFrame({"row_id": val_df["row_id"].tolist()})
    for j, c in enumerate(RISK_LEVELS):
        out[f"prob_{c}"] = P[:, j]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    gold = [normalize_label(x) for x in val_df["risk_level"]]
    pred = [RISK_LEVELS[i] for i in P.argmax(1)]
    f1 = risk_weighted_f1(gold, pred)
    print(f"\nwrote {args.out}")
    print(f"NLI alone on val: weighted-F1 = {f1:.4f}")
    print("  (NLI alone generalises worse than this val figure suggests, "
          "which is why it enters the champion blend at weight 0.30 rather "
          "than as an equal member)")


if __name__ == "__main__":
    main()
