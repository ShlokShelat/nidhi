"""
compute_boundary_weights.py

Emits a per-example difficulty weight for risk training: difficulty-aware
sampling only, with no loss or architecture changes.

Design points:
  1. AVERAGE across all 3 existing risk seeds (risk_v1, risk_v1_s14,
     risk_v1_s15) rather than scoring from one model. A single model's
     confidently-wrong judgments would otherwise be labelled "hard" and
     upweighted, feeding that model's own bias back into training.
     Probabilities are averaged first, matching every other seed-ensemble
     here, then margin / predicted_label / runner_up_label are all derived
     from that single averaged distribution.
  2. BUCKETED weights, not a continuous inverse-margin scale. A continuous
     scale sends the weight of a near-zero-margin example arbitrarily high,
     so a handful of noisy or mislabelled rows would dominate the gradient;
     bucketing caps that at 4x.
       margin > 0.60        -> weight 1
       0.30 <= margin <= 0.60 -> weight 2
       0.10 <= margin < 0.30  -> weight 3
       margin < 0.10          -> weight 4
  3. Saves predicted_label AND runner_up_label alongside margin, so
     later analysis can ask "which boundary is hardest" (Indicator-
     Ideation? Ideation-Behavior? Behavior-Attempt?), not just "how
     hard overall".

Usage:
  python src/compute_boundary_weights.py \
      --train artifacts/train_split.csv \
      --checkpoints runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
      --out artifacts/train_boundary_weights.csv
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS
from train_risk import DualHeadModel, PostDataset


@torch.no_grad()
def score_train(checkpoint, tr_df, device, use_bf16, batch_size):
    with open(os.path.join(checkpoint, "run_config.json")) as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    loader = DataLoader(
        PostDataset(tr_df, tok, cfg["max_len"], cfg["user_context"]),
        batch_size=batch_size, shuffle=False, num_workers=0)

    model = DualHeadModel(cfg["model"], alpha_ordinal=cfg["alpha_ordinal"]).to(device)
    state = torch.load(os.path.join(checkpoint, "model_final.pt"), map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    all_probs, all_labels = [], []
    for batch in loader:
        batch.pop("idx")
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(**batch)
        probs = torch.exp(out["logits"].float())  # logits are log-probs (see DualHeadModel._combine)
        all_probs.append(probs.cpu())
        all_labels.append(labels)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(all_probs), torch.cat(all_labels)


def bucket_weight(margin):
    if margin > 0.60:
        return 1
    elif margin >= 0.30:
        return 2
    elif margin >= 0.10:
        return 3
    else:
        return 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--checkpoints", required=True,
                    help="comma-sep, e.g. runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")

    tr_df = pd.read_csv(args.train)
    checkpoints = [c.strip() for c in args.checkpoints.split(",")]

    all_probs = []
    labels = None
    for ckpt in checkpoints:
        print(f"scoring {ckpt}...")
        probs, lbl = score_train(ckpt, tr_df, device, use_bf16, args.batch)
        all_probs.append(probs)
        labels = lbl  # identical across checkpoints — same train_df, same order

    avg_probs = torch.stack(all_probs).mean(dim=0)  # (N, 4) — ensemble judgment

    margins = np.zeros(len(tr_df))
    predicted_labels = []
    runner_up_labels = []
    for i in range(len(tr_df)):
        gold_idx = labels[i].item()
        p = avg_probs[i]
        p_gold = p[gold_idx].item()
        p_others = p.clone()
        p_others[gold_idx] = -1
        best_other_idx = int(p_others.argmax().item())
        margins[i] = p_gold - p_others[best_other_idx].item()

        top_idx = int(p.argmax().item())
        predicted_labels.append(RISK_LEVELS[top_idx])
        # runner-up = second-highest prob overall (not "best other than gold")
        p2 = p.clone()
        p2[top_idx] = -1
        runner_up_labels.append(RISK_LEVELS[int(p2.argmax().item())])

    weights = np.array([bucket_weight(m) for m in margins])

    out_df = tr_df.copy()
    out_df["margin"] = margins
    out_df["predicted_label"] = predicted_labels
    out_df["runner_up_label"] = runner_up_labels
    out_df["sample_weight"] = weights
    out_df.to_csv(args.out, index=False)

    print(f"\nwrote {args.out} (ensemble of {len(checkpoints)} checkpoints)")
    print(f"margin stats: min={margins.min():.3f} max={margins.max():.3f} mean={margins.mean():.3f}")
    from collections import Counter
    print(f"weight distribution: {dict(sorted(Counter(weights.tolist()).items()))}")

    print(f"\n{'-' * 70}")
    print("HARDEST 10 training examples (lowest ensemble margin):")
    hardest = out_df.nsmallest(10, "margin")
    for _, row in hardest.iterrows():
        print(f"  [margin={row['margin']:+.3f}] gold={row['risk_level']:<10} "
             f"pred={row['predicted_label']:<10} runner_up={row['runner_up_label']:<10}")
        print(f"    {str(row['post'])[:100]}")

    print(f"\n{'-' * 70}")
    print("WHICH BOUNDARY IS HARDEST (among the weight=4 hardest bucket):")
    hard_bucket = out_df[out_df["sample_weight"] == 4]
    boundary_counts = Counter(zip(hard_bucket["risk_level"].str.lower().str.strip(),
                                  hard_bucket["predicted_label"]))
    for (gold, pred), cnt in boundary_counts.most_common(10):
        print(f"  gold={gold:<10} predicted={pred:<10} count={cnt}")


if __name__ == "__main__":
    main()
