"""
mine_train_errors.py

Finds GENUINE model errors to seed "error-neighbor" synthetic
generation — without ever touching val_split (that would leak val
into training and invalidate the final evaluation).

Method: split train_split itself into train_sub (80%) / holdout_sub
(20%), train factor-NLI on train_sub ONLY, then score holdout_sub
(never seen by this training run) and collect false negatives and
false positives for the 8 targeted factors. Those are real,
held-out failures — legitimate seeds for "generate examples that
resemble what the model actually gets wrong," per the team's
diagnosis, without contaminating val_split.

This trains one smaller model (80% of train data, few epochs) purely
as a DIAGNOSTIC tool to surface errors — it is NOT the model that
gets deployed or evaluated for the real result.

Usage:
  python src/mine_train_errors.py \
      --train artifacts/train_split.csv \
      --outdir runs/error_miner \
      --seeds-out artifacts/train_error_seeds.csv \
      --holdout-frac 0.2 --epochs 3
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from train_factors_nli import (NLIScorer, PairDataset, make_pairs, score_all,
                               tune_thresholds, gold_matrix, FACTOR_TAXONOMY, N)

TARGETED = {
    "poor social support", "psychological capital", "interpersonal difficulty",
    "hopelessness", "prior self-harm or suicidal thought/attempt",
    "suicide means (with access)", "mental health issues", "stressful life event",
}
TARGETED_IDX = [FACTOR_TAXONOMY.index(f) for f in TARGETED]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds-out", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--neg-ratio", type=int, default=3)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-errors-per-factor", type=int, default=25,
                    help="cap on FN + FP examples kept per targeted factor")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    full = pd.read_csv(args.train)
    idx = rng.permutation(len(full))
    n_holdout = int(len(full) * args.holdout_frac)
    holdout_df = full.iloc[idx[:n_holdout]].reset_index(drop=True)
    train_sub = full.iloc[idx[n_holdout:]].reset_index(drop=True)
    print(f"train_sub: {len(train_sub)}  holdout_sub (never trained on): {len(holdout_df)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    pw = float(np.clip(args.neg_ratio, 1.0, 5.0))
    model = NLIScorer(args.model, pos_weight=pw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    probe = make_pairs(train_sub, args.neg_ratio, rng, 0.5)
    steps_per_epoch = len(probe) // args.batch // args.grad_accum + 1
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    for epoch in range(args.epochs):
        train_rows = make_pairs(train_sub, args.neg_ratio, rng, 0.5)
        loader = DataLoader(PairDataset(train_rows, tok, args.max_len),
                            batch_size=args.batch, shuffle=True, num_workers=2)
        model.train(); opt.zero_grad()
        run = []
        for b, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch)["loss"]
            (loss / args.grad_accum).backward()
            run.append(loss.item())
            if (b + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        print(f"epoch {epoch+1}: mean loss {np.mean(run):.4f}")

    print("\nscoring holdout_sub (genuinely unseen by this training run)...")
    probs = score_all(model, holdout_df, tok, args.max_len, device, use_bf16)
    gold = gold_matrix(holdout_df)
    ths = tune_thresholds(probs, gold)
    np.save(os.path.join(args.outdir, "thresholds.npy"), ths)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pt"))
    with open(os.path.join(args.outdir, "tok"), "w") as f:
        f.write(args.model)

    seeds = []
    for j in TARGETED_IDX:
        factor = FACTOR_TAXONOMY[j]
        p = probs[:, j]
        g = gold[:, j]
        t = ths[j]
        pred = (p >= t).astype(int)

        fn_idx = np.where((pred == 0) & (g == 1))[0]  # model MISSED it — real thing
        fp_idx = np.where((pred == 1) & (g == 0))[0]  # model FALSELY fired — real near-miss

        # rank by "closest to the threshold" — the genuinely hard, ambiguous
        # cases, not flukes far from the boundary
        fn_idx = fn_idx[np.argsort(np.abs(p[fn_idx] - t))]
        fp_idx = fp_idx[np.argsort(np.abs(p[fp_idx] - t))]

        half = args.max_errors_per_factor // 2
        for i in fn_idx[:half]:
            seeds.append({
                "factor": factor, "error_type": "false_negative",
                "model_prob": float(p[i]), "threshold": float(t),
                "post": holdout_df.iloc[i]["post"],
            })
        for i in fp_idx[:half]:
            seeds.append({
                "factor": factor, "error_type": "false_positive",
                "model_prob": float(p[i]), "threshold": float(t),
                "post": holdout_df.iloc[i]["post"],
            })

    seeds_df = pd.DataFrame(seeds)
    os.makedirs(os.path.dirname(args.seeds_out) or ".", exist_ok=True)
    seeds_df.to_csv(args.seeds_out, index=False)

    print(f"\nwrote {args.seeds_out}: {len(seeds_df)} genuine held-out errors")
    print(seeds_df.groupby(["factor", "error_type"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
