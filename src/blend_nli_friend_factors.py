"""
Blend NLI factor probabilities with the teammate's MentalRoBERTa factor
probabilities — validated honestly on val_split.

Cross-architecture factor blending gave +0.02 S2 with our OLD factor model
(0.4329 -> 0.4549 at weight 0.6). It has never been tried with the NLI
models, which are far stronger (S2 0.6301) and architecturally different
again. Two independently-built systems of differing but real quality is
exactly the regime where a single global blend weight has transferred for
us; per-label weighting and val-selected combinations have not.

Validation is legitimate: our NLI models trained on train_split only, and
the teammate's numbers for val_split rows come from their out-of-fold
export, so neither model saw these posts in training.

This experiment moves MANY posts (unlike the 5-post evidence backoff that
looked good on val and failed on test), so val is measurable here.

Usage:
  python src/blend_nli_friend_factors.py \
      --val artifacts/val_split.csv --test data/test.csv \
      --nli-run runs/factors_nli,runs/factors_nli_s2 \
      --friend-oof artifacts/friend_oof_export.csv \
      --friend-test artifacts/friend_factor_probs.csv \
      --out artifacts/nli_friend_factors.csv
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
from scoring import FACTOR_TAXONOMY, parse_factors, factors_macro_f1
from train_factors_nli import NLIScorer, score_all, N


def to_lists(binary):
    return [[FACTOR_TAXONOMY[j] for j in range(N) if row[j]] for row in binary]


def gold_matrix(df):
    Y = np.zeros((len(df), N), dtype=int)
    for r, cell in enumerate(df["factors"]):
        for f in parse_factors(cell):
            if f in FACTOR_TAXONOMY:
                Y[r, FACTOR_TAXONOMY.index(f)] = 1
    return Y


def nli_probs(run_dirs, df, max_len, device, use_bf16):
    """Averaged probability matrix across the given NLI seed runs."""
    acc, ths = [], []
    for rd in run_dirs:
        model_name = open(os.path.join(rd, "tok")).read().strip()
        tok = AutoTokenizer.from_pretrained(model_name)
        model = NLIScorer(model_name).to(device)
        model.load_state_dict(
            torch.load(os.path.join(rd, "model.pt"), map_location=device),
            strict=False)
        model.eval()
        acc.append(score_all(model, df, tok, max_len, device, use_bf16))
        ths.append(np.load(os.path.join(rd, "thresholds.npy")))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.mean(acc, axis=0), np.mean(ths, axis=0)


def align_friend(path, row_ids, fallback):
    fp = pd.read_csv(path)
    fp.columns = [c.strip() for c in fp.columns]
    fp = fp.set_index("row_id")
    missing = [f for f in FACTOR_TAXONOMY if f not in fp.columns]
    if missing:
        raise SystemExit(f"{path} missing factor columns: {missing[:3]}...")
    out = np.zeros((len(row_ids), N), dtype=np.float64)
    n_miss = 0
    for i, rid in enumerate(row_ids):
        if rid in fp.index:
            out[i] = fp.loc[rid, list(FACTOR_TAXONOMY)].values.astype(float)
        else:
            out[i] = fallback[i]
            n_miss += 1
    if n_miss:
        print(f"  ({n_miss} row_ids absent from {os.path.basename(path)}; "
              f"used our probs there)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--nli-run", required=True)
    ap.add_argument("--friend-oof", required=True)
    ap.add_argument("--friend-test", required=True)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--out", default="artifacts/nli_friend_factors.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    runs = [r.strip() for r in args.nli_run.split(",")]

    val_df = load(args.val)
    gold = gold_matrix(val_df)
    print(f"scoring NLI models on val ({len(val_df)} posts)...")
    v_nli, ths = nli_probs(runs, val_df, args.max_len, device, use_bf16)
    v_friend = align_friend(args.friend_oof, val_df["row_id"].tolist(), v_nli)

    base = factors_macro_f1(to_lists(gold), to_lists((v_nli >= ths).astype(int)))
    fr_only = factors_macro_f1(to_lists(gold),
                               to_lists((v_friend >= 0.5).astype(int)))
    print(f"\n  NLI ensemble alone:      {base:.4f}")
    print(f"  friend alone (th=0.5):   {fr_only:.4f}\n")

    print("  weight sweep (w = weight on OURS):")
    best_w, best_f = 1.0, base
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * v_nli + (1 - w) * v_friend
        f = factors_macro_f1(to_lists(gold), to_lists((blend >= ths).astype(int)))
        mark = ""
        if f > best_f:
            best_f, best_w, mark = f, w, "  *"
        print(f"    w={w:.2f}: {f:.4f}{mark}")

    print(f"\n  BEST w_ours={best_w:.2f} -> {best_f:.4f} "
          f"(vs NLI alone {base:.4f}, delta {best_f-base:+.4f})")
    if best_f <= base + 0.005:
        print("  *** blend does not clearly beat NLI alone — skip the upload ***")
        return

    print(f"\nscoring NLI models on test ({378} posts)...")
    test_df = load(args.test)
    t_nli, _ = nli_probs(runs, test_df, args.max_len, device, use_bf16)
    t_friend = align_friend(args.friend_test, test_df["row_id"].tolist(), t_nli)
    blend = best_w * t_nli + (1 - best_w) * t_friend
    on = (blend >= ths).astype(int)
    out = pd.DataFrame({
        "row_id": test_df["row_id"].tolist(),
        "factors": [str(list(f)) for f in to_lists(on)],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: mean {on.sum(1).mean():.2f} factors/post, "
          f"{(on.sum(1)==0).sum()} posts with none")


if __name__ == "__main__":
    main()
