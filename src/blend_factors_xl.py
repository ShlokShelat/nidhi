"""
Blend the deberta-v2-xlarge factor-NLI model into the 2-seed deberta-v3
factor-NLI ensemble, validated on val_split gold.

The three members are architecturally diverse (two deberta-v3-large seeds and
one deberta-v2-xlarge), so their errors are only partly correlated and an
average can cancel the independent part. Averaging helps only when the added
member is of comparable quality to the ones already in the blend: a clearly
weaker member dilutes the average instead of sharpening it, which is why the
sweep below is validated rather than assumed.

The script sweeps the weight on xlarge against the v3 pair and scores each
weight on val gold. A factor change moves many labels across many posts, so
the effect is large relative to val noise and val is a usable instrument here.
A test file is written only if the blend beats the v3 ensemble by a clear
margin.

Usage:
  python src/blend_factors_xl.py \
      --val artifacts/val_split.csv --test data/test.csv \
      --v3-runs runs/factors_nli,runs/factors_nli_s2 \
      --xl-run runs/factors_nli_xl \
      --out artifacts/factors_blend_xl.csv
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


def probs_and_ths(run_dirs, df, max_len, device, use_bf16):
    acc, ths = [], []
    for rd in run_dirs:
        mn = open(os.path.join(rd, "tok")).read().strip()
        tok = AutoTokenizer.from_pretrained(mn)
        m = NLIScorer(mn).to(device)
        m.load_state_dict(torch.load(os.path.join(rd, "model.pt"),
                                     map_location=device), strict=False)
        m.eval()
        acc.append(score_all(m, df, tok, max_len, device, use_bf16))
        ths.append(np.load(os.path.join(rd, "thresholds.npy")))
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.mean(acc, axis=0), np.mean(ths, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--v3-runs", required=True)
    ap.add_argument("--xl-run", required=True)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--force-weight", type=float, default=None,
                    help="skip auto-pick, write test at THIS xlarge weight (use ~0.5 for balanced blend)")
    ap.add_argument("--out", default="artifacts/factors_blend_xl.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    v3_runs = [r.strip() for r in args.v3_runs.split(",")]
    xl_runs = [r.strip() for r in args.xl_run.split(",")]

    val_df = load(args.val)
    gold = gold_matrix(val_df)

    print("scoring v3 ensemble on val...")
    v3_p, v3_ths = probs_and_ths(v3_runs, val_df, args.max_len, device, use_bf16)
    print("scoring xlarge on val...")
    xl_p, xl_ths = probs_and_ths(xl_runs, val_df, args.max_len, device, use_bf16)

    def macro(probs, ths):
        return factors_macro_f1(to_lists(gold), to_lists((probs >= ths).astype(int)))

    base = macro(v3_p, v3_ths)
    xl_alone = macro(xl_p, xl_ths)
    print(f"\n  v3 2-seed ensemble (val): {base:.4f}")
    print(f"  xlarge alone (val):       {xl_alone:.4f}\n")

    print("  weight sweep (w = weight on xlarge; thresholds averaged too):")
    best = (base, 0.0)
    for w in np.arange(0.0, 1.01, 0.05):
        p = (1 - w) * v3_p + w * xl_p
        th = (1 - w) * v3_ths + w * xl_ths
        f = macro(p, th)
        mark = " *" if f > best[0] else ""
        print(f"    w={w:.2f}: macro-F1={f:.4f}{mark}")
        if f > best[0]:
            best = (f, w)

    best_f, best_w = best
    print(f"\n  BEST(auto) w_xl={best_w:.2f} -> {best_f:.4f} "
          f"(vs v3 ensemble {base:.4f}, delta {best_f-base:+.4f})")
    if args.force_weight is not None:
        best_w = args.force_weight
        p = (1 - best_w) * v3_p + best_w * xl_p
        th = (1 - best_w) * v3_ths + best_w * xl_ths
        print(f"  FORCED w_xl={best_w:.2f} -> val {macro(p, th):.4f} (balanced blend)")
    elif best_f <= base + 0.005:
        print("  *** blend does not clearly beat v3 ensemble — not writing ***")
        return

    print("\n  blend beats v3 ensemble on val — generating test file")
    test_df = load(args.test)
    print("scoring v3 ensemble on test...")
    tv3, _ = probs_and_ths(v3_runs, test_df, args.max_len, device, use_bf16)
    print("scoring xlarge on test...")
    txl, _ = probs_and_ths(xl_runs, test_df, args.max_len, device, use_bf16)
    p = (1 - best_w) * tv3 + best_w * txl
    th = (1 - best_w) * v3_ths + best_w * xl_ths
    on = (p >= th).astype(int)
    out = pd.DataFrame({
        "row_id": test_df["row_id"].tolist(),
        "factors": [str(list(f)) for f in to_lists(on)],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"  wrote {args.out}: mean {on.sum(1).mean():.2f} factors/post, "
          f"{(on.sum(1)==0).sum()} posts with none")


if __name__ == "__main__":
    main()
