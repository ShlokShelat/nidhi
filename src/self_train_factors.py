"""
Self-training for factors — pseudo-label the already-crawled external posts
using the existing champion factor-NLI ensemble as teacher, instead of Qwen.

WHY THIS FIXES THE PRIOR FAILURE: the external-data experiment already run
(label_external.py, Qwen2.5-14B zero-shot) shows a val/test mismatch (val
0.622 but test S2 0.559, below the no-external 0.590 baseline) — the
organizers' own paper warned LLMs have conservative labeling bias, and it
showed here.

Classic self-training (Lee 2013; BOND, Liang et al. 2020; LIME, Park et al.
2022) uses a VALIDATED model as teacher and keeps only HIGH-CONFIDENCE
pseudo-labels (commonly >=0.85-0.9) to avoid propagating teacher error into
the student. The factor-NLI ensemble here is exactly this kind of teacher: it is
validated (transfers cleanly, val 0.6038 -> test 0.5901, only slight
shrinkage) and calibrated by construction (per-label thresholds tuned on
gold). Using IT instead of a noisy generic LLM to label the same 1055
external posts should produce much cleaner pseudo-labels.

Design:
  - Score all 1055 external posts with the 2-seed factor-NLI ensemble
    (the champion config).
  - For each post, keep a factor as a pseudo-positive ONLY if its
    probability clears a STRICT threshold (--conf-gate, default 0.85 —
    well above the per-label thresholds used for normal prediction).
  - Keep only posts that gain at least one RARE-factor pseudo-positive
    (same rare-factor targeting philosophy as the original external-data
    script — inject signal where scarce, not where already abundant).
  - Cap examples per factor to avoid over-concentration (a lesson from the
    6x-augmentation failure).
  - Output in the schema train_factors_nli.py --external expects
    (row_id, post, factors).

Usage:
  python src/self_train_factors.py \
      --raw artifacts/external_raw.csv \
      --nli-run runs/factors_nli,runs/factors_nli_s2 \
      --conf-gate 0.85 --per-factor-cap 60 \
      --out artifacts/external_selftrain.csv
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
from scoring import FACTOR_TAXONOMY
from train_factors_nli import NLIScorer, score_all, N

RARE_TARGETS = {
    "sexual orientation related issues", "exposure to others' suicide",
    "substance use", "meaning in life", "poor school performance",
    "sense of responsibility", "physical health/characteristic",
    "cognitive deficits", "low socio-economic status", "traumatic experience",
}


def probs_for(run_dirs, posts_df, max_len, device, use_bf16):
    acc = []
    for rd in run_dirs:
        mn = open(os.path.join(rd, "tok")).read().strip()
        tok = AutoTokenizer.from_pretrained(mn)
        m = NLIScorer(mn).to(device)
        m.load_state_dict(torch.load(os.path.join(rd, "model.pt"),
                                     map_location=device), strict=False)
        m.eval()
        acc.append(score_all(m, posts_df, tok, max_len, device, use_bf16))
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.mean(acc, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--nli-run", required=True)
    ap.add_argument("--conf-gate", type=float, default=0.85)
    ap.add_argument("--per-factor-cap", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--out", default="artifacts/external_selftrain.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    run_dirs = [r.strip() for r in args.nli_run.split(",")]

    raw = pd.read_csv(args.raw)
    print(f"scoring {len(raw)} external posts with {len(run_dirs)}-model teacher...")
    probs = probs_for(run_dirs, raw.rename(columns={"post": "post"}), args.max_len,
                      device, use_bf16)

    per_factor_count = {f: 0 for f in FACTOR_TAXONOMY}
    kept_rows = []
    for i, row in raw.iterrows():
        p = probs[i]
        pos = [FACTOR_TAXONOMY[j] for j in range(N) if p[j] >= args.conf_gate]
        rare_pos = [f for f in pos if f in RARE_TARGETS
                    and per_factor_count[f] < args.per_factor_cap]
        if not rare_pos:
            continue
        for f in pos:
            per_factor_count[f] += 1
        kept_rows.append({
            "row_id": row.get("ext_id", f"SELF{i:06d}"),
            "post": row["post"],
            "factors": str(list(pos)),
        })

    out = pd.DataFrame(kept_rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(out)} self-labeled posts "
          f"(conf_gate={args.conf_gate}, cap={args.per_factor_cap})")
    print("per-factor kept counts (rare targets):")
    for f in sorted(RARE_TARGETS, key=lambda x: per_factor_count[x], reverse=True):
        print(f"  {per_factor_count[f]:4d}  {f}")
    print("\ncompare to Qwen external labeling counts: substance use 64, "
          "physical health 37, traumatic exp 36, low socio-econ 29, "
          "exposure 17, poor school 13 (that run underperformed on test)")


if __name__ == "__main__":
    main()
