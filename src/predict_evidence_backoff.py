"""
Emit test-set evidence using the validated backoff decoding strategy.

Validated on val_split against the OFFICIAL per-post macro metric:
  t_hi=0.65 (normal span selection, already optimal)
  t_lo=0.40 (fallback: if NO candidate clears t_hi, emit the single best
             candidate that clears t_lo)

Rationale: phrase-F1 is averaged per post, and a post where gold has spans
but nothing is emitted scores a hard 0.0 no matter how good the model is. The
backoff rescues some of those posts without disturbing the ones that already
score well.

Output is a CSV (row_id, evidence) for predict.py --evidence-override.

Usage:
  python src/predict_evidence_backoff.py \
      --test data/test.csv --run runs/evidence_v2 \
      --t-hi 0.65 --t-lo 0.40 \
      --out artifacts/evidence_backoff.csv
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

import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from train_evidence import TaggerModel, collect_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--t-hi", type=float, default=0.65)
    ap.add_argument("--t-lo", type=float, default=0.40)
    ap.add_argument("--max-extra", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", default="artifacts/evidence_backoff.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(args.run, "run_config.json")))
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = TaggerModel(cfg["model"])
    model.load_state_dict(
        torch.load(os.path.join(args.run, "model_final.pt"), map_location=device),
        strict=False)
    model.to(device).eval()

    df = load(args.test)
    cands = collect_candidates(model, tok, df["post"].astype(str).tolist(),
                               args.max_len, device)

    spans, n_rescued = [], 0
    for post_c in cands:
        keep = [s for s, c in post_c if c >= args.t_hi]
        if not keep:
            fb = sorted([(c, s) for s, c in post_c if c >= args.t_lo],
                        reverse=True)[: args.max_extra]
            keep = [s for _, s in fb]
            if keep:
                n_rescued += 1
        spans.append(keep)

    out = pd.DataFrame({
        "row_id": df["row_id"].tolist(),
        "evidence": ["; ".join(s) for s in spans],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    tot = sum(len(s) for s in spans)
    empty = sum(1 for s in spans if not s)
    print(f"wrote {args.out}")
    print(f"  t_hi={args.t_hi} t_lo={args.t_lo} max_extra={args.max_extra}")
    print(f"  {tot} spans, {empty} posts with none, "
          f"{n_rescued} posts rescued by backoff")


if __name__ == "__main__":
    main()
