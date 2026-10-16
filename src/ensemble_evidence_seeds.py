"""
Evidence seed ensemble — confidence averaging across independently-seeded
BIO taggers, with min_conf swept and validated on val_split gold.

WHY THIS, AND WHY IT DIFFERS FROM THE UNION THAT FAILED:
An earlier evidence "union" merged local spans with a teammate's ungated
model's spans. That ADDED every span from both sources, roughly doubling
span count, and cost S1 0.024 because containment scoring punishes false
positives. This is the opposite operation: it AVERAGES per-span confidence
across seeds. A span proposed by only 1 of N seeds has its confidence
divided by N and falls below threshold (filtered out); a span all seeds
agree on retains full confidence. Ensembling here raises PRECISION and
consensus rather than inflating recall.

This is the same probability-averaging mechanism that transferred cleanly
for risk and for the NLI factor models, applied to the one component not yet
ensembled.

Evidence carries a 0.3 weight in the composite — identical to factors —
yet has received far less attention than factors in this campaign.

Validation is honest: min_conf is swept against gold spans on val_split,
and the ensemble is compared to each constituent seed.

Usage:
  python src/ensemble_evidence_seeds.py \
      --val artifacts/val_split.csv --test data/test.csv \
      --runs runs/evidence_v2,runs/evidence_s14,runs/evidence_s15 \
      --out artifacts/evidence_ens.csv
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
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from scoring import normalize_span, parse_evidence, phrase_f1
from train_evidence import collect_candidates, spans_at


def load_tagger(run_dir, device):
    """Load a trained evidence tagger from its run dir."""
    cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
    model_name = cfg["model"]
    tok = AutoTokenizer.from_pretrained(model_name)
    from train_evidence import TaggerModel
    model = TaggerModel(model_name)
    state = torch.load(os.path.join(run_dir, "model_final.pt"), map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, tok, cfg


def average_candidates(cands_per_model):
    """cands_per_model: list (per model) of list (per post) of [(span, conf)].

    Returns per-post list of (span, averaged_conf) where a span missing from
    a model contributes 0 to its average — so single-seed spans are damped
    by 1/N and consensus spans keep their confidence.
    """
    n_models = len(cands_per_model)
    n_posts = len(cands_per_model[0])
    out = []
    for i in range(n_posts):
        acc = defaultdict(float)
        surface = {}
        for m in range(n_models):
            for span, conf in cands_per_model[m][i]:
                key = normalize_span(span)
                acc[key] += float(conf)
                surface.setdefault(key, span)  # keep first seen casing
        out.append([(surface[k], v / n_models) for k, v in acc.items()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated evidence run dirs")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", default="artifacts/evidence_ens.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dirs = [r.strip() for r in args.runs.split(",")]
    print(f"ensembling {len(run_dirs)} evidence taggers: {run_dirs}\n")

    val_df = load(args.val)
    test_df = load(args.test)
    gold = [parse_evidence(c) for c in val_df["evidence"]]
    grid = np.arange(0.30, 0.90, 0.025)

    val_cands, test_cands = [], []
    for rd in run_dirs:
        model, tok, cfg = load_tagger(rd, device)
        vc = collect_candidates(model, tok, val_df["post"].astype(str).tolist(),
                                args.max_len, device)
        tc = collect_candidates(model, tok, test_df["post"].astype(str).tolist(),
                                args.max_len, device)
        val_cands.append(vc)
        test_cands.append(tc)
        # solo score at its own stored min_conf, and at best sweep point
        stored = cfg.get("evidence_min_conf", 0.65)
        solo_stored = phrase_f1(gold, spans_at(vc, stored)).f1
        solo_best = max(phrase_f1(gold, spans_at(vc, t)).f1 for t in grid)
        print(f"  {rd}: solo phrase-F1 @stored({stored:.2f})={solo_stored:.4f}, "
              f"best-swept={solo_best:.4f}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    best_solo = max(
        max(phrase_f1(gold, spans_at(vc, t)).f1 for t in grid) for vc in val_cands)

    val_avg = average_candidates(val_cands)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = phrase_f1(gold, spans_at(val_avg, t)).f1
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\n  best SOLO seed (swept):  {best_solo:.4f}")
    print(f"  ENSEMBLE (avg conf):     {best_f1:.4f} @min_conf={best_t:.3f}  "
          f"(delta {best_f1-best_solo:+.4f})")

    if best_f1 <= best_solo + 0.002:
        print("\n  *** ensemble does NOT beat best solo — not worth an upload ***")
    else:
        print("\n  ensemble validated on gold val — worth spending a submission on")

    test_avg = average_candidates(test_cands)
    spans = spans_at(test_avg, best_t)
    out = pd.DataFrame({
        "row_id": test_df["row_id"].tolist(),
        "evidence": ["; ".join(s) for s in spans],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    tot = sum(len(s) for s in spans)
    print(f"\nwrote {args.out}: {tot} spans, "
          f"{sum(1 for s in spans if not s)} posts with none")


if __name__ == "__main__":
    main()
