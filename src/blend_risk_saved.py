"""
Blend risk-NLI into the risk ensemble.

The ensemble side is read from SAVED probability files rather than
reconstructed by re-running the loaders, so the blend's w=0 point is exactly
the ensemble that is already validated. Rebuilding it would make the baseline
a second, unverified model and every reported delta would be measured against
the wrong thing.

Ensemble val probs:  average of runs/risk_*/val_predictions.csv
                     (columns prob_indicator/ideation/behavior/attempt)
Ensemble test probs: artifacts/ensembled_risk.csv
                     (columns p_Indicator/p_Ideation/p_Behavior/p_Attempt)
Risk-NLI probs:      recomputed from the run checkpoints.

The blend weight is swept on val gold. Val overstates a standalone risk-NLI
member relative to how it generalises, so a high weight would hand most of
the decision to the less reliable side; the question this script answers is
whether a SMALL dose (low w) lifts the ensemble while changing few posts.
It therefore prints the full curve for a human to inspect, and auto-writes
the test file only for the best LOW-weight point (w <= 0.35) that also beats
the ensemble by more than the noise floor.

Usage:
  python src/blend_risk_saved.py \
      --val artifacts/val_split.csv --test data/test.csv \
      --nli-run runs/risk_nli,runs/risk_nli_s2 \
      --ens-val runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
      --ens-test artifacts/ensembled_risk.csv \
      --out artifacts/risk_blend.csv
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
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1

PROBCOLS_VAL = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]
PROBCOLS_TEST = ["p_Indicator", "p_Ideation", "p_Behavior", "p_Attempt"]


def ens_val_probs(paths, row_ids):
    """Average saved per-model val probabilities, aligned to row_ids."""
    mats = []
    for p in paths:
        df = pd.read_csv(p).set_index("row_id")
        m = df.loc[row_ids, PROBCOLS_VAL].values.astype(float)
        mats.append(m)
    return np.mean(mats, axis=0)


def ens_test_probs(path, row_ids):
    df = pd.read_csv(path).set_index("row_id")
    return df.loc[row_ids, PROBCOLS_TEST].values.astype(float)


def nli_probs_for(df, run_dirs, max_len, device, use_bf16):
    from train_risk_nli import RiskNLIScorer, score_posts
    acc = []
    for rd in run_dirs:
        mn = open(os.path.join(rd, "tok")).read().strip()
        tok = AutoTokenizer.from_pretrained(mn)
        m = RiskNLIScorer(mn).to(device)
        # strict=False is needed because the checkpoint carries buffers the
        # scorer does not declare -- but silently tolerating MISSING weights
        # would mean scoring a blend member with a randomly initialised head
        # and raising no error anywhere. Report what was dropped, and refuse
        # outright if the classification head itself is absent.
        _st = torch.load(os.path.join(rd, "model.pt"), map_location=device)
        _missing, _unexpected = m.load_state_dict(_st, strict=False)
        if _missing:
            print(f"    {rd}: MISSING keys {list(_missing)[:6]}")
        if _unexpected:
            print(f"    {rd}: unexpected keys {list(_unexpected)[:6]}")
        if any(("head" in k or "classifier" in k or "score" in k)
               for k in _missing):
            raise SystemExit(
                f"{rd}: the classification head is missing from the checkpoint; "
                "this member would score with random weights. Refusing.")
        m.eval()
        acc.append(score_posts(m, df, tok, max_len, device, use_bf16))
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.mean(acc, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--nli-run", required=True)
    ap.add_argument("--ens-val", required=True, help="comma-sep val_predictions.csv paths")
    ap.add_argument("--ens-test", required=True, help="ensembled_risk.csv")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--force-weight", type=float, default=None,
                    help="skip the low-weight auto-pick and write test at THIS exact w")
    ap.add_argument("--out", default="artifacts/risk_blend.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    nli_runs = [r.strip() for r in args.nli_run.split(",")]
    ens_val_paths = [p.strip() for p in args.ens_val.split(",")]

    val_df = load(args.val)
    row_ids = val_df["row_id"].tolist()
    gold = [normalize_label(x) for x in val_df["risk_level"]]

    v_ens = ens_val_probs(ens_val_paths, row_ids)
    print("scoring risk-NLI on val...")
    v_nli = nli_probs_for(val_df, nli_runs, args.max_len, device, use_bf16)

    def wf1(probs):
        pred = [RISK_LEVELS[i] for i in probs.argmax(1)]
        return risk_weighted_f1(gold, pred), pred

    ens_f1, ens_pred = wf1(v_ens)
    nli_f1, _ = wf1(v_nli)
    print(f"\n  REAL ensemble (val, from saved probs): {ens_f1:.4f}  "
          f"(should be ~0.7575, matches transferred 0.7648)")
    print(f"  risk-NLI (val):                        {nli_f1:.4f}\n")
    print("  weight sweep (w = weight on risk-NLI):")
    rows = []
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * v_nli + (1 - w) * v_ens
        f1, pred = wf1(blend)
        nchg = sum(1 for a, b in zip(pred, ens_pred) if a != b)
        rows.append((w, f1, nchg))
        lowtag = "  <-low" if w <= 0.35 else ""
        print(f"    w={w:.2f}: wF1={f1:.4f}  ({nchg} posts changed){lowtag}")

    if args.force_weight is not None:
        w = args.force_weight
        blend = w * v_nli + (1 - w) * v_ens
        f1, pred = wf1(blend)
        nchg = sum(1 for a, b in zip(pred, ens_pred) if a != b)
        print(f"\n  FORCED w={w:.2f} -> val wF1 {f1:.4f} ({nchg} posts changed)")
        test_df = load(args.test)
        t_ids = test_df["row_id"].tolist()
        t_ens = ens_test_probs(args.ens_test, t_ids)
        print("scoring risk-NLI on test...")
        t_nli = nli_probs_for(test_df, nli_runs, args.max_len, device, use_bf16)
        blend_t = w * t_nli + (1 - w) * t_ens
        predt = [RISK_LEVELS[i] for i in blend_t.argmax(1)]
        out = pd.DataFrame({"row_id": t_ids,
                            "pred_risk_level": [p.capitalize() for p in predt]})
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out.to_csv(args.out, index=False)
        from collections import Counter
        print(f"  wrote {args.out}: {dict(Counter(predt))}")
        return

    # Pick the best LOW-weight point (w <= 0.35) that beats the ensemble by
    # more than the noise floor. The point must clear BOTH a score margin and
    # a minimum number of changed posts: a gain carried by one or two posts is
    # within val sampling noise and does not evidence a better model.
    low = [(f1, w, nchg) for (w, f1, nchg) in rows if 0 < w <= 0.35]
    low.sort(reverse=True)
    print()
    if low and low[0][0] - ens_f1 >= 0.008 and low[0][2] >= 8:
        best_f1, best_w, nchg = low[0]
        print(f"  BEST LOW-weight: w={best_w:.2f} -> {best_f1:.4f} "
              f"(+{best_f1-ens_f1:.4f} vs ensemble, {nchg} posts) — writing test")
        test_df = load(args.test)
        t_ids = test_df["row_id"].tolist()
        t_ens = ens_test_probs(args.ens_test, t_ids)
        print("scoring risk-NLI on test...")
        t_nli = nli_probs_for(test_df, nli_runs, args.max_len, device, use_bf16)
        blend = best_w * t_nli + (1 - best_w) * t_ens
        pred = [RISK_LEVELS[i] for i in blend.argmax(1)]
        out = pd.DataFrame({"row_id": t_ids,
                            "pred_risk_level": [p.capitalize() for p in pred]})
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out.to_csv(args.out, index=False)
        from collections import Counter
        print(f"  wrote {args.out}: {dict(Counter(pred))}")
    else:
        print("  No low-weight point clears the noise floor (+0.008 & >=8 posts).")
        print("  Low-dose risk-NLI does NOT reliably lift the real ensemble.")
        print("  (High-w points are untrustworthy: at large w the blend is")
        print("   dominated by the NLI member alone, which is the weaker of")
        print("   the two components.)")


if __name__ == "__main__":
    main()
