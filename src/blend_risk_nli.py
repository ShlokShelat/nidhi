"""
Blend risk-NLI softmax probabilities INTO the existing risk ensemble
(augment, don't replace).

Rationale: raw risk-NLI overfit val_split (val 0.80 -> test 0.745). But
the existing risk ensemble transferred cleanly (val 0.7646 -> test 0.7648).
Cross-architecture probability blending at a single global weight is the
combination that has most reliably transferred on this task. A low-weight
mix of risk-NLI into the stable ensemble may lift the posts risk-NLI gets
right while the ensemble anchors its overfit.

Validated on val_split gold (both model families held val out of training).
Reports weighted-F1 across a weight sweep and, because a full risk change
moves MANY posts (unlike the evidence backoff), val is measurable here —
but the blend must still beat the ensemble by a clear margin before it is
worth recommending, and the script reports how many posts change.

Inputs:
  --nli-probs   : artifacts/risk_nli_ens_probs.npy  (N_test x 4, from predict_risk_nli)
  --nli-val-run : risk-NLI run dir(s) to score val   (comma-sep)
  --ens-val     : existing ensemble's val risk probs (risk_v1 etc.) — if no
                  saved file is available, this script re-scores them.

Because probability files differ across the pipeline, this script recomputes
BOTH families' probabilities on val_split from their run dirs, so alignment
is guaranteed.

Usage:
  python src/blend_risk_nli.py \
      --val artifacts/val_split.csv --test data/test.csv \
      --nli-run runs/risk_nli,runs/risk_nli_s2 \
      --ens-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
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


def nli_probs_for(df, run_dirs, max_len, device, use_bf16):
    from train_risk_nli import RiskNLIScorer, score_posts
    acc = []
    for rd in run_dirs:
        mn = open(os.path.join(rd, "tok")).read().strip()
        tok = AutoTokenizer.from_pretrained(mn)
        m = RiskNLIScorer(mn).to(device)
        m.load_state_dict(torch.load(os.path.join(rd, "model.pt"),
                                     map_location=device), strict=False)
        m.eval()
        acc.append(score_posts(m, df, tok, max_len, device, use_bf16))
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.mean(acc, axis=0)


def ens_probs_for(df, run_dirs, max_len, device, use_bf16):
    """Softmax risk probs from the conventional DualHead risk models."""
    from train_risk import DualHeadModel, build_text
    acc = []
    for rd in run_dirs:
        # model name stored in run_config.json or default
        import json
        cfgp = os.path.join(rd, "run_config.json")
        mn = json.load(open(cfgp))["model"] if os.path.exists(cfgp) \
            else "microsoft/deberta-v3-large"
        tok = AutoTokenizer.from_pretrained(mn)
        m = DualHeadModel(mn).to(device)
        # checkpoint filename varies
        ckpt = None
        for name in ("model_final.pt", "model.pt", "best_model.pt"):
            if os.path.exists(os.path.join(rd, name)):
                ckpt = os.path.join(rd, name); break
        m.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        m.eval()
        probs = np.zeros((len(df), len(RISK_LEVELS)))
        d = df.copy()
        d["user_post_count"] = d.groupby("anon_user_id")["post"].transform("size")
        d["post_index_in_user"] = d.groupby("anon_user_id").cumcount() + 1
        with torch.no_grad():
            for i, (_, row) in enumerate(d.iterrows()):
                enc = tok(build_text(row, True), truncation=True,
                          max_length=max_len, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()
                       if k in ("input_ids", "attention_mask")}
                with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    logits = m(**enc)["logits"]
                probs[i] = torch.softmax(logits.float(), -1).cpu().numpy()[0]
        acc.append(probs)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.mean(acc, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--nli-run", required=True)
    ap.add_argument("--ens-run", required=True)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--out", default="artifacts/risk_blend.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    nli_runs = [r.strip() for r in args.nli_run.split(",")]
    ens_runs = [r.strip() for r in args.ens_run.split(",")]

    val_df = load(args.val)
    gold = [normalize_label(x) for x in val_df["risk_level"]]

    print("scoring risk-NLI on val...")
    v_nli = nli_probs_for(val_df, nli_runs, args.max_len, device, use_bf16)
    print("scoring conventional ensemble on val...")
    v_ens = ens_probs_for(val_df, ens_runs, args.max_len, device, use_bf16)

    def wf1(probs):
        pred = [RISK_LEVELS[i] for i in probs.argmax(1)]
        return risk_weighted_f1(gold, pred), pred

    ens_f1, ens_pred = wf1(v_ens)
    nli_f1, _ = wf1(v_nli)
    print(f"\n  ensemble alone (val): {ens_f1:.4f}")
    print(f"  risk-NLI alone (val): {nli_f1:.4f}\n")
    print("  weight sweep (w = weight on risk-NLI):")
    best = (ens_f1, 0.0, ens_pred)
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * v_nli + (1 - w) * v_ens
        f1, pred = wf1(blend)
        nchg = sum(1 for a, b in zip(pred, ens_pred) if a != b)
        mark = " *" if f1 > best[0] else ""
        print(f"    w={w:.2f}: wF1={f1:.4f}  (changes {nchg} posts vs ensemble){mark}")
        if f1 > best[0]:
            best = (f1, w, pred)

    best_f1, best_w, _ = best
    nchg = sum(1 for a, b in zip(best[2], ens_pred) if a != b)
    print(f"\n  BEST w_nli={best_w:.2f} -> {best_f1:.4f} "
          f"(vs ensemble {ens_f1:.4f}, delta {best_f1-ens_f1:+.4f}, "
          f"{nchg} posts changed)")

    # noise-floor guard: require both a real margin AND enough posts moved
    if best_w == 0.0 or best_f1 - ens_f1 < 0.008 or nchg < 12:
        print("\n  *** blend does not clearly beat the ensemble (margin/"
              "posts below noise floor) — NOT worth an upload ***")
        return

    print("\n  blend beats ensemble on val by a clear margin — generating test")
    test_df = load(args.test)
    t_nli = nli_probs_for(test_df, nli_runs, args.max_len, device, use_bf16)
    t_ens = ens_probs_for(test_df, ens_runs, args.max_len, device, use_bf16)
    blend = best_w * t_nli + (1 - best_w) * t_ens
    pred = [RISK_LEVELS[i] for i in blend.argmax(1)]
    out = pd.DataFrame({
        "row_id": test_df["row_id"].tolist(),
        "pred_risk_level": [p.capitalize() for p in pred],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    from collections import Counter
    print(f"  wrote {args.out}: {dict(Counter(pred))}")


if __name__ == "__main__":
    main()
