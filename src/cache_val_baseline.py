"""
cache_val_baseline.py

Computes the deployed champion's val predictions ONCE and caches them, so
rule_ablation_v2.py (and anything else that needs "what does the champion
currently predict on val") never has to reload the risk-NLI ensemble again.

The blend reproduced here is byte-for-byte the one in mine_champion_failures.py
and rule_ablation.py:

    base     = 0.5 * mean(3 risk seeds) + 0.5 * friend
    deployed = 0.30 * risk_NLI_ensemble + 0.70 * base
    champion = apply_rules(post, argmax(deployed))     # urgency, completed_act

It also closes a real gap in mine_champion_failures.py. That script emits a
case only when the base risk classifiers are unanimous-and-WRONG (P1) or
non-unanimous (P2). A post where all four base voters agree on the CORRECT
label but the NLI blend or a rule then flips the final answer to a wrong one
falls through both branches and is silently invisible in the failure files.
On the current val split that bucket is 222 posts, and at least two of them
(P01847 via urgency, P00755 via completed_act) do have a deployed rule firing
on them. This script reports that bucket explicitly under BLIND SPOT.

Run once on the GPU box:
  python src/cache_val_baseline.py \
      --val artifacts/val_split.csv \
      --val-probs runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
      --nli-run runs/risk_nli,runs/risk_nli_s2 \
      --friend-probs-val artifacts/friend_risk_probs_val.csv \
      --out artifacts/val_deployed_baseline.csv
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1
from apply_rules_to_test import apply_rules

PROBCOLS = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]
FRIEND_COLS = ["Indicator", "Ideation", "Behavior", "Attempt"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-probs", required=True)
    ap.add_argument("--nli-run", required=True)
    ap.add_argument("--friend-probs-val", required=True)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--rules", default="urgency,completed_act")
    ap.add_argument("--out", default="artifacts/val_deployed_baseline.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")

    val_df = pd.read_csv(args.val)
    row_ids = val_df["row_id"].tolist()
    gold = [normalize_label(x) for x in val_df["risk_level"]]

    per_model = []
    for p in [p.strip() for p in args.val_probs.split(",")]:
        d = pd.read_csv(p).set_index("row_id")
        per_model.append(d.loc[row_ids, PROBCOLS].values.astype(float))
    equal_avg = np.mean(per_model, axis=0)

    print("scoring risk-NLI on val (this is the only slow part)...")
    nli_probs = nli_probs_for(val_df, [r.strip() for r in args.nli_run.split(",")],
                              args.max_len, device, use_bf16)

    friend_df = pd.read_csv(args.friend_probs_val).set_index("row_id")
    friend = np.zeros((len(row_ids), 4))
    n_missing = 0
    for i, rid in enumerate(row_ids):
        if rid in friend_df.index:
            p = friend_df.loc[rid, FRIEND_COLS].values.astype(float)
        else:
            p = equal_avg[i]
            n_missing += 1
        friend[i] = p / p.sum()
    if n_missing:
        print(f"WARNING: {n_missing} val rows missing friend prediction, "
              f"used our-avg fallback (same policy as the champion scripts)")

    deployed = 0.30 * nli_probs + 0.70 * (0.5 * equal_avg + 0.5 * friend)
    pre_rule = [RISK_LEVELS[int(np.argmax(p))] for p in deployed]

    rule_keys = tuple(r.strip() for r in args.rules.split(","))
    champion = [apply_rules(post, lbl, rule_keys)[0]
                for post, lbl in zip(val_df["post"], pre_rule)]

    votes = np.stack([np.argmax(x, axis=1) for x in per_model]
                     + [np.argmax(friend, axis=1)], axis=1)
    unanimous = np.array([len(set(v)) == 1 for v in votes])
    base_lbl = np.array([RISK_LEVELS[v[0]] for v in votes])
    unan_correct = unanimous & (base_lbl == np.array(gold))

    out = pd.DataFrame({
        "row_id": row_ids,
        "gold": gold,
        "pre_rule_pred": pre_rule,
        "champion_final": champion,
        "base_unanimous": unanimous,
        "base_unanimous_correct": unan_correct,
        "approximate": False,
    })
    for j, c in enumerate(PROBCOLS):
        out[c] = deployed[:, j]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    f1_pre = risk_weighted_f1(gold, pre_rule)
    f1_post = risk_weighted_f1(gold, champion)
    print(f"\nval weighted-F1  pre-rule {f1_pre:.4f} -> champion {f1_post:.4f} "
          f"({f1_post - f1_pre:+.4f})")
    print(f"posts changed by the rule layer: "
          f"{sum(1 for a, b in zip(pre_rule, champion) if a != b)}")

    blind = out[out.base_unanimous_correct & (out.champion_final != out.gold)]
    print(f"\nBLIND SPOT (base voters unanimous AND correct, but champion ends "
          f"up wrong):\n  {len(blind)} posts -- these never appear in "
          f"champion_p1/p2_failures.txt")
    for _, r in blind.iterrows():
        print(f"    {r.row_id}  gold={r.gold:<9} pre_rule={r.pre_rule_pred:<9} "
              f"champion={r.champion_final}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
