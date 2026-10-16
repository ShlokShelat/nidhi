"""
combined_rule_ablation.py

Companion to rule_ablation.py. Independent per-rule evaluation can
hide overlap: two rules that each look good alone might correct the
SAME posts (no additive benefit) or interact badly when stacked. This
applies a chosen ORDERED SEQUENCE of rules cumulatively and reports
weighted-F1 at each stage:

  baseline
  baseline + RuleA
  baseline + RuleA + RuleB
  baseline + RuleA + RuleB + RuleC
  ...

Only run this AFTER rule_ablation.py has told you which individual
rules have positive net corrections and positive weighted-F1 delta —
pass just those, in whatever order you want to test (--rules is a
comma-separated list of rule keys; order matters since later rules
in the sequence apply to already-modified predictions).

Available rule keys: urgency, completed_act, passive_fatalism, suicide_note

Usage:
  python src/combined_rule_ablation.py \
      --val artifacts/val_split.csv \
      --val-probs runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
      --nli-run runs/risk_nli,runs/risk_nli_s2 \
      --friend-probs-val artifacts/friend_risk_probs_val.csv \
      --rules urgency,completed_act
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1

PROBCOLS = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]
FRIEND_COLS = ["Indicator", "Ideation", "Behavior", "Attempt"]

URGENCY_PHRASES = [
    r"\babout to\b", r"\bgoing to end\b", r"\bgoing to kill\b",
    r"\bget(ting)? up the nerve\b", r"\bbefore .{0,20} (gets?|comes?) back\b",
    r"\btonight\b.{0,30}\b(kill|end|die|suicide)\b",
    r"\bwanna end it\b", r"\bwant to end it\b",
    r"\bgave myself\b.{0,20}(month|week|day|deadline)",
    r"\bsoon i'?ll be gone\b", r"\bi don'?t know if i'?m doing it\b",
]

METHOD_PHRASES = [
    r"\bpills?\b", r"\bgun\b", r"\bloaded gun\b",
    r"\baccess to (a |an )?(loaded )?(gun|weapon|firearm)\b",
    r"\brope\b", r"\bhang(ing|ed)?\b",
    r"\boverdos(e|ed|ing)\b", r"\bcut(ting)? my\b", r"\bcut my (wrist|neck|arm)",
    r"\bbridge\b", r"\bjump(ing|ed)? (off|from)\b", r"\bpoison\b",
    r"\bblade\b", r"\bknife\b", r"\bnoose\b",
    r"\b(took|taking|taken|take)\b.{0,15}\d+\b", r"\bswallow(ed|ing)?\b",
]

COMPLETED_ACT_PHRASES = [
    r"\bi'?ve cut\b", r"\bi cut my\b",
    r"\b(i|just|already)\b.{0,10}\btook\b.{0,15}\d+\b",
    r"\bi'?ve taken\b", r"\bi have taken\b", r"\btaken \d+\b",
    r"\bi swallowed\b", r"\bi overdosed\b",
]

PASSIVE_DEATH_PHRASES = [
    r"\bwait(ing)? for (my )?death\b", r"\blet my body decay\b",
    r"\bi'?ll probably die\b", r"\bnothing to live for\b",
    r"\bi give up\b.{0,30}\bdone\b",
]

SUICIDE_NOTE_TITLE = [r"^suicidal notes?\b"]

RULE_DEFS = {
    "urgency": {
        "name": "Urgency present + method absent -> cap at IDEATION",
        "present": URGENCY_PHRASES, "absent": METHOD_PHRASES,
        "from_labels": ["behavior", "attempt"], "to_label": "ideation",
    },
    "completed_act": {
        "name": "Completed act phrasing -> escalate BEHAVIOR to ATTEMPT",
        "present": COMPLETED_ACT_PHRASES, "absent": None,
        "from_labels": ["behavior"], "to_label": "attempt",
    },
    "passive_fatalism": {
        "name": "Passive/fatalistic death phrasing -> cap at INDICATOR",
        "present": PASSIVE_DEATH_PHRASES, "absent": None,
        "from_labels": ["ideation"], "to_label": "indicator",
    },
    "suicide_note": {
        "name": "'Suicidal notes' title -> escalate IDEATION to BEHAVIOR",
        "present": SUICIDE_NOTE_TITLE, "absent": None,
        "from_labels": ["ideation"], "to_label": "behavior",
    },
}


def match_any(text, patterns):
    text_l = str(text).lower()
    return any(re.search(p, text_l) for p in patterns)


def apply_one_rule(posts, preds, rule_key):
    rd = RULE_DEFS[rule_key]
    new_preds = list(preds)
    n_fired_and_changed = 0
    for i, post in enumerate(posts):
        has_present = match_any(post, rd["present"])
        has_absent = rd["absent"] is not None and match_any(post, rd["absent"])
        fires = has_present and not has_absent
        if not fires:
            continue
        if rd["from_labels"] is not None and preds[i] not in rd["from_labels"]:
            continue
        if new_preds[i] != rd["to_label"]:
            n_fired_and_changed += 1
        new_preds[i] = rd["to_label"]
    return new_preds, n_fired_and_changed


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


def load_val_probs(path, row_ids):
    df = pd.read_csv(path).set_index("row_id")
    return df.loc[row_ids, PROBCOLS].values.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-probs", required=True)
    ap.add_argument("--nli-run", required=True)
    ap.add_argument("--friend-probs-val", required=True)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--rules", required=True,
                    help="comma-sep, ORDERED, from {urgency,completed_act,"
                         "passive_fatalism,suicide_note} — only pass rules "
                         "rule_ablation.py already showed as net-positive")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")

    val_df = pd.read_csv(args.val)
    row_ids = val_df["row_id"].tolist()
    posts = val_df["post"].tolist()
    gold = [normalize_label(x) for x in val_df["risk_level"]]

    prob_paths = [p.strip() for p in args.val_probs.split(",")]
    per_model_probs = [load_val_probs(p, row_ids) for p in prob_paths]
    equal_avg = np.mean(per_model_probs, axis=0)

    print("scoring risk-NLI on val...")
    nli_runs = [r.strip() for r in args.nli_run.split(",")]
    nli_probs = nli_probs_for(val_df, nli_runs, args.max_len, device, use_bf16)

    friend_df = pd.read_csv(args.friend_probs_val).set_index("row_id")
    friend_probs = np.zeros((len(row_ids), 4))
    for i, rid in enumerate(row_ids):
        if rid in friend_df.index:
            p = friend_df.loc[rid, FRIEND_COLS].values.astype(float)
        else:
            p = equal_avg[i]
        friend_probs[i] = p / p.sum()

    base_ensemble = 0.5 * equal_avg + 0.5 * friend_probs
    deployed_probs = 0.30 * nli_probs + 0.70 * base_ensemble
    baseline_preds = [RISK_LEVELS[int(np.argmax(p))] for p in deployed_probs]

    rule_keys = [r.strip() for r in args.rules.split(",")]
    for k in rule_keys:
        if k not in RULE_DEFS:
            raise SystemExit(f"unknown rule key: {k}. choices: {list(RULE_DEFS)}")

    print(f"\n{'=' * 90}")
    print("SEQUENTIAL ABLATION")
    print(f"{'=' * 90}")

    current_preds = baseline_preds
    stage_name = "baseline (no rules)"
    f1 = risk_weighted_f1(gold, current_preds)
    print(f"{stage_name:<55}weighted-F1 = {f1:.4f}")

    applied = []
    for k in rule_keys:
        prev_preds = current_preds
        current_preds, n_changed = apply_one_rule(posts, current_preds, k)
        applied.append(RULE_DEFS[k]["name"])
        f1 = risk_weighted_f1(gold, current_preds)

        corrected = sum(1 for i in range(len(gold))
                        if prev_preds[i] != gold[i] and current_preds[i] == gold[i])
        broken = sum(1 for i in range(len(gold))
                     if prev_preds[i] == gold[i] and current_preds[i] != gold[i])

        stage_name = "+ " + " + ".join(rule_keys[:len(applied)])
        print(f"{stage_name:<55}weighted-F1 = {f1:.4f}   "
             f"(this stage: {n_changed} changed, {corrected} corrected, {broken} broken)")

    print(f"\nFinal: baseline {risk_weighted_f1(gold, baseline_preds):.4f} -> "
         f"combined {risk_weighted_f1(gold, current_preds):.4f}  "
         f"(delta {risk_weighted_f1(gold, current_preds) - risk_weighted_f1(gold, baseline_preds):+.4f})")


if __name__ == "__main__":
    main()
