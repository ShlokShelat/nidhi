"""
rule_ablation.py

Per team's methodology: doesn't measure gold-label precision on raw
text (that only tells you if a pattern correlates with a label, not
whether it would help your ACTUAL deployed system). Instead:

  1. Compute the REAL deployed ensemble's baseline prediction for
     every val post (0.30*risk-NLI + 0.70*(0.5*equal-3-model + 0.5*friend),
     the exact formula confirmed to reproduce the champion byte-for-byte).
  2. For each candidate rule, find which val posts the trigger fires on.
  3. Apply the rule's override to get a "with-rule" prediction.
  4. Compare against gold and the baseline:
       CORRECTED = baseline wrong, rule makes it right
       BROKEN    = baseline right, rule makes it wrong
       NEUTRAL   = rule fires but prediction unchanged, or fires and
                   is wrong before and after (no net effect)
  5. Report per-rule: matches, corrected, broken, net, and the
     resulting change in official risk weighted-F1.

Usage:
  python src/rule_ablation.py \
      --val artifacts/val_split.csv \
      --val-probs runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
      --nli-run runs/risk_nli,runs/risk_nli_s2 \
      --friend-probs-val artifacts/friend_risk_probs_val.csv
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


def match_any(text, patterns):
    text_l = str(text).lower()
    return any(re.search(p, text_l) for p in patterns)


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


def apply_rule(baseline_label, trigger_fires, action):
    from_labels, to_label = action
    if not trigger_fires:
        return baseline_label
    if from_labels is not None and baseline_label not in from_labels:
        return baseline_label
    return to_label


def evaluate_rule(name, df, baseline_preds, gold, trigger_patterns, action,
                  compound_absent=None):
    if compound_absent is not None:
        present, absent = trigger_patterns, compound_absent
        fires = df["post"].apply(
            lambda t: match_any(t, present) and not match_any(t, absent))
    else:
        fires = df["post"].apply(lambda t: match_any(t, trigger_patterns))

    n_fire = int(fires.sum())
    if n_fire == 0:
        print(f"\n{name}: 0 matches on val -- cannot evaluate")
        return None

    new_preds = [apply_rule(b, f, action) for b, f in zip(baseline_preds, fires)]

    corrected, broken, neutral_still_wrong, unaffected = 0, 0, 0, 0
    for i in range(len(df)):
        if not fires.iloc[i]:
            continue
        b, n, g = baseline_preds[i], new_preds[i], gold[i]
        if b == n:
            unaffected += 1
        elif b != g and n == g:
            corrected += 1
        elif b == g and n != g:
            broken += 1
        else:
            neutral_still_wrong += 1

    base_f1 = risk_weighted_f1(gold, baseline_preds)
    new_f1 = risk_weighted_f1(gold, new_preds)

    print(f"\n{'=' * 90}")
    print(f"RULE: {name}")
    print(f"{'=' * 90}")
    print(f"  posts matching trigger: {n_fire}")
    print(f"  unaffected (rule fired, no change vs baseline): {unaffected}")
    print(f"  CORRECTED (baseline wrong -> rule right): {corrected}")
    print(f"  BROKEN    (baseline right -> rule wrong): {broken}")
    print(f"  changed but still wrong either way: {neutral_still_wrong}")
    print(f"  NET: {corrected - broken:+d}")
    print(f"  baseline weighted-F1: {base_f1:.4f}  |  with-rule weighted-F1: {new_f1:.4f}"
         f"  |  delta: {new_f1 - base_f1:+.4f}")

    if corrected > 0:
        print(f"  example CORRECTED posts:")
        shown = 0
        for i in range(len(df)):
            if shown >= 3:
                break
            if fires.iloc[i] and baseline_preds[i] != gold[i] and new_preds[i] == gold[i]:
                print(f"    [{gold[i]}] baseline={baseline_preds[i]} -> rule={new_preds[i]}: "
                     f"{str(df.iloc[i]['post'])[:100]}")
                shown += 1
    if broken > 0:
        print(f"  example BROKEN posts:")
        shown = 0
        for i in range(len(df)):
            if shown >= 3:
                break
            if fires.iloc[i] and baseline_preds[i] == gold[i] and new_preds[i] != gold[i]:
                print(f"    [{gold[i]}] baseline={baseline_preds[i]} -> rule={new_preds[i]}: "
                     f"{str(df.iloc[i]['post'])[:100]}")
                shown += 1

    return {"name": name, "matches": n_fire, "corrected": corrected,
            "broken": broken, "net": corrected - broken,
            "f1_delta": new_f1 - base_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-probs", required=True)
    ap.add_argument("--nli-run", required=True)
    ap.add_argument("--friend-probs-val", required=True)
    ap.add_argument("--max-len", type=int, default=384)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")

    val_df = pd.read_csv(args.val)
    row_ids = val_df["row_id"].tolist()
    gold = [normalize_label(x) for x in val_df["risk_level"]]

    prob_paths = [p.strip() for p in args.val_probs.split(",")]
    per_model_probs = [load_val_probs(p, row_ids) for p in prob_paths]
    equal_avg = np.mean(per_model_probs, axis=0)

    print("scoring risk-NLI on val...")
    nli_runs = [r.strip() for r in args.nli_run.split(",")]
    nli_probs = nli_probs_for(val_df, nli_runs, args.max_len, device, use_bf16)

    friend_df = pd.read_csv(args.friend_probs_val).set_index("row_id")
    friend_probs = np.zeros((len(row_ids), 4))
    n_missing = 0
    for i, rid in enumerate(row_ids):
        if rid in friend_df.index:
            p = friend_df.loc[rid, FRIEND_COLS].values.astype(float)
        else:
            p = equal_avg[i]
            n_missing += 1
        friend_probs[i] = p / p.sum()
    if n_missing:
        print(f"WARNING: {n_missing} val rows missing friend prediction, used our-avg fallback")

    base_ensemble = 0.5 * equal_avg + 0.5 * friend_probs
    deployed_probs = 0.30 * nli_probs + 0.70 * base_ensemble
    baseline_preds = [RISK_LEVELS[int(np.argmax(p))] for p in deployed_probs]

    baseline_f1 = risk_weighted_f1(gold, baseline_preds)
    print(f"\nDEPLOYED ENSEMBLE baseline (real formula) weighted-F1 on val: {baseline_f1:.4f}\n")

    results = []

    results.append(evaluate_rule(
        "Urgency present + method absent -> cap at IDEATION (from Behavior/Attempt)",
        val_df, baseline_preds, gold,
        URGENCY_PHRASES, action=(["behavior", "attempt"], "ideation"),
        compound_absent=METHOD_PHRASES))

    results.append(evaluate_rule(
        "Completed self-harm act phrasing -> escalate BEHAVIOR to ATTEMPT",
        val_df, baseline_preds, gold,
        COMPLETED_ACT_PHRASES, action=(["behavior"], "attempt")))

    results.append(evaluate_rule(
        "Passive/fatalistic death phrasing -> cap at INDICATOR (from Ideation)",
        val_df, baseline_preds, gold,
        PASSIVE_DEATH_PHRASES, action=(["ideation"], "indicator")))

    results.append(evaluate_rule(
        "Post starts with 'Suicidal notes' -> escalate IDEATION to BEHAVIOR",
        val_df, baseline_preds, gold,
        SUICIDE_NOTE_TITLE, action=(["ideation"], "behavior")))

    print(f"\n{'=' * 90}")
    print("SUMMARY (sorted by net corrections)")
    print(f"{'=' * 90}")
    valid = [r for r in results if r]
    valid.sort(key=lambda r: -r["net"])
    print(f"{'rule':<70}{'matches':>9}{'net':>6}{'f1_delta':>10}")
    for r in valid:
        print(f"{r['name'][:68]:<70}{r['matches']:>9}{r['net']:>6}{r['f1_delta']:>+10.4f}")


if __name__ == "__main__":
    main()
