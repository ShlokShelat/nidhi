"""
mine_champion_failures.py

Stage 1 of the Error-Driven Symbol Discovery framework: mine the
CURRENT, real error set against the actual deployed champion (rules
included). Imports apply_rules() from apply_rules_to_test.py -- the
single source of truth for rule logic -- rather than re-implementing
the regexes here, so this script can never silently drift out of sync
with what's actually deployed.

TERMINOLOGY FIX: "P1" here means RISK-CLASSIFIER UNANIMITY (3 base
seeds + friend model), NOT "every model including NLI agrees" -- NLI
only enters later via the 0.30/0.70 blend, it doesn't vote in this
agreement check. Labeled explicitly as such throughout, rather than
the more ambiguous "all agree" from the earlier version.

BLIND-SPOT FIX (P0): the previous version asked "where did the models
disagree?" when the question that matters is "where did the DEPLOYED
SYSTEM get it wrong?". Those are not the same, and the difference was
silently dropping cases. A post where all four base voters agree on the
CORRECT label, but the 0.30 NLI blend or a rule then flips the final
answer to a wrong one, satisfied neither the old P1 branch (which
required the base voters to be wrong) nor the P2 branch (which required
them to disagree) -- so it vanished from the failure files entirely. On
the current val split that bucket held 222 candidate posts, and at least
two real errors: P01847 (flipped by urgency) and P00755 (flipped by
completed_act), both self-inflicted by the rule layer and both invisible.

The three buckets are now an EXHAUSTIVE partition of every post the
deployed champion gets wrong:

  P0 - base voters unanimous AND correct; the blend and/or the rule
       layer introduced the error. Highest-value bucket: these are
       regressions our own pipeline caused, not model failures.
  P1 - base voters unanimous and wrong.
  P2 - base voters disagree.

P0 cases are additionally attributed to whichever stage caused the
flip (BLEND, RULE, or both).

Every P1/P2 case's output includes a RULE TRACE showing which rules
were checked, which fired, and whether they changed the prediction --
including the important "rule fired, changed prediction, STILL wrong"
category, which is flagged separately: these are cases where a rule
helped but wasn't sufficient, often the richest source for the NEXT
rule to discover.

Usage:
  python src/mine_champion_failures.py \
      --val artifacts/val_split.csv \
      --val-probs runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
      --nli-run runs/risk_nli,runs/risk_nli_s2 \
      --friend-probs-val artifacts/friend_risk_probs_val.csv \
      --out-p1 artifacts/champion_p1_failures.txt \
      --out-p2 artifacts/champion_p2_failures.txt
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS, normalize_label
from data_audit import load
from apply_rules_to_test import apply_rules, RULE_DEFS  # single source of truth

PROBCOLS = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]
FRIEND_COLS = ["Indicator", "Ideation", "Behavior", "Attempt"]


def wrap(text, width=95, indent="    "):
    return ("\n" + indent).join(textwrap.wrap(str(text), width=width))


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


def format_trace(trace):
    lines = ["RULE TRACE (deployed order):"]
    for t in trace:
        status = []
        status.append("FIRED" if t["fired"] else "did not fire")
        if t["fired"] and not t["applicable"]:
            status.append("but label wasn't eligible for this rule")
        if t["changed"]:
            status.append("-> CHANGED prediction")
        lines.append(f"    {t['rule']:<20} {', '.join(status)}")
    return "\n".join(lines)


def format_case(n, label, rid, gold, post, votes_str, ens_pred_pre_rules,
                after_rules, trace, insufficient, cause=None):
    lines = []
    tag = "  [RULE FIRED BUT INSUFFICIENT]" if insufficient else ""
    if cause:
        tag += f"  [ERROR INTRODUCED BY: {cause}]"
    lines.append("=" * 100)
    lines.append(f"{label} {n}  row_id={rid}  GOLD={gold}{tag}")
    lines.append("=" * 100)
    lines.append(f"POST:\n    {wrap(post)}")
    lines.append("")
    lines.append(votes_str)
    lines.append("")
    lines.append(f"ENSEMBLE PREDICTION (before rules): {ens_pred_pre_rules}")
    lines.append(format_trace(trace))
    lines.append(f"FINAL PREDICTION (after rules): {after_rules}   [STILL WRONG vs gold={gold}]")
    if insufficient:
        lines.append("")
        lines.append(">>> A rule changed this prediction but it's still wrong -- this is a")
        lines.append(">>> strong candidate source for discovering the NEXT rule/family.")
    lines.append("")
    lines.append("YOUR NOTES: _______________________________________________")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-probs", required=True)
    ap.add_argument("--names", default="risk_v1,risk_v1_s14,risk_v1_s15")
    ap.add_argument("--nli-run", required=True)
    ap.add_argument("--friend-probs-val", required=True)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--rules", default="urgency,completed_act")
    ap.add_argument("--out-p0", default="artifacts/champion_p0_failures.txt")
    ap.add_argument("--out-p1", default="artifacts/champion_p1_failures.txt")
    ap.add_argument("--out-p2", default="artifacts/champion_p2_failures.txt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")

    rule_keys = [r.strip() for r in args.rules.split(",")]
    for k in rule_keys:
        if k not in RULE_DEFS:
            raise SystemExit(f"unknown rule: {k}")

    val_df = load(args.val)
    row_ids = val_df["row_id"].tolist()
    gold = [normalize_label(x) for x in val_df["risk_level"]]
    posts = val_df["post"].tolist()

    names = [n.strip() for n in args.names.split(",")]
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

    p0_cases, p1_cases, p2_cases, fixed_cases = [], [], [], []
    still_wrong_p1, fixed_by_rules_p1 = 0, 0
    insufficient_count = 0
    p0_cause = {"BLEND": 0, "RULE": 0, "BLEND+RULE": 0}

    for i, rid in enumerate(row_ids):
        probs_list = [per_model_probs[m][i] for m in range(len(names))] + [friend_probs[i]]
        votes = [RISK_LEVELS[int(np.argmax(p))] for p in probs_list]
        all_names = names + ["friend_model"]

        ens_pred = RISK_LEVELS[int(np.argmax(deployed_probs[i]))]
        after_rules, trace = apply_rules(posts[i], ens_pred, rule_keys)

        risk_classifiers_unanimous = len(set(votes)) == 1  # 3 seeds + friend only, NOT NLI
        risk_models_wrong = risk_classifiers_unanimous and votes[0] != gold[i]
        ens_wrong_after = after_rules != gold[i]
        any_rule_changed = any(t["changed"] for t in trace)
        insufficient = any_rule_changed and ens_wrong_after

        votes_str = ("MODEL VOTES (3 risk seeds + friend -- NLI enters separately "
                     "via the 0.30/0.70 blend, not counted here):\n" +
                     "\n".join(f"    {n:<18} {v}" for n, v in zip(all_names, votes)))

        # P0: the base voters had it RIGHT and our own pipeline broke it.
        if risk_classifiers_unanimous and not risk_models_wrong:
            if ens_wrong_after:
                blend_flipped = ens_pred != votes[0]
                rule_flipped = after_rules != ens_pred
                cause = ("BLEND+RULE" if blend_flipped and rule_flipped
                         else "RULE" if rule_flipped else "BLEND")
                p0_cause[cause] += 1
                p0_cases.append((rid, gold[i], posts[i], votes_str, ens_pred,
                                 after_rules, trace, insufficient, cause))
            continue

        if risk_models_wrong:
            if ens_wrong_after:
                still_wrong_p1 += 1
                if insufficient:
                    insufficient_count += 1
                p1_cases.append((rid, gold[i], posts[i], votes_str, ens_pred, after_rules, trace, insufficient))
            else:
                fixed_by_rules_p1 += 1
                fixed_cases.append((rid, gold[i], trace))  # for per-rule "fully fixed" attribution
        elif ens_wrong_after:
            p2_cases.append((rid, gold[i], posts[i], votes_str, ens_pred, after_rules, trace, insufficient))

    # sort so [RULE FIRED BUT INSUFFICIENT] cases come first in both files --
    # these are the richest source for the next rule, read them first
    p1_cases.sort(key=lambda c: not c[7])
    p2_cases.sort(key=lambda c: not c[7])

    # P0 first: errors our own pipeline introduced on posts the base models
    # already had right. Sorted rule-caused first -- those are directly and
    # cheaply fixable by tightening the rule that caused them.
    p0_cases.sort(key=lambda c: (c[8] == "BLEND", c[0]))
    out0 = []
    for n, (rid, g, post, vstr, ep, ar, tr, ins, cause) in enumerate(p0_cases, start=1):
        out0.append(format_case(
            n, "P0 CASE (base voters unanimous AND CORRECT -- our pipeline broke it)",
            rid, g, post, vstr, ep, ar, tr, ins, cause))
    os.makedirs(os.path.dirname(args.out_p0) or ".", exist_ok=True)
    with open(args.out_p0, "w") as f:
        f.write("\n".join(out0))

    out1 = []
    for n, (rid, g, post, vstr, ep, ar, tr, ins) in enumerate(p1_cases, start=1):
        out1.append(format_case(n, "P1 CASE (risk classifiers unanimous, still wrong after rules)",
                                rid, g, post, vstr, ep, ar, tr, ins))
    os.makedirs(os.path.dirname(args.out_p1) or ".", exist_ok=True)
    with open(args.out_p1, "w") as f:
        f.write("\n".join(out1))

    out2 = []
    for n, (rid, g, post, vstr, ep, ar, tr, ins) in enumerate(p2_cases, start=1):
        out2.append(format_case(n, "P2 CASE (disagreement, still wrong after rules)",
                                rid, g, post, vstr, ep, ar, tr, ins))
    os.makedirs(os.path.dirname(args.out_p2) or ".", exist_ok=True)
    with open(args.out_p2, "w") as f:
        f.write("\n".join(out2))

    total_wrong = len(p0_cases) + len(p1_cases) + len(p2_cases)
    print(f"\nwrote {args.out_p0}: {len(p0_cases)} P0 failures -- posts the base "
          f"voters got RIGHT that the deployed pipeline then broke "
          f"(cause: {p0_cause}). These were invisible to every previous "
          f"mining pass; rule-caused ones are sorted first and are the "
          f"cheapest wins available.")
    print(f"wrote {args.out_p1}: {len(p1_cases)} genuine P1 failures "
         f"(rules already fixed {fixed_by_rules_p1} of the original risk-classifier-unanimous cases), "
         f"sorted with [RULE FIRED BUT INSUFFICIENT] cases first")
    print(f"wrote {args.out_p2}: {len(p2_cases)} genuine P2 failures (disagreement, still wrong), "
         f"same sort order")
    print(f"\n{insufficient_count} cases have a rule that FIRED and CHANGED the prediction "
         f"but it's still wrong -- flagged [RULE FIRED BUT INSUFFICIENT] and sorted to the top. "
         f"These are the highest-value cases to read first for discovering the next rule.")

    # rule summary table -- quantitative picture before any manual reading,
    # also directly usable as report material
    print("\n" + "=" * 60)
    print("RULE SUMMARY (across all P1 + P2 cases mined)")
    print("=" * 60)
    print(f"total deployed-champion errors on val: {total_wrong} "
          f"(P0 {len(p0_cases)} + P1 {len(p1_cases)} + P2 {len(p2_cases)}) "
          f"-- this partition is exhaustive over wrong predictions")
    # every case here is STILL WRONG by construction; P0 included so the
    # per-rule table counts rules that fired on posts they then broke
    still_wrong_cases = [c[:8] for c in p0_cases] + p1_cases + p2_cases
    for k in rule_keys:
        # fired/changed measured across ALL touched posts (fixed + still-wrong)
        fired_wrong = sum(1 for c in still_wrong_cases if any(t["rule"] == k and t["fired"] for t in c[6]))
        fired_fixed = sum(1 for _, _, tr in fixed_cases if any(t["rule"] == k and t["fired"] for t in tr))
        fired = fired_wrong + fired_fixed
        changed_wrong = sum(1 for c in still_wrong_cases if any(t["rule"] == k and t["changed"] for t in c[6]))
        # fully fixed: this rule changed the prediction on a case that ended up CORRECT
        fully_fixed = sum(1 for _, _, tr in fixed_cases if any(t["rule"] == k and t["changed"] for t in tr))
        # partially fixed: this rule changed the prediction but the case is still wrong
        partially_fixed = sum(1 for c in still_wrong_cases
                              if any(t["rule"] == k and t["changed"] for t in c[6]) and c[7])
        no_effect = fired - fully_fixed - partially_fixed
        print(f"\n{k}")
        print(f"  fired:            {fired}")
        print(f"  fully fixed:      {fully_fixed}  (fired, changed, case is now CORRECT)")
        print(f"  partially fixed:  {partially_fixed}  (fired, changed, but case is still wrong overall)")
        print(f"  no effect:        {no_effect}  (fired but didn't change the prediction)")
    print("=" * 60)


if __name__ == "__main__":
    main()
