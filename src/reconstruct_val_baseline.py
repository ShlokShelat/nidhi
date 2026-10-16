"""
reconstruct_val_baseline.py

Produces an APPROXIMATE stand-in for cache_val_baseline.py's output on a
machine with no GPU / no risk-NLI weights, so candidate rules can be
pre-screened before booking GPU time. It never replaces the real thing --
every row it cannot pin down exactly is flagged, and the output carries an
`approximate` column that rule_ablation_v2.py prints a warning for.

How it reconstructs the champion's val predictions without the NLI model:

  * champion_p1/p2_failures.txt already record, for every case they contain,
    the deployed blend's pre-rule prediction AND the post-rule final. Those
    rows are recovered EXACTLY by parsing those files.
  * By construction the miner emits a case whenever the champion's final
    answer is wrong -- EXCEPT for posts where the base risk classifiers were
    unanimous and correct, which fall through both its branches. So:
      - not listed, and NOT (unanimous and base-correct)  -> final == gold, exact
      - not listed, and unanimous and base-correct        -> UNCERTAIN; estimated
        as apply_rules(post, gold), which correctly recovers the cases where a
        deployed rule fires on an otherwise-correct post, but cannot see a
        prediction the NLI blend alone flipped.

On the current split that leaves 222 uncertain rows out of 372. That is a
large uncertain set in absolute terms, but it is confined to posts the base
models already get right, so it mostly matters for measuring how often a new
rule BREAKS something -- which is exactly why a PASS here is provisional.

Usage:
  python src/reconstruct_val_baseline.py --out artifacts/val_baseline_reconstructed.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1
from apply_rules_to_test import apply_rules

PROBCOLS = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]
FRIEND_COLS = ["Indicator", "Ideation", "Behavior", "Attempt"]

CASE_RE = re.compile(
    r"row_id=(?P<rid>\S+)\s+GOLD=(?P<gold>\w+).*?"
    r"ENSEMBLE PREDICTION \(before rules\): (?P<pre>\w+).*?"
    r"FINAL PREDICTION \(after rules\): (?P<fin>\w+)",
    re.DOTALL)


def parse_failures(paths):
    listed = {}
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"missing failure file: {p}")
        for m in CASE_RE.finditer(open(p, encoding="utf-8").read()):
            listed[m.group("rid")] = (m.group("gold"), m.group("pre"), m.group("fin"))
    return listed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--val-probs",
                    default="runs/risk_v1/val_predictions.csv,"
                            "runs/risk_v1_s14/val_predictions.csv,"
                            "runs/risk_v1_s15/val_predictions.csv")
    ap.add_argument("--friend-probs-val", default="artifacts/friend_risk_probs_val.csv")
    ap.add_argument("--p1", default="artifacts/champion_p1_failures.txt")
    ap.add_argument("--p2", default="artifacts/champion_p2_failures.txt")
    ap.add_argument("--rules", default="urgency,completed_act")
    ap.add_argument("--out", default="artifacts/val_baseline_reconstructed.csv")
    args = ap.parse_args()

    val = pd.read_csv(args.val)
    row_ids = val["row_id"].tolist()
    gold = [normalize_label(x) for x in val["risk_level"]]
    rule_keys = tuple(r.strip() for r in args.rules.split(","))

    per = []
    for p in [x.strip() for x in args.val_probs.split(",")]:
        d = pd.read_csv(p).set_index("row_id")
        per.append(d.loc[row_ids, PROBCOLS].values.astype(float))
    equal_avg = np.mean(per, axis=0)

    fr = pd.read_csv(args.friend_probs_val).set_index("row_id")
    friend = np.zeros((len(row_ids), 4))
    for i, rid in enumerate(row_ids):
        p = (fr.loc[rid, FRIEND_COLS].values.astype(float)
             if rid in fr.index else equal_avg[i])
        friend[i] = p / p.sum()

    votes = np.stack([np.argmax(x, axis=1) for x in per]
                     + [np.argmax(friend, axis=1)], axis=1)
    unanimous = np.array([len(set(v)) == 1 for v in votes])
    base_lbl = np.array([RISK_LEVELS[v[0]] for v in votes])
    unan_correct = unanimous & (base_lbl == np.array(gold))

    listed = parse_failures([args.p1, args.p2])
    bad = [r for r in listed if r in row_ids and listed[r][0] != gold[row_ids.index(r)]]
    if bad:
        raise SystemExit(f"failure files disagree with val gold on {len(bad)} rows: {bad[:5]}")

    champion, pre_rule, approx = [], [], []
    for i, rid in enumerate(row_ids):
        if rid in listed:
            pre_rule.append(listed[rid][1])
            champion.append(listed[rid][2])
            approx.append(False)
        elif unan_correct[i]:
            est, _ = apply_rules(val["post"].iloc[i], gold[i], rule_keys)
            pre_rule.append(gold[i])
            champion.append(est)
            approx.append(True)
        else:
            pre_rule.append(gold[i])
            champion.append(gold[i])
            approx.append(False)

    out = pd.DataFrame({
        "row_id": row_ids, "gold": gold,
        "pre_rule_pred": pre_rule, "champion_final": champion,
        "base_unanimous": unanimous, "base_unanimous_correct": unan_correct,
        "approximate": approx,
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"val rows                : {len(out)}")
    print(f"recovered exactly       : {(~out.approximate).sum()} "
          f"({len(listed)} from failure files, "
          f"{(~out.approximate).sum() - len(listed)} implied correct)")
    print(f"UNCERTAIN (estimated)   : {out.approximate.sum()}")
    flipped = out[out.approximate & (out.champion_final != out.gold)]
    print(f"  of those, a deployed rule fires and breaks a correct post: {len(flipped)}")
    for _, r in flipped.iterrows():
        print(f"    {r.row_id}  gold={r.gold} -> champion={r.champion_final}")
    print(f"\nreconstructed champion val wF1: "
          f"{risk_weighted_f1(gold, list(out.champion_final)):.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
