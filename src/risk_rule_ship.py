"""
risk_rule_ship.py

Keeps the symbolic rule that helps and drops the one that hurts.

risk_rule_ablation.py decomposed the symbolic layer on 1590 OOF posts. It
changes 30 predictions, and the two rules inside it point opposite ways:

    from        to           n   fixed  broke   net
    behavior    ideation    20     6      12     -6    HARMFUL
    behavior    attempt      8     6       2     +4    this is route_attempt
    attempt     ideation     2     0       1     -1    harmful

    per class:  attempt F1 0.638 -> 0.670  (recall 0.545 -> 0.591)
                behavior F1 0.729 -> 0.719 (recall 0.701 -> 0.665)
                weighted   0.7858 -> 0.7841

So the layer as a whole is a liability on held-out authors even though the
behavior->attempt routing on its own is beneficial: the demotion rule
(behavior->ideation) is what drags the combined layer down.

THE SURGICAL CHANGE, WHICH NEEDS NO KNOWLEDGE OF WHICH RULE IS WHICH:

    start from pre_rule_pred, and keep the shipped label ONLY where the rules
    routed behavior -> attempt.

Every other transition is reverted. That preserves the attempt F1 gain and
gives back the behavior recall, for a deletion -- zero new parameters.

EVIDENCE IS UNTOUCHED BY CONSTRUCTION. Suppression fires on predicted
`indicator`, and no reverted transition involves indicator in either
direction, so the evidence column and S2 must come back bit-identical. That is
asserted below, not hoped for.

Usage:
  python src/risk_rule_ship.py                       # diagnose only
  python src/risk_rule_ship.py --build submissions/NIDHI_attonly.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
CANDS = [
    ("artifacts/ensembled_risk.csv", "pred_risk_level"),
    ("artifacts/risk_champion_qwen.csv", "pred_risk_level"),
    ("artifacts/ensembled_risk_full.csv", "pred_risk_level"),
    ("artifacts/ensembled_risk_rat.csv", "pred_risk_level"),
]
PCOLS = [["p_Indicator", "p_Ideation", "p_Behavior", "p_Attempt"],
         ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"],
         ["Indicator", "Ideation", "Behavior", "Attempt"]]


def load_pre(path, col):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, keep_default_na=False)
    if "row_id" not in d.columns:
        return None
    d["row_id"] = d["row_id"].astype(str)
    if col in d.columns:
        return dict(zip(d["row_id"],
                        [normalize_label(x) for x in d[col]]))
    for cs in PCOLS:                      # fall back to argmax of the probs
        if all(c in d.columns for c in cs):
            A = d[cs].to_numpy(dtype=float)
            return dict(zip(d["row_id"],
                            [LEVELS[i] for i in A.argmax(1)]))
    return None


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--champion", default="submissions/NIDHI_factfit.csv")
    a_.add_argument("--pre", default=None,
                    help="file holding the PRE-RULE test risk; "
                         "auto-discovered if omitted")
    a_.add_argument("--pre-col", default="pred_risk_level")
    a_.add_argument("--build", default=None)
    args = a_.parse_args()

    ch = pd.read_csv(args.champion, keep_default_na=False)
    ch["row_id"] = ch["row_id"].astype(str)
    rc = "risk_level" if "risk_level" in ch.columns else "pred_risk_level"
    ship = {r: normalize_label(x) for r, x in zip(ch["row_id"], ch[rc])}
    print(f"champion {args.champion}  {len(ch)} rows, risk column '{rc}'\n")

    tries = ([(args.pre, args.pre_col)] if args.pre else CANDS)
    print("looking for the PRE-RULE test risk column:")
    best = None
    for p, c in tries:
        pre = load_pre(p, c)
        if pre is None:
            print(f"  {p:<45} missing / unusable")
            continue
        common = [r for r in ch["row_id"] if r in pre]
        if not common:
            print(f"  {p:<45} no overlapping row_id")
            continue
        diff = [r for r in common if pre[r] != ship[r]]
        pat = {}
        for r in diff:
            pat[(pre[r], ship[r])] = pat.get((pre[r], ship[r]), 0) + 1
        print(f"  {p:<45} {len(common)} rows, {len(diff)} differ  {pat}")
        # the pre-rule source is the one differing ONLY by rule transitions
        ok = diff and all(a != "indicator" and b != "indicator"
                          for a, b in pat)
        if ok and (best is None or len(diff) < best[2]):
            best = (p, pre, len(diff), pat)

    if best is None:
        raise SystemExit(
            "\nNo file looks like the pre-rule risk column. Pass one with "
            "--pre. It must share row_ids with the champion and differ from "
            "it ONLY on behavior/ideation/attempt transitions -- any file "
            "differing on `indicator` is a different model, not the same "
            "model before rules, and using it would change evidence too.")

    path, pre, ndiff, pat = best
    print(f"\nusing {path}  ({ndiff} posts differ)")
    print(f"transition pattern on test: {pat}")

    keep = [(a, b) for a, b in pat if a == "behavior" and b == "attempt"]
    revert = [(a, b) for a, b in pat if (a, b) not in keep]
    print(f"\n  KEEP   {keep}   (route_attempt: +0.032 attempt F1 on OOF)")
    print(f"  REVERT {revert}   (net -7 correct predictions on OOF)")

    new = []
    n_rev = n_keep = 0
    for r in ch["row_id"]:
        p, s = pre.get(r), ship[r]
        if p is None or p == s:
            new.append(s)
            continue
        if (p, s) in keep:
            new.append(s)
            n_keep += 1
        else:
            new.append(p)
            n_rev += 1
    print(f"\n  kept {n_keep} routed posts, reverted {n_rev} posts")

    ind_before = sum(1 for r in ch["row_id"] if ship[r] == "indicator")
    ind_after = sum(1 for x in new if x == "indicator")
    print(f"  indicator count {ind_before} -> {ind_after}")
    if ind_before != ind_after:
        raise SystemExit(
            "REFUSING: the indicator count moved, so evidence suppression "
            "changes and S2 would not be bit-identical. That makes this a "
            "two-variable submission and the delta unattributable.")
    print("  indicator unchanged -> evidence and S2 are bit-identical, so any")
    print("  S1 movement is attributable to the risk column alone.")

    if args.build:
        out = ch.copy()
        # preserve the original label casing convention of the champion file
        seen = {normalize_label(x): x for x in ch[rc]}
        out[rc] = [seen.get(x, x) for x in new]
        os.makedirs(os.path.dirname(args.build) or ".", exist_ok=True)
        out.to_csv(args.build, index=False)
        chg = int(sum(1 for a, b in zip(ch[rc], out[rc])
                      if normalize_label(a) != normalize_label(b)))
        print(f"\nwrote {args.build}   {chg} risk labels changed")
        print("Only the risk column changes. S2 must return EXACTLY 0.6908.")
        print("Expected S1 movement is small and positive (~+0.0034 risk wF1")
        print("on OOF, one-draw spread wider than that) -- judge it against")
        print("the champion's 0.8061, and do not read a single draw as proof.")


if __name__ == "__main__":
    main()
