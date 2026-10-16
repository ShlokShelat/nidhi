"""
factor_risk_consistency.py

Applies the annotation scheme's risk-conditional factor structure as a
post-hoc correction -- the same intervention class as
evidence_risk_consistency.py, which measured a macro phrase-F1 gain.

Measured over all 1635 labelled posts, P(factor present | risk level):

    factor                                    ind    idea    beh    att
    prior self-harm or suicidal thought/...  0.015  0.077  0.166  0.886
    suicide means (with access)              0.003  0.048  0.578  0.342
    exposure to others' suicide              0.003  0.015  0.008  0.009

Some conditionals are close to deterministic: 88.6% of Attempt posts carry
"prior self-harm...", against 1.5% of Indicator posts -- a 59x ratio. The
factor model never sees any of this: it scores (post, factor-definition)
pairs with no access to the risk head, so nothing in the pipeline enforces
consistency between the two outputs.

Two corrections, both driven purely by training-set statistics:

    ADD   P(factor | predicted risk) >= --hi  and the factor was not predicted
    DROP  P(factor | predicted risk) <= --lo  and the factor was predicted

Defaults hi=0.80 / lo=0.02 are deliberately extreme round numbers, chosen so
the rule only fires where the conditional is near-deterministic. They are NOT
tuned on val -- this campaign has 7 documented failures from val-tuned
parameters and 2 successes from parameter-free consistency fixes. Use
--sweep to SEE the sensitivity, but changing the defaults to whatever
maximises val is exactly the mistake to avoid.

Subtask 2 is macro-F1 over 24 labels, so a correction concentrated on one or
two labels moves the score meaningfully: each label carries 1/24 = 4.2%.

Usage:
  python src/factor_risk_consistency.py --report          # measure on val
  python src/factor_risk_consistency.py --report --sweep  # sensitivity only
  python src/factor_risk_consistency.py --apply \
      --factors artifacts/nli_ensemble_factors.csv \
      --risk artifacts/risk_blend_scheduledfix.csv \
      --out artifacts/factors_consistent_test.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors, normalize_label
from rule_ablation_v2 import author_split

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def conditional_table(labeled_path="data/labeled.csv"):
    """P(factor | risk level) from the FULL labelled set. No val involved."""
    lab = pd.read_csv(labeled_path)
    rcol = "suicide risk" if "suicide risk" in lab.columns else "risk_level"
    fcol = "factors"
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    M = np.zeros((len(FACTOR_TAXONOMY), 4))
    tot = np.zeros(4)
    for _, r in lab.iterrows():
        li = LEVELS.index(normalize_label(r[rcol]))
        tot[li] += 1
        for f in set(parse_factors(r[fcol])):
            if f.strip().lower() in idx:
                M[idx[f.strip().lower()], li] += 1
    return M / np.clip(tot, 1, None), tot


def apply_rule(pred_sets, risks, P, hi, lo):
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    out, n_add, n_drop = [], 0, 0
    for s, rk in zip(pred_sets, risks):
        li = LEVELS.index(rk)
        cur = set(s)
        for t in FACTOR_TAXONOMY:
            p = P[idx[t], li]
            if p >= hi and t not in cur:
                cur.add(t)
                n_add += 1
            elif p <= lo and t in cur:
                cur.discard(t)
                n_drop += 1
        out.append(sorted(cur))
    return out, n_add, n_drop


def macro(gold_sets, pred_sets):
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    def mat(rows):
        m = np.zeros((len(rows), len(FACTOR_TAXONOMY)), dtype=int)
        for i, s in enumerate(rows):
            for f in s:
                if f.strip().lower() in idx:
                    m[i, idx[f.strip().lower()]] = 1
        return m
    return float(f1_score(mat(gold_sets), mat(pred_sets),
                          average=None, zero_division=0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--val-factors", default="artifacts/nli_factors_val.csv")
    ap.add_argument("--val-baseline", default=None)
    ap.add_argument("--factors", default=None)
    ap.add_argument("--risk", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hi", type=float, default=0.80)
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    P, tot = conditional_table()

    if args.report:
        val = pd.read_csv(args.val)
        rids = list(val["row_id"])
        gold = [set(parse_factors(x)) for x in val["factors"]]

        cands = ([args.val_baseline] if args.val_baseline else
                 ["artifacts/val_deployed_baseline.csv",
                  "artifacts/val_baseline_reconstructed.csv"])
        bp = next((p for p in cands if p and os.path.exists(p)), None)
        if bp is None:
            raise SystemExit("no val baseline found; expected one of " + str(cands))
        rec = pd.read_csv(bp).set_index("row_id")
        rc = next(c for c in ("champion_final", "pred_risk_level", "risk_level")
                  if c in rec.columns)
        risks = [normalize_label(rec.loc[r, rc]) for r in rids]

        if not os.path.exists(args.val_factors):
            raise SystemExit(
                f"missing {args.val_factors}. Generate it with:\n"
                f"  python src/predict_factors_nli.py --test {args.val} "
                f"--nli-run runs/factors_nli,runs/factors_nli_s2 "
                f"--out {args.val_factors}")
        fv = pd.read_csv(args.val_factors, keep_default_na=False).set_index("row_id")
        pred = [set(parse_factors(fv.loc[r, "factors"])) for r in rids]

        print(f"val risk from {bp} ('{rc}'); factors from {args.val_factors}\n")
        base = macro(gold, pred)
        half = author_split(val)[0].values
        A = [i for i in range(len(rids)) if half[i] == "A"]
        B = [i for i in range(len(rids)) if half[i] == "B"]

        def show(name, sp, extra=""):
            a = macro([gold[i] for i in A], [sp[i] for i in A])
            b = macro([gold[i] for i in B], [sp[i] for i in B])
            m = macro(gold, sp)
            print(f"{name:<30}{m:>9.4f}{a:>9.4f}{b:>9.4f}   {extra}")
            return m

        print(f"{'system':<30}{'macro':>9}{'halfA':>9}{'halfB':>9}")
        show("deployed factors", pred)
        if args.sweep:
            for hi in (0.70, 0.80, 0.90):
                for lo in (0.01, 0.02, 0.05):
                    g, na, nd = apply_rule(pred, risks, P, hi, lo)
                    show(f"hi={hi} lo={lo}", g, f"+{na} add / -{nd} drop")
            print("\nSWEEP IS DIAGNOSTIC ONLY. Picking hi/lo off this table is "
                  "val-tuning,\nwhich has failed 7/7 in this campaign. Ship the "
                  "defaults or nothing.")
        else:
            g, na, nd = apply_rule(pred, risks, P, args.hi, args.lo)
            m = show(f"+ rule (hi={args.hi} lo={args.lo})", g,
                     f"+{na} add / -{nd} drop")
            print(f"\ndelta {m - base:+.4f} macro-F1 = {0.3*(m-base):+.4f} composite")
        return

    if not args.apply:
        raise SystemExit("pass --report or --apply")
    for r in ("factors", "risk", "out"):
        if not getattr(args, r):
            raise SystemExit(f"--apply needs --{r}")

    fa = pd.read_csv(args.factors, keep_default_na=False)
    rk = pd.read_csv(args.risk)
    rc = next(c for c in ("pred_risk_level", "risk_level") if c in rk.columns)
    rmap = dict(zip(rk["row_id"], rk[rc]))
    rids = list(fa["row_id"])
    pred = [set(parse_factors(x)) for x in fa["factors"]]
    risks = [normalize_label(rmap[r]) for r in rids]
    out, na, nd = apply_rule(pred, risks, P, args.hi, args.lo)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame({"row_id": rids,
                  "factors": [str(list(s)) for s in out]}).to_csv(args.out, index=False)
    changed = sum(1 for a, b in zip(pred, out) if set(a) != set(b))
    print(f"added {na}, dropped {nd}; {changed}/{len(rids)} posts changed")
    print(f"mean factors/post {np.mean([len(s) for s in pred]):.2f} -> "
          f"{np.mean([len(s) for s in out]):.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
