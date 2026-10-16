"""
factor_signal_for_risk.py

Measures whether the plan/attempt distinctions the risk labels depend on are
already carried by the hand-annotated factor column.

The official risk definitions are compositional:

    indicator = no explicit mention of suicide
    ideation  = explicit suicidal expressions but NO suicidal plan
    behavior  = explicit suicidal expressions AND self-harm or a suicidal PLAN
    attempt   = explicit mentions of recent or past suicidal ATTEMPTS

and two of the 24 factors encode exactly those distinctions:

    suicide means (with access)                 "description of potential suicide
                                                 means or access to those means"
    prior self-harm or suicidal thought/attempt "past, resolved, previous self-harm
                                                 or suicidal thought, plan, or attempt"

They are independent human annotations rather than derivations of risk_level,
so a factor -> risk mechanism is not circular.

WHAT IT MEASURES
  1. How well each gold factor separates the confusable risk pairs, over all
     labelled posts -- the ceiling if the signal were used perfectly.
  2. The same restricted to the posts the risk model gets CONFIDENTLY WRONG.
     This is the decisive table: a factor that separates easy cases adds
     nothing the risk representation does not already hold.
  3. How accurately the factor model predicts those two factors, since at test
     time only predictions are available. Gold separation x prediction quality
     bounds what any factor->risk mechanism can deliver.

Separation on the confident errors means the signal is real and missing from
the risk representation, so the architecture should consume it (multi-task
head, auxiliary binary head, or factor probabilities as risk blend features).
No separation means those posts are genuinely ambiguous.

Usage:
  python src/factor_signal_for_risk.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_audit import load
from scoring import FACTOR_TAXONOMY, normalize_label, parse_factors

LEV = ["indicator", "ideation", "behavior", "attempt"]
PLAN_FACTORS = ["suicide means (with access)",
                "prior self-harm or suicidal thought/attempt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--heads", default="artifacts/risk_heads/heads_full.npz",
                    help="risk_head_diagnosis.py output on the labelled set")
    ap.add_argument("--factor-oof", default="artifacts/fac_oof_f{f}.npz",
                    help="factor OOF probabilities, for the prediction-quality half")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    lab = load(args.labeled)
    lab["risk"] = [normalize_label(x) for x in lab["risk_level"]]
    TAX = set(FACTOR_TAXONOMY)
    fs = [{str(f).strip().lower() for f in parse_factors(x)} & TAX
          for x in lab["factors"]]
    lab["fs"] = fs
    rid = list(lab["row_id"])
    print(f"{len(lab)} labelled posts\n")

    # ---------------- 1. gold separation over ALL posts --------------------
    print("=" * 94)
    print("1. GOLD FACTOR SEPARATION over all posts")
    print("=" * 94)
    pairs = [("ideation", "behavior"), ("behavior", "attempt"),
             ("indicator", "ideation")]
    for lo, hi in pairs:
        m = lab["risk"].isin([lo, hi]).to_numpy()
        y = (lab["risk"][m] == hi).to_numpy().astype(int)
        print(f"\n  {lo} vs {hi}   ({int(m.sum())} posts, "
              f"{int(y.sum())} are {hi})")
        print(f"    {'factor':<46}{'P|'+lo:>9}{'P|'+hi:>9}{'lift':>7}"
              f"{'sep':>7}")
        rows = []
        for t in FACTOR_TAXONOMY:
            v = np.array([t in s for s in np.array(fs, dtype=object)[m]])
            p_lo = v[y == 0].mean() if (y == 0).any() else 0.0
            p_hi = v[y == 1].mean() if (y == 1).any() else 0.0
            rows.append((t, p_lo, p_hi, abs(p_hi - p_lo)))
        for t, p_lo, p_hi, sep in sorted(rows, key=lambda r: -r[3])[:6]:
            lift = p_hi / p_lo if p_lo > 1e-9 else float("inf")
            star = "  <-- plan/attempt factor" if t in PLAN_FACTORS else ""
            print(f"    {t:<46}{100*p_lo:>8.1f}%{100*p_hi:>8.1f}%"
                  f"{lift:>7.2f}{sep:>7.2f}{star}")

    # ---------------- 2. the confident errors ------------------------------
    if not os.path.exists(args.heads):
        print(f"\nSKIPPING part 2: {args.heads} not found. Run "
              "src/risk_head_diagnosis.py --data artifacts/full_train.csv "
              "--tag full first.")
        return
    h = np.load(args.heads, allow_pickle=True)
    hrid = list(h["row_id"])
    o = {r: i for i, r in enumerate(hrid)}
    mix = h["mix"][[o[r] for r in rid]]
    pred = np.array([LEV[i] for i in mix.argmax(1)])
    gold = lab["risk"].to_numpy()

    print("\n" + "=" * 94)
    print("2. THE SAME FACTORS, RESTRICTED TO THE CONFIDENTLY WRONG POSTS")
    print("=" * 94)
    print("   This is the decisive table. A factor that separates the whole")
    print("   population but NOT these posts adds nothing the model lacks.")
    for lo, hi in pairs:
        a, b = LEV.index(lo), LEV.index(hi)
        err = (((gold == lo) & (pred == hi)) | ((gold == hi) & (pred == lo)))
        conf = err & (np.abs(mix[:, a] - mix[:, b]) > 0.3)
        n = int(conf.sum())
        print(f"\n  {lo} vs {hi}: {int(err.sum())} errors, "
              f"{n} of them confident (margin > 0.3)")
        if n < 8:
            print("    too few to measure")
            continue
        y = (gold[conf] == hi).astype(int)
        sub = np.array(fs, dtype=object)[conf]
        print(f"    {'factor':<46}{'P|gold='+lo:>13}{'P|gold='+hi:>13}"
              f"{'sep':>7}")
        rows = []
        for t in FACTOR_TAXONOMY:
            v = np.array([t in s for s in sub])
            p_lo = v[y == 0].mean() if (y == 0).any() else 0.0
            p_hi = v[y == 1].mean() if (y == 1).any() else 0.0
            rows.append((t, p_lo, p_hi, abs(p_hi - p_lo)))
        for t, p_lo, p_hi, sep in sorted(rows, key=lambda r: -r[3])[:6]:
            star = "  <-- plan/attempt factor" if t in PLAN_FACTORS else ""
            print(f"    {t:<46}{100*p_lo:>12.1f}%{100*p_hi:>12.1f}%"
                  f"{sep:>7.2f}{star}")
        # How far could a perfect user of these two factors get -- AGAINST THE
        # MAJORITY BASELINE. Without the baseline a figure like "fixes 56%"
        # reads as progress when guessing the majority class already gets 61%.
        base = max(y.mean(), 1 - y.mean())
        print(f"    baseline: always guess "
              f"`{hi if y.mean() > 0.5 else lo}` -> {base:.0%} "
              f"({int(y.sum())} gold={hi}, {int((1-y).sum())} gold={lo})")
        for t in PLAN_FACTORS:
            v = np.array([t in s for s in sub]).astype(int)
            acc = max((v == y).mean(), (v != y).mean())
            verdict = "BEATS baseline" if acc > base + 0.02 else "no better than guessing"
            print(f"    -> best rule on `{t}`: {acc:.0%}  ({verdict})")

    # ---------------- 3. are those factors predictable? --------------------
    print("\n" + "=" * 94)
    print("3. HOW WELL ARE THE TWO KEY FACTORS PREDICTED?")
    print("=" * 94)
    print("   At test time only predictions are available, so gold separation")
    print("   must be multiplied by this.")
    P, prid = [], []
    for f in range(args.k):
        p = args.factor_oof.format(f=f)
        if not os.path.exists(p):
            print(f"   skipping: {p} missing")
            return
        d = np.load(p, allow_pickle=True)
        prid.extend(list(d["row_id"]))
        P.append(d["probs"])
    P = np.concatenate(P)
    om = {r: i for i, r in enumerate(prid)}
    keep = [r for r in rid if r in om]
    Pk = P[[om[r] for r in keep]]
    fk = [fs[rid.index(r)] for r in keep]
    from sklearn.metrics import average_precision_score, f1_score
    for t in PLAN_FACTORS:
        j = FACTOR_TAXONOMY.index(t)
        y = np.array([t in s for s in fk]).astype(int)
        ap_ = average_precision_score(y, Pk[:, j])
        best = max(f1_score(y, (Pk[:, j] >= th).astype(int), zero_division=0)
                   for th in np.arange(0.05, 0.96, 0.01))
        print(f"   {t:<46} n={int(y.sum()):>4}  AP {ap_:.3f}  best F1 {best:.3f}")

    print("\n" + "=" * 94)
    print("VERDICT GUIDE")
    print("=" * 94)
    print("  Separation in part 2 -> the signal is real and MISSING from the")
    print("  risk representation. Consume it: multi-task (risk + the 24 factor")
    print("  labels on a shared encoder), an auxiliary binary head for these")
    print("  two factors, or the factor probabilities as risk-blend features.")
    print("  Part 3 bounds how much of it survives prediction error.")
    print()
    print("  No separation in part 2 -> those posts are genuinely ambiguous and")
    print("  no factor-derived supervision can fix the boundary.")


if __name__ == "__main__":
    main()
