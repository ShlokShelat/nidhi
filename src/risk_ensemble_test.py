"""
risk_ensemble_test.py

Measures whether ADDING a Qwen probability source to a risk ensemble helps,
as opposed to replacing an existing member with it.

Blending probabilities is a different operation from a label override: an
override lets a model that is 55% sure overrule one that is 90% sure, which
discards exactly the information an average uses. Decorrelated errors between
members are what makes an average pay, so the members here are deliberately
heterogeneous: the friend model, the v2 DeBERTa control, and Qwen.

Scope: this scores a REBUILT ensemble from the probability sources available
on disk. A gain here is evidence that the addition is worth testing on a
stronger base, not proof that it survives there.

Validation controls:
  - blend weights are cross-fitted across author-disjoint halves, so the
    reported delta is never read off the split that chose the weight
  - an arm must win on BOTH halves, not just overall
  - a first check scores the friend model alone: a score near 0.95 means the
    file is IN-SAMPLE rather than out-of-fold, and every number below it would
    then be meaningless

Usage:
  python src/risk_ensemble_test.py
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1

COLS = ["Indicator", "Ideation", "Behavior", "Attempt"]


def author_halves(auth):
    s = pd.Series(auth)
    counts = s.value_counts()
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    half, size = {}, {"A": 0, "B": 0}
    for a in order:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


def load_run(tmpl, k=5):
    parts = []
    for f in range(k):
        p = os.path.join(tmpl.replace("{f}", str(f)), "oof_predictions.csv")
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True)
    pc = [f"prob_{c}" for c in RISK_LEVELS]
    if any(c not in d.columns for c in pc):
        return None
    return d.set_index("row_id")[pc].rename(
        columns=dict(zip(pc, RISK_LEVELS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--friend", default="artifacts/friend_risk_probs_val.csv")
    ap.add_argument("--control",
                    default="v2_rebuild/runs/risk_ctrl_f{f}")
    ap.add_argument("--qwen",
                    default="v2_rebuild/runs/risk_llm2_ord_f{f}")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled)
    gold = {r: normalize_label(x) for r, x in zip(lab["row_id"], lab["risk_level"])}
    amap = dict(zip(lab["row_id"], lab["anon_user_id"]))

    src = {}
    fr = pd.read_csv(args.friend)
    src["friend"] = fr.set_index("row_id")[COLS].rename(
        columns={c: c.lower() for c in COLS})
    for name, tmpl in (("control", args.control), ("qwen", args.qwen)):
        d = load_run(tmpl, args.k)
        if d is None:
            print(f"WARNING: no usable probs for '{name}' at {tmpl}")
            continue
        src[name] = d
    for n, d in src.items():
        print(f"  {n:<9} {len(d)} rows")

    rows = set(gold)
    for d in src.values():
        rows &= set(d.index)
    rows = sorted(rows)
    print(f"\ncommon posts: {len(rows)}")
    g = [gold[r] for r in rows]
    half = author_halves([amap[r] for r in rows])

    # Every source was renamed to have exactly the RISK_LEVELS column names, so
    # index by RISK_LEVELS explicitly. Relying on the frames' own column order
    # would silently mis-align friend (Indicator..Attempt) against the run
    # files (prob_indicator..prob_attempt) if either ever changed.
    P = {n: d.loc[rows, RISK_LEVELS].to_numpy(float) for n, d in src.items()}
    for n in P:                      # normalise unnormalised score columns
        P[n] = P[n] / P[n].sum(axis=1, keepdims=True)

    def sc(M, mask=None):
        pred = [RISK_LEVELS[i] for i in M.argmax(1)]
        if mask is None:
            return risk_weighted_f1(g, pred)
        return risk_weighted_f1([x for x, m in zip(g, mask) if m],
                                [x for x, m in zip(pred, mask) if m])

    print(f"\n{'system':<28}{'wF1':>9}{'halfA':>9}{'halfB':>9}")
    print("-" * 55)
    for n in src:
        print(f"{n:<28}{sc(P[n]):>9.4f}{sc(P[n], half == 'A'):>9.4f}"
              f"{sc(P[n], half == 'B'):>9.4f}")
    if sc(P["friend"]) > 0.93:
        print("\n*** friend alone scores above 0.93 -- that file is almost")
        print("*** certainly IN-SAMPLE, not out-of-fold. Stop here; every")
        print("*** number below would be meaningless.")

    # equal-weight blends of every subset of size >= 2
    names = list(src)
    print()
    combos = []
    for r in range(2, len(names) + 1):
        for c in itertools.combinations(names, r):
            M = np.mean([P[n] for n in c], axis=0)
            combos.append((c, sc(M), sc(M, half == "A"), sc(M, half == "B")))
            print(f"{'+'.join(c):<28}{combos[-1][1]:>9.4f}"
                  f"{combos[-1][2]:>9.4f}{combos[-1][3]:>9.4f}")

    # The question that matters: does adding qwen to friend+control help?
    if all(n in src for n in ("friend", "control", "qwen")):
        base = np.mean([P["friend"], P["control"]], axis=0)
        print(f"\n{'':<28}{'overall':>9}{'halfA':>9}{'halfB':>9}")
        print(f"{'friend+control':<28}{sc(base):>9.4f}"
              f"{sc(base, half == 'A'):>9.4f}{sc(base, half == 'B'):>9.4f}")
        print(f"\nadding qwen at weight w (cross-fitted below):")
        print(f"{'w':>7}{'overall':>10}{'delta':>9}{'halfA':>9}{'halfB':>9}")
        grid = np.round(np.arange(0.0, 0.71, 0.05), 2)
        b, ba, bb = sc(base), sc(base, half == "A"), sc(base, half == "B")
        for w in grid:
            M = (1 - w) * base + w * P["qwen"]
            print(f"{w:>7.2f}{sc(M):>10.4f}{sc(M) - b:>+9.4f}"
                  f"{sc(M, half == 'A') - ba:>+9.4f}"
                  f"{sc(M, half == 'B') - bb:>+9.4f}")

        print("\nCROSS-FITTED (weight chosen on the OTHER half)")
        tot = []
        for tgt, srch in (("A", "B"), ("B", "A")):
            mt, ms = half == tgt, half == srch
            bt = sc(base, mt)
            best = max(grid, key=lambda w:
                       sc((1 - w) * base + w * P["qwen"], ms))
            got = sc((1 - best) * base + best * P["qwen"], mt)
            tot.append(got - bt)
            print(f"  tune on {srch}, apply to {tgt}: w={best:.2f}  "
                  f"{bt:.4f} -> {got:.4f}  ({got - bt:+.4f})")
        m = float(np.mean(tot))
        print(f"  mean cross-fitted delta {m:+.4f}")
        print("\n" + "=" * 62)
        if m > 0 and all(x > 0 for x in tot):
            print("Qwen ADDS to this ensemble on both halves. Worth testing on")
            print("the shipped ensemble -- but this rebuilt base is weaker, so")
            print("the gain may not survive against a stronger one.")
        else:
            print("Qwen does not add to this ensemble: the cross-fitted delta")
            print("is not positive on both halves.")
        print("=" * 62)


if __name__ == "__main__":
    main()
