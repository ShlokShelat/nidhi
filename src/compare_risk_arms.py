"""
compare_risk_arms.py

Checks whether a hard-mining or class-weighting arm reduces the misses it
targets.

Compares any risk arm against the champion's three seeds on `val_split`, which
neither arm trains on. Reports PER CLASS, because hard mining and class
weighting target `attempt` specifically: it is the rarest class and the one
with the weakest recall, so a change to it is invisible in an overall number.

WHAT TO LOOK FOR, in order:
  1. `attempt` recall. That is the target of the intervention.
  2. `attempt` precision. Upweighting buys recall at some cost to precision;
     the trade only pays if F1 rises.
  3. The other three classes. If they fall by more than attempt gains, the model
     is chasing hard cases at the expense of the easy majority -- which is what
     a mean sample weight far above 1 causes.
  4. Weighted F1 overall, which is the scored quantity.

CAVEAT ON SIZE. val_split is 372 posts with roughly 26 `attempt` among them, so
a couple of posts move recall by ~8 points. Read the DIRECTION and whether the
change is large relative to that granularity; do not read three decimals.

Usage:
  python src/compare_risk_arms.py --arm hard
  python src/compare_risk_arms.py --arm cw --arm hard
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402


def load_arm(dirs):
    """Average the seeds' val probabilities, as deployment does."""
    P, rid = [], None
    for d in dirs:
        p = os.path.join(d, "val_predictions.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- has {d} finished?")
        v = pd.read_csv(p)
        v["row_id"] = v["row_id"].astype(str)
        v = v.sort_values("row_id").reset_index(drop=True)
        cols = [c for c in v.columns
                if c.lower().replace("prob_", "") in
                [r.lower() for r in RISK_LEVELS]]
        if len(cols) != 4:
            raise SystemExit(f"{p}: expected 4 prob columns, got {cols}")
        cols = sorted(cols, key=lambda c:
                      [r.lower() for r in RISK_LEVELS].index(
                          c.lower().replace("prob_", "")))
        P.append(v[cols].to_numpy(dtype=float))
        if rid is None:
            rid = v["row_id"].tolist()
        elif rid != v["row_id"].tolist():
            raise SystemExit(f"{p}: row order differs from the first seed")
    return np.mean(P, axis=0), rid


def report(name, P, rid, gold_map):
    gold = [gold_map[r] for r in rid]
    pred = [RISK_LEVELS[i] for i in P.argmax(1)]
    w = risk_weighted_f1(gold, pred)
    rows = []
    for c in RISK_LEVELS:
        g = np.array([x == c for x in gold])
        p = np.array([x == c for x in pred])
        tp = int((g & p).sum())
        prec = tp / max(int(p.sum()), 1)
        rec = tp / max(int(g.sum()), 1)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append((c, int(g.sum()), int(p.sum()), prec, rec, f1))
    return w, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="tag, e.g. hard -> runs/risk_hard_s13/14/15")
    ap.add_argument("--champ", default="runs/risk_v1,runs/risk_v1_s14,"
                                       "runs/risk_v1_s15")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    args = ap.parse_args()
    if not args.arm:
        raise SystemExit("pass at least one --arm")

    va = pd.read_csv(args.val, keep_default_na=False)
    gold_map = {str(r): normalize_label(x)
                for r, x in zip(va["row_id"], va["risk_level"])}

    arms = {"champion": [d.strip() for d in args.champ.split(",")]}
    for t in args.arm:
        arms[t] = [f"runs/risk_{t}_s{s}" for s in (13, 14, 15)]

    out = {}
    for name, dirs in arms.items():
        P, rid = load_arm(dirs)
        out[name] = report(name, P, rid, gold_map)
        print(f"{name:<12}{len(rid)} val posts   "
              f"weighted-F1 {out[name][0]:.4f}")

    base_w = out["champion"][0]
    for name in arms:
        if name == "champion":
            continue
        print("\n" + "=" * 70)
        print(f"{name.upper()} vs CHAMPION   "
              f"weighted-F1 {out[name][0]:.4f} vs {base_w:.4f}  "
              f"({out[name][0] - base_w:+.4f})")
        print("=" * 70)
        print(f"{'class':<12}{'gold':>5}{'pred':>6}{'prec':>8}{'rec':>8}"
              f"{'F1':>8}   {'(champion: prec/rec/F1)':<26}")
        print("-" * 78)
        for (c, g, p, pr, rc, f1), (_, _, p0, pr0, rc0, f10) in zip(
                out[name][1], out["champion"][1]):
            mark = "  <-- the target" if c == "attempt" else ""
            print(f"{c:<12}{g:>5}{p:>6}{pr:>8.3f}{rc:>8.3f}{f1:>8.3f}   "
                  f"{pr0:.3f}/{rc0:.3f}/{f10:.3f} (n={p0}){mark}")

        a = [r for r in out[name][1] if r[0] == "attempt"][0]
        a0 = [r for r in out["champion"][1] if r[0] == "attempt"][0]
        print(f"\n  attempt recall {a0[4]:.3f} -> {a[4]:.3f}   "
              f"({a[4] - a0[4]:+.3f})")
        print(f"  attempt F1     {a0[5]:.3f} -> {a[5]:.3f}   "
              f"({a[5] - a0[5]:+.3f})")
        others = sum(r[5] for r in out[name][1] if r[0] != "attempt") - \
            sum(r[5] for r in out["champion"][1] if r[0] != "attempt")
        print(f"  other 3 classes, summed F1 change {others:+.3f}")
        if a[5] > a0[5] and others > -0.05:
            print("\n  Mechanism working: attempt improves without the other")
            print("  classes paying for it.")
        elif a[5] > a0[5]:
            print("\n  attempt improved but the other classes gave up more.")
            print("  Lower --hard-weight and retrain rather than adopting this.")
        else:
            print("\n  attempt did NOT improve. The mining did not transfer.")

    print(f"\nval_split holds only ~{sum(1 for v in gold_map.values() if v == 'attempt')}"
          f" attempt posts, so one post moves recall by several points. Read")
    print("direction and magnitude, not decimals.")


if __name__ == "__main__":
    main()
