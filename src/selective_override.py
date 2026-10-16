"""
selective_override.py

Can Qwen ADD to the champion rather than replace it?

WHERE THIS COMES FROM. compare_vs_champion_risk.py settled the replacement
question on 295 val posts with honest held-out predictions on both sides:

    CHAMPION   0.8202
    ord Qwen   0.7777
    base Qwen  0.7323

Replacement is dead. But the ordinal arm is not uniformly worse:

    18 posts Qwen gets right and the champion misses
    31 posts the champion gets right and Qwen misses

Those 18 are real, non-overlapping skill. Taking Qwen everywhere trades 18 for
31 and loses. The only way it pays is to take Qwen ONLY where it is likely to
be one of the 18 -- and the natural signal for that is its own confidence.

THIS IS A TUNED THRESHOLD, THE KIND OF CHANGE THAT MOST OFTEN FAILS TO
TRANSFER. So it is held to the strictest standard used in this codebase:

    - the confidence cut-off is CROSS-FITTED across author-disjoint halves
      (chosen on one half, applied blind to the other, both directions)
    - it must beat the champion on BOTH halves, not just overall
    - the sweep is printed so a gain that exists at exactly one cut-off is
      visible as noise rather than reported as a result

  If the gain only appears at one threshold, or only on one half, it is noise
  and it does not get a submission slot. On this metric the champion is at
  0.8202, and 295 posts give this instrument coarse resolution, so a small
  apparent gain here is not distinguishable from chance and should not be
  shipped either.

Usage:
  python src/selective_override.py \
      --arm 'v2_rebuild/runs/risk_llm2_ord_f{f}'
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion",
                    default="artifacts/val_champion_reconstructed.csv")
    ap.add_argument("--champion-col", default="champion_final")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--arm", required=True, help="run dir template with {f}")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    ch = pd.read_csv(args.champion)
    val = pd.read_csv(args.val)
    amap = dict(zip(val["row_id"], val["anon_user_id"]))

    parts = []
    for f in range(args.k):
        p = os.path.join(args.arm.replace("{f}", str(f)), "oof_predictions.csv")
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    if not parts:
        raise SystemExit(f"no fold files under {args.arm}")
    q = pd.concat(parts, ignore_index=True)
    pcols = [f"prob_{c}" for c in RISK_LEVELS]
    missing = [c for c in pcols if c not in q.columns]
    if missing:
        raise SystemExit(f"arm OOF has no {missing}")

    qp = {r: normalize_label(p) for r, p in zip(q["row_id"], q["pred"])}
    qc = {r: float(v) for r, v in zip(q["row_id"], q[pcols].to_numpy().max(1))}

    rows = sorted(set(ch["row_id"]) & set(qp) & set(amap))
    if not rows:
        raise SystemExit("no overlap")
    gold = {r: normalize_label(g) for r, g in zip(ch["row_id"], ch["gold"])}
    champ = {r: normalize_label(p)
             for r, p in zip(ch["row_id"], ch[args.champion_col])}

    g = [gold[r] for r in rows]
    base = risk_weighted_f1(g, [champ[r] for r in rows])
    half = author_halves([amap[r] for r in rows])
    print(f"{len(rows)} posts   champion {base:.4f}   "
          f"halves A={int((half == 'A').sum())} B={int((half == 'B').sum())}")

    def score(th, mask=None):
        p = [qp[r] if (qp[r] != champ[r] and qc[r] >= th) else champ[r]
             for r in rows]
        if mask is None:
            return risk_weighted_f1(g, p)
        gg = [x for x, m in zip(g, mask) if m]
        pp = [x for x, m in zip(p, mask) if m]
        return risk_weighted_f1(gg, pp)

    grid = np.round(np.arange(0.50, 1.001, 0.025), 3)

    print(f"\n{'conf':>7}{'overall':>10}{'delta':>9}{'halfA':>9}{'halfB':>9}"
          f"{'n_over':>8}{'gained':>8}{'lost':>8}")
    print("-" * 68)
    bA = score(2.0, half == "A")
    bB = score(2.0, half == "B")
    for th in grid:
        over = [r for r in rows if qp[r] != champ[r] and qc[r] >= th]
        gain = sum(1 for r in over if qp[r] == gold[r] and champ[r] != gold[r])
        lost = sum(1 for r in over if champ[r] == gold[r] and qp[r] != gold[r])
        print(f"{th:>7.3f}{score(th):>10.4f}{score(th) - base:>+9.4f}"
              f"{score(th, half == 'A') - bA:>+9.4f}"
              f"{score(th, half == 'B') - bB:>+9.4f}"
              f"{len(over):>8}{gain:>8}{lost:>8}")

    # Cross-fitted: pick the cut-off on one half, apply blind to the other.
    print("\nCROSS-FITTED (cut-off chosen on the OTHER half -- the honest number)")
    tot = []
    for tgt, src in (("A", "B"), ("B", "A")):
        mt, ms = half == tgt, half == src
        bt = score(2.0, mt)
        best = max(grid, key=lambda t: score(t, ms))
        d = score(best, mt) - bt
        tot.append(d)
        print(f"  tune on {src}, apply to {tgt}: cut-off {best:.3f}  "
              f"{bt:.4f} -> {score(best, mt):.4f}  ({d:+.4f})")
    mean = float(np.mean(tot))
    print(f"  mean cross-fitted delta {mean:+.4f}")

    ceiling = risk_weighted_f1(
        g, [qp[r] if qp[r] == gold[r] else champ[r] for r in rows])
    print(f"\noracle ceiling (take Qwen only when it is right): {ceiling:.4f} "
          f"({ceiling - base:+.4f})")
    print("  -- unreachable, shown only to bound what any confidence rule "
          "could ever buy")

    print("\n" + "=" * 68)
    if mean > 0.02 and all(x > 0 for x in tot):
        print("WORTH A SUBMISSION: cross-fitted gain on both halves, above the")
        print("~0.02 resolution of 295 posts.")
        print("Next: run the ord arm over test, apply the same rule to the")
        print("champion risk column, patch risk ONLY.")
    else:
        print("NOT worth a submission. Either the gain is one-sided, or it is")
        print("inside the noise floor of 295 posts. The LLM risk line closes")
        print("here -- replacement lost, and selective override does not pay.")
    print("=" * 68)


if __name__ == "__main__":
    main()
