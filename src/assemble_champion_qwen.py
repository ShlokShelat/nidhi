"""
assemble_champion_qwen.py

Build the test risk column: the base ensemble plus Qwen as an extra member.

    ensembled = artifacts/ensembled_risk.csv        (already on disk:
                0.5 x mean(3 risk_v1 seeds) + 0.5 x friend, p_* columns)
    blended   = 0.70 x ensembled + 0.30 x NLI
    withqwen  = (1 - w) x blended + w x qwen_ord
    final     = apply_rules(withqwen, [urgency, completed_act])

WHY w DEFAULTS BELOW THE SWEEP PEAK. The argmax of a weight sweep is the value
most fitted to the particular validation split it was swept on, so it is the
value least likely to transfer. The default takes the conservative end of the
range that cross-fitting supports, which keeps the blend close to the base
ensemble: when the expected gain is small, over-shooting the weight costs more
than under-shooting it.

SAFETY. The script refuses to write if the reconstructed pre-Qwen blend does
not closely reproduce the reference risk column on test. Without that guard, a
reconstruction that silently differs from the intended base would measure the
Qwen contribution against the wrong thing and write a column nobody validated.

Usage:
  python src/assemble_champion_qwen.py --w 0.30
  python src/patch_submission_risk.py \
      --base submissions/NIDHI_evidence_consistent.csv \
      --risk artifacts/risk_champion_qwen.csv \
      --out  submissions/NIDHI_champion_qwen.csv --max-changes 60
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scoring import RISK_LEVELS, normalize_label
from apply_rules_to_test import apply_rules

PCOLS = [f"prob_{c}" for c in RISK_LEVELS]
ENSCOLS = ["p_Indicator", "p_Ideation", "p_Behavior", "p_Attempt"]


def norm(X):
    return X / X.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--ensembled", default="artifacts/ensembled_risk.csv")
    ap.add_argument("--nli", default="artifacts/nli_risk_test_probs.csv")
    ap.add_argument("--qwen", default="artifacts/qwen_ord_test_probs.csv")
    ap.add_argument("--champion-risk", default="artifacts/risk_blend_equal.csv",
                    help="the reference pre-rule risk labels, for the "
                         "reconstruction guard")
    ap.add_argument("--w-nli", type=float, default=0.30)
    ap.add_argument("--w", type=float, default=0.30, help="Qwen weight")
    ap.add_argument("--rules", default="urgency,completed_act")
    ap.add_argument("--out", default="artifacts/risk_champion_qwen.csv")
    ap.add_argument("--min-agree", type=float, default=0.97)
    args = ap.parse_args()

    te = pd.read_csv(args.test).set_index("row_id")
    ens = pd.read_csv(args.ensembled).set_index("row_id")
    nli = pd.read_csv(args.nli).set_index("row_id")
    qw = pd.read_csv(args.qwen).set_index("row_id")

    rows = list(te.index)
    for nm, d in (("ensembled", ens), ("nli", nli), ("qwen", qw)):
        miss = [r for r in rows if r not in d.index]
        if miss:
            raise SystemExit(f"{nm} missing {len(miss)} test rows, "
                             f"e.g. {miss[:3]}")

    E = norm(ens.loc[rows, ENSCOLS].to_numpy(float))
    N = norm(nli.loc[rows, PCOLS].to_numpy(float))
    Q = norm(qw.loc[rows, PCOLS].to_numpy(float))

    blended = (1 - args.w_nli) * E + args.w_nli * N
    posts = list(te.loc[rows, "post"])
    keys = [r.strip() for r in args.rules.split(",")]

    def finalize(P):
        lab = [RISK_LEVELS[i] for i in P.argmax(1)]
        return [apply_rules(posts[i], lab[i], keys)[0] for i in range(len(lab))]

    # --- guard: does the reconstruction reproduce the reference risk column? ---
    pre = [RISK_LEVELS[i] for i in blended.argmax(1)]
    if os.path.exists(args.champion_risk):
        cr = pd.read_csv(args.champion_risk).set_index("row_id")
        col = next((c for c in ("pred_risk_level", "risk_level")
                    if c in cr.columns), None)
        if col:
            ref = [normalize_label(cr.loc[r, col]) for r in rows]
            agree = float(np.mean([a == b for a, b in zip(pre, ref)]))
            print(f"reconstruction agrees with {args.champion_risk} "
                  f"on {100 * agree:.1f}% of test posts")
            if agree < args.min_agree:
                raise SystemExit(
                    f"below --min-agree {args.min_agree:.2f}. The rebuilt "
                    f"blend is NOT the reference blend, so adding Qwen to it "
                    f"would measure the wrong thing. Refusing to write.")
    else:
        print(f"WARNING: {args.champion_risk} not found -- reconstruction "
              f"guard SKIPPED. Verify before submitting.")

    final = finalize((1 - args.w) * blended + args.w * Q)
    base = finalize(blended)
    changed = sum(1 for a, b in zip(final, base) if a != b)

    out = pd.DataFrame({"row_id": rows,
                        "pred_risk_level": [x.capitalize() for x in final]})
    P = (1 - args.w) * blended + args.w * Q
    for j, c in enumerate(RISK_LEVELS):
        out[f"prob_{c}"] = P[:, j]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    from collections import Counter
    print(f"\nw_nli={args.w_nli}  w_qwen={args.w}")
    print(f"Qwen changed {changed} of {len(rows)} posts vs the base blend")
    print(f"base blend     : {dict(Counter(base))}")
    print(f"with qwen      : {dict(Counter(final))}")
    print(f"\nwrote {args.out}")
    print("\nNext:")
    print("  python src/patch_submission_risk.py \\")
    print("      --base submissions/NIDHI_evidence_consistent.csv \\")
    print(f"      --risk {args.out} \\")
    print("      --out submissions/NIDHI_champion_qwen.csv --max-changes 60")


if __name__ == "__main__":
    main()
