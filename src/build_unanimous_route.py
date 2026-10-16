"""
build_unanimous_route.py

Ship the one rule that met all four criteria: overrule the champion to a SINGLE
class, only when EVERY listed member agrees.

THE RULE. qwen72 + llama70 + llama8, unanimous, -> indicator.

    vs route_attempt(oof_base) 0.8055:  29 moved  19 fix / 6 brk
    vs oof_base (raw)          0.7985:  29 moved  19 fix / 6 brk

Identical fix/break ratio on both baselines is the robustness check this rule
passes that others in the same family have failed. A ~3:1 fix ratio on 1.8%
of posts, one class, zero fitted parameters.

WHY UNANIMITY RATHER THAN MAJORITY. With k challengers, unanimity IS "champion
weight k, ties to the champion". It degrades SAFELY as the champion gets
stronger, which is exactly the property majority lacks: on test the deployed
champion is 0.8203 while the OOF encoders it was validated against are
0.7985/0.7858, so a bare majority lets two weaker models outvote a better one,
which is a documented failure mode of majority-vote overrides in this chain.

REGIME MATCHING IS MANDATORY FOR llama8. Its OOF is ONE fold model per post;
the 5-fold AVERAGE is a different distribution, because averaging checkpoints
narrows the probability distribution and can shift the argmax. Pass a
SINGLE-FOLD test file -- artifacts/lora_loraL8_test_f{k}.csv -- not
lora_loraL8_test.csv. The script refuses the averaged file by name.

INDICATOR PAYS TWICE. Gold evidence is empty exactly on indicator posts and the
champion suppresses spans there, so every post moved INTO indicator also changes
its evidence score. The evidence column must be rebuilt afterwards -- WITH THE
_rc RUNS, which is what the champion's evidence actually comes from (verified:
the _rc vote reproduces it on 0 of 378 posts differing).

Usage:
  python src/build_unanimous_route.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label  # noqa: E402

PROBE = {"indicator": (112, 170), "ideation": (99, 142),
         "behavior": (69, 113), "attempt": (13, 41)}


def read_labels(path, name):
    d = pd.read_csv(path, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    col = next((c for c in ("cot", "pred_risk_level", "risk_level")
                if c in d.columns), None)
    if col is None:
        raise SystemExit(f"{path} has no risk label column")
    print(f"  {name:<10}{os.path.basename(path):<40}(column '{col}')")
    return {r: normalize_label(v) for r, v in zip(d["row_id"], d[col])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--target", default="indicator", choices=RISK_LEVELS)
    ap.add_argument("--members", nargs="+", default=[
        "qwen72=artifacts/cot_soft_oof.csv.extra.csv",
        "llama70=artifacts/cot_soft_llama_oof.csv.extra.csv",
        "llama8=artifacts/lora_loraL8_test_f0.csv",
    ])
    ap.add_argument("--out", default="submissions/NIDHI_unaind.csv")
    args = ap.parse_args()

    ch = pd.read_csv(args.champion, keep_default_na=False)
    ch["row_id"] = ch["row_id"].astype(str)
    print(f"champion {args.champion}: {len(ch)} rows")
    print(f"target class: {args.target}   rule: ALL members must agree")
    print("members:")

    M = {}
    for spec in args.members:
        n, p = spec.split("=", 1)
        if p.endswith("lora_loraL8_test.csv"):
            raise SystemExit(
                "REFUSING the 5-fold AVERAGED file for llama8. Its OOF is one "
                "fold model per post; the average is a different distribution "
                "and that mismatch is half of why bag3 lost. Pass a "
                "single-fold file: artifacts/lora_loraL8_test_f0.csv")
        if not os.path.exists(p):
            raise SystemExit(f"{n}: {p} not found")
        M[n] = read_labels(p, n)

    for n, m in M.items():
        absent = [r for r in ch["row_id"] if r not in m]
        if absent:
            raise SystemExit(f"{n} lacks {len(absent)} test rows "
                             f"(e.g. {absent[:3]})")

    spell = {}
    for v in ch["risk_level"]:
        spell.setdefault(normalize_label(v), str(v))
    tgt = spell.get(args.target, args.target)

    new, moved = [], 0
    for r, raw in zip(ch["row_id"], ch["risk_level"]):
        cur = normalize_label(raw)
        if cur != args.target and all(M[n][r] == args.target for n in M):
            new.append(tgt)
            moved += 1
        else:
            new.append(raw)

    out = ch.copy()
    out["risk_level"] = new
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    before = [normalize_label(v) for v in ch["risk_level"]]
    after = [normalize_label(v) for v in out["risk_level"]]
    bc, ac = Counter(before), Counter(after)
    print(f"\nunanimously routed {moved} of {len(ch)} posts to {args.target}")
    print(f"  (OOF rate was 29/1635 = 1.8%, so expect roughly "
          f"{round(0.018 * len(ch))})")
    print(f"\n{'class':<12}{'champ':>7}{'new':>6}{'probe 90%':>14}")
    print("-" * 40)
    for c in RISK_LEVELS:
        lo, hi = PROBE[c]
        flag = "" if lo <= ac[c] <= hi else "  <- OUTSIDE"
        print(f"{c:<12}{bc[c]:>7}{ac[c]:>6}{f'{lo}-{hi}':>14}{flag}")
    for col in ("evidence", "factors"):
        if col in ch.columns:
            same = int((ch[col].astype(str) == out[col].astype(str)).sum())
            print(f"  {col}: {same}/{len(ch)} unchanged")
    print(f"\nwrote {args.out}")

    if args.target == "indicator" and moved:
        print("\n" + "=" * 72)
        print("EVIDENCE MUST BE REBUILT -- and with the _rc RUNS")
        print("=" * 72)
        print(f"{moved} posts moved INTO indicator, so the suppression rule now")
        print("has to empty their spans. Rebuilding with the ORIGINAL eight")
        print("taggers instead of the _rc ones changes 119 of 378 posts at")
        print("identical risk and silently reverts the evcond win -- that is")
        print("what sank bag3_ev. The _rc vote reproduces the champion exactly")
        print("(verified: 0 of 378 differ).\n")
        print("  RC8=evidence_v2_rc,evidence_v1_rc,evidence_s14_rc,"
              "evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,"
              "evidence_lr2_rc,evidence_ep9_rc")
        print("  python src/evidence_vote_test.py --runs \"$RC8\" \\")
        print("      --dir artifacts/evidence_runs --k 2 --cluster --rep modal \\")
        print(f"      --test data/test.csv --risk-from {args.out} \\")
        print("      --out artifacts/evvote_test_unaind.csv")
        print(f"  python src/patch_submission_risk.py --base {args.out} \\")
        print("      --evidence artifacts/evvote_test_unaind.csv \\")
        print("      --out submissions/NIDHI_unaind_ev.csv")


if __name__ == "__main__":
    main()
