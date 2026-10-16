"""
build_risk_majority.py

Build the 3-way MAJORITY risk vote: champion + CoT-72B + LoRA-Llama-8B.

THE RESULT THIS DEPLOYS. `risk_bag_eval.py`, measured on 1635 out-of-fold
posts / 153 authors:

    encoder + cot72b + llama, MAJORITY
      vs oof_base      0.7985 -> 0.8259
      vs champion_oof  0.7858 -> 0.8206

Majority beat prob-avg, logit-avg and maxconf in nearly every subset, and the
winning trio is exactly the three lowest-overlap members (pairwise error
Jaccard 0.376-0.471); adding the 4th member, `qwen`, made it WORSE because it
correlates 0.502 with llama.

TIE-BREAK. Three voters over four classes can split 1-1-1. Offline that was
broken by mean probability, but the champion member here is a hard label from a
submission file with no probability vector, so ties fall back to the CHAMPION --
the strongest single member and the safe default. The count is reported; if it
is large the rule is doing more than voting.

THE EVIDENCE COUPLING, which is the real trap. Gold evidence is empty exactly on
indicator posts and `evidence_vote_test.py` suppresses all spans when the
predicted level is indicator. So a post CROSSING that boundary needs its
evidence rebuilt:

    into indicator   -> spans must be emptied
    out of indicator -> spans must be GENERATED, and it currently has none

This script reports both counts and refuses to pretend the evidence column is
still valid. Re-run the evidence vote against the new risk column afterwards --
the command is printed.

Usage:
  python src/build_risk_majority.py
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


def read_labels(path: str, name: str) -> dict:
    d = pd.read_csv(path, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    col = next((c for c in ("risk_level", "pred_risk_level", "cot")
                if c in d.columns), None)
    if col is None:
        raise SystemExit(f"{path} has no risk label column")
    print(f"  {name:<10}{path}  (column '{col}')")
    return {r: normalize_label(v) for r, v in zip(d["row_id"], d[col])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--cot", default="artifacts/cot_soft_oof.csv.extra.csv")
    ap.add_argument("--lora", default="artifacts/lora_loraL8_test.csv")
    ap.add_argument("--out", default="submissions/NIDHI_bag3.csv")
    args = ap.parse_args()

    ch = pd.read_csv(args.champion, keep_default_na=False)
    ch["row_id"] = ch["row_id"].astype(str)
    print(f"champion {args.champion}: {len(ch)} rows")
    print("members:")
    A = read_labels(args.champion, "champion")
    B = read_labels(args.cot, "cot72b")
    C = read_labels(args.lora, "llama")

    for nm, m in (("cot72b", B), ("llama", C)):
        absent = [r for r in ch["row_id"] if r not in m]
        if absent:
            raise SystemExit(
                f"{nm} lacks predictions for {len(absent)} test rows "
                f"(e.g. {absent[:3]}). Voting on part of the test set would "
                f"ship an unmeasured mixture.")

    spell = {}
    for v in ch["risk_level"]:
        spell.setdefault(normalize_label(v), str(v))
    missing_spell = [c for c in RISK_LEVELS if c not in spell]
    if missing_spell:
        print(f"  note: champion never predicts {missing_spell}; using the "
              f"canonical spelling for those")
        for c in missing_spell:
            spell[c] = c

    new, n_tie, n_moved = [], 0, 0
    for r, raw in zip(ch["row_id"], ch["risk_level"]):
        a, b, c = A[r], B[r], C[r]
        cnt = Counter([a, b, c])
        top, k = cnt.most_common(1)[0]
        if k >= 2:
            pick = top
        else:
            pick = a          # 1-1-1 split: keep the champion
            n_tie += 1
        if pick != a:
            n_moved += 1
        new.append(spell[pick] if pick != a else raw)

    out = ch.copy()
    out["risk_level"] = new
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\nmajority moved {n_moved} of {len(ch)} posts")
    print(f"  three-way ties (kept champion): {n_tie}")

    before = [normalize_label(v) for v in ch["risk_level"]]
    after = [normalize_label(v) for v in out["risk_level"]]
    bc, ac = Counter(before), Counter(after)
    print(f"\n{'class':<12}{'champ':>7}{'new':>6}{'probe 90%':>14}")
    print("-" * 40)
    for c in RISK_LEVELS:
        lo, hi = PROBE[c]
        flag = "" if lo <= ac[c] <= hi else "  <- OUTSIDE"
        print(f"{c:<12}{bc[c]:>7}{ac[c]:>6}{f'{lo}-{hi}':>14}{flag}")

    into = sum(1 for b, a in zip(before, after)
               if a == "indicator" and b != "indicator")
    outof = sum(1 for b, a in zip(before, after)
                if b == "indicator" and a != "indicator")
    print(f"\nINDICATOR BOUNDARY CROSSINGS -- these break evidence consistency")
    print(f"  into indicator  : {into}  (their spans must be emptied)")
    print(f"  out of indicator: {outof}  (they have NO spans and need some)")
    print(f"\nwrote {args.out}")

    if into or outof:
        print("\n" + "=" * 70)
        print("THE EVIDENCE COLUMN IN THAT FILE IS NOW INCONSISTENT.")
        print("=" * 70)
        print("Rebuild it against the new risk labels before uploading.\n")
        print("USE THE _rc RUNS. The champion's evidence comes from the")
        print("RISK-CONDITIONED taggers, NOT the original eight. Rebuilding")
        print("with the originals silently reverts to a worse evidence")
        print("column AND changes 119 of 378 posts at identical risk, which")
        print("is a regression this exact mismatch has caused before.")
        print("VERIFY the run list first: rebuild at the CHAMPION'S OWN risk")
        print("and confirm it reproduces the champion's evidence column")
        print("EXACTLY. If it does not, the run list is wrong.\n")
        print("  RC8=evidence_v2_rc,evidence_v1_rc,evidence_s14_rc,"
              "evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,"
              "evidence_lr2_rc,evidence_ep9_rc")
        print(f"  python src/evidence_vote_test.py --runs \"$RC8\" \\")
        print("      --dir artifacts/evidence_runs --k 2 --cluster --rep modal \\")
        print(f"      --test data/test.csv --risk-from {args.out} \\")
        print("      --out artifacts/evvote_test_bag3.csv")
        print(f"  python src/patch_submission_risk.py --base {args.out} \\")
        print("      --evidence artifacts/evvote_test_bag3.csv \\")
        print("      --out submissions/NIDHI_bag3_ev.csv\n")
        print("k=2 over the _rc EIGHT -- not 17 at k=4, since a validation-only")
        print("gain on a larger cluster config is not sufficient evidence it")
        print("transfers to held-out evaluation.")
    else:
        print("\nNo indicator crossings, so the evidence column stays valid "
              "and this is a pure risk change.")


if __name__ == "__main__":
    main()
