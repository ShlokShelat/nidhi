"""
build_route_attempt.py

Ship `route_attempt`: take the CoT-72B's ATTEMPT calls, keep the champion's
risk label everywhere else.

WHY THIS RULE AND NOT THE OTHER THREE. Measured on honest OOF against BOTH
available baselines, zero fitted parameters:

    baseline            posts moved
    oof_base   0.7985       28
    champ_oof  0.7841       26

Only rule positive and stable across both. `route_behavior` FLIPS SIGN between
them because champion_oof has much weaker behavior recall, and
`route_indicator` is indistinguishable from noise on both baselines, despite
the LLM's better indicator RECALL, because its indicator PRECISION is worse
(0.873 vs 0.888).

THE MECHANISM, from the disagreement matrix (encoder-right / llm-right):
    encoder said behavior, llm said attempt :  5/15  and  4/11
The LLM wins that cell ~3:1 on both baselines. That is the behavior/attempt
near-tie (0.401 vs 0.426) that four definition arms and five symbolic rules
failed to fix. Reasoning reaches it.

TWO SAFEGUARDS BUILT IN.

1. INDICATOR POSTS ARE NEVER ROUTED. Gold evidence is empty exactly on
   indicator posts and the champion suppresses spans there, so promoting an
   indicator post to attempt would leave it with EMPTY evidence against
   non-empty gold -- scoring 0 on that post's evidence. The OOF matrix shows
   indicator->attempt is 1 post, so this costs nothing and keeps the change
   confined to the risk column.

2. PREVALENCE IS REPORTED AGAINST THE PROBE INTERVALS. Offline-validated
   changes that move predictions TOWARD TRAIN PREVALENCE are a known failure
   pattern: an apparent offline gain can be an artifact of matching the
   training distribution rather than a real improvement. This rule moves
   attempt UP (champion 17), and train prevalence is 6.9% = ~26 posts on 378.
   That coincidence is worth seeing before uploading, even though this rule is
   per-post routing rather than a calibration change.

Usage:
  python src/build_route_attempt.py
  python src/build_route_attempt.py --out submissions/NIDHI_routeatt.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label  # noqa: E402

# 90% intervals from the marginal probes, posts out of 378
PROBE = {"indicator": (112, 170), "ideation": (99, 142),
         "behavior": (69, 113), "attempt": (13, 41)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="submissions/NIDHI_evcond_bag8.csv",
                    help="base submission; defaults to the merged 0.7549 file")
    ap.add_argument("--llm", default="artifacts/cot_soft_oof.csv.extra.csv",
                    help="CoT-72B TEST predictions (row_id, cot[, prob_*])")
    ap.add_argument("--out", default="submissions/NIDHI_routeatt.csv")
    args = ap.parse_args()

    ch = pd.read_csv(args.champion, keep_default_na=False)
    ch["row_id"] = ch["row_id"].astype(str)
    lm = pd.read_csv(args.llm, keep_default_na=False)
    lm["row_id"] = lm["row_id"].astype(str)
    col = "cot" if "cot" in lm.columns else "pred_risk_level"
    llm = {r: normalize_label(v) for r, v in zip(lm["row_id"], lm[col])}

    absent = [r for r in ch["row_id"] if r not in llm]
    if absent:
        raise SystemExit(
            f"{args.llm} lacks predictions for {len(absent)} test rows "
            f"(e.g. {absent[:3]}). Routing part of the test set would ship an "
            f"unmeasured mixture.")

    # preserve the champion's own casing for the label strings
    spell = {}
    for v in ch["risk_level"]:
        spell.setdefault(normalize_label(v), str(v))
    if "attempt" not in spell:
        raise SystemExit("champion has no attempt rows; cannot infer its "
                         "spelling of the label")
    att = spell["attempt"]
    print(f"champion label spellings: {spell}")

    cur = [normalize_label(v) for v in ch["risk_level"]]
    new, n_moved, n_skipped_ind = [], 0, 0
    for r, c, raw in zip(ch["row_id"], cur, ch["risk_level"]):
        want = llm[r]
        if want == "attempt" and c != "attempt":
            if c == "indicator":
                # never promote an indicator post: its evidence is suppressed
                # to empty and would then score 0 against non-empty gold
                n_skipped_ind += 1
                new.append(raw)
            else:
                n_moved += 1
                new.append(att)
        else:
            new.append(raw)

    out = ch.copy()
    out["risk_level"] = new
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\nrouted {n_moved} posts to attempt")
    print(f"  skipped {n_skipped_ind} indicator posts (evidence-consistency "
          f"guard)")
    for c in ("evidence", "factors"):
        if c in ch.columns:
            same = int((ch[c].astype(str) == out[c].astype(str)).sum())
            print(f"  {c}: {same}/{len(ch)} unchanged"
                  + ("" if same == len(ch) else "   <- SHOULD BE ALL"))

    before = Counter(cur)
    after = Counter(normalize_label(v) for v in out["risk_level"])
    print(f"\n{'class':<12}{'before':>8}{'after':>7}{'probe 90%':>14}")
    print("-" * 41)
    for c in ("indicator", "ideation", "behavior", "attempt"):
        lo, hi = PROBE[c]
        flag = "" if lo <= after[c] <= hi else "  <- OUTSIDE"
        print(f"{c:<12}{before[c]:>8}{after[c]:>7}{f'{lo}-{hi}':>14}{flag}")

    print(f"\nwrote {args.out}")
    print("\nEXPECTED: risk +0.0062 to +0.0070 wF1 -> composite +0.0025 to")
    print("+0.0028 over the base file. Zero fitted parameters, replicated on")
    print("two independent baselines.")
    print("\nWATCH: attempt moves UP toward train prevalence (6.9% = ~26 of")
    print("378). A prediction shift toward train prevalence is a known false-")
    print("positive pattern for offline-validated changes. This is per-post")
    print("routing rather than a calibration change, but if this submission")
    print("underperforms, that pattern is the first suspect.")


if __name__ == "__main__":
    main()
