"""
mine_attempt_patterns.py

Extend `completed_act` to cover the attempt phrasings it misses.

THE MEASURED WEAKNESS. `attempt` is the worst cell in the risk system:
P 0.75 / R 0.24 / F1 0.364 on val -- six of 25 gold attempts found, fifteen
called `behavior`. The champion predicts 17 attempts on 378 test posts where
training prevalence implies about 26. Fixing behavior<->attempt confusion
accounts for more than half of the ideation/behavior/attempt headroom, and
unlike the ideation/behavior boundary it is a RECALL problem with a mechanism
a regex can reach: the official definition is "explicit mentions of RECENT OR
PAST suicidal attempts", which is a tense distinction.

WHY THE DEPLOYED RULE MISSES THEM. `completed_act` has eight patterns and every
one is about overdose or cutting -- i've cut, i cut my, took N, i've taken,
i have taken, taken N, i swallowed, i overdosed. Nothing for "tried to kill
myself", "my attempt", "survived", "woke up in the hospital", "they found me",
"failed attempt". It changed exactly ONE test post.

WHY THIS IS AN ALLOWED MOVE. New symbolic rule FAMILIES were closed after
several failed on a prevalence argument -- too few eligible posts. This is an
EXTENSION of a rule already deployed and validated, which is the same shape
as the only rule changes that have ever transferred: additions to an existing
rule's pattern list. It adds no fitted parameters: a pattern either matches or
it does not.

THE DISCIPLINE. Every candidate is scored on the population the rule can
actually act on -- posts the model predicted `behavior` -- over the 1590-post
out-of-fold instrument, not the 372-post val split that misread three changes.
A candidate is reported as SAFE only if its net (corrected minus broken) is
positive on BOTH author-disjoint halves. That is exactly the bar that rejected
a related guarded-rule candidate whose net was negative on too small a
footprint to trust.

Usage:
  python src/mine_attempt_patterns.py
  python src/mine_attempt_patterns.py --from-labels behavior ideation
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_audit import load
from scoring import normalize_label

# Hand-written candidates covering the phrasings the deployed list omits.
# Grouped so the report shows which family of language each one belongs to.
# These come from the OFFICIAL definition ("recent or past suicidal attempts"),
# not from a score search.
CANDIDATES = {
    "explicit attempt verb": [
        r"\b(i|i've|i have)\s+(tried|attempted)\s+(to\s+)?(kill|end|off)\b",
        r"\btried\s+to\s+(kill|end)\s+(my\s?self|me|my life)\b",
        r"\battempted\s+suicide\b",
        r"\bmy\s+(suicide\s+)?attempts?\b",
        r"\b(first|second|third|last|previous|failed)\s+attempt\b",
        r"\b\d+\s+attempts?\b",
    ],
    "survived / aftermath": [
        r"\bi\s+survived\b",
        r"\bwoke\s+up\s+in\s+(the\s+)?(hospital|hospital bed|icu|er)\b",
        r"\b(they|someone|my\s+\w+)\s+found\s+me\b",
        r"\bgot\s+my\s+stomach\s+pumped\b",
        r"\bstomach\s+pumped\b",
        r"\b(was|been)\s+(hospitalised|hospitalized|committed)\b",
        r"\bpsych\s+ward\b",
        r"\b72\s+hour\s+hold\b",
    ],
    "failed attempt framing": [
        r"\b(it|the attempt)\s+didn'?t\s+work\b",
        r"\bfailed\s+to\s+(kill|end)\b",
        r"\bi\s+(didn'?t|couldn'?t)\s+die\b",
        r"\bstill\s+(here|alive)\s+(after|because)\b",
        r"\bshould\s+have\s+(died|worked)\b",
    ],
    "past-tense method": [
        r"\bi\s+(hung|hanged)\s+my\s?self\b",
        r"\bi\s+jumped\b",
        r"\bi\s+slit\b",
        r"\bi\s+shot\s+my\s?self\b",
        r"\bi\s+drank\s+(bleach|antifreeze)\b",
        r"\bi\s+tried\s+(hanging|asphyxiation|overdosing)\b",
    ],
}

DEPLOYED = [
    r"\bi'?ve cut\b", r"\bi cut my\b",
    r"\b(i|just|already)\b.{0,10}\btook\b.{0,15}\d+\b",
    r"\bi'?ve taken\b", r"\bi have taken\b", r"\btaken \d+\b",
    r"\bi swallowed\b", r"\bi overdosed\b",
]


def fires(pats, text):
    t = text.lower()
    return any(re.search(p, t) for p in pats)


def author_halves(auth):
    s = pd.Series(auth)
    c = s.value_counts()
    half, size = {}, {"A": 0, "B": 0}
    for a in sorted(c.index, key=lambda x: (-c[x], str(x))):
        p = "A" if size["A"] <= size["B"] else "B"
        half[a] = p
        size[p] += c[a]
    return s.map(half).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="artifacts/champion_oof.csv",
                    help="pre-rule OOF predictions (champion_oof.py output)")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--from-labels", nargs="+", default=["behavior"],
                    help="predicted labels the rule may promote from")
    ap.add_argument("--min-net", type=int, default=2)
    args = ap.parse_args()

    if not os.path.exists(args.baseline):
        raise SystemExit(f"missing {args.baseline} -- run "
                         "python src/champion_oof.py --export "
                         f"{args.baseline}")
    pred = pd.read_csv(args.baseline, keep_default_na=False)
    col = next((c for c in ("pred_risk_level", "pred", "risk_level")
                if c in pred.columns), None)
    if col is None:
        raise SystemExit(f"{args.baseline} has no prediction column: "
                         f"{list(pred.columns)}")
    lab = load(args.labeled).set_index("row_id")

    rid = [r for r in pred["row_id"] if r in lab.index]
    p = {r: normalize_label(v) for r, v in zip(pred["row_id"], pred[col])}
    gold = {r: normalize_label(lab.loc[r, "risk_level"]) for r in rid}
    post = {r: str(lab.loc[r, "post"]) for r in rid}
    auth = [lab.loc[r, "anon_user_id"] for r in rid]
    half = author_halves(auth)

    elig = [r for r in rid if p[r] in args.from_labels]
    recoverable = [r for r in elig if gold[r] == "attempt"]
    print(f"{len(rid)} OOF posts   halves "
          f"{int((half=='A').sum())}/{int((half=='B').sum())}")
    print(f"predicted {args.from_labels}: {len(elig)} posts")
    print(f"  of those, gold=attempt (RECOVERABLE): {len(recoverable)}")
    print(f"  of those, gold!=attempt (AT RISK)   : {len(elig)-len(recoverable)}")
    natt = sum(1 for r in rid if gold[r] == "attempt")
    found = sum(1 for r in rid if gold[r] == "attempt" and p[r] == "attempt")
    print(f"\ngold attempts overall {natt}, model finds {found} "
          f"(recall {found/natt:.3f})")
    print(f"the deployed 8 patterns already promote: "
          f"{sum(1 for r in elig if fires(DEPLOYED, post[r]))} of {len(elig)} "
          f"eligible")

    hmap = dict(zip(rid, half))
    print("\n" + "=" * 96)
    print("CANDIDATE PATTERNS, scored on the population the rule can act on")
    print("=" * 96)
    print(f"{'pattern':<52}{'fires':>6}{'corr':>6}{'brk':>5}{'net':>5}"
          f"{'netA':>6}{'netB':>6}  verdict")
    print("-" * 96)

    keep = []
    for fam, pats in CANDIDATES.items():
        print(f"\n  [{fam}]")
        for pat in pats:
            f = [r for r in elig if re.search(pat, post[r].lower())]
            corr = [r for r in f if gold[r] == "attempt"]
            brk = [r for r in f if gold[r] != "attempt"]
            net = len(corr) - len(brk)
            nA = sum(1 for r in corr if hmap[r] == "A") - \
                sum(1 for r in brk if hmap[r] == "A")
            nB = sum(1 for r in corr if hmap[r] == "B") - \
                sum(1 for r in brk if hmap[r] == "B")
            safe = net >= args.min_net and nA >= 0 and nB >= 0
            if safe:
                keep.append(pat)
            v = "SAFE" if safe else ("neutral" if net >= 0 else "harmful")
            print(f"    {pat:<50}{len(f):>6}{len(corr):>6}{len(brk):>5}"
                  f"{net:>+5}{nA:>+6}{nB:>+6}  {v}")

    print("\n" + "=" * 96)
    print(f"COMBINED EFFECT of the {len(keep)} SAFE patterns")
    print("=" * 96)
    if not keep:
        print("  none passed. The attempt-recall gap is not regex-reachable,")
        print("  and this direction closes.")
        return

    f = [r for r in elig if fires(keep, post[r])]
    corr = [r for r in f if gold[r] == "attempt"]
    brk = [r for r in f if gold[r] != "attempt"]
    print(f"  fires on {len(f)} eligible posts: "
          f"{len(corr)} corrected, {len(brk)} broken, net {len(corr)-len(brk):+d}")
    for h in ("A", "B"):
        c = sum(1 for r in corr if hmap[r] == h)
        b = sum(1 for r in brk if hmap[r] == h)
        print(f"    half {h}: {c} corrected, {b} broken, net {c-b:+d}")

    from sklearn.metrics import f1_score
    LEV = ["indicator", "ideation", "behavior", "attempt"]
    y = np.array([LEV.index(gold[r]) for r in rid])
    before = np.array([LEV.index(p[r]) for r in rid])
    after = before.copy()
    fs = set(f)
    for i, r in enumerate(rid):
        if r in fs:
            after[i] = LEV.index("attempt")
    wb = f1_score(y, before, average="weighted")
    wa = f1_score(y, after, average="weighted")
    print(f"\n  weighted-F1  {wb:.4f} -> {wa:.4f}   ({wa-wb:+.4f})")
    print(f"  composite impact if it transfers: {0.4*(wa-wb):+.4f}")
    fb = f1_score((y == 3).astype(int), (before == 3).astype(int), zero_division=0)
    fa = f1_score((y == 3).astype(int), (after == 3).astype(int), zero_division=0)
    print(f"  attempt F1   {fb:.3f} -> {fa:.3f}")

    print("\n  Patterns to add to completed_act['present'] in "
          "src/candidate_patterns.py:")
    for pat in keep:
        print(f"      r\"{pat}\",")
    print("\n  Then re-verify with:")
    print("    python src/rule_ablation_v2.py --baseline artifacts/champion_oof.csv \\")
    print("        --val artifacts/full_train_oof.csv --chain urgency,completed_act")
    print("\n  NOTE these patterns were selected on the same 1590 posts they are")
    print("  scored on, so the net figures are optimistic. The both-halves gate")
    print("  is the guard, and the honest test is held-out evaluation -- but only")
    print("  ship if the combined net is comfortably above the handful of posts")
    print("  that noise alone would move.")


if __name__ == "__main__":
    main()
