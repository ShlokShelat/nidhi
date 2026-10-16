"""
retest_rules_on_twin.py

RE-JUDGE EVERY CANDIDATE SYMBOLIC RULE ON THE MATCHED BASE.

The layer decomposition from `champion_oof_twin.py` + `retest_routes_on_twin.py`
is the reason this script exists: the blend-only stage, the two deployed rules
(urgency, completed_act), and route_attempt are three separate lifts, and
`champion_oof.csv` -- the base every candidate rule has historically been
judged against -- contains NEITHER of the rules layers. So every rejection on
that shelf is suspect in both directions: a rule could look good by
duplicating what urgency already does, or look worthless because the base it
was added to was missing the very layer it belongs in. `route_behavior`
flipping sign between two rules-free bases is the proof that this matters --
a candidate has to be judged on top of the FULLY deployed base, not a base
missing pieces of it.

`apply_rules_to_test.py` exposes ALL_DEFS = the two deployed rules plus a shelf
of CANDIDATE_DEFS. This script adds each candidate, one at a time, ON TOP of
the fully deployed base (rules + route_attempt), and measures with author
resamples.

IT ALSO TESTS REMOVAL. A candidate that only duplicates `urgency` shows up as
"changes 0 posts". A deployed rule that is now redundant shows up in the
leave-one-out block. Both are worth knowing before adding anything.

WHAT COUNTS AS A PASS -- unchanged from retest_routes_on_twin.py:
  * fix/break ratio FIRST, not just win rate -- a candidate that fixes and
    breaks in roughly equal measure is not earning its place even at a
    positive mean delta.
  * win rate over author resamples, never the CI of the mean.
  * `ind` counts rows crossing the indicator boundary. Evidence is suppressed
    exactly on indicator posts, so those rows move S2 too and the risk delta
    shown is NOT the composite delta.
  * 372 posts / 32 authors is coarse. Under ~90% win with f/b under 2:1 is
    UNTESTED, not a result.

Usage:
  python src/champion_oof_twin.py          # writes champion_val_probs.csv
  python src/retest_rules_on_twin.py
  python src/retest_rules_on_twin.py --pairs      # also try rule pairs
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rules_to_test import ALL_DEFS, RULE_DEFS, apply_rules  # noqa: E402
from scoring import normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
DEPLOYED = ["urgency", "completed_act"]

# ---------------------------------------------------------------------------
# THE PROVENANCE PROBLEM, AND WHY THIS TABLE EXISTS.
#
# Every candidate below was WRITTEN BY READING val_split POSTS, and
# candidate_patterns.py names them in its own comments:
#
#   "Motivating val cases (champion says ideation, gold is behavior):
#      P00876 P00878 P00879 P00880 P01949 P01950 P01953 P01955"
#
# The twin is built on val_split. So scoring these rules there is CIRCULAR:
# a rule that fires 7 times and fixes 7 has, at worst, merely memorised the 7
# posts it was written against. `farewell_note` fixing 7 with 0 breaks is
# exactly what a perfectly fitted rule looks like -- and exactly what a
# genuinely good rule looks like too. The aggregate cannot tell them apart.
#
# What CAN tell them apart is that the motivating posts are NAMED. Split every
# firing into:
#   NAMED -- the post is in this rule's own motivating list. Zero evidential
#            value: the rule was built to fix it.
#   NOVEL -- the post is not. This is the only honest evidence here, and the
#            only number worth reading.
#
# A rule whose firings are all NAMED generalises to nothing observed. A rule
# with novel fixes and no novel breaks has earned a real test.
# ---------------------------------------------------------------------------
MOTIVATING = {
    "farewell_note": {"P00876", "P00878", "P00879", "P00880",
                      "P01949", "P01950", "P01953", "P01955"},
    "farewell_note_broad": {"P00876", "P00878", "P00879", "P00880",
                            "P01949", "P01950", "P01953", "P01955"},
    "third_person_death": {"P00118", "P00119", "P00559", "P00568"},
    "protective_negation": {"P00753", "P00990"},
    "urgency_guarded": {"P00679", "P01073", "P01156"},
}


def wf1(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    tot = 0.0
    for c in LEVELS:
        g, p = (gold == c), (pred == c)
        tp = float((g & p).sum())
        pr = tp / max(p.sum(), 1e-9)
        rc = tp / max(g.sum(), 1e-9)
        tot += 2 * pr * rc / max(pr + rc, 1e-9) * g.sum()
    return tot / max(len(gold), 1)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--qwen72", default="artifacts/cot_soft_oof.csv")
    a_.add_argument("--splits", type=int, default=400)
    a_.add_argument("--seed", type=int, default=17)
    a_.add_argument("--pairs", action="store_true")
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold = np.array([normalize_label(v) for v in va["risk_level"]])
    posts = list(va["post"].astype(str))
    auth = np.array([str(x) for x in va["anon_user_id"]])

    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    M = pv.set_index("row_id").loc[rows,
                                   [f"prob_{c}" for c in LEVELS]].to_numpy(float)
    raw = [LEVELS[i] for i in M.argmax(1)]

    route = {}
    if os.path.exists(args.qwen72):
        d = pd.read_csv(args.qwen72, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        c = "cot" if "cot" in d.columns else "pred_risk_level"
        route = {r: normalize_label(v) for r, v in zip(d["row_id"], d[c])}

    def chain(keys):
        lab = [normalize_label(apply_rules(p, l, keys)[0])
               for p, l in zip(posts, raw)]
        return np.array([("attempt" if (route.get(r) == "attempt"
                                        and x != "indicator") else x)
                         for r, x in zip(rows, lab)])

    base = chain(DEPLOYED)
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([by[a] for a in rng.choice(ua, len(ua), True)])
             for _ in range(args.splits)]
    print(f"val {len(rows)} posts / {len(ua)} authors")
    print(f"  no rules        {wf1(gold, chain([])):.4f}")
    print(f"  DEPLOYED base   {wf1(gold, base):.4f}\n")

    def judge(nm, new, ref=None, key=None):
        ref = base if ref is None else ref
        ch = new != ref
        if not ch.sum():
            print(f"  {nm:<34}{0:>7}   (no effect on top of the base)")
            return None
        fix = int(((new == gold) & (ref != gold) & ch).sum())
        brk = int(((ref == gold) & (new != gold) & ch).sum())
        d = np.array([wf1(gold[b], new[b]) - wf1(gold[b], ref[b])
                      for b in boots])
        ind = int(sum(1 for i in np.flatnonzero(ch)
                      if ref[i] == "indicator" or new[i] == "indicator"))
        win = float(np.mean(d > 0))
        # with 0 breaks the delta cannot go negative, so a sub-100% win rate is
        # resamples that contain NONE of the moved posts -- ties, not losses.
        loss = float(np.mean(d < 0))
        seen = MOTIVATING.get(key or nm.replace("+ ", "").strip(), set())
        mv = [rows[i] for i in np.flatnonzero(ch)]
        nov = [r for r in mv if r not in seen]
        nfix = int(sum(1 for i in np.flatnonzero(ch)
                       if rows[i] not in seen and new[i] == gold[i]
                       and ref[i] != gold[i]))
        nbrk = int(sum(1 for i in np.flatnonzero(ch)
                       if rows[i] not in seen and ref[i] == gold[i]
                       and new[i] != gold[i]))
        print(f"  {nm:<34}{int(ch.sum()):>7}{fix:>6}{brk:>6}"
              f"{wf1(gold, new):>9.4f}{0.4*d.mean():>+9.4f}{win:>5.0%}"
              f"{loss:>6.0%}{ind:>5}"
              f"{(f'{len(nov)}/{nfix}/{nbrk}' if seen else '-'):>12}")
        return (0.4 * d.mean(), win, fix, brk, nm, nfix, nbrk, len(seen) > 0)

    hdr = (f"  {'variant':<34}{'moved':>7}{'fix':>6}{'brk':>6}"
           f"{'wF1':>9}{'d comp':>9}{'win':>6}{'LOSS':>6}{'ind':>5}"
           f"{'novel m/f/b':>12}")
    print("  -- 1. LEAVE ONE DEPLOYED RULE OUT (is each still earning its "
          "place?) --")
    print(hdr)
    for k in DEPLOYED:
        judge(f"drop {k}", chain([x for x in DEPLOYED if x != k]))

    print("\n  -- 2. EACH CANDIDATE, ADDED TO THE DEPLOYED BASE --")
    print(hdr)
    cands = [k for k in ALL_DEFS if k not in RULE_DEFS]
    res = []
    for k in sorted(cands):
        r = judge(f"+ {k}", chain(DEPLOYED + [k]), key=k)
        if r:
            res.append(r)

    print("\n  -- 3. CANDIDATE REPLACING ITS DEPLOYED SIBLING --")
    print(hdr)
    for k in sorted(cands):
        if "urgency" in k:
            judge(f"{k} instead of urgency",
                  chain(["completed_act", k]))

    if args.pairs and res:
        top = [r[4][2:] for r in sorted(res, reverse=True)[:4]]
        if len(top) > 1:
            print("\n  -- 4. PAIRS OF THE BEST CANDIDATES --")
            print(hdr)
            for a, b in combinations(top, 2):
                judge(f"+ {a} + {b}", chain(DEPLOYED + [a, b]))

    print("\n  READ THE LAST COLUMN FIRST: novel moved/fixed/broken, counting")
    print("  ONLY posts absent from that rule's own motivating list. Every")
    print("  candidate was written by reading val posts, and the twin IS val,")
    print("  so `fix` and `d comp` are contaminated by construction. A rule")
    print("  whose novel column reads 0/0/0 fixed nothing it had not already")
    print("  been shown; its perfect record is memory, not skill.")
    print("  LOSS is the share of resamples that actually LOSE. With 0 breaks")
    print("  it is 0% and a sub-100% win rate just means some resamples missed")
    print("  the moved posts -- ties, not losses. Judge win against LOSS.")
    print("  ind > 0 also moves S2, so that row's risk delta is not composite.")
    honest = [r for r in res if r[7] and r[5] > 0]
    if honest:
        b = max(honest)
        print(f"\n  best with NOVEL fixes: {b[4]}  {b[5]} novel fix / "
              f"{b[6]} novel break")
        print("  -> worth a real held-out test. Novel fixes on a val set the")
        print("     rule was mined from are suggestive, not conclusive: the")
        print("     author overlap means a 'novel' post can still share an")
        print("     author with a named one.")
    else:
        print("\n  NO candidate produced a single novel fix. Every improvement")
        print("  above is the rule recognising the posts it was written from.")
        print("  The rules shelf is closed -- and this is why it looked open.")


if __name__ == "__main__":
    main()
