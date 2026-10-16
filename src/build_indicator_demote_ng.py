"""
build_indicator_demote_ng.py

`indicator_demote` moves a post from indicator to ideation when the text
explicitly mentions suicide -- which by the task's own definition it cannot be
an indicator post if it does ("Indicator: the post contains no explicit mentions
of suicide"). The rule is guarded against NEGATED mentions, which are the
failure case: a post can contain the exact surface string the rule keys on
while asserting its opposite -- an author writing that they feel the urge NOT
to kill themselves names the phrase without expressing the intent. A plain
substring test reads the phrase and demotes the post; the gold label is
indicator, so the demotion is wrong.

Guarding against negation balances the effect across the two author halves,
which matters more than the point estimate -- the unguarded rule was carried
by only one half.

WHY THIS IS A DIFFERENT KIND OF OVERRIDE. Model-vs-model overrides pit a
second opinion from a network that shares the champion's blind spots against
the champion, and that correlation limits their value. This is a DEFINITIONAL
check against the post text instead -- no second model, no correlated errors.

TWO THINGS THIS SCRIPT ESTABLISHES BEFORE IT WRITES ANYTHING:

  1. THE MARGIN=0 SCORE. The cross-fitted rule carries a probability term
     (P[ideation] >= margin * P[indicator]) that requires the champion's test
     probabilities, which are not available. At margin=0 that term vanishes
     and the rule is purely lexical, which is the only version deployable today.
     So it must be scored AT MARGIN 0, not at whatever the cross-fit chose.

  2. THE TEST FOOTPRINT. 18 of 1635 is 1.1%, so expect ~4 posts on 378. A
     positive expectation over so few posts carries enormous variance relative
     to the point estimate. The count is printed before the file is written so
     the decision is made with it in hand.

AND THE EVIDENCE KNOCK-ON IS NOT OPTIONAL. Gold evidence is empty exactly on
indicator posts and the champion suppresses spans there, so every post moved OFF
indicator gets its spans UNSUPPRESSED. That doubles the effect of any wrong
move in whichever direction it goes. The rebuild must use the `_rc` runs: the
original eight move 119 of 378 posts at identical risk and would silently
revert the evidence-conditioning fix.

Usage:
  python src/build_indicator_demote_ng.py                       # measure only
  python src/build_indicator_demote_ng.py --write               # + build it
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_symbolic_fences import EXPLICIT, explicit_unnegated  # noqa: E402
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402

PROBE = {"indicator": (112, 170), "ideation": (99, 142),
         "behavior": (69, 113), "attempt": (13, 41)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="artifacts/factors_tfit_test.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default="artifacts/ng_risk_base.csv")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--evdir-oof", default="artifacts/evidence_oof")
    ap.add_argument("--evdir-test", default="artifacts/evidence_runs")
    ap.add_argument("--require-evidence", action="store_true",
                    help="ALSO require that at least one span tagger finds "
                         "quotable text. Two INDEPENDENT signals for the same "
                         "decision: the lexical cue reads the words, the "
                         "taggers were trained on span labels. Posts where NO "
                         "tagger fires are 94.2%% gold indicator (measured), so "
                         "demoting one of those is very likely wrong -- and 2 "
                         "of the 4 the lexical rule fires on for test are "
                         "exactly that case.")
    args = ap.parse_args()

    def any_candidate(d, tags, keys):
        """row_ids where at least one tagger produced a surviving candidate."""
        hit = set()
        for t in tags:
            for suf in ("_oof_candidates.csv", "_val_candidates.csv",
                        "_test_candidates.csv"):
                p = f"{d}/{t}{suf}"
                if not os.path.exists(p):
                    continue
                c = pd.read_csv(p, keep_default_na=False)
                c["row_id"] = c["row_id"].astype(str)
                if "conf" in c.columns and "min_conf" in c.columns and len(c):
                    c = c[c["conf"] >= float(c["min_conf"].iloc[0])]
                hit |= set(c["row_id"])
                break
        return {k: (k in hit) for k in keys}

    # ---------------- 1. score the DEPLOYABLE (margin=0) version on OOF -----
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    d = pd.read_csv(args.oof, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    d = d.drop_duplicates("row_id").set_index("row_id")
    col = next((c for c in ("pred_risk_level", "risk_level")
                if c in d.columns), None)
    pc = [f"prob_{c}" for c in RISK_LEVELS]
    rows = [r for r in d.index if r in lab.index]
    if col is not None:
        pred = np.array([normalize_label(d.loc[r, col]) for r in rows])
    else:
        M = d.loc[rows, pc].to_numpy(float)
        pred = np.array([RISK_LEVELS[i] for i in M.argmax(1)])
    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    posts = lab.loc[rows, "post"].astype(str).to_numpy()

    plain = np.array([bool(EXPLICIT.search(t)) for t in posts])
    ng = np.array([explicit_unnegated(t) for t in posts])

    base = risk_weighted_f1(list(gold), list(pred))
    print(f"OOF {len(rows)} posts / {pd.unique(auth).size} authors   "
          f"base wF1 {base:.4f}\n")
    print("=" * 84)
    print("MARGIN=0 -- the purely lexical version, the only one deployable")
    print("=" * 84)

    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    rng = np.random.default_rng(args.seed)
    IXS = [np.concatenate([by[uauth[i]] for i in
                           rng.choice(len(uauth), len(uauth), replace=True)])
           for _ in range(args.boot)]
    half = set(uauth[: len(uauth) // 2])
    HA = np.concatenate([by[a] for a in uauth if a in half])
    HB = np.concatenate([by[a] for a in uauth if a not in half])

    OOF_TAGS = ["v2", "ep12", "lr3", "bs16", "ml256", "friend", "elec"]
    hasev_map = any_candidate(args.evdir_oof, OOF_TAGS, rows)
    hasev = np.array([hasev_map.get(r, False) for r in rows])
    print(f"  posts with >=1 tagger candidate: {int(hasev.sum())} of "
          f"{len(rows)}\n")

    print(f"{'variant':<18}{'fired':>7}{'fix':>6}{'brk':>6}{'wF1':>9}"
          f"{'delta':>9}{'95% CI':>20}{'win':>6}{'halfA':>9}{'halfB':>9}")
    best = None
    for name, cue in (("plain", plain), ("negation-guarded", ng),
                      ("ng + has-evidence", ng & hasev)):
        m = (pred == "indicator") & cue
        new = pred.copy()
        new[m] = "ideation"
        fix = int(((gold != "indicator") & m).sum())
        brk = int(((gold == "indicator") & m).sum())
        v = risk_weighted_f1(list(gold), list(new))
        dd = np.array([risk_weighted_f1(list(gold[ix]), list(new[ix]))
                       - risk_weighted_f1(list(gold[ix]), list(pred[ix]))
                       for ix in IXS])
        lo, hi = np.percentile(dd, [2.5, 97.5])
        dA = (risk_weighted_f1(list(gold[HA]), list(new[HA]))
              - risk_weighted_f1(list(gold[HA]), list(pred[HA])))
        dB = (risk_weighted_f1(list(gold[HB]), list(new[HB]))
              - risk_weighted_f1(list(gold[HB]), list(pred[HB])))
        print(f"{name:<18}{int(m.sum()):>7}{fix:>6}{brk:>6}{v:>9.4f}"
              f"{v-base:>+9.4f}{f'[{lo:+.4f},{hi:+.4f}]':>20}"
              f"{(dd > 0).mean():>6.0%}{dA:>+9.4f}{dB:>+9.4f}")
        want = "ng + has-evidence" if args.require_evidence else \
            "negation-guarded"
        if name == want:
            best = (v - base, lo, hi, (dd > 0).mean(), dA, dB, fix, brk)

    dlt, lo, hi, win, dA, dB, fix, brk = best
    print(f"\n  fix:brk = {fix}:{brk}"
          f"{'  (ratio ' + format(fix/max(brk,1), '.1f') + ':1)' if brk else ''}")
    print(f"  composite from risk alone = 0.4 x {dlt:+.4f} = {0.4*dlt:+.4f}")
    print("\n  A model-vs-model override's challenger is another risk model,")
    print("  so a favorable offline fix ratio there is not decisive by itself:")
    print("  the challenger can share the champion's blind spots. This rule is")
    print("  a definitional check on the text instead, so its errors are not")
    print("  the champion's errors.")

    # ---------------- 2. the test footprint --------------------------------
    ch = pd.read_csv(args.champion, keep_default_na=False)
    ch["row_id"] = ch["row_id"].astype(str)
    te = pd.read_csv(args.test, keep_default_na=False)
    te["row_id"] = te["row_id"].astype(str)
    tp = {r: str(p) for r, p in zip(te["row_id"], te["post"])}

    RC8 = ["evidence_v2_rc", "evidence_v1_rc", "evidence_s14_rc",
           "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
           "evidence_lr2_rc", "evidence_ep9_rc"]
    tev = any_candidate(args.evdir_test, RC8, list(ch["row_id"]))
    cur = np.array([normalize_label(v) for v in ch["risk_level"]])
    lex = np.array([(c == "indicator") and explicit_unnegated(tp.get(r, ""))
                    for r, c in zip(ch["row_id"], cur)])
    ev = np.array([tev.get(r, False) for r in ch["row_id"]])
    fire = (lex & ev) if args.require_evidence else lex
    print(f"  lexical rule alone fires on: {int(lex.sum())}")
    print(f"  of those, taggers ALSO find evidence: {int((lex & ev).sum())}")
    print(f"  the other {int((lex & ~ev).sum())} have ZERO candidates from all "
          f"8 taggers --")
    print(f"    posts with no tagger hit are 94.2% gold indicator, so "
          f"demoting those is")
    print(f"    very likely WRONG. --require-evidence drops them.")
    print("\n" + "=" * 84)
    print("TEST FOOTPRINT")
    print("=" * 84)
    print(f"  champion indicator posts: {int((cur == 'indicator').sum())}")
    print(f"  of those, affirmative explicit mention: {int(fire.sum())}")
    print(f"  OOF rate {100*(pred == 'indicator').mean() * 0 + 100*18/1635:.1f}%"
          f" would predict ~{round(0.011*len(ch))} on {len(ch)} posts")
    if fire.sum() <= 6:
        print(f"\n  *** {int(fire.sum())} posts. The expectation is positive but "
              f"the VARIANCE over so")
        print("      few posts is larger than the effect. This is a coin flip "
              "with a")
        print("      favourable mean, not a reliable gain. ***")

    new = cur.copy()
    new[fire] = "ideation"
    spell = {}
    for v in ch["risk_level"]:
        spell.setdefault(normalize_label(v), str(v))
    print(f"\n{'class':<12}{'champ':>7}{'new':>6}{'probe 90%':>14}")
    for c in RISK_LEVELS:
        lo_, hi_ = PROBE[c]
        n2 = int((new == c).sum())
        flag = "" if lo_ <= n2 <= hi_ else "  <- OUTSIDE"
        print(f"{c:<12}{int((cur == c).sum()):>7}{n2:>6}"
              f"{f'{lo_}-{hi_}':>14}{flag}")

    if args.write:
        out = ch.copy()
        out["risk_level"] = [spell.get(x, x) for x in new]
        out.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}  ({int(fire.sum())} risk labels changed)")
        print("\n" + "=" * 84)
        print("EVIDENCE MUST BE REBUILT -- these posts leave indicator, so their")
        print("spans are UNSUPPRESSED. Use the _rc runs.")
        print("=" * 84)
        print('  RC8=evidence_v2_rc,evidence_v1_rc,evidence_s14_rc,'
              'evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,'
              'evidence_lr2_rc,evidence_ep9_rc')
        print('  python src/evidence_vote_test.py --runs "$RC8" \\')
        print('      --dir artifacts/evidence_runs --k 2 --cluster --rep modal \\')
        print(f'      --test data/test.csv --risk-from {args.out} \\')
        print('      --out artifacts/evvote_test_ng.csv')
        print(f'  python src/patch_submission_risk.py --base {args.out} \\')
        print('      --evidence artifacts/evvote_test_ng.csv \\')
        print('      --out submissions/NIDHI_ngdemote.csv')
        print("\n  Then factors MUST return 0.6908 exactly -- that is the check")
        print("  that only risk and its evidence knock-on moved.")
    else:
        print("\n(measure-only; pass --write to build it)")


if __name__ == "__main__":
    main()
