"""
risk_symbolic_fences.py

Apply symbolic fences AFTER the model instead of inside the hypothesis text.

Editing hypothesis definitions is ZERO-SUM: the model scores K hypotheses and
softmaxes them, so fencing `behavior` necessarily drags `ideation` with it. A
post-hoc rule is surgical -- it touches only the posts that match it and leaves
every other prediction bit-identical.

Two properties of this error surface constrain the rule design:
  - A rule that fires on every post carries its own error rate onto the whole
    set, so blanket lexical rules on the indicator boundary are catastrophic.
  - Behavior<->ideation errors are CONFIDENTLY wrong, and a rule cannot outvote
    a confident model without also overturning its correct predictions.

Every rule here is therefore GATED ON UNCERTAINTY: it fires only when the target
class is already close behind the predicted one. The attempt boundary is where
that gate is justified -- on gold-attempt posts the model sits at 0.401 behavior
vs 0.426 attempt, a near-tie, unlike the confident errors elsewhere.

Testing 5 rules over 1635 posts will find something by chance, so the evaluation
is built to expose that:
  - every margin threshold is CROSS-FITTED over author halves (tuned on half A,
    applied to half B, and vice versa), so no rule is scored at a threshold
    chosen on the posts it is scoring
  - the delta is reported with a paired bootstrap over AUTHORS, not posts,
    because posts from one author are not independent
  - every rule is reported, not just the winner, so the multiple-comparison
    problem is visible rather than hidden

PRE-STATED BAR: a rule is accepted only if its cross-fitted delta clears a
meaningful margin AND it wins on BOTH author halves.

Usage:
  python src/risk_symbolic_fences.py
  python src/risk_symbolic_fences.py --oof artifacts/nligaur/oof_ladder.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402

# an act ALREADY CARRIED OUT -- measured at 0.480 on gold attempt spans vs
# 0.021 on gold behavior spans (risk_attempt_boundary.py), a 23x separation
DONE = re.compile(
    r"\b(tried|attempted|survived|woke up|hospital|emergency|ambulance|icu|"
    r"stomach pump|overdosed|od'?d|swallowed|resuscitat\w*|last time|"
    r"tried once|didn'?t work|failed attempt|second time|again)\b", re.I)

# plan / method / preparation -- the behavior criterion
PLAN = re.compile(
    r"\b(pills|rope|noose|gun|knife|blade|razor|bleach|jump|hang|overdose|"
    r"tonight|tomorrow|next week|suicide note|goodbye|method|ways to (?:go|die)|"
    r"how to die|cut(?:ting)? myself|hurt(?:ing)? myself|self.?harm)\b", re.I)

EXPLICIT = re.compile(
    r"\b(suicid\w*|kill (?:my ?self|me)|end (?:my life|it all)|want to die|"
    r"wanna die|wish i (?:was|were) dead|better off dead|self.?harm|"
    r"hurt(?:ing)? my ?self|kms|ctb)\b", re.I)

# NEGATION. `indicator_demote` moves indicator -> ideation whenever EXPLICIT
# matches, so a NEGATED mention of the explicit vocabulary is a guaranteed wrong
# move: the post disclaims the very act the rule keys on. Suppressing those
# firings removes errors the rule can be certain about.
NEG = re.compile(
    r"\b(not|never|no|nor|don'?t|doesn'?t|didn'?t|won'?t|wouldn'?t|shouldn'?t|"
    r"isn'?t|aren'?t|wasn'?t|cannot|can'?t|couldn'?t|without|rather than|"
    r"instead of|no longer)\b", re.I)

# ...but these constructions are AFFIRMATIVE despite containing a negation
# token -- "can't stop <act>" asserts the act rather than disclaiming it.
# Suppressing them would trade one error class for another.
NEG_EXEMPT = re.compile(
    r"\b(can'?t stop|cannot stop|couldn'?t stop|can'?t help|cannot help|"
    r"can'?t take|can'?t keep|can'?t go on|can'?t live|don'?t know how much "
    r"longer|no longer want to live|not sure i can)\b", re.I)


def explicit_unnegated(t: str) -> bool:
    """EXPLICIT fires and is NOT negated in the ~8 words before it.

    Scans every match rather than the first, because one negated mention does
    not cancel an affirmative one elsewhere in the post. The window is clipped
    at sentence boundaries so a negation in the previous sentence cannot reach
    across.
    """
    for m in EXPLICIT.finditer(t):
        pre = t[max(0, m.start() - 60):m.start()]
        pre = re.split(r"[.!?\n]", pre)[-1]          # same sentence only
        if NEG_EXEMPT.search(pre) or NEG_EXEMPT.search(
                t[max(0, m.start() - 60):m.end() + 20]):
            return True                              # affirmative after all
        if not NEG.search(pre):
            return True
    return False


MARGIN_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9]


def rule_mask(name, pred, post, P, idx, margin):
    """Which posts this rule fires on, and what it changes them to."""
    pb = P[:, idx["behavior"]]
    pa = P[:, idx["attempt"]]
    pi = P[:, idx["indicator"]]
    top = P.max(1)
    done = np.array([bool(DONE.search(t)) for t in post])
    plan = np.array([bool(PLAN.search(t)) for t in post])
    expl = np.array([bool(EXPLICIT.search(t)) for t in post])
    expl_ng = np.array([explicit_unnegated(t) for t in post])

    if name == "attempt_promote":
        # behavior -> attempt when a completed act is stated and attempt is
        # close behind. The near-tie boundary; the one rule with a mechanism.
        return (pred == "behavior") & done & (pa >= margin * pb), "attempt"
    if name == "attempt_demote":
        # attempt -> behavior when NO completed act is stated anywhere
        return (pred == "attempt") & ~done & (pb >= margin * pa), "behavior"
    if name == "behavior_promote":
        # ideation -> behavior on a plan cue (the confidently-wrong boundary)
        return (pred == "ideation") & plan & (pb >= margin * top), "behavior"
    if name == "indicator_promote":
        # anything -> indicator when the post never mentions suicide at all
        return (pred != "indicator") & ~expl & (pi >= margin * top), "indicator"
    if name == "indicator_demote":
        # indicator -> ideation when the post plainly does mention it
        return (pred == "indicator") & expl & (P[:, idx["ideation"]]
                                               >= margin * pi), "ideation"
    if name == "indicator_demote_ng":
        # SAME rule, but a NEGATED mention no longer counts. The only change
        # from `indicator_demote`, so the delta between the two isolates
        # negation exactly.
        return (pred == "indicator") & expl_ng & (P[:, idx["ideation"]]
                                                  >= margin * pi), "ideation"
    if name == "indicator_promote_ng":
        # the mirror: -> indicator when the post never AFFIRMATIVELY mentions
        # suicide. Negated mentions should not protect a post from being
        # called indicator either.
        return (pred != "indicator") & ~expl_ng & (pi >= margin * top), \
            "indicator"
    raise ValueError(name)


def apply_rule(name, pred, post, P, idx, margin):
    m, to = rule_mask(name, pred, post, P, idx, margin)
    out = pred.copy()
    out[m] = to
    return out, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    d = pd.read_csv(args.oof, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    d = d[d["row_id"].isin(lab.index)].reset_index(drop=True)
    y = np.array([normalize_label(x) for x in d["gold"]])
    pred = np.array([normalize_label(x) for x in d["pred_risk_level"]])
    post = [" ".join(str(lab.loc[r, "post"]).split()) for r in d["row_id"]]
    author = np.array([str(lab.loc[r, "anon_user_id"]) for r in d["row_id"]])
    pc = [f"prob_{c}" for c in RISK_LEVELS]
    if any(c not in d.columns for c in pc):
        raise SystemExit(f"{args.oof} lacks {pc}")
    P = d[pc].to_numpy(float)
    idx = {c: i for i, c in enumerate(RISK_LEVELS)}

    base = risk_weighted_f1(list(y), list(pred))
    print(f"{len(d)} OOF posts / {len(set(author))} authors   "
          f"base weighted-F1 {base:.4f}")

    # author halves, fixed once so every rule sees the same split
    auths = sorted(set(author))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(auths))
    hA = {auths[i] for i in perm[:len(auths) // 2]}
    inA = np.array([a in hA for a in author])
    print(f"author halves: {int(inA.sum())} / {int((~inA).sum())} posts")

    rules = ["attempt_promote", "attempt_demote", "behavior_promote",
             "indicator_promote", "indicator_demote",
             # negation-guarded variants -- compare each against its plain twin
             # directly above/below; the delta between them IS the effect of
             # negation, with everything else held fixed
             "indicator_demote_ng", "indicator_promote_ng"]

    print("\n" + "=" * 78)
    print("CROSS-FITTED RULE EVALUATION  (threshold tuned on the OTHER half)")
    print("=" * 78)
    print(f"{'rule':<20}{'fired':>7}{'xfit F1':>10}{'delta':>9}"
          f"{'95% CI':>20}{'A':>7}{'B':>7}")
    print("-" * 78)

    for name in rules:
        # tune margin on each half, apply to the other
        xpred = pred.copy()
        chosen = []
        for tune, appl in ((inA, ~inA), (~inA, inA)):
            best_m, best_f = MARGIN_GRID[0], -1.0
            for m in MARGIN_GRID:
                p2, _ = apply_rule(name, pred, post, P, idx, m)
                f = risk_weighted_f1(list(y[tune]), list(p2[tune]))
                if f > best_f:
                    best_f, best_m = f, m
            chosen.append(best_m)
            p2, _ = apply_rule(name, pred, post, P, idx, best_m)
            xpred[appl] = p2[appl]
        n_fired = int((xpred != pred).sum())
        f_x = risk_weighted_f1(list(y), list(xpred))
        dA = (risk_weighted_f1(list(y[inA]), list(xpred[inA]))
              - risk_weighted_f1(list(y[inA]), list(pred[inA])))
        dB = (risk_weighted_f1(list(y[~inA]), list(xpred[~inA]))
              - risk_weighted_f1(list(y[~inA]), list(pred[~inA])))

        # paired bootstrap over AUTHORS
        deltas = []
        aidx = {a: np.where(author == a)[0] for a in auths}
        for _ in range(args.boot):
            pick = rng.choice(len(auths), len(auths), replace=True)
            rows = np.concatenate([aidx[auths[i]] for i in pick])
            deltas.append(risk_weighted_f1(list(y[rows]), list(xpred[rows]))
                          - risk_weighted_f1(list(y[rows]), list(pred[rows])))
        lo, hi = np.percentile(deltas, [2.5, 97.5])

        ok = (f_x - base) >= 0.005 and lo > 0 and dA > 0 and dB > 0
        print(f"{name:<20}{n_fired:>7}{f_x:>10.4f}{f_x - base:>+9.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>20}{dA:>+7.4f}{dB:>+7.4f}"
              f"{'  <- PASSES' if ok else ''}")

    print("\nBAR: delta >= +0.005 AND CI excludes 0 AND both halves positive.")
    print("Five rules are tested, so something will look good by chance; the")
    print("cross-fit, the CI and the two-half agreement are what separate a")
    print("real rule from that. A rule that fires on very few posts cannot")
    print("clear +0.005 no matter how correct it is -- check `fired` first.")

    # diagnostic: what is the ceiling for the attempt rule at all?
    done = np.array([bool(DONE.search(t)) for t in post])
    m = (pred == "behavior") & done
    if m.sum():
        would_fix = int((y[m] == "attempt").sum())
        would_break = int((y[m] == "behavior").sum())
        print(f"\nATTEMPT RULE CEILING: {int(m.sum())} posts are predicted "
              f"behavior AND contain a completed-act marker.")
        print(f"  flipping ALL of them to attempt: fixes {would_fix}, "
              f"breaks {would_break}")
        print("  If breaks >= fixes the cue is not discriminative ON POSTS,")
        print("  whatever it measured on SPANS -- the span was the annotator's")
        print("  chosen clause; the post is everything around it too.")


if __name__ == "__main__":
    main()
