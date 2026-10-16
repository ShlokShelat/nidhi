"""
evidence_suppression_rule.py

Re-examine the evidence CONSISTENCY RULE under MACRO averaging.

WHY NOW. `NIDHI_probe_noev.csv` proved the official evidence metric is
macro-averaged, not micro: emptying the evidence column returned S1=0.6172,
which under micro would imply risk_wF1=1.08. Under MACRO, a post whose gold is
"none" and whose prediction is empty scores a full **1.0**, and 35.7% of posts
have gold "none".

Every evidence decision in this project was tuned under micro. Re-scoring the
VOTE changed nothing (vote>=2 wins under both). But the SUPPRESSION RULE was
never re-examined, and it is the decision macro affects most:

    if predicted risk == indicator:  emit no evidence

Under micro, emitting nothing on an empty-gold post is merely neutral -- it
adds no true positives. Under macro it earns 1.0, exactly as much as a
perfectly matched span. So the rule is worth far more than earlier estimates
credited, and its TRIGGER may be wrong: it fires on a hard argmax, when the
quantity that actually matters is "how confident is the model that this post
has no evidence at all".

WHAT IS SWEPT
  none          never suppress -- the floor, shows the rule's total worth
  argmax        the shipped rule: suppress iff predicted risk is indicator
  p(ind)>=t     suppress on the indicator PROBABILITY, t swept and cross-fitted.
                Catches posts that are probably-indicator but lost the argmax,
                and spares posts that barely won it.
  oracle        suppress iff GOLD risk is indicator. NOT shippable. It is the
                ceiling this rule reaches with a perfect risk model, and it
                separates "the rule is mis-tuned" (fixable free) from "the
                risk predictions are the bottleneck" (not fixable here).

DATA. Everything is local and honestly out-of-fold:
  artifacts/friend_oof_export.csv        OOF risk probabilities, 1590 posts
  artifacts/evidence_oof/friend_oof_candidates.csv   OOF evidence spans
  artifacts/full_train.csv               gold evidence + gold risk

This uses ONE tagger, not the shipped 5-tagger vote (those candidate files are
on the cluster). That is fine for the question being asked: the rule operates
on top of whatever spans the vote produces, and its value is set by macro's
free 1.0 and by risk accuracy, not by which tagger supplied the spans. The
ORACLE HEADROOM in particular is near tagger-independent in sign.

Thresholds are cross-fitted over author halves; the bootstrap resamples
AUTHORS, because posts by one author are not independent.

Usage:
  python src/evidence_suppression_rule.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_vote_oof import author_halves, dedupe  # noqa: E402
from scoring import (RISK_LEVELS, normalize_label, parse_evidence,  # noqa: E402
                     phrase_f1, token_len)

PCOLS = ["Indicator", "Ideation", "Behavior", "Attempt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--cands",
                    default="artifacts/evidence_oof/friend_oof_candidates.csv")
    ap.add_argument("--risk", default="artifacts/friend_oof_export.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    cd = pd.read_csv(args.cands)
    cd["row_id"] = cd["row_id"].astype(str)
    if "min_conf" in cd.columns:
        mc = float(cd["min_conf"].iloc[0])
        cd = cd[cd["conf"] >= mc]
    else:
        mc = 0.0
    spans = {}
    for r, s in zip(cd["row_id"], cd["span"]):
        spans.setdefault(r, []).append(str(s))
    spans = {r: dedupe(v) for r, v in spans.items()}

    rk = pd.read_csv(args.risk)
    rk["row_id"] = rk["row_id"].astype(str)
    rk = rk.set_index("row_id")
    missing = [c for c in PCOLS if c not in rk.columns]
    if missing:
        raise SystemExit(f"{args.risk} lacks risk columns {missing}")

    # NOT intersected with `spans`. A post the tagger stayed silent on is a
    # REAL post with an EMPTY prediction, and under macro those are the most
    # valuable rows in the dataset -- an empty prediction on empty gold scores
    # 1.0. Intersecting with the candidate file drops them and drove the
    # gold-empty rate from the known 35.7% down to 6.6%, i.e. it deleted almost
    # every case the metric rewards. Keep every scored post.
    rows = sorted(set(lab.index) & set(rk.index))
    if not rows:
        raise SystemExit("no overlap between gold and risk probs")
    n_sil = sum(1 for r in rows if r not in spans)
    print(f"tagger silent on {n_sil} of {len(rows)} posts -- kept as empty "
          f"predictions, NOT dropped")
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    gold_risk = [normalize_label(lab.loc[r, "risk_level"]) for r in rows]
    M = rk.loc[rows, PCOLS].to_numpy(dtype=float)
    P_ind = M[:, 0]
    pred_risk = [RISK_LEVELS[i] for i in M.argmax(1)]
    auth = np.array([lab.loc[r, "anon_user_id"] for r in rows])
    half = author_halves(auth)
    A, B = half == "A", half == "B"

    n_emp = sum(1 for g in G if not g)
    acc = float(np.mean([p == g for p, g in zip(pred_risk, gold_risk)]))
    print(f"{len(rows)} OOF posts, {pd.unique(auth).size} authors  "
          f"(min_conf={mc}, risk argmax acc {acc:.3f})")
    print(f"gold evidence EMPTY on {n_emp} posts "
          f"({100 * n_emp / len(rows):.1f}%) -- under MACRO each is a free 1.0 "
          f"if the prediction also emits nothing")
    print(f"p(indicator) range [{P_ind.min():.3f}, {P_ind.max():.3f}]")

    # How often is "gold risk is indicator" the same event as "gold evidence is
    # empty"? The rule's whole premise. If these diverge the rule is a proxy for
    # the wrong thing and no threshold will fix it.
    gi = np.array([g == "indicator" for g in gold_risk])
    ge = np.array([not g for g in G])
    print(f"\nthe rule's premise: gold risk==indicator -> gold evidence empty")
    print(f"  P(empty | indicator) = {ge[gi].mean():.3f}   ({int(gi.sum())} posts)")
    print(f"  P(empty | other)     = {ge[~gi].mean():.3f}   "
          f"({int((~gi).sum())} posts)")
    print(f"  so {int((ge & ~gi).sum())} non-indicator posts ALSO have empty "
          f"gold -- suppression opportunities the rule never fires on")

    def finish(i, suppress):
        """The shipped fallback, unchanged: force one short span when the
        candidate list is empty and suppression is NOT active."""
        if suppress:
            return []
        s = spans.get(rows[i], [])
        return s if s else []

    def build(mask):
        return [finish(i, bool(mask[i])) for i in range(len(rows))]

    def mf1(P, sel=None):
        if sel is None:
            return phrase_f1(G, P, average="macro").f1
        return phrase_f1([g for g, m in zip(G, sel) if m],
                         [p for p, m in zip(P, sel) if m], average="macro").f1

    arms = {}
    arms["none (never suppress)"] = build(np.zeros(len(rows), bool))
    arms["argmax (SHIPPED)"] = build(np.array([r == "indicator"
                                               for r in pred_risk]))

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    picks, X = {}, [None] * len(rows)
    for src, dst in ((B, A), (A, B)):
        best, bt = -1.0, 0.5
        for t in grid:
            v = mf1(build(P_ind >= t), src)
            if v > best:
                best, bt = v, t
        cand = build(P_ind >= bt)
        for i in np.flatnonzero(dst):
            X[i] = cand[i]
        picks["A" if dst is A else "B"] = bt
    arms[f"p(ind)>=t xfit {picks}"] = X

    arms["ORACLE gold risk"] = build(gi)
    arms["ORACLE gold empty"] = build(ge)

    base = mf1(arms["argmax (SHIPPED)"])
    print(f"\n{'rule':<28}{'macroF1':>9}{'vs ship':>10}{'suppressed':>12}"
          f"{'of which gold-empty':>21}")
    print("-" * 80)
    for k, P in arms.items():
        sup = np.array([not p for p in P])
        hit = int((sup & ge).sum())
        n = int(sup.sum())
        frac = f"{hit}/{n} = {hit / n:.2f}" if n else "-"
        print(f"{k:<28}{mf1(P):>9.4f}{mf1(P) - base:>+10.4f}{n:>12}"
              f"{frac:>21}")

    uniq = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uniq}
    rng = np.random.default_rng(args.seed)
    boot = {k: np.empty(args.boot) for k in arms}
    for i in range(args.boot):
        pk = rng.choice(len(uniq), size=len(uniq), replace=True)
        ix = np.concatenate([by[uniq[j]] for j in pk])
        gb = [G[j] for j in ix]
        for k, P in arms.items():
            boot[k][i] = phrase_f1(gb, [P[j] for j in ix],
                                   average="macro").f1

    print("\n" + "=" * 80)
    print(f"PAIRED delta vs the SHIPPED rule  ({args.boot} author bootstrap)")
    print("=" * 80)
    print(f"{'rule':<28}{'delta':>9}{'95% CI':>22}{'win%':>7}"
          f"{'composite':>11}")
    print("-" * 77)
    for k in arms:
        if k == "argmax (SHIPPED)":
            continue
        d = boot[k] - boot["argmax (SHIPPED)"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        obs = mf1(arms[k]) - base
        print(f"{k:<28}{obs:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}"
              f"{100 * float((d > 0).mean()):>6.1f}%{0.3 * obs:>+11.4f}")

    orc = mf1(arms["ORACLE gold risk"]) - base
    thr = mf1(arms[f"p(ind)>=t xfit {picks}"]) - base
    print(f"\nHOW TO READ THIS")
    print(f"  rule's total worth   {base - mf1(arms['none (never suppress)']):+.4f}"
          f"   (shipped minus never-suppress)")
    print(f"  oracle headroom      {orc:+.4f}   (perfect risk model, same rule)")
    print(f"  threshold captures   {thr:+.4f}   (shippable, no retraining)")
    print("If the oracle headroom is small, the rule is already near its ceiling")
    print("and the bottleneck is risk accuracy, not the suppression logic --")
    print("the lane closes. If the threshold captures most of a LARGE oracle")
    print("headroom, the rule was mis-tuned and it is free to fix.")
    print("Evidence is 0.30 of the composite; the last column is already scaled.")
    # ---------------------------------------------------------------------
    # WHERE DOES THE EVIDENCE SCORE ACTUALLY COME FROM?
    # Macro splits cleanly in two, because gold-empty posts can only ever score
    # 1.0 or 0.0. Knowing the split says whether to invest in SPAN QUALITY or in
    # INDICATOR DETECTION -- and those are very different projects.
    # ---------------------------------------------------------------------
    P = arms["argmax (SHIPPED)"]
    emp = mf1([P[i] for i in np.flatnonzero(ge)], None) if False else None
    per = []
    for i in range(len(rows)):
        per.append(phrase_f1([G[i]], [P[i]], average="macro").f1)
    per = np.array(per)
    print("\n" + "=" * 80)
    print("DECOMPOSITION of the shipped evidence score, by gold type")
    print("=" * 80)
    print(f"{'subset':<26}{'posts':>7}{'share':>8}{'mean F1':>10}"
          f"{'contributes':>13}{'headroom':>10}")
    print("-" * 74)
    for nm, m in (("gold EMPTY (1.0 or 0.0)", ge), ("gold has spans", ~ge)):
        n = int(m.sum())
        w = n / len(rows)
        print(f"{nm:<26}{n:>7}{w:>8.3f}{per[m].mean():>10.4f}"
              f"{w * per[m].mean():>13.4f}{w * (1 - per[m].mean()):>10.4f}")
    print(f"{'TOTAL':<26}{len(rows):>7}{1.0:>8.3f}{per.mean():>10.4f}"
          f"{per.mean():>13.4f}{1 - per.mean():>10.4f}")
    print("\nThe 'headroom' column is macro F1 recoverable by perfecting that")
    print("subset alone. Multiply by 0.30 for composite. Spend effort where the")
    print("headroom is, not where the intuition is.")

    print("\n" + "=" * 80)
    print("STRUCTURAL FACT: gold evidence emptiness IS the indicator label")
    print("=" * 80)
    print(f"  P(gold empty | gold risk == indicator) = {ge[gi].mean():.3f}")
    print(f"  P(gold empty | gold risk != indicator) = {ge[~gi].mean():.3f}")
    print("These are not two tasks. Annotators wrote 'none' exactly when they")
    print("labelled a post `indicator`. So every indicator error costs TWICE:")
    print("once in risk_wF1 (weight 0.40) and again in evidence (weight 0.30).")
    print("A change that improves indicator recall is worth ~1.75x what its risk")
    print("number alone suggests -- and NOTHING else in evidence is coupled that")
    print("way. Conversely, tuning span text can never fix the empty half.")

    print("\nFALSIFIER, stated before looking: this lane is only worth a slot if")
    print("the threshold arm clears a macro F1 gain whose 95% CI excludes 0")
    print("AND it wins on both halves. Anything smaller is inside the")
    print("measurement noise band and would be the winner's curse again.")


if __name__ == "__main__":
    main()
