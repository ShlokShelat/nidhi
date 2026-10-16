"""
confusion_audit.py

Validates the hand-written CONFUSABLE map against the confusions the model
actually makes.

WHAT IT IS. train_factors_nli.CONFUSABLE decides half of every post's negative
examples -- hard_frac=0.5, so roughly 7,300 of the ~14,600 training pairs per
epoch are chosen by it. It was written by hand from the published definitions,
by judging which ones look alike. Whether those are the pairs the model
confuses is an empirical question, and the data to answer it is already on
disk: artifacts/fac_oof_f{0..4}.npz hold the incumbent's probabilities for all
1635 labeled posts, each produced by a model that never saw that post.

THE MEASUREMENT. For an ordered pair (g, j), g present in gold and j absent,
"j is a hard negative for g" should mean: j lights up BECAUSE g is present.
Not simply that j lights up -- a high-prevalence label fires on nearly
everything, so a raw probability ranking would just rediscover the prevalence
order. The measurement is therefore a lift, not an absolute score:

    lift(g, j) =  mean prob(j) over posts where g IS present and j is absent
                  ----------------------------------------------------------
                  mean prob(j) over posts where g is ABSENT and j is absent

lift > 1 means the presence of g genuinely drags j upward. Normalising by the
absent-g base rate is what keeps globally-rare labels from being ranked purely
on rarity and globally-common ones purely on frequency.

LEAKAGE. Each post's OOF probability comes from a model trained without that
post, so nothing about the post itself leaks. The mined map is an aggregate
over all 1635, and the fold-k model will consume a map that was partly informed
by the gold labels of fold k's own validation posts. That channel is real but
very weak -- it moves which negatives get sampled, not what the labels are, and
it is averaged over 1635 posts. It is worth stating rather than pretending to
zero. If you want it airtight, pass --per-fold to mine fold k's map from the
other four folds only; it costs nothing but a little noise from the smaller
sample.

OUTPUT. artifacts/confusable_mined.json, consumable by setting
CONFUSABLE_JSON=... in the environment before training.

READ THE OVERLAP LINE FIRST. If the mined map largely reproduces the hand map,
this direction is closed and the hard negatives were never the problem. If it
disagrees substantially, half the training pairs have been aimed at the wrong
boundaries and there is real headroom.

Usage:
  python src/confusion_audit.py
  python src/confusion_audit.py --per-fold --topk 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from train_factors_nli import CONFUSABLE, N

PROT = {"social support", "coping strategy", "psychological capital",
        "sense of responsibility", "meaning in life"}


def load_oof(pattern="artifacts/fac_oof_f{f}.npz", k=5):
    """Concatenate the five out-of-fold probability blocks, keeping fold ids."""
    rid, P, fold = [], [], []
    for f in range(k):
        p = pattern.format(f=f)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = np.load(p, allow_pickle=True)
        rid.extend(list(d["row_id"]))
        P.append(d["probs"])
        fold.extend([f] * len(d["row_id"]))
    return rid, np.concatenate(P), np.array(fold)


def gold_matrix(rid, lab):
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(rid), N), dtype=int)
    for i, r in enumerate(rid):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    return Y


def lift_matrix(P, Y):
    """L[g, j] = how much the presence of g inflates the model's score for an
    ABSENT j. Diagonal and undefined cells are left at 0."""
    L = np.zeros((N, N), dtype=float)
    for g in range(N):
        present = Y[:, g] == 1
        absent = ~present
        if present.sum() < 3:
            continue
        for j in range(N):
            if j == g:
                continue
            # restrict BOTH sides to posts where j is genuinely absent, so the
            # comparison is about false positives only
            a = present & (Y[:, j] == 0)
            b = absent & (Y[:, j] == 0)
            if a.sum() < 3 or b.sum() < 3:
                continue
            base = P[b, j].mean()
            if base <= 1e-6:
                continue
            L[g, j] = P[a, j].mean() / base
    return L


def mine(L, topk, min_lift):
    out = {}
    for g in range(N):
        order = np.argsort(-L[g])
        picks = [FACTOR_TAXONOMY[j] for j in order[:topk]
                 if L[g, j] >= min_lift]
        if picks:
            out[FACTOR_TAXONOMY[g]] = picks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--oof", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--topk", type=int, default=3,
                    help="hard-negative neighbours to keep per factor "
                         "(the hand map averages 2.1)")
    ap.add_argument("--min-lift", type=float, default=1.15,
                    help="ignore neighbours the presence of g barely moves")
    ap.add_argument("--per-fold", action="store_true",
                    help="also write a map per fold, mined from the OTHER four "
                         "folds only -- removes the weak aggregate leak")
    ap.add_argument("--out", default="artifacts/confusable_mined.json")
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")
    rid, P, fold = load_oof(args.oof, args.k)
    Y = gold_matrix(rid, lab)
    print(f"{len(rid)} OOF posts, {int(Y.sum())} gold factor assignments\n")

    L = lift_matrix(P, Y)
    mined = mine(L, args.topk, args.min_lift)

    print("PER FACTOR: hand-written map vs measured lift")
    print("=" * 96)
    hit = tot = 0
    hand_only, mined_only = [], []
    for g, t in enumerate(FACTOR_TAXONOMY):
        hand = [x for x in CONFUSABLE.get(t, []) if x in FACTOR_TAXONOMY]
        got = mined.get(t, [])
        tot += len(hand)
        hit += sum(1 for x in hand if x in got)
        hand_only += [(t, x, L[g, FACTOR_TAXONOMY.index(x)])
                      for x in hand if x not in got]
        mined_only += [(t, x, L[g, FACTOR_TAXONOMY.index(x)])
                       for x in got if x not in hand]
        print(f"\n  {t}{'   [protective]' if t in PROT else ''}   n={int(Y[:, g].sum())}")
        print(f"    hand   {', '.join(hand) if hand else '(none)'}")
        print("    lift   " + ", ".join(
            f"{x} {L[g, FACTOR_TAXONOMY.index(x)]:.2f}" for x in got)
            if got else "    lift   (nothing above threshold)")

    print("\n" + "=" * 96)
    print("AGREEMENT")
    print("=" * 96)
    pct = 100.0 * hit / tot if tot else 0.0
    print(f"  hand-written pairs also found by lift: {hit}/{tot}  ({pct:.0f}%)")
    print(f"  hand-written but NOT confirmed:        {len(hand_only)}")
    print(f"  found by lift but NOT hand-written:    {len(mined_only)}")

    print("\n  worst hand-written guesses (lowest measured lift):")
    for t, x, v in sorted(hand_only, key=lambda z: z[2])[:10]:
        print(f"    {t:<44} -> {x:<40} lift {v:.2f}")

    print("\n  strongest confusions the hand map MISSED:")
    for t, x, v in sorted(mined_only, key=lambda z: -z[2])[:10]:
        print(f"    {t:<44} -> {x:<40} lift {v:.2f}")

    json.dump(mined, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")

    if args.per_fold:
        base = os.path.splitext(args.out)[0]
        for f in range(args.k):
            keep = fold != f
            m = mine(lift_matrix(P[keep], Y[keep]), args.topk, args.min_lift)
            json.dump(m, open(f"{base}_f{f}.json", "w"), indent=2)
        print(f"wrote {base}_f{{0..{args.k-1}}}.json  (each excludes its own fold)")

    print("\nHOW TO READ THIS")
    print("  agreement high (>70%)  -> the hand map was already right; hard")
    print("                            negatives are not the constraint")
    print("  agreement low          -> half of every epoch's negatives have been")
    print("                            aimed at boundaries the model does not")
    print("                            actually struggle with, and the mined map")
    print("                            is worth a CV arm")


if __name__ == "__main__":
    main()
