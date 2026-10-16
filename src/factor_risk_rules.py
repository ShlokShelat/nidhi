"""
factor_risk_rules.py

Apply the evidence-consistency method to FACTORS, and measure it honestly.

BACKGROUND. The evidence-consistency fix works because gold obeys a hard
structural rule: risk == indicator implies no evidence spans, anything else
implies at least one. That is a property of the ANNOTATION SCHEME, not of any
model, which is why it generalizes where repeated threshold-tuned changes do
not.

Factors have the same structure, and it is sharper. Mining P(factor | risk) over
all 1635 labelled posts, exactly two of the 24 labels are risk-conditional:

                                              indicator ideation behavior attempt
    suicide means (with access)                    0.3%     4.8%    57.8%   34.2%
    prior self-harm or suicidal thought/attempt    1.5%     7.7%    16.6%   88.6%

At indicator, `suicide means` appears in 2 of 611 posts. Predicting it there is
a false positive essentially every time. At attempt, `prior self-harm` appears
in 101 of 114 -- missing it is a false negative essentially every time.

Every other label is flat across risk levels (hopelessness 36-53%, emotion
dysregulation 32-40%). Rules built on those would be fitting noise, so the
mining below applies a support floor and a margin requirement rather than
taking whatever the table happens to show.

THREE THINGS THIS DOES NOT DO, EACH DELIBERATELY.

  It does not use gold risk. At submission time the rule fires on PREDICTED
  risk, so measuring with gold risk would overstate the gain by exactly the
  amount the risk model is wrong. Predicted risk comes from the same
  author-disjoint OOF folds.

  It does not mine on the fold it scores. For each fold, P(factor | risk) is
  estimated on the other four. Mining and scoring on the same rows is the
  val-tuning failure mode that has repeatedly failed to generalize.

  It does not tune `lo` and `hi`. They are fixed at 0.02 and 0.80 as structural
  choices -- "essentially never" and "essentially always". The sweep printed at
  the end is a diagnostic for reading stability, NOT a menu to pick from. If
  the gain only exists at one setting, it is noise.

Ship criterion, same as the evidence fix: positive on BOTH author-disjoint
halves, not just overall.

Usage:
  python src/factor_risk_rules.py \
      --factor-oof 'artifacts/fac_oof_f{f}.npz' \
      --risk-oof   'v2_rebuild/runs/risk_ctrl_f{f}/oof_predictions.csv'
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, normalize_label, parse_factors

RISKS = ["indicator", "ideation", "behavior", "attempt"]


def macro_f1(P, Y):
    out = []
    for j in range(Y.shape[1]):
        tp = int((P[:, j] & Y[:, j]).sum())
        fp = int((P[:, j] & (1 - Y[:, j])).sum())
        fn = int(((1 - P[:, j]) & Y[:, j]).sum())
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out)), out


def author_halves(auth):
    """Deterministic author-disjoint 2-way split, balanced on post count.
    Same policy as rule_ablation_v2.py so the two analyses are comparable."""
    s = pd.Series(auth)
    counts = s.value_counts()
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    half, size = {}, {"A": 0, "B": 0}
    for a in order:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


def mine(Y, risk, rows, lo, hi, min_support, min_prev=0.05):
    """P(label | risk) over `rows` only, judged by LIFT against the label's own
    base rate. Returns (suppress, force) sets of (risk, label_index).

    WHY LIFT AND NOT AN ABSOLUTE FLOOR. The first version of this function used
    `P(label|risk) <= lo` with lo=0.02, and it destroyed the metric.
    `poor school performance` has a 1.0% base rate, so its conditional
    probability is under 2% at EVERY risk level -- the rule fired on all four
    and deleted the label, taking its F1 from 0.706 to 0.000. Same for
    `exposure to others' suicide` (0.9%) and `sexual orientation related
    issues`. That criterion was detecting GLOBAL RARITY, not conditional
    absence, which is a different thing entirely.

    A label is conditionally absent when it falls far below ITS OWN base rate:

        suicide means      base 17.9%, at indicator 0.3%  -> lift 0.02  REAL
        prior self-harm    base 13.1%, at indicator 1.5%  -> lift 0.11  REAL
        poor school perf   base  1.0%, at indicator 1.0%  -> lift 1.00  NOTHING

    `min_prev` additionally refuses to mine labels too rare to estimate a
    conditional on at all. With 1635 posts and a 1% base rate there are ~16
    positives spread over four risk levels; no threshold can separate signal
    from noise there.
    """
    sup, forc = set(), set()
    base = Y[rows].mean(axis=0)
    for r in RISKS:
        m = rows & (risk == r)
        n = int(m.sum())
        if n < min_support:
            continue
        for j in range(Y.shape[1]):
            if base[j] < min_prev:
                continue                      # too rare to condition on
            p = float(Y[m, j].mean())
            lift = p / base[j] if base[j] > 0 else 1.0
            if lift <= lo:
                sup.add((r, j))
            elif p >= hi:
                forc.add((r, j))
    return sup, forc


def apply_rules(P, risk, sup, forc):
    out = P.copy()
    for (r, j) in sup:
        out[(risk == r), j] = 0
    for (r, j) in forc:
        out[(risk == r), j] = 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--factor-oof", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--risk-oof",
                    default="v2_rebuild/runs/risk_ctrl_f{f}/oof_predictions.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lo", type=float, default=0.15,
                    help="suppress when P(label|risk) falls to this FRACTION "
                         "of the label's own base rate (lift), not an absolute "
                         "probability -- see mine()")
    ap.add_argument("--hi", type=float, default=0.70,
                    help="force when P(label|risk) exceeds this. Lower than the "
                         "gold-conditional would suggest because the rule fires "
                         "on PREDICTED risk, and a 74.7%%-accurate risk model "
                         "dilutes every conditional toward the base rate.")
    ap.add_argument("--min-support", type=int, default=50)
    ap.add_argument("--min-prev", type=float, default=0.05,
                    help="skip labels below this base rate: too few positives "
                         "to estimate a conditional on")
    ap.add_argument("--gold-risk", action="store_true",
                    help="diagnostic only: use GOLD risk instead of predicted. "
                         "Shows the ceiling if the risk model were perfect. "
                         "Never the number to ship on.")
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled)
    gmap = dict(zip(lab["row_id"], lab["factors"]))
    amap = dict(zip(lab["row_id"], lab["anon_user_id"]))
    rmap = {r: normalize_label(x) for r, x in zip(lab["row_id"], lab["risk_level"])}
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

    rids, PR, TH, FD = [], [], [], []
    for f in range(args.k):
        p = args.factor_oof.format(f=f)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run src/factors_oof_cv.py first")
        d = np.load(p, allow_pickle=True)
        rids.extend(list(d["row_id"]))
        PR.append(d["probs"])
        TH.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
        FD.extend([f] * len(d["row_id"]))
    PR, TH, FD = np.concatenate(PR), np.concatenate(TH), np.array(FD)
    P0 = (PR >= TH).astype(int)

    Y = np.zeros_like(P0)
    for i, r in enumerate(rids):
        for fa in parse_factors(gmap.get(r, "")):
            j = idx.get(str(fa).strip().lower())
            if j is not None:
                Y[i, j] = 1

    if args.gold_risk:
        risk = np.array([rmap[r] for r in rids])
        print("*** using GOLD risk -- diagnostic ceiling, not shippable ***")
    else:
        pr = {}
        # --risk-oof may be a per-fold pattern containing {f}, or a single
        # already-pooled file such as artifacts/champion_oof.csv. In the second
        # case format() is a no-op and the same file is read k times, which is
        # harmless, so both forms work without a separate flag.
        seen = set()
        for f in range(args.k):
            p = args.risk_oof.format(f=f)
            if p in seen:
                continue
            seen.add(p)
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
            d = pd.read_csv(p)
            # champion_oof.py writes `pred_risk_level`; the per-fold risk dumps
            # write `pred`. Accept either rather than making the caller rename.
            col = next((c for c in ("pred", "pred_risk_level", "risk_level")
                        if c in d.columns), None)
            if col is None:
                raise SystemExit(
                    f"{p} has no risk-prediction column "
                    f"(looked for pred / pred_risk_level / risk_level; "
                    f"found {list(d.columns)})")
            pr.update({r: normalize_label(x)
                       for r, x in zip(d["row_id"], d[col])})
        missing = [r for r in rids if r not in pr]
        if missing:
            # champion_oof.py covers the 1590-post INTERSECTION of the five
            # risk members (the friend model is short 45 posts), while the
            # factor OOF covers all 1635. Restrict to the intersection rather
            # than aborting -- but say so, because every number below is then
            # computed on a smaller set than the factor arm tables.
            keep = np.array([r in pr for r in rids])
            print(f"NOTE: no predicted risk for {len(missing)} of {len(rids)} "
                  f"posts (champion_oof.csv is the 1590-post intersection). "
                  f"Restricting to {int(keep.sum())} posts.")
            rids = [r for r, k in zip(rids, keep) if k]
            PR, TH, P0, Y, FD = PR[keep], TH[keep], P0[keep], Y[keep], FD[keep]
        risk = np.array([pr[r] for r in rids])
        agree = float((risk == np.array([rmap[r] for r in rids])).mean())
        print(f"predicted risk agrees with gold on {100 * agree:.1f}% of posts "
              f"-- the rules inherit these errors, which is the point of not "
              f"measuring with gold risk")

    auth = np.array([amap[r] for r in rids])
    half = author_halves(auth)

    # cross-fitted: fold f is scored with rules mined on the other four
    P1 = P0.copy()
    all_sup, all_forc = {}, {}
    for f in range(args.k):
        tr = FD != f
        sup, forc = mine(Y, risk, tr, args.lo, args.hi, args.min_support,
                         args.min_prev)
        te = FD == f
        P1[te] = apply_rules(P0[te], risk[te], sup, forc)[:, :]
        all_sup[f], all_forc[f] = sup, forc

    b0, per0 = macro_f1(P0, Y)
    b1, per1 = macro_f1(P1, Y)
    print(f"\nmacro-F1 before {b0:.4f}   after {b1:.4f}   "
          f"delta {b1 - b0:+.4f}")
    print(f"  -> {0.3 * (b1 - b0):+.4f} composite if it transfers "
          f"(factors carry 0.3)")
    print(f"cells changed: {int((P0 != P1).sum())} of {P0.size}")

    print("\nBY AUTHOR-DISJOINT HALF (the ship criterion -- both must be > 0)")
    ok = True
    for h in ("A", "B"):
        m = half == h
        a, _ = macro_f1(P0[m], Y[m])
        c, _ = macro_f1(P1[m], Y[m])
        print(f"  half {h}: {a:.4f} -> {c:.4f}  ({c - a:+.4f})  n={int(m.sum())}")
        ok &= (c - a) > 0

    print("\nPER-LABEL (only labels that moved)")
    for j, t in enumerate(FACTOR_TAXONOMY):
        if abs(per1[j] - per0[j]) > 1e-9:
            print(f"  {t:<46} {per0[j]:.3f} -> {per1[j]:.3f} "
                  f"({per1[j] - per0[j]:+.3f})")

    print("\nRULES MINED (fold 0's set; check they are stable across folds)")
    for r, j in sorted(all_sup[0]):
        print(f"  SUPPRESS  {FACTOR_TAXONOMY[j]:<46} when risk = {r}")
    for r, j in sorted(all_forc[0]):
        print(f"  FORCE     {FACTOR_TAXONOMY[j]:<46} when risk = {r}")
    same = all(all_sup[f] == all_sup[0] and all_forc[f] == all_forc[0]
               for f in range(args.k))
    print(f"  rule set identical across all {args.k} folds: {same}"
          + ("" if same else "   <- unstable mining, treat with suspicion"))

    print("\nLIFT TABLE for labels above the base-rate floor "
          f"({100 * args.min_prev:.0f}%)")
    base = Y.mean(axis=0)
    print(f"{'factor':<46}{'base':>7}" + "".join(f"{r[:5]:>8}" for r in RISKS))
    for j, t in enumerate(FACTOR_TAXONOMY):
        if base[j] < args.min_prev:
            continue
        lifts = [Y[risk == r, j].mean() / base[j] for r in RISKS]
        print(f"{t:<46}{100 * base[j]:>6.1f}%"
              + "".join(f"{x:>8.2f}" for x in lifts))
    print("  (1.00 = exactly the base rate. Suppression needs a value far "
          "below 1, not a small absolute probability.)")

    print("\nSTABILITY SWEEP (diagnostic only -- do NOT pick the best cell)")
    print(f"{'lo':>6}{'hi':>6}{'delta':>10}{'halfA':>10}{'halfB':>10}")
    for lo in (0.10, 0.15, 0.25):
        for hi in (0.60, 0.70, 0.80):
            Q = P0.copy()
            for f in range(args.k):
                s2, f2 = mine(Y, risk, FD != f, lo, hi, args.min_support,
                              args.min_prev)
                Q[FD == f] = apply_rules(P0[FD == f], risk[FD == f], s2, f2)
            d = macro_f1(Q, Y)[0] - b0
            da = macro_f1(Q[half == "A"], Y[half == "A"])[0] - \
                macro_f1(P0[half == "A"], Y[half == "A"])[0]
            db = macro_f1(Q[half == "B"], Y[half == "B"])[0] - \
                macro_f1(P0[half == "B"], Y[half == "B"])[0]
            print(f"{lo:>6.2f}{hi:>6.2f}{d:>+10.4f}{da:>+10.4f}{db:>+10.4f}")

    print("\n" + "=" * 70)
    if ok and (b1 - b0) > 0:
        print("SHIP CRITERION MET: positive overall AND on both author halves.")
        print("Build it into a submission with --apply, patch factors ONLY on")
        print("top of NIDHI_evidence_consistent.csv (the 0.7422 champion).")
    else:
        print("Ship criterion NOT met -- do not build a submission from this.")
    print("=" * 70)


if __name__ == "__main__":
    main()
