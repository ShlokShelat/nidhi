"""
champion_oof.py

Rebuild a champion-like risk ensemble over ~1590 OUT-OF-FOLD posts, so that
every subsequent decision is made on roughly 4.5x the data of the 372-post
val split.

WHY THIS EXISTS -- the measurement problem, quantified.

  Predicted gains fitted on the 372-post val split have repeatedly failed to
  transfer to the actual result, while a change that encodes a structural
  property of the ANNOTATION SCHEME (risk == indicator implies no evidence
  spans), independent of any model or split, transferred cleanly.

  The difference is not the size of the predicted gain, it is the KIND of
  change. Choices FITTED TO VAL -- which taggers vote, what weight a member
  gets, which rules fire -- are not well supported by 372 posts across 32
  authors. The rule re-mining made this obvious: seven of ten corrections
  from one mined rule came from TWO authors, and half the split saw zero
  firings, yet it passed the stability gate because zero counts as
  non-negative.

  All four champion components now have honest out-of-fold predictions:

      v1-architecture DeBERTa   v2_rebuild/runs/risk_ctrl_f*      1635
      friend model              artifacts/friend_oof_export.csv   1590
      ordinal Qwen              v2_rebuild/runs/risk_llm2_ord_f*  1635
      risk-NLI                  artifacts/nli_risk_oof_all.csv    1635

  So the champion recipe can be assembled over their intersection and used to
  re-decide the Qwen weight and re-mine the symbolic layer with ~1590 posts
  and all 153 authors behind every verdict.

THE RECIPE, from scripts/rebuild_and_apply_rules.sh:
    ensembled = 0.5 x DeBERTa + 0.5 x friend
    blended   = 0.70 x ensembled + 0.30 x NLI
    final     = apply_rules(blended, [urgency, completed_act])

HONEST CAVEAT, stated once and loudly: this uses ONE v1-architecture seed where
the champion averages three, and its NLI models are fold-trained rather than
the two shipped seeds. So it is champion-LIKE, not the champion. Absolute
numbers here will not match the official score. What it is good for is RANKING --
which rule helps, which weight is best -- with enough data that the ranking
is trustworthy. Final ship decisions should still be cross-checked on the
354-post true-champion reconstruction (src/champion_plus_qwen.py).

Usage:
  python src/champion_oof.py --export artifacts/champion_oof.csv
  python src/rule_ablation_v2.py --baseline artifacts/champion_oof.csv \
      --val artifacts/full_train_oof.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1
from apply_rules_to_test import apply_rules

PCOLS = [f"prob_{c}" for c in RISK_LEVELS]
FRIEND_COLS = ["Indicator", "Ideation", "Behavior", "Attempt"]


def author_halves(auth):
    s = pd.Series(auth)
    counts = s.value_counts()
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    half, size = {}, {"A": 0, "B": 0}
    for a in order:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


def load_folds(tmpl, k=5):
    parts = []
    for f in range(k):
        p = os.path.join(tmpl.replace("{f}", str(f)), "oof_predictions.csv")
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True)
    if any(c not in d.columns for c in PCOLS):
        return None
    return d.set_index("row_id")[PCOLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--deberta", default="v2_rebuild/runs/risk_ctrl_f{f}")
    ap.add_argument("--friend", default="artifacts/friend_oof_export.csv")
    ap.add_argument("--nli", default="artifacts/nli_risk_oof_all.csv")
    ap.add_argument("--qwen", default="v2_rebuild/runs/risk_llm2_ord_f{f}")
    ap.add_argument("--qwen2", default="v2_rebuild/runs/risk_llm2_cw_f{f}",
                    help="second Qwen arm (class-weighted); optional")
    ap.add_argument("--w-friend", type=float, default=0.5)
    ap.add_argument("--w-nli", type=float, default=0.30)
    ap.add_argument("--rules", default="urgency,completed_act")
    ap.add_argument("--export", default="artifacts/champion_oof.csv")
    ap.add_argument("--export-val", default="artifacts/full_train_oof.csv",
                    help="the matching gold frame, for rule_ablation_v2")
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")

    deb = load_folds(args.deberta)
    if deb is None:
        raise SystemExit(f"no DeBERTa OOF at {args.deberta}")
    fr = pd.read_csv(args.friend).set_index("row_id")[FRIEND_COLS]
    fr.columns = PCOLS
    nli = pd.read_csv(args.nli).set_index("row_id")[PCOLS]
    qw = load_folds(args.qwen)
    qw2 = load_folds(args.qwen2)

    src = {"deberta": deb, "friend": fr, "nli": nli}
    if qw is not None:
        src["qwen_ord"] = qw
    if qw2 is not None:
        src["qwen_cw"] = qw2
    for n, d in src.items():
        print(f"  {n:<10} {len(d)} posts")

    rows = set(lab.index)
    for d in src.values():
        rows &= set(d.index)
    rows = sorted(rows)
    print(f"\nintersection: {len(rows)} posts, "
          f"{lab.loc[rows, 'anon_user_id'].nunique()} authors")

    def M(d):
        x = d.loc[rows, PCOLS].to_numpy(float)
        return x / x.sum(axis=1, keepdims=True)

    gold = [normalize_label(x) for x in lab.loc[rows, "risk_level"]]
    posts = list(lab.loc[rows, "post"])
    keys = [r.strip() for r in args.rules.split(",")]
    half = author_halves(list(lab.loc[rows, "anon_user_id"]))

    def finalize(P):
        lb = [RISK_LEVELS[i] for i in P.argmax(1)]
        return [apply_rules(posts[i], lb[i], keys)[0] for i in range(len(lb))]

    def sc(labels, mask=None):
        if mask is None:
            return risk_weighted_f1(gold, labels)
        return risk_weighted_f1([x for x, m in zip(gold, mask) if m],
                                [x for x, m in zip(labels, mask) if m])

    ens = (1 - args.w_friend) * M(deb) + args.w_friend * M(fr)
    blended = (1 - args.w_nli) * ens + args.w_nli * M(nli)
    base_lbl = finalize(blended)

    print(f"\n{'stage':<22}{'wF1':>9}{'halfA':>9}{'halfB':>9}")
    print("-" * 49)
    for nm, P in (("deberta", M(deb)), ("friend", M(fr)), ("nli", M(nli)),
                  ("+friend", ens), ("+nli", blended)):
        lb = [RISK_LEVELS[i] for i in P.argmax(1)]
        print(f"{nm:<22}{sc(lb):>9.4f}{sc(lb, half == 'A'):>9.4f}"
              f"{sc(lb, half == 'B'):>9.4f}")
    print(f"{'+rules (BASELINE)':<22}{sc(base_lbl):>9.4f}"
          f"{sc(base_lbl, half == 'A'):>9.4f}{sc(base_lbl, half == 'B'):>9.4f}")

    base, bA, bB = (sc(base_lbl), sc(base_lbl, half == "A"),
                    sc(base_lbl, half == "B"))

    # ---- Qwen weight, now on ~1590 posts instead of 354 -----------------
    grid = np.round(np.arange(0.0, 0.61, 0.05), 2)
    for arm in ("qwen_ord", "qwen_cw"):
        if arm not in src:
            continue
        Q = M(src[arm])
        print(f"\nadding {arm}:")
        print(f"{'w':>7}{'wF1':>10}{'delta':>9}{'halfA':>9}{'halfB':>9}")
        cache = {}
        for w in grid:
            lb = finalize((1 - w) * blended + w * Q)
            cache[w] = lb
            print(f"{w:>7.2f}{sc(lb):>10.4f}{sc(lb) - base:>+9.4f}"
                  f"{sc(lb, half == 'A') - bA:>+9.4f}"
                  f"{sc(lb, half == 'B') - bB:>+9.4f}")
        tot = []
        for tgt, s2 in (("A", "B"), ("B", "A")):
            mt, ms = half == tgt, half == s2
            bt = sc(base_lbl, mt)
            best = max(grid, key=lambda w: sc(cache[w], ms))
            tot.append(sc(cache[best], mt) - bt)
            print(f"  xfit on {s2} -> {tgt}: w={best:.2f}  {tot[-1]:+.4f}")
        print(f"  mean cross-fitted {np.mean(tot):+.4f}")

    # both Qwen arms bagged together
    if "qwen_ord" in src and "qwen_cw" in src:
        Q = 0.5 * M(src["qwen_ord"]) + 0.5 * M(src["qwen_cw"])
        print(f"\nadding BOTH Qwen arms (bagged):")
        print(f"{'w':>7}{'wF1':>10}{'delta':>9}{'halfA':>9}{'halfB':>9}")
        for w in grid:
            lb = finalize((1 - w) * blended + w * Q)
            print(f"{w:>7.2f}{sc(lb):>10.4f}{sc(lb) - base:>+9.4f}"
                  f"{sc(lb, half == 'A') - bA:>+9.4f}"
                  f"{sc(lb, half == 'B') - bB:>+9.4f}")

    # ---- export for rule re-mining --------------------------------------
    pre = [RISK_LEVELS[i] for i in blended.argmax(1)]
    out = pd.DataFrame({
        "row_id": rows,
        "pred_risk_level": [x.capitalize() for x in base_lbl],
        "pre_rule_pred": [x.capitalize() for x in pre],
        "gold": gold,
    })
    for j, c in enumerate(RISK_LEVELS):
        out[f"prob_{c}"] = blended[:, j]
    os.makedirs(os.path.dirname(args.export) or ".", exist_ok=True)
    out.to_csv(args.export, index=False)
    lab.loc[rows].reset_index().to_csv(args.export_val, index=False)
    print(f"\nwrote {args.export} and {args.export_val} ({len(rows)} posts)")
    print("\nRe-mine the symbolic layer with 4.5x the data:")
    print(f"  python src/rule_ablation_v2.py --baseline {args.export} \\")
    print(f"      --val {args.export_val}")
    print("\nA rule that passed on 354 posts because two authors happened to")
    print("carry it will NOT pass here. That is the point.")


if __name__ == "__main__":
    main()
