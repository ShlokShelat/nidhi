"""
champion_plus_qwen.py

Rebuild the CHAMPION risk ensemble on val, verify it, then ask whether Qwen
adds anything as an extra member.

THE RECIPE, from scripts/rebuild_and_apply_rules.sh:

    ensembled = 0.5 x mean(risk_v1, risk_v1_s14, risk_v1_s15)
              + 0.5 x friend
    blended   = 0.70 x ensembled + 0.30 x mean(risk_nli, risk_nli_s2)
    final     = apply_rules(blended, [urgency, completed_act])

Six models at two blend levels. Qwen must not be compared against a SINGLE
DeBERTa scoring 0.7387 when this stack scores 0.8202: a gain measured against
a stand-in that is not in the submission is not a real gain, and evaluating
against the wrong control is how a large regression can reach the
final score undetected.

WHY THE VERIFICATION STEP IS NOT OPTIONAL. Before any Qwen number is produced,
the rebuild is scored against artifacts/val_champion_reconstructed.csv. If the
rebuilt labels do not closely match champion_final, then whatever Qwen is
being added to is not the champion, and the answer would be as misleading as
comparing against the wrong control. Skipping this check risks a submission
slot on a misleading answer.

WHAT WOULD MAKE THIS WORTH SHIPPING. Qwen has already failed as a wholesale
replacement and as a selective label override, cross-fitted, and it only won
as a blend against a weak two-model ensemble. The open question is whether
that blend gain survives against a strong six-model base, where a seventh
voice matters much less. So:

    - the blend weight is CROSS-FITTED across author-disjoint halves
    - it must win on BOTH halves
    - 295-odd val posts resolve about +-0.02, so anything smaller is noise

Note the ordinal Qwen arm only ran folds 1-4, so fold-0 val posts have no Qwen
prediction and drop out of the comparison. All arms are scored on the same
surviving rows.

Usage:
  python src/champion_plus_qwen.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

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


def load_qwen(tmpl, k=5):
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
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--seeds",
                    default="runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15")
    ap.add_argument("--friend", default="artifacts/friend_risk_probs_val.csv")
    ap.add_argument("--nli", default="artifacts/nli_risk_val_probs.csv")
    ap.add_argument("--qwen", default="v2_rebuild/runs/risk_llm2_ord_f{f}")
    ap.add_argument("--champion",
                    default="artifacts/val_champion_reconstructed.csv")
    ap.add_argument("--w-friend", type=float, default=0.5)
    ap.add_argument("--w-nli", type=float, default=0.30)
    ap.add_argument("--rules", default="urgency,completed_act")
    ap.add_argument("--export", default=None,
                    help="write the champion+Qwen val predictions so the "
                         "SYMBOLIC LAYER can be re-mined against THIS model's "
                         "errors. The current rules were mined against "
                         "DeBERTa's mistakes; the risk model has changed, so "
                         "a rule that paid then may now be firing on posts the "
                         "ensemble already gets right. Feed the output to "
                         "src/rule_ablation_v2.py --baseline")
    ap.add_argument("--export-w", type=float, default=0.40,
                    help="Qwen weight to export at (the deployed one)")
    args = ap.parse_args()

    val = pd.read_csv(args.val).set_index("row_id")
    ch = pd.read_csv(args.champion).set_index("row_id")

    seeds = []
    for r in args.seeds.split(","):
        p = os.path.join(r.strip(), "val_predictions.csv")
        seeds.append(pd.read_csv(p).set_index("row_id")[PCOLS])
    fr = pd.read_csv(args.friend).set_index("row_id")[FRIEND_COLS]
    fr.columns = PCOLS
    nli = pd.read_csv(args.nli).set_index("row_id")[PCOLS]
    qw = load_qwen(args.qwen)

    rows = set(val.index) & set(ch.index) & set(fr.index) & set(nli.index)
    for s in seeds:
        rows &= set(s.index)
    have_qwen = qw is not None
    if have_qwen:
        rows &= set(qw.index)
    rows = sorted(rows)
    print(f"{len(rows)} val posts with every component"
          + ("" if have_qwen else "   (NO QWEN PROBS FOUND)"))

    def M(d):
        x = d.loc[rows, PCOLS].to_numpy(float)
        return x / x.sum(axis=1, keepdims=True)

    ours = np.mean([M(s) for s in seeds], axis=0)
    ens = (1 - args.w_friend) * ours + args.w_friend * M(fr)
    blended = (1 - args.w_nli) * ens + args.w_nli * M(nli)

    gold = [normalize_label(x) for x in val.loc[rows, "risk_level"]]
    posts = list(val.loc[rows, "post"])
    rule_keys = [r.strip() for r in args.rules.split(",")]

    def finalize(P):
        lab = [RISK_LEVELS[i] for i in P.argmax(1)]
        return [apply_rules(posts[i], lab[i], rule_keys)[0]
                for i in range(len(lab))]

    def sc(labels, mask=None):
        if mask is None:
            return risk_weighted_f1(gold, labels)
        return risk_weighted_f1([x for x, m in zip(gold, mask) if m],
                                [x for x, m in zip(labels, mask) if m])

    rebuilt = finalize(blended)
    champ = [normalize_label(x) for x in ch.loc[rows, "champion_final"]]

    print("\n--- STAGE SCORES ---")
    for nm, P in (("seeds only", ours), ("+friend", ens), ("+nli", blended)):
        print(f"  {nm:<14} {risk_weighted_f1(gold, [RISK_LEVELS[i] for i in P.argmax(1)]):.4f}")
    print(f"  +rules         {sc(rebuilt):.4f}   <- rebuilt champion")
    print(f"  champion_final {sc(champ):.4f}   <- reference")

    agree = float(np.mean([a == b for a, b in zip(rebuilt, champ)]))
    print(f"\nrebuild agrees with champion_final on {100 * agree:.1f}% of posts")
    if agree < 0.95:
        print("*** REBUILD DOES NOT MATCH THE CHAMPION. Everything below would")
        print("*** be measured against the wrong baseline -- the exact mistake")
        print("*** that cost a submission today. Fix the recipe before reading on.")
    if not have_qwen:
        return

    half = author_halves(list(val.loc[rows, "anon_user_id"]))
    base = sc(rebuilt)
    bA, bB = sc(rebuilt, half == "A"), sc(rebuilt, half == "B")
    print(f"\nbase {base:.4f}   halfA {bA:.4f} (n={int((half=='A').sum())})"
          f"   halfB {bB:.4f} (n={int((half=='B').sum())})")

    Q = M(qw)
    print(f"\nadding Qwen as a 7th member at weight w:")
    print(f"{'w':>7}{'overall':>10}{'delta':>9}{'halfA':>9}{'halfB':>9}")
    grid = np.round(np.arange(0.0, 0.51, 0.05), 2)
    for w in grid:
        lab = finalize((1 - w) * blended + w * Q)
        print(f"{w:>7.2f}{sc(lab):>10.4f}{sc(lab) - base:>+9.4f}"
              f"{sc(lab, half == 'A') - bA:>+9.4f}"
              f"{sc(lab, half == 'B') - bB:>+9.4f}")

    print("\nCROSS-FITTED (weight chosen on the OTHER half)")
    cache = {w: finalize((1 - w) * blended + w * Q) for w in grid}
    tot = []
    for tgt, srch in (("A", "B"), ("B", "A")):
        mt, ms = half == tgt, half == srch
        bt = sc(rebuilt, mt)
        best = max(grid, key=lambda w: sc(cache[w], ms))
        got = sc(cache[best], mt)
        tot.append(got - bt)
        print(f"  tune on {srch}, apply to {tgt}: w={best:.2f}  "
              f"{bt:.4f} -> {got:.4f}  ({got - bt:+.4f})")
    m = float(np.mean(tot))
    print(f"  mean cross-fitted delta {m:+.4f}")

    if args.export:
        W = args.export_w
        mixed = (1 - W) * blended + W * Q
        pre = [RISK_LEVELS[i] for i in mixed.argmax(1)]
        post = finalize(mixed)
        out = pd.DataFrame({
            "row_id": rows,
            "pred_risk_level": [x.capitalize() for x in post],
            "pre_rule_pred": [x.capitalize() for x in pre],
            "gold": gold,
        })
        for j, c in enumerate(RISK_LEVELS):
            out[f"prob_{c}"] = mixed[:, j]
        os.makedirs(os.path.dirname(args.export) or ".", exist_ok=True)
        out.to_csv(args.export, index=False)
        print(f"\nexported {len(out)} val predictions at w={W} to {args.export}")
        print(f"  pre-rule  wF1 {risk_weighted_f1(gold, pre):.4f}")
        print(f"  post-rule wF1 {risk_weighted_f1(gold, post):.4f}"
              f"   (rules move {sum(1 for a, b in zip(pre, post) if a != b)} posts)")
        print("  Re-mine the symbolic layer against THIS error profile:")
        print(f"    python src/rule_ablation_v2.py --baseline {args.export} \\")
        print("        --val artifacts/val_split.csv")

    print("\n" + "=" * 66)
    if agree >= 0.95 and m > 0.02 and all(x > 0 for x in tot):
        print("WORTH A SUBMISSION: Qwen adds to the REAL champion ensemble,")
        print("cross-fitted, on both halves, above the noise floor.")
        print("Next: bag the ord adapters over test, rebuild the champion test")
        print("probs the same way, blend at this w, apply rules, patch risk only.")
    else:
        print("NOT worth a submission. The gain seen earlier came from")
        print("adding Qwen to a weak two-model ensemble; against the real")
        print("six-model champion it does not survive. Replacement, override")
        print("and blending have now all been measured -- the LLM risk line")
        print("is closed for good.")
    print("=" * 66)


if __name__ == "__main__":
    main()
