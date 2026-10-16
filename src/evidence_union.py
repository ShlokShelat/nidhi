"""
evidence_union.py

Raise evidence recall by pooling spans across the evidence models.

THE DIAGNOSIS THIS FOLLOWS FROM (src/evidence_diagnosis.py, 372 val posts):

    phrase-F1 0.6902   P 0.6991   R 0.6816      <- RECALL-limited
    93 gold spans recoverable in principle
    break-even hit-rate ~ F1/2 = 0.35
    matches lost to the 3x length cap: 2 of 358 (length costs nothing here)

Micro-F1 with precision approx equal to recall has an exploitable asymmetry:
adding a prediction is net positive whenever its chance of matching exceeds
about F1/2. At 0.69 that is 0.35 -- an extra span pays even if it is WRONG
almost two-thirds of the time.

The diagnostic could not say whether the tagger's own runner-up clears 0.35.
But there are five evidence runs on disk (v2, full, nli, s14, s15), and a span
proposed by a DIFFERENT model is a much stronger candidate than the same
model's second choice. That is the same ensemble logic that pays on risk --
except here it has a mechanical reason to work rather than a hope: union
strictly increases recall, and the metric's shape decides whether the precision
cost is worth it.

ARMS (all parameter-free except the vote count, a small integer):

    v2 alone      the shipped tagger, baseline
    union         every span any model proposes -- max recall, min precision
    vote>=2       spans proposed by at least two models
    vote>=3       stricter still

CONTROLS. Spans are deduplicated after normalisation, and a span already
CONTAINED in another kept span is dropped -- otherwise the union would submit
near-duplicates that cannot both match (matching is one-to-one) and would only
cost precision. Every arm is judged overall AND on both author-disjoint halves.

The consistency fix is applied to every arm, so this measures the union on top
of what is actually shipped rather than on top of the old pre-fix system.

Usage:
  python src/evidence_union.py
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span, parse_evidence,
                     phrase_f1, token_len)

RUNS = ["evidence_v2", "evidence_full", "evidence_nli",
        "evidence_s14", "evidence_s15"]


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


def dedupe(spans):
    """Drop exact duplicates and spans contained in a longer kept span.
    Matching is one-to-one, so two near-identical predictions can never both
    score -- keeping both only costs precision."""
    out = []
    for s in sorted(spans, key=lambda x: -token_len(x)):
        n = normalize_span(s)
        if not n:
            continue
        if any(n in normalize_span(k) for k in out):
            continue
        out.append(s)
    return out


def consistency(spans, risk, post_spans_fallback):
    """The shipped rule: indicator -> no spans; anything else -> at least one."""
    if risk == "indicator":
        return []
    if spans:
        return spans
    return post_spans_fallback[:1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--pred-col", default="pred_evidence")
    args = ap.parse_args()

    val = pd.read_csv(args.val)
    preds = {}
    for r in [x.strip() for x in args.runs.split(",")]:
        p = os.path.join("runs", r, "val_predictions.csv")
        if not os.path.exists(p):
            print(f"  (skip {r}: no val_predictions.csv)")
            continue
        d = pd.read_csv(p)
        col = args.pred_col if args.pred_col in d.columns else next(
            (c for c in ("pred_evidence", "evidence", "pred")
             if c in d.columns), None)
        if col is None:
            print(f"  (skip {r}: no prediction column)")
            continue
        preds[r] = dict(zip(d["row_id"], d[col]))
        print(f"  loaded {r} ({len(d)} rows, col {col!r})")
    if not preds:
        raise SystemExit("no evidence runs with val predictions")

    rows = [r for r in val["row_id"] if all(r in m for m in preds.values())]
    v = val.set_index("row_id").loc[rows]
    G = [parse_evidence(x) for x in v["evidence"]]
    risk = [normalize_label(x) for x in v["risk_level"]]
    half = author_halves(list(v["anon_user_id"]))
    print(f"\n{len(rows)} posts common to all {len(preds)} runs")

    per_run = {n: [parse_evidence(m[r]) for r in rows] for n, m in preds.items()}

    def arm(pick):
        """pick(i) -> list of spans for post i, before the consistency rule."""
        out = []
        for i in range(len(rows)):
            s = dedupe(pick(i))
            # fallback for FORCE: prefer the shipped tagger's shortest span
            fb = sorted(per_run.get("evidence_v2", per_run[list(per_run)[0]])[i],
                        key=token_len) or sorted(s, key=token_len)
            out.append(consistency(s, risk[i], fb))
        return out

    def sc(P, mask=None):
        if mask is None:
            return phrase_f1(G, P)
        return phrase_f1([g for g, m in zip(G, mask) if m],
                         [p for p, m in zip(P, mask) if m])

    names = list(per_run)

    def votes(i, k):
        c = {}
        for n in names:
            for s in dedupe(per_run[n][i]):
                key = normalize_span(s)
                c.setdefault(key, [0, s])
                c[key][0] += 1
        return [s for cnt, s in c.values() if cnt >= k]

    arms = {}
    base_name = "evidence_v2" if "evidence_v2" in per_run else names[0]
    arms[f"{base_name} alone"] = arm(lambda i: per_run[base_name][i])
    arms["union all"] = arm(lambda i: [s for n in names for s in per_run[n][i]])
    # Sweep every vote count. With 4 models vote>=3 was far too strict (recall
    # 0.55); with 8 the useful threshold moves up, and where it lands is the
    # thing to read off this table rather than assume.
    for k in range(2, len(names) + 1):
        arms[f"vote>={k}"] = arm(lambda i, k=k: votes(i, k))

    base = sc(arms[f"{base_name} alone"])
    bA = sc(arms[f"{base_name} alone"], half == "A")
    bB = sc(arms[f"{base_name} alone"], half == "B")

    print(f"\n{'arm':<20}{'F1':>8}{'P':>8}{'R':>8}{'pred':>7}"
          f"{'delta':>9}{'halfA':>9}{'halfB':>9}")
    print("-" * 78)
    for nm, P in arms.items():
        r = sc(P)
        a = sc(P, half == "A")
        b = sc(P, half == "B")
        print(f"{nm:<20}{r.f1:>8.4f}{r.precision:>8.4f}{r.recall:>8.4f}"
              f"{r.n_pred:>7}{r.f1 - base.f1:>+9.4f}"
              f"{a.f1 - bA.f1:>+9.4f}{b.f1 - bB.f1:>+9.4f}")

    print(f"\nbaseline: {base_name} + consistency = {base.f1:.4f} "
          f"(P {base.precision:.4f} R {base.recall:.4f})")

    winners = [(nm, sc(P).f1 - base.f1) for nm, P in arms.items()
               if nm != f"{base_name} alone"
               and sc(P).f1 > base.f1
               and sc(P, half == "A").f1 > bA.f1
               and sc(P, half == "B").f1 > bB.f1]

    print("\n" + "=" * 78)
    if winners:
        nm, d = max(winners, key=lambda x: x[1])
        print(f"BEST: {nm}  {d:+.4f} phrase-F1  ({0.3 * d:+.4f} composite), "
              f"positive on both halves")
        print("Next: rebuild the test evidence column with this arm, apply the")
        print("consistency rule, patch evidence ONLY on top of the champion.")
        print("This is parameter-free (a union, or a small integer vote), which")
        print("is the category that has transferred here -- unlike the seven")
        print("tuned-threshold changes that all failed.")
    else:
        print("No pooling arm beats the shipped tagger on both halves.")
        print("Evidence recall cannot be bought from these models, and the")
        print("last live component closes.")
    print("=" * 78)


if __name__ == "__main__":
    main()
