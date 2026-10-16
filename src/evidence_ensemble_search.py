"""
evidence_ensemble_search.py

Evidence carries 0.3 of the composite and the error decomposition
(evidence_error_analysis.py) showed only ~4% of gold spans are boundary-
fixable -- so a symbolic post-processing layer cannot move it. What the
decomposition DOES show is a large spurious-prediction count (105 predicted
spans matching nothing on val, precision 0.699) alongside comparable recall.
That is the profile seed-agreement filtering is good at: keep spans several
independently-trained taggers agree on, drop the ones only one seed invented.

This scores every available evidence run individually and then tries
span-level combination strategies, all against the official phrase metric,
on CPU from cached val predictions.

Strategies:
  union      -- every span any run predicts (max recall, min precision)
  vote>=k    -- keep a span if at least k runs predict a span that MATCHES it
                under the official containment rule (not string equality --
                seeds rarely agree on exact boundaries)
  intersect  -- vote>=n_runs

Usage:
  python src/evidence_ensemble_search.py
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1, _span_pair_matches, token_len

RUNS = ["runs/evidence_v2/val_predictions.csv",
        "runs/evidence_v1/val_predictions.csv",
        "runs/evidence_s14/val_predictions.csv",
        "runs/evidence_s15/val_predictions.csv"]


def load(paths, val):
    out = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"  (missing, skipped) {p}")
            continue
        d = pd.read_csv(p)
        col = next((c for c in d.columns if "pred" in c.lower()), None)
        if col is None:
            continue
        m = val[["row_id"]].merge(d[["row_id", col]], on="row_id", how="left")
        out[p.split("/")[1]] = [parse_evidence(x) for x in m[col]]
    return out


def vote_combine(per_run_spans, k):
    """Keep spans agreed on by >= k runs, using the official match rule.

    The representative kept for an agreed group is the SHORTEST span, since
    the metric's 3x cap punishes over-long spans and containment favours the
    tighter one.
    """
    out = []
    for i in range(len(next(iter(per_run_spans.values())))):
        cands = []
        for spans in per_run_spans.values():
            cands.append(spans[i])
        pool = [s for spans in cands for s in spans]
        kept, used = [], set()
        for j, s in enumerate(pool):
            if j in used:
                continue
            group = [s]
            used.add(j)
            for j2 in range(j + 1, len(pool)):
                if j2 in used:
                    continue
                if _span_pair_matches(pool[j2], s):
                    group.append(pool[j2])
                    used.add(j2)
            # count DISTINCT runs contributing to this group
            n_runs = sum(1 for spans in cands
                         if any(any(_span_pair_matches(x, g) for g in group)
                                for x in spans))
            if n_runs >= k:
                kept.append(min(group, key=token_len))
        out.append(kept)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    args = ap.parse_args()

    val = pd.read_csv(args.val)[["row_id", "evidence"]]
    gold = [parse_evidence(x) for x in val["evidence"]]

    print("loading evidence runs...")
    runs = load(RUNS, val)
    if not runs:
        raise SystemExit("no evidence runs found")

    print(f"\n{'=' * 78}\nINDIVIDUAL RUNS (official phrase metric, val)\n{'=' * 78}")
    print(f"{'run':<18}{'P':>9}{'R':>9}{'F1':>9}{'n_pred':>9}")
    for name, spans in runs.items():
        r = phrase_f1(gold, spans)
        print(f"{name:<18}{r.precision:>9.4f}{r.recall:>9.4f}{r.f1:>9.4f}{r.n_pred:>9}")

    print(f"\n{'=' * 78}\nCOMBINATIONS\n{'=' * 78}")
    print(f"{'strategy':<34}{'P':>9}{'R':>9}{'F1':>9}{'n_pred':>9}")
    best = None
    n = len(runs)
    for k in range(1, n + 1):
        combined = vote_combine(runs, k)
        r = phrase_f1(gold, combined)
        tag = "union (vote>=1)" if k == 1 else (
            f"intersect (vote>={k})" if k == n else f"vote>={k}")
        print(f"{f'{n} runs, {tag}':<34}{r.precision:>9.4f}{r.recall:>9.4f}"
              f"{r.f1:>9.4f}{r.n_pred:>9}")
        if best is None or r.f1 > best[1]:
            best = (f"{n} runs {tag}", r.f1)

    # also try 3-run subsets, in case one run is dragging the pool down
    names = list(runs)
    for subset in itertools.combinations(names, 3):
        sub = {s: runs[s] for s in subset}
        for k in (2, 3):
            r = phrase_f1(gold, vote_combine(sub, k))
            label = f"{'+'.join(s.replace('evidence_', '') for s in subset)} vote>={k}"
            print(f"{label:<34}{r.precision:>9.4f}{r.recall:>9.4f}"
                  f"{r.f1:>9.4f}{r.n_pred:>9}")
            if r.f1 > best[1]:
                best = (label, r.f1)

    base = phrase_f1(gold, runs["evidence_v2"]).f1
    print(f"\ndeployed evidence_v2 alone : {base:.4f}")
    print(f"best combination           : {best[0]} -> {best[1]:.4f} "
          f"({best[1] - base:+.4f})")
    print(f"composite impact if it transfers: {0.3 * (best[1] - base):+.4f}")


if __name__ == "__main__":
    main()
