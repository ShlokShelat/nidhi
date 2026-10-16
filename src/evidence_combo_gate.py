"""
evidence_combo_gate.py

Re-runs the evidence combination search AND the author-disjoint stability gate
against the FRESH per-run predictions cached by evidence_vote_ensemble.py,
rather than the runs/*/val_predictions.csv files.

This matters: those cached files are not all produced by the deployed decoder.
evidence_v1 scores 0.6610 from its cached val_predictions.csv but 0.6634 when
re-scored through predict.py's predict_evidence(), which is the function that
will actually run on test. The earlier offline search and stability gate used
the stale numbers, so both need redoing on the fresh cache before anything is
submitted.

Everything here is CPU-only; evidence_vote_ensemble.py already wrote the
per-run predictions to artifacts/evidence_per_run/.

Usage:
  python src/evidence_combo_gate.py
  python src/evidence_combo_gate.py --cache-dir artifacts/evidence_per_run
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1
from evidence_vote_ensemble import vote_combine
from rule_ablation_v2 import author_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--cache-dir", default="artifacts/evidence_per_run")
    ap.add_argument("--tag", default="val_split")
    ap.add_argument("--baseline", default="evidence_v2")
    args = ap.parse_args()

    val = pd.read_csv(args.val)
    gold = [parse_evidence(x) for x in val["evidence"]]
    half = author_split(val)[0].values
    idxA = [i for i in range(len(val)) if half[i] == "A"]
    idxB = [i for i in range(len(val)) if half[i] == "B"]

    runs = {}
    for f in sorted(glob.glob(os.path.join(args.cache_dir, f"*__{args.tag}.csv"))):
        name = os.path.basename(f).split("__")[0]
        d = pd.read_csv(f, keep_default_na=False).set_index("row_id")["evidence"]
        runs[name] = [parse_evidence(d.get(r, "")) for r in val["row_id"]]
    if not runs:
        raise SystemExit(f"no cached per-run predictions in {args.cache_dir} "
                         f"for tag {args.tag}. Run evidence_vote_ensemble.py first.")
    print(f"loaded {len(runs)} runs from cache: {list(runs)}\n")

    def score(spans):
        return (phrase_f1(gold, spans).f1,
                phrase_f1([gold[i] for i in idxA], [spans[i] for i in idxA]).f1,
                phrase_f1([gold[i] for i in idxB], [spans[i] for i in idxB]).f1)

    print(f"{'run (alone)':<40}{'full':>8}{'halfA':>8}{'halfB':>8}")
    for n, s in runs.items():
        f, a, b = score(s)
        print(f"{n:<40}{f:>8.4f}{a:>8.4f}{b:>8.4f}")

    if args.baseline not in runs:
        raise SystemExit(f"baseline {args.baseline} not in cache")
    base = score(runs[args.baseline])
    print(f"\nbaseline = {args.baseline}: full {base[0]:.4f} "
          f"(A {base[1]:.4f} / B {base[2]:.4f})")

    names = list(runs)
    results = []
    for r in range(2, len(names) + 1):
        for subset in itertools.combinations(names, r):
            for k in range(2, r + 1):
                spans = vote_combine([runs[s] for s in subset], k)
                f, a, b = score(spans)
                results.append({
                    "combo": "+".join(s.replace("evidence_", "") for s in subset)
                             + f" k>={k}",
                    "full": f, "A": a, "B": b,
                    "dFull": f - base[0], "dA": a - base[1], "dB": b - base[2],
                })

    results.sort(key=lambda x: -x["full"])
    print(f"\n{'combination':<34}{'full':>8}{'halfA':>8}{'halfB':>8}"
          f"{'dFull':>9}{'dA':>8}{'dB':>8}  gate")
    passing = []
    for r in results:
        gate = "PASS" if (r["dFull"] > 0 and r["dA"] > 0 and r["dB"] > 0) else "REJECT"
        if gate == "PASS":
            passing.append(r)
        print(f"{r['combo']:<34}{r['full']:>8.4f}{r['A']:>8.4f}{r['B']:>8.4f}"
              f"{r['dFull']:>+9.4f}{r['dA']:>+8.4f}{r['dB']:>+8.4f}  {gate}")

    print(f"\n{'=' * 92}")
    if not passing:
        print("NOTHING PASSES the author-disjoint gate -- do not ship an "
              "evidence ensemble on this evidence.")
        return
    best = max(passing, key=lambda r: min(r["dA"], r["dB"]))
    top = passing[0]
    print(f"highest full-val PASS : {top['combo']}  {top['full']:.4f} "
          f"({top['dFull']:+.4f})")
    print(f"most BALANCED PASS    : {best['combo']}  {best['full']:.4f} "
          f"(worst half {min(best['dA'], best['dB']):+.4f})")
    print(f"\nRecommend the balanced one unless full-val is much higher: a "
          f"combination whose\ngain sits almost entirely in one author half is "
          f"the failure mode this gate exists to catch.")
    print(f"composite estimate for balanced pick: "
          f"{0.3 * best['dFull']:+.4f}")


if __name__ == "__main__":
    main()
