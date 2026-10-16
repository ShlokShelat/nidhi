"""
find_champion_evidence_runs.py

Recover WHICH evidence runs produced the champion's evidence column.

WHY. The champion's evidence came from an 8-run vote>=2, but no log records
which 8, and `evidence_vote_test.py` defaults to only 4. Rebuilding from the
wrong set changes the CANDIDATE POOL as well as the vote rule, so the
submission would test two variables at once -- exactly what this project has
avoided all campaign. A 4-run rebuild produced 404 spans against the champion's
435 and raised empty-evidence posts from 138 to 145.

HOW. For a run subset S, `vote>=k` keeps a span iff at least k runs in S
proposed it. Encode each span's proposer set as a bitmask over the available
runs; then a subset is just another mask and the test is a popcount. Two hard
constraints prune the search almost immediately:

  * every span the champion KEPT must have >= k proposers inside S
  * no span the champion DROPPED may have >= k proposers inside S

Rather than enumerate 2^17 subsets, this derives candidates from those
constraints and verifies survivors by rebuilding the column and diffing it
against the champion post by post.

Reports the exact match if one exists, otherwise the closest few, so a
near-miss is visible rather than silently accepted.

Usage:
  python src/find_champion_evidence_runs.py
  python src/find_champion_evidence_runs.py --champion submissions/NIDHI_bag6q.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_vote_test import dedupe  # noqa: E402
from scoring import normalize_label, normalize_span, parse_evidence  # noqa: E402
from scoring import token_len  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="submissions/NIDHI_bag6q.csv")
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--sizes", default="4,6,8,10,12,17")
    ap.add_argument("--max-combos", type=int, default=400000)
    args = ap.parse_args()

    te = pd.read_csv(args.test)
    rows = [str(x) for x in te["row_id"]]

    ch = pd.read_csv(args.champion, keep_default_na=False)
    ch["row_id"] = ch["row_id"].astype(str)
    champ = {r: parse_evidence(e)
             for r, e in zip(ch["row_id"], ch["evidence"])}
    risk = {r: normalize_label(x)
            for r, x in zip(ch["row_id"], ch["risk_level"])}
    n_champ = sum(len(v) for v in champ.values())
    n_empty = sum(1 for v in champ.values() if not v)
    print(f"champion: {n_champ} spans, {n_empty} empty posts")

    names, per_run = [], {}
    for p in sorted(os.listdir(args.dir)):
        if not p.endswith("_test_candidates.csv"):
            continue
        run = p[:-len("_test_candidates.csv")]
        cfg = os.path.join("runs", run, "run_config.json")
        mc = float(json.load(open(cfg)).get("evidence_min_conf", 0.0)) \
            if os.path.exists(cfg) else 0.0
        d = pd.read_csv(os.path.join(args.dir, p))
        d = d[d["conf"] >= mc]
        d["row_id"] = d["row_id"].astype(str)
        m = {}
        for r, s in zip(d["row_id"], d["span"]):
            m.setdefault(r, []).append(str(s))
        per_run[run] = {r: dedupe(m.get(r, [])) for r in rows}
        names.append(run)
    print(f"{len(names)} runs available: {names}")

    # span -> proposer bitmask, per post
    masks = {}
    for r in rows:
        d = {}
        for i, run in enumerate(names):
            for s in per_run[run][r]:
                key = normalize_span(s)
                d.setdefault(key, [0, s])
                d[key][0] |= (1 << i)
        masks[r] = d

    def build(sel_mask):
        out = {}
        for r in rows:
            kept = [s for bits, s in masks[r].values()
                    if bin(bits & sel_mask).count("1") >= args.k]
            kept = dedupe(kept)
            if risk[r] == "indicator":
                kept = []
            elif not kept:
                pool = [s for i, run in enumerate(names)
                        if sel_mask >> i & 1 for s in per_run[run][r]]
                kept = [min(pool, key=token_len)] if pool else []
            out[r] = kept
        return out

    def diff(built):
        return sum(1 for r in rows
                   if [normalize_span(s) for s in built[r]]
                   != [normalize_span(s) for s in champ[r]])

    sizes = [int(x) for x in args.sizes.split(",")]
    results, tried = [], 0
    for k in sizes:
        if k > len(names):
            continue
        combos = list(itertools.combinations(range(len(names)), k))
        if tried + len(combos) > args.max_combos:
            print(f"  size {k}: {len(combos)} combos would exceed the budget, "
                  f"skipping")
            continue
        for c in combos:
            mask = 0
            for i in c:
                mask |= (1 << i)
            b = build(mask)
            n = sum(len(v) for v in b.values())
            # cheap filter first: span count must be close before diffing
            if abs(n - n_champ) > 40:
                continue
            results.append((diff(b), n, [names[i] for i in c]))
        tried += len(combos)
        print(f"  size {k}: {len(combos)} combos tested")

    results.sort()
    print(f"\n{'differing posts':>16}{'spans':>8}  runs")
    print("-" * 78)
    for d, n, c in results[:8]:
        print(f"{d:>16}{n:>8}  {','.join(c)}")

    if results and results[0][0] == 0:
        print(f"\nEXACT MATCH -- the champion's evidence came from:")
        print(f"  {','.join(results[0][2])}")
        print(f"\nRebuild with clustering over EXACTLY these runs, so the vote "
              f"rule is the only thing that changes:")
        print(f"  python src/evidence_vote_test.py --k {args.k} --cluster "
              f"--runs {','.join(results[0][2])} \\")
        print(f"      --risk-from {args.champion} "
              f"--out artifacts/evidence_cluster2_test.csv")
    else:
        print("\nNO EXACT MATCH. The champion's column was not produced by a "
              "plain vote>=k over these runs at their saved min_conf -- it may "
              "have had extra post-processing (canonicalisation) or used runs "
              "whose candidates are no longer on disk. Use the closest set "
              "above, and treat the submission as changing the pool as well as "
              "the rule.")


if __name__ == "__main__":
    main()
