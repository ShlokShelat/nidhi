#!/usr/bin/env python
"""FREE GO/NO-GO on the span-reranker lane. CPU, seconds, no GPU, no re-scoring.

WHY THIS EXISTS. `train_span_reranker.py` reported OOF AUC 0.890 against
"hand-feature ceiling 0.752 / n_votes 0.658". Those two comparators are a
HARDCODED STRING at train_span_reranker.py:228 -- legacy numbers from the RC7
tagger pool, which has no counterpart in this enumerated pool. The `conf`
column here is the constant 1.0 (evidence_enum_candidates.py:141), so the
"confidence 0.506" line is random by construction. **Only 0.890 is a real
measurement on this pool**, and AUC is not the competition metric.

THE POOL IS SUBSAMPLED, SO EVERYTHING HERE IS AN UPPER BOUND.
`artifacts/span_rerank_enum_oof.csv` is the 4:1 negative-subsampled TRAINING
pool: 128906/1632 = 79 candidates/post at 21.3% positive, where the
deployment pool is ~219/post at ~5% positive. Roughly two-thirds of the
distractors a deployed selector must reject were deleted before this file was
written. Selection precision measured here is therefore FICTITIOUS-HIGH.

That one-sidedness is exactly what makes this cheap test worth running:

    if the best cross-fitted macro here is BELOW RC7's 0.8045, the lane is
    DEAD -- the honest number on the full pool can only be lower, and no
    GPU time is warranted.

A number ABOVE 0.8045 proves nothing on its own; it only earns the right to
rebuild the full pool (`evidence_enum_candidates.py --no-subsample`) and
re-score on GPU.

The reranker scores for every post are OOF and author-disjoint, so the
selection measured here is honest with respect to authors even though the
candidate pool is not honest with respect to distractors.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1  # noqa: E402


def norm(s):
    return " ".join(str(s).lower().split())


def dedupe_nested(spans, keep="long"):
    """Drop any span contained in an already-kept one.

    Mirrors apply_span_reranker.py:72-78, which keeps the LONGEST. On an
    n-gram pool this is load-bearing: nested near-duplicates all match the
    same gold span, win one bipartite match, and the rest score as false
    positives. The direction is a real knob -- gold averages 4.6 tokens and
    the matcher caps pred at 3x gold tokens, so keep-longest can push a
    candidate out of the cap on short golds. Hence `keep` is swept.
    """
    order = sorted(spans, key=lambda s: len(s.split()),
                   reverse=(keep == "long"))
    out = []
    for s in order:
        ns = norm(s)
        if not any(ns in norm(k) or norm(k) in ns for k in out):
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="artifacts/span_rerank_enum_oof.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--val", default="artifacts/val_split.csv",
                    help="the 372-post subset RC7's 0.8045 was measured on")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--rc7", type=float, default=0.8045,
                    help="the incumbent to beat, on the val subset")
    args = ap.parse_args()

    df = pd.read_csv(args.scores, keep_default_na=False)
    df["row_id"] = df["row_id"].astype(str)
    need = {"row_id", "span", "risk", "y", "author", "rerank"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{args.scores} missing {need - set(df.columns)}")

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    gold_by = {r: parse_evidence(e)
               for r, e in zip(lab["row_id"], lab["evidence"])}

    rows = sorted(df["row_id"].unique(), key=lambda r: str(r))
    missing = [r for r in rows if r not in gold_by]
    if missing:
        raise SystemExit(f"{len(missing)} scored posts have no gold row "
                         f"in {args.labeled}; first: {missing[:3]}")

    gold = [gold_by[r] for r in rows]
    idx = {r: i for i, r in enumerate(rows)}

    # per-post candidates, pre-sorted by descending reranker score.
    # Single pass -- a df.loc filter per post would be 1632 scans of 129k rows.
    cand = [[] for _ in rows]
    pos = [[] for _ in rows]
    risk_of = {}
    for r, s, sc, rk, yy in zip(df["row_id"], df["span"], df["rerank"],
                                df["risk"], df["y"]):
        i = idx[r]
        cand[i].append((float(sc), str(s)))
        if int(yy) == 1:
            pos[i].append(str(s))
        risk_of[r] = str(rk)
    for c in cand:
        c.sort(key=lambda t: -t[0])

    author = {}
    for r, a in zip(df["row_id"], df["author"]):
        author[r] = str(a)
    authors = sorted(set(author.values()))

    val_ids = set()
    if os.path.exists(args.val):
        v = pd.read_csv(args.val, keep_default_na=False)
        val_ids = {str(x) for x in v["row_id"]}
    in_val = np.array([r in val_ids for r in rows])

    n_empty_gold = sum(1 for g in gold if not g)
    print(f"{len(rows)} posts / {len(authors)} authors / "
          f"{len(df)} candidates ({len(df)/len(rows):.1f} per post, "
          f"{100*df['y'].mean():.1f}% positive)")
    print(f"empty-gold posts: {n_empty_gold} "
          f"({100*n_empty_gold/len(rows):.1f}%) -- each scores 1.0 if we "
          f"correctly emit nothing")
    print(f"val subset present in this pool: {int(in_val.sum())} posts")
    print()
    print("!! POOL IS 4:1 NEGATIVE-SUBSAMPLED. Every macro below is an "
          "UPPER BOUND. !!")
    print()

    # ---- reference points ---------------------------------------------------
    always_empty = phrase_f1(gold, [[] for _ in rows], average="macro").f1
    oracle_sel = [dedupe_nested(p, "long") if p else [] for p in pos]
    oracle = phrase_f1(gold, oracle_sel, average="macro").f1
    print(f"REFERENCE  always-empty        macro {always_empty:.4f}")
    print(f"REFERENCE  pool oracle (y==1)  macro {oracle:.4f}   "
          f"<- ceiling of ANY selector on this pool")
    print()

    # ---- grid ---------------------------------------------------------------
    grid = []
    for keep in ("long", "short"):
        for abst in (True, False):
            for k in (1, 2, 3, 4, 99):
                for t in np.arange(0.10, 0.96, 0.05):
                    grid.append((keep, abst, k, round(float(t), 2)))

    # per-config, per-post F1 computed ONCE; splits are then pure numpy.
    F = np.zeros((len(grid), len(rows)), dtype=float)
    for ci, (keep, abst, k, t) in enumerate(grid):
        pred = []
        for i, r in enumerate(rows):
            if abst and risk_of.get(r) == "indicator":
                pred.append([])
                continue
            sel = [s for sc, s in cand[i] if sc >= t][:k]
            pred.append(dedupe_nested(sel, keep) if sel else [])
        res = phrase_f1(gold, pred, average="macro")
        for p in res.per_post:
            F[ci, p["idx"]] = p["f1"]

    # ---- cross-fitted selection by author -----------------------------------
    rng = np.random.default_rng(args.seed)
    post_author = np.array([author[r] for r in rows])
    a_arr = np.array(authors)

    def crossfit(mask, tag):
        """Pick the config on FIT authors, score on HELD-OUT authors."""
        sub = np.where(mask)[0]
        if len(sub) == 0:
            print(f"  {tag}: no posts")
            return
        held, chosen = [], {}
        for _ in range(args.splits):
            perm = rng.permutation(len(a_arr))
            fit_a = set(a_arr[perm[:len(a_arr) // 2]])
            fit_m = np.array([post_author[i] in fit_a for i in sub])
            if fit_m.all() or not fit_m.any():
                continue
            fi, hi = sub[fit_m], sub[~fit_m]
            ci = int(np.argmax(F[:, fi].mean(1)))
            held.append(F[ci, hi].mean())
            chosen[grid[ci]] = chosen.get(grid[ci], 0) + 1
        held = np.array(held)
        best = max(chosen, key=chosen.get)
        print(f"  {tag}")
        print(f"    held-out macro   {held.mean():.4f}  "
              f"sd {held.std(ddof=1):.4f}  over {len(held)} splits")
        print(f"    modal config     keep={best[0]} abstain={best[1]} "
              f"k={best[2]} t={best[3]}  chosen {chosen[best]}/{len(held)}")
        return held.mean()

    print("CROSS-FITTED (config chosen on FIT authors, scored on HELD-OUT):")
    all_m = crossfit(np.ones(len(rows), bool), "all posts")
    print()
    val_m = crossfit(in_val, f"val subset ({int(in_val.sum())} posts)") \
        if in_val.any() else None

    # ---- verdict ------------------------------------------------------------
    print()
    print("=" * 70)
    ref = val_m if val_m is not None else all_m
    where = "val subset" if val_m is not None else "all posts"
    print(f"UPPER BOUND on {where}: {ref:.4f}   RC7 incumbent: {args.rc7:.4f}")
    if ref < args.rc7:
        print()
        print("VERDICT: LANE IS DEAD. This is an upper bound on a pool with")
        print("two-thirds of the distractors removed, and it ALREADY loses to")
        print("RC7. The honest number on the full pool can only be lower.")
        print("Do not spend GPU on rebuilding the pool.")
    else:
        print()
        print(f"VERDICT: SURVIVES by {ref - args.rc7:+.4f}. This proves nothing")
        print("on its own -- it is an inflated bound. It earns exactly one")
        print("thing: the right to build the HONEST pool and re-score.")
        print("  1. python src/evidence_enum_candidates.py --data "
              "artifacts/val_split.csv \\")
        print("         --lo 2 --hi 8 --no-subsample --out "
              "artifacts/evidence_oof/enum_val_candidates.csv")
        print("  2. GPU re-score with the f0..f4 checkpoints, then re-run this.")
        print("Then divide any val win by ~2.8 for val->test attenuation")
        print("before it earns a submission slot.")
    if oracle < args.rc7:
        print()
        print(f"NOTE: even the pool ORACLE ({oracle:.4f}) is below RC7. No")
        print("selector over these candidates can win; the lane is generation-")
        print("limited, not selection-limited.")
    print("=" * 70)


if __name__ == "__main__":
    main()
