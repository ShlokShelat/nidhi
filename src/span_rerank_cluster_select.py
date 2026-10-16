#!/usr/bin/env python
"""CLUSTER-THEN-RANK selection over the enumerated pool. CPU, no GPU.

WHY THIS SHAPE. `evidence_true_oracle.py` proved the pool's exact ceiling is
0.9864 on val against RC7's 0.8045 -- the candidates are essentially perfect
and the entire evidence gap is SELECTION. `span_rerank_upper_bound.py` reached
only 0.7494 with a flat `score >= t` rule, and the reason is structural:

    79 candidates/post at 21.3% positive = ~17 "positive" candidates
    against ~1.8 gold spans.

`y==1` means "matches SOME gold", not "should be emitted". Seventeen
overlapping n-grams all match the same one or two spans, so a threshold rule
emits redundant variants at ~10% precision. The reranker's AUC 0.890 was never
the problem. RC7 wins because it CLUSTERS and emits one representative per
cluster; the reranker was plugged into the wrong decision rule.

THE RULE HERE, which mirrors the oracle's structure (one candidate per gold):

    1. cluster a post's candidates by overlap (union-find on containment or
       high token-Jaccard) -- one cluster should correspond to one gold span
    2. pick ONE representative per cluster
    3. rank clusters by score, emit the top k
    4. abstain entirely when the best cluster looks weak

Every knob is swept and chosen on FIT authors, scored on HELD-OUT authors, as
`ci-of-the-mean-error` requires -- report WIN RATE and per-split sd, never the
CI of the mean.

STILL AN UPPER BOUND: the pool is 4:1 negative-subsampled, so precision here
is optimistic. Beating RC7 is necessary, not sufficient. Divide any val win by
~2.8 for val->test attenuation before it earns a submission slot.
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


def cluster(spans, jac=0.5):
    """Union-find over containment OR token-Jaccard >= jac.

    One cluster should stand for one gold span. Containment alone is too
    strict on an n-gram pool: two windows over the same phrase can overlap
    heavily without either containing the other.
    """
    n = len(spans)
    toks = [set(norm(s).split()) for s in spans]
    ns = [norm(s) for s in spans]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if ns[i] in ns[j] or ns[j] in ns[i]:
                same = True
            else:
                u = len(toks[i] | toks[j])
                same = u > 0 and len(toks[i] & toks[j]) / u >= jac
            if same:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="artifacts/span_rerank_enum_oof.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--rc7", type=float, default=0.8045)
    ap.add_argument("--jac", type=float, default=0.5,
                    help="token-Jaccard at which two candidates join one "
                         "cluster. Too high fragments one gold into several "
                         "clusters (redundant emissions); too low merges two "
                         "distinct golds into one (caps recall).")
    ap.add_argument("--prefilter", type=float, default=0.10,
                    help="drop candidates below this score BEFORE clustering; "
                         "keeps the O(n^2) union-find cheap and stops obvious "
                         "junk from bridging two real clusters")
    args = ap.parse_args()

    df = pd.read_csv(args.scores, keep_default_na=False)
    df["row_id"] = df["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    gold_by = {r: parse_evidence(e)
               for r, e in zip(lab["row_id"], lab["evidence"])}

    per = {}
    risk_of, author = {}, {}
    for r, s, sc, rk, a in zip(df["row_id"], df["span"], df["rerank"],
                               df["risk"], df["author"]):
        per.setdefault(r, []).append((float(sc), str(s)))
        risk_of[r] = str(rk)
        author[r] = str(a)

    rows = sorted(per)
    gold = [gold_by.get(r, []) for r in rows]

    val_ids = set()
    if os.path.exists(args.val):
        v = pd.read_csv(args.val, keep_default_na=False)
        val_ids = {str(x) for x in v["row_id"]}
    in_val = np.array([r in val_ids for r in rows])

    # ---- cluster once per post; representatives are cheap variants ----------
    print(f"clustering {len(rows)} posts (prefilter {args.prefilter}) ...")
    post_cl = []
    for r in rows:
        items = [(sc, s) for sc, s in per[r] if sc >= args.prefilter]
        if not items:
            items = sorted(per[r], reverse=True)[:1]
        spans = [s for _, s in items]
        scores = [sc for sc, _ in items]
        groups = cluster(spans, args.jac)
        cl = []
        for g in groups:
            # cluster score = its best candidate. Three representative rules:
            #   best   -- highest-scoring member
            #   short  -- fewest tokens (the 3x cap punishes long preds)
            #   long   -- most tokens (what the old dedupe did)
            best = max(g, key=lambda i: scores[i])
            short = min(g, key=lambda i: len(spans[i].split()))
            longi = max(g, key=lambda i: len(spans[i].split()))
            cl.append({"score": scores[best],
                       "best": spans[best],
                       "short": spans[short],
                       "long": spans[longi]})
        cl.sort(key=lambda d: -d["score"])
        post_cl.append(cl)

    ncl = np.mean([len(c) for c in post_cl])
    ng = np.mean([len(g) for g in gold])
    print(f"  mean clusters/post {ncl:.1f}   mean gold spans/post {ng:.2f}")
    if ncl < ng:
        print("  WARNING: fewer clusters than gold spans -- clustering is "
              "merging distinct golds and caps recall. Lower --jac.")

    # ---- grid ---------------------------------------------------------------
    # THE FIRST GRID HIT ITS OWN BOUNDARY: it chose t=0.95, the maximum, and
    # never abstained. The reranker trained to loss ~0.008, so nearly every
    # positive scores ~0.99 and an absolute threshold barely separates
    # anything. An ABSOLUTE cut is the wrong parameterisation for a saturated
    # scorer -- WITHIN-POST RANK is. Two modes are swept:
    #   abs  score >= t                      (extended to 0.999, log-spaced
    #                                         near the top where the mass is)
    #   rel  score >= alpha * best_in_post   (scale-free; survives saturation)
    T_ABS = [round(float(x), 4) for x in np.arange(0.10, 0.96, 0.05)] + \
            [0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.998, 0.999]
    T_REL = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 1.0]
    # abstain on the BEST cluster's score, also extended into the saturated
    # region -- 36% of posts have empty gold and score 1.0 for emitting nothing
    TMIN = [0.0, 0.5, 0.9, 0.98, 0.99, 0.995, 0.998]

    grid = []
    for rep in ("best", "short", "long"):
        for kmax in (1, 2, 3, 4):
            for mode, ts in (("abs", T_ABS), ("rel", T_REL)):
                for t in ts:
                    for tmin in TMIN:
                        grid.append((rep, kmax, mode, t, tmin))
    print(f"  grid: {len(grid)} configs "
          f"(abs thresholds to 0.999, plus scale-free relative mode)")

    F = np.zeros((len(grid), len(rows)))
    for ci, (rep, kmax, mode, t, tmin) in enumerate(grid):
        pred = []
        for cl in post_cl:
            # abstain: if even the best cluster is weak, emit nothing
            if not cl or cl[0]["score"] < tmin:
                pred.append([])
                continue
            cut = t if mode == "abs" else t * cl[0]["score"]
            pred.append([c[rep] for c in cl[:kmax] if c["score"] >= cut])
        res = phrase_f1(gold, pred, average="macro")
        for p in res.per_post:
            F[ci, p["idx"]] = p["f1"]

    # ---- cross-fit by author ------------------------------------------------
    rng = np.random.default_rng(args.seed)
    post_author = np.array([author[r] for r in rows])
    a_arr = np.array(sorted(set(post_author)))

    def crossfit(mask, tag):
        sub = np.where(mask)[0]
        held, chosen = [], {}
        for _ in range(args.splits):
            perm = rng.permutation(len(a_arr))
            fit_a = set(a_arr[perm[:len(a_arr) // 2]])
            fm = np.array([post_author[i] in fit_a for i in sub])
            if fm.all() or not fm.any():
                continue
            fi, hi = sub[fm], sub[~fm]
            ci = int(np.argmax(F[:, fi].mean(1)))
            held.append(F[ci, hi].mean())
            chosen[grid[ci]] = chosen.get(grid[ci], 0) + 1
        held = np.array(held)
        best = max(chosen, key=chosen.get)
        print(f"  {tag}")
        print(f"    held-out macro  {held.mean():.4f}  "
              f"sd {held.std(ddof=1):.4f}  over {len(held)} splits")
        print(f"    modal config    rep={best[0]} kmax={best[1]} "
              f"{best[2]}-cut {best[3]} abstain<{best[4]}   "
              f"{chosen[best]}/{len(held)}")
        # A boundary pick means the optimum is OUTSIDE the grid and the number
        # below understates what this rule can do -- the first version of this
        # sweep chose t=0.95 with 0.95 as its maximum and cost a whole round.
        edge = []
        if best[2] == "abs" and best[3] >= max(T_ABS):
            edge.append("abs threshold at grid MAX")
        if best[2] == "rel" and best[3] >= max(T_REL):
            edge.append("rel threshold at grid MAX")
        if best[1] == 4:
            edge.append("kmax at grid MAX")
        if best[4] >= max(TMIN):
            edge.append("abstain at grid MAX")
        if edge:
            print(f"    !! BOUNDARY: {', '.join(edge)} -- extend the grid; "
                  f"this result is a floor, not the rule's best")
        return held.mean()

    print()
    print("CROSS-FITTED (config on FIT authors, scored on HELD-OUT):")
    allm = crossfit(np.ones(len(rows), bool), "all posts")
    print()
    valm = crossfit(in_val, f"val subset ({int(in_val.sum())})") \
        if in_val.any() else None

    ref = valm if valm is not None else allm
    print()
    print("=" * 68)
    print(f"cluster-then-rank {ref:.4f}   RC7 {args.rc7:.4f}   "
          f"oracle 0.9864")
    print(f"flat-threshold reranker was 0.7494 -- delta {ref-0.7494:+.4f}")
    if ref > args.rc7:
        print()
        print(f"BEATS RC7 by {ref-args.rc7:+.4f} on an INFLATED pool. Next: "
              f"rebuild the")
        print("honest pool (evidence_enum_candidates.py --no-subsample), "
              "re-score on")
        print("GPU, and re-run this. Divide by ~2.8 for val->test before "
              "believing it.")
    else:
        print()
        print("Still below RC7 even with clustering. If the delta vs 0.7494 is")
        print("large, the structure was right and the remaining loss is the")
        print("abstain/k rule; if it is small, the reranker's ranking does not")
        print("survive set construction and the lane closes.")
    print("=" * 68)


if __name__ == "__main__":
    main()
