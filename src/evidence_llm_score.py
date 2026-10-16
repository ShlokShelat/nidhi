"""
evidence_llm_score.py

Graded variant of the LLM span selector. Asks the model to score each
candidate 0-100 instead of making a binary keep/drop call, then chooses the
cut-off with ONE parameter fitted on an author-disjoint half.

Why: the binary prompt (evidence_llm_reranker.py) measured on val

    LLM binary   P 0.7374  R 0.6117  F1 0.6687
    vote>=2      P 0.6799  R 0.7179  F1 0.6984
    best tagger                      F1 0.6969

Best-in-class precision, worst-in-class recall -- the model was over-
rejecting because the prompt told it to "be selective" and anchored it at
"zero to three" quotations per post. That is an operating-point error, not
evidence that the model cannot judge spans.

Scoring separates the two possibilities. Average precision measures whether
the RANKING carries signal independently of any cut-off:

  * high AP, poor binary F1  -> ranking is good, only the cut was wrong; fix
    the cut and the method works.
  * low AP                   -> the model cannot tell good spans from bad and
    the whole idea is dead, prompt wording notwithstanding.

Reported honestly: the threshold is fitted on one author half and applied to
the other, both directions. The oracle threshold is printed too, clearly
marked as circular.

Usage:
  export DEEPSEEK_API_KEY=sk-...
  python src/evidence_llm_score.py --split artifacts/val_split.csv \
      --risk-pred artifacts/val_baseline_reconstructed.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1, _span_pair_matches
from rule_ablation_v2 import author_split
from evidence_llm_reranker import (SYSTEM_SCORE, USER_SCORE, parse_scores,
                                   load_runs, candidates_for)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="artifacts/val_split.csv")
    ap.add_argument("--risk-pred", default=None)
    ap.add_argument("--cache-dir", default="artifacts/evidence_per_run")
    ap.add_argument("--tag", default="val_split")
    ap.add_argument("--cache", default="artifacts/llm_score_cache.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.split)
    gold = [parse_evidence(x) for x in df["evidence"]]
    runs = load_runs(df, args.cache_dir, args.tag)
    if not runs:
        raise SystemExit("no tagger predictions found")

    if args.risk_pred:
        rp = pd.read_csv(args.risk_pred).set_index("row_id")
        col = next(c for c in ("pred_risk_level", "champion_final", "risk_level")
                   if c in rp.columns)
        risks = [str(rp.loc[r, col]) for r in df["row_id"]]
    else:
        risks = [str(x) for x in df["risk_level"]]

    cache = {}
    if os.path.exists(args.cache):
        for line in open(args.cache, encoding="utf-8"):
            try:
                r = json.loads(line)
                cache[r["key"]] = r["response"]
            except Exception:
                pass
    print(f"cache entries: {len(cache)}")

    from deepseek_client import DeepSeekClient
    client = DeepSeekClient()

    n_posts = len(df) if not args.limit else min(args.limit, len(df))
    rows, n_calls, n_fail = [], 0, 0
    fh = open(args.cache, "a", encoding="utf-8")

    for i in range(n_posts):
        cands = candidates_for(runs, i)
        if not cands:
            continue
        # gold match label, greedy one-to-one like the official matcher
        used = set()
        for c in cands:
            c["y"] = 0
        for gi, g in enumerate(gold[i]):
            for c in cands:
                if c["y"] or gi in used:
                    continue
                if _span_pair_matches(c["span"], g):
                    c["y"] = 1
                    used.add(gi)
                    break

        listing = "\n".join(f"[{j}] {c['span']}" for j, c in enumerate(cands))
        prompt = USER_SCORE.format(risk=risks[i],
                                   post=str(df["post"].iloc[i])[:4000],
                                   cands=listing)
        key = hashlib.md5((SYSTEM_SCORE + prompt).encode("utf-8")).hexdigest()
        if key in cache:
            raw = cache[key]
        else:
            raw = client.generate(SYSTEM_SCORE, prompt)
            n_calls += 1
            fh.write(json.dumps({"key": key, "response": raw}) + "\n")
            fh.flush()

        sc = parse_scores(raw, len(cands))
        if sc is None:
            n_fail += 1
            sc = [100.0 if len(c["by"]) >= 2 else 0.0 for c in cands]
        for c, s in zip(cands, sc):
            rows.append({"post_idx": i, "span": c["span"], "y": c["y"],
                         "score": s, "n_models": len(c["by"])})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_posts} ({n_calls} new calls)")
    fh.close()

    cand = pd.DataFrame(rows)
    print(f"\ncalls {n_calls}   parse failures {n_fail}   "
          f"candidates {len(cand)}   positives {int(cand.y.sum())}")

    ap_llm = average_precision_score(cand.y, cand.score)
    ap_vote = average_precision_score(cand.y, cand.n_models)
    base = cand.y.mean()
    print(f"\nRANKING QUALITY (threshold-free)")
    print(f"  base rate                : {base:.4f}")
    print(f"  average precision, LLM   : {ap_llm:.4f}  (lift {ap_llm/base:.2f}x)")
    print(f"  average precision, votes : {ap_vote:.4f}  (lift {ap_vote/base:.2f}x)")

    half = author_split(df)[0].values
    ph = np.array([half[i] for i in cand.post_idx])
    grid = list(range(0, 101, 5))

    def decode(sub, thr):
        keep = {}
        for pi, s, sc_ in zip(sub.post_idx, sub.span, sub.score):
            if sc_ >= thr:
                keep.setdefault(pi, []).append(s)
        return keep

    def f1_on(posts, sub, thr):
        k = decode(sub, thr)
        return phrase_f1([gold[i] for i in posts],
                         [k.get(i, []) for i in posts]).f1

    picked = []
    sel = {}
    for tr, te in (("A", "B"), ("B", "A")):
        s_tr, s_te = cand[ph == tr], cand[ph == te]
        p_tr = sorted(set(s_tr.post_idx)); p_te = sorted(set(s_te.post_idx))
        best_t = max(grid, key=lambda t: f1_on(p_tr, s_tr, t))
        picked.append(best_t)
        sel.update(decode(s_te, best_t))
    all_posts = list(range(n_posts))
    spans = [sel.get(i, []) for i in all_posts]
    s = phrase_f1([gold[i] for i in all_posts], spans)

    orc = max(grid, key=lambda t: f1_on(all_posts, cand, t))
    s_orc = phrase_f1([gold[i] for i in all_posts],
                      [decode(cand, orc).get(i, []) for i in all_posts])

    print(f"\nDECODED")
    print(f"  cross-fitted (thresholds {picked}) : P {s.precision:.4f} "
          f"R {s.recall:.4f}  F1 {s.f1:.4f}")
    print(f"  oracle threshold {orc:<3} (CIRCULAR)      : F1 {s_orc.f1:.4f}")
    v2 = [[c["span"] for c in candidates_for(runs, i) if len(c["by"]) >= 2]
          for i in all_posts]
    f_v2 = phrase_f1([gold[i] for i in all_posts], v2).f1
    print(f"  vote>=2 baseline                  : F1 {f_v2:.4f}")
    print(f"\n  vs vote>=2: {s.f1 - f_v2:+.4f} phrase-F1 "
          f"= {0.3 * (s.f1 - f_v2):+.4f} composite")

    for h in ("A", "B"):
        posts = [i for i in all_posts if half[i] == h]
        print(f"  half {h}: LLM {phrase_f1([gold[i] for i in posts], [sel.get(i, []) for i in posts]).f1:.4f}"
              f"   vote>=2 {phrase_f1([gold[i] for i in posts], [v2[i] for i in posts]).f1:.4f}")

    if args.out:
        pd.DataFrame({"row_id": df["row_id"][:n_posts],
                      "evidence": ["; ".join(x) for x in spans]}).to_csv(
            args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
