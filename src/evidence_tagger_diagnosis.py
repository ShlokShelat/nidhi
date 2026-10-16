"""
evidence_tagger_diagnosis.py

WHAT, EXACTLY, WOULD MAKE THE EVIDENCE TAGGER BETTER?

The tagger is responsible for more than the zero-scoring posts suggested. At
votes>=2 on 372 val posts, macro is 0.8824, so the gap to perfect is 0.1176:

    31 posts scoring 0        0.0833   (27 of them are RISK errors -> 0.0108)
    42 posts scoring 0<F1<1   0.0343   (ALL tagger error)
                              ------
    tagger-attributable       ~0.045 evidence composite-weighted

So a perfect tagger is worth meaningfully more than the ~1% an earlier pass
claimed by counting only total failures. The partial-credit posts are where
its errors live and they were never examined.

THIS SCRIPT ANSWERS THREE THINGS, EACH DECIDING A DIFFERENT TRAINING CHOICE:

  1. TRUNCATION. The tagger encodes the risk prefix as sequence 1 and the post
     as sequence 2 with truncation="only_second" at max_len. Any gold span past
     the cutoff is UNREACHABLE -- a hard recall ceiling no amount of training
     can lift. Measured with the tagger's OWN tokenizer, not a character
     heuristic. If gold spans are being cut, a longer-context run has a
     measurable target; if not, max_len is ruled out for good.

  2. PRECISION vs RECALL on the 42 partial posts. Is the tagger MISSING gold
     spans (recall -> train for coverage, lower the decision threshold, more seeds)
     or emitting SURPLUS ones (precision -> tighten)? These need opposite
     fixes and the aggregate F1 hides which.

  3. THE CANDIDATE ORACLE. Before any span is voted on, each tagger proposes
     candidates. If the RIGHT span is already among the candidates and the
     vote discards it, that is a selection failure fixable for free. If the
     right span is never proposed, only a better model helps. This is the
     generation-vs-selection split, and it is the single most important number
     for deciding whether to spend GPU on training.

Nothing here needs a GPU: the tokenizer is CPU, and the candidates are cached.

Usage:
  python src/evidence_tagger_diagnosis.py
  python src/evidence_tagger_diagnosis.py --max-len 512
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     normalize_label, normalize_span, parse_evidence,
                     token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def post_prf(gold, pred):
    if not gold and not pred:
        return 1.0, 1.0, 1.0, 0
    if not gold or not pred:
        return 0.0, 0.0, 0.0, 0
    adj = [[j for j, g in enumerate(gold) if _span_pair_matches(p, g)]
           for p in pred]
    m = _max_bipartite_matching(adj, len(gold))
    pr, rc = m / len(pred), m / len(gold)
    f = 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)
    return pr, rc, f, m


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--cache", default="artifacts/evcond_val_spans.json")
    a_.add_argument("--runs", default="runs/evcond5_cond_f0")
    a_.add_argument("--max-len", type=int, default=0,
                    help="0 = read it from the run's run_config.json")
    a_.add_argument("--min-conf", type=float, default=0.35)
    a_.add_argument("--votes", type=int, default=2)
    a_.add_argument("--show", type=int, default=8)
    args = a_.parse_args()

    S = json.load(open(args.cache))
    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    va = va[va["row_id"].isin(S)].reset_index(drop=True)
    rows = va["row_id"].tolist()
    gold = [parse_evidence(v) for v in va["evidence"]]
    posts = list(va["post"].astype(str))
    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pred_risk = {r: normalize_label(v)
                 for r, v in zip(pv["row_id"], pv["pred_risk_level"])}

    # ---------------------------------------------------------------- 1
    print("=" * 78)
    print("1. TRUNCATION -- are gold spans past max_len, hence unreachable?")
    print("=" * 78)
    ml, base_model = args.max_len, "microsoft/deberta-v3-large"
    cfg = os.path.join(args.runs, "run_config.json")
    if os.path.exists(cfg):
        c = json.load(open(cfg))
        ml = ml or int(c.get("max_len", 512))
        base_model = c.get("model", base_model)
    ml = ml or 512
    try:
        from transformers import AutoTokenizer
        from train_evidence import RISK_PREFIX
        tok = AutoTokenizer.from_pretrained(base_model)
        print(f"  tokenizer {base_model}   max_len {ml}   "
              f"truncation=only_second")
        cut, lost, tot_spans, long_posts = 0, 0, 0, 0
        worst = []
        for i, r in enumerate(rows):
            lv = pred_risk.get(r, "indicator")
            enc = tok(RISK_PREFIX.format(lv), posts[i],
                      truncation="only_second", max_length=ml,
                      return_offsets_mapping=True)
            seq = enc.sequence_ids()
            offs = [o for o, s in zip(enc["offset_mapping"], seq) if s == 1]
            reach = max((b for a, b in offs), default=0)
            full = len(tok(RISK_PREFIX.format(lv), posts[i],
                           truncation="do_not_truncate")["input_ids"])
            if full > ml:
                long_posts += 1
            for g in gold[i]:
                tot_spans += 1
                pos = posts[i].find(g)
                if pos < 0:
                    pos = posts[i].lower().find(g.lower())
                if pos >= 0 and pos + len(g) > reach:
                    lost += 1
                    if len(worst) < args.show:
                        worst.append((r, full, pos, g[:70]))
            if reach < len(posts[i]):
                cut += 1
        print(f"  posts exceeding max_len            {long_posts:>4} / "
              f"{len(rows)}  ({long_posts/len(rows):.1%})")
        print(f"  posts whose text is cut at all     {cut:>4}")
        print(f"  GOLD SPANS beyond the cutoff       {lost:>4} / {tot_spans}"
              f"  ({lost/max(tot_spans,1):.1%})   <- the hard ceiling")
        if worst:
            print("\n  examples of unreachable gold spans:")
            for r, full, pos, g in worst:
                print(f"    [{r}] post {full} tokens, span at char {pos}: "
                      f"{g!r}")
        if lost == 0:
            print("\n  ZERO gold spans lost. max_len is RULED OUT -- a longer")
            print("  context window cannot help, do not spend GPU on it.")
        else:
            print(f"\n  {lost} spans are unreachable at any training budget.")
            print("  Upper bound on a longer-context run, and it is a RECALL")
            print("  ceiling: those posts can never score 1.0 today.")
    except Exception as e:                                  # noqa: BLE001
        print(f"  could not run the tokenizer check: {e}")

    # spans at the shipped decode
    def build(votes, min_conf):
        out = []
        for r in rows:
            lv = pred_risk.get(r, "indicator")
            if lv == "indicator":
                out.append([])
                continue
            c = {}
            for s, cf in S[r][lv]:
                if cf < min_conf:
                    continue
                e = c.setdefault(normalize_span(s), [0, s])
                e[0] += 1
            keep = sorted([v[1] for v in c.values() if v[0] >= votes],
                          key=lambda x: -token_len(x))
            o = []
            for s in keep:
                n = normalize_span(s)
                if n and not any(n in normalize_span(k) for k in o):
                    o.append(s)
            out.append(o)
        return out

    P = build(args.votes, args.min_conf)

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 78)
    print("2. THE PARTIAL-CREDIT POSTS -- missing spans or surplus ones?")
    print("=" * 78)
    stats = [post_prf(g, p) for g, p in zip(gold, P)]
    part = [i for i, (pr, rc, f, m) in enumerate(stats) if 0 < f < 1]
    print(f"  {len(part)} posts score strictly between 0 and 1")
    if part:
        pr = np.mean([stats[i][0] for i in part])
        rc = np.mean([stats[i][1] for i in part])
        surplus = sum(max(len(P[i]) - stats[i][3], 0) for i in part)
        missed = sum(max(len(gold[i]) - stats[i][3], 0) for i in part)
        print(f"  mean precision {pr:.3f}   mean recall {rc:.3f}")
        print(f"  surplus predicted spans (unmatched) {surplus}")
        print(f"  gold spans never matched             {missed}")
        if missed > surplus * 1.3:
            print("\n  RECALL-bound: gold spans are being missed. Fixes are more")
            print("  seeds, a lower decision threshold, a looser vote.")
        elif surplus > missed * 1.3:
            print("\n  PRECISION-bound: spans are emitted that the annotator")
            print("  did not write.")
            print("  Fixes are a tighter vote or a higher threshold -- but")
            print("  note a stricter vote threshold measurably costs accuracy,")
            print("  so it is not free.")
        else:
            print("\n  BALANCED: precision and recall errors are comparable,")
            print("  so no threshold move helps -- both sides move together.")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 78)
    print("3. GENERATION vs SELECTION -- is the right span already proposed?")
    print("=" * 78)
    gen_ok, sel_lost, never = 0, 0, 0
    miss_examples = []
    for i, r in enumerate(rows):
        lv = pred_risk.get(r, "indicator")
        if lv == "indicator" or not gold[i]:
            continue
        allc = [s for s, cf in S[r][lv]]
        chosen = P[i]
        for g in gold[i]:
            in_chosen = any(_span_pair_matches(p, g) for p in chosen)
            in_any = any(_span_pair_matches(s, g) for s in allc)
            if in_chosen:
                gen_ok += 1
            elif in_any:
                sel_lost += 1
            else:
                never += 1
                if len(miss_examples) < args.show:
                    miss_examples.append((r, g[:80],
                                          [s[:40] for s in allc[:3]]))
    tot = gen_ok + sel_lost + never
    print(f"  gold spans matched by the shipped decode   {gen_ok:>4}"
          f"  {gen_ok/max(tot,1):>6.1%}")
    print(f"  PROPOSED by some tagger but not selected   {sel_lost:>4}"
          f"  {sel_lost/max(tot,1):>6.1%}   <- FREE if selection improves")
    print(f"  never proposed by ANY tagger               {never:>4}"
          f"  {never/max(tot,1):>6.1%}   <- needs a better MODEL")
    print("\n  This is the number that decides whether to spend GPU. A large")
    print("  `not selected` share means the information already exists and the")
    print("  vote is discarding it. A large `never proposed` share means the")
    print("  taggers genuinely cannot find these clauses and only training")
    print("  helps -- and training on train_split IS validatable on val_split,")
    print("  so it can be measured honestly before shipping.")
    if miss_examples:
        print("\n  gold spans NO tagger ever proposed:")
        for r, g, cs in miss_examples:
            print(f"    [{r}] gold {g!r}")
            print(f"          proposed: {cs}")


if __name__ == "__main__":
    main()
