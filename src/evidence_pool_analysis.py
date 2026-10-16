"""
evidence_pool_analysis.py

Diagnose what is missing from the evidence candidate POOL, and what would put
it there.

The decode can only choose among candidates it is given, so the distance
between the shipped score and the best-subset oracle of the pool bounds what
any selection rule can win. Gold spans that no tagger ever proposes are outside
that bound entirely. This script measures the POOL, not the decode:

  1. EVERY unmatched gold span, categorised:
       TRUNCATED  past the tokenizer cutoff -- unreachable at any training
                  budget; only a longer context window helps
       CAP        a candidate CONTAINS it but exceeds 3x its tokens. The
                  information is in the pool; a trimmed variant would match
       PARTIAL    a candidate overlaps it, neither contains the other
       ABSENT     nothing in the pool is close -- a real model failure
     Only ABSENT justifies a better tagger. TRUNCATED justifies longer context.
     CAP justifies pool augmentation, which is tested below.

  2. DOES min_conf STARVE THE POOL? If the best-subset oracle keeps rising as
     the confidence floor drops, the pool is artificially small and the
     measured ceiling was an artefact of the gate. If it plateaus, the floor is
     already generous and lowering it only adds noise.

  3. POOL AUGMENTATION WITH TRIMMED VARIANTS. For each candidate, also offer
     its stopword-stripped core and its head/tail sub-spans. Costs no GPU. A
     long candidate can fail the scorer's 3x token cap against a short gold
     span while a trimmed form of the same candidate passes it, so trimming
     makes short golds reachable. Reported as an ORACLE first -- whether a
     decode can exploit it is a separate question, and the ceiling has to be
     established before a rule is worth writing.

Usage:
  python src/evidence_pool_analysis.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     normalize_label, normalize_span, parse_evidence,
                     token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
STOP = set("i my me a an the and or but to of in it is am was be been so that "
           "this if for on with just not no do dont don't have has had will "
           "would can cant can't im i'm you your he she they we at as all any "
           "what when how why like really very much more too now then".split())


def f1_of(gold, pred):
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    adj = [[j for j, g in enumerate(gold) if _span_pair_matches(p, g)]
           for p in pred]
    m = _max_bipartite_matching(adj, len(gold))
    pr, rc = m / len(pred), m / len(gold)
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)


def dedupe_nested(spans):
    out = []
    for s in sorted(spans, key=lambda x: -token_len(x)):
        n = normalize_span(s)
        if n and not any(n in normalize_span(k) for k in out):
            out.append(s)
    return out


def best_subset(cands, gold, cap=13):
    if not gold or not cands:
        return []
    if len(cands) <= cap:
        best, bs = 0.0, []
        for k in range(1, len(cands) + 1):
            for c in combinations(cands, k):
                v = f1_of(gold, list(c))
                if v > best:
                    best, bs = v, list(c)
            if best == 1.0:
                break
        return bs
    cur, best, pool = [], 0.0, list(cands)
    while pool:
        v, s = max(((f1_of(gold, cur + [x]), x) for x in pool),
                   key=lambda z: z[0])
        if v <= best:
            break
        best, cur = v, cur + [s]
        pool.remove(s)
    return cur


def variants(s):
    """Trimmed forms of a span: stopword-stripped core, and head/tail windows.
    All are SUBSTRINGS of the original, so any that lands inside a gold span
    matches automatically -- the 3x cap cannot reject a substring."""
    w = s.split()
    out = {s}
    a, b = 0, len(w)
    while a < b and w[a].lower().strip(".,!?'\"") in STOP:
        a += 1
    while b > a and w[b - 1].lower().strip(".,!?'\"") in STOP:
        b -= 1
    if b - a >= 1 and (a or b != len(w)):
        out.add(" ".join(w[a:b]))
    for L in (len(w) - 1, len(w) - 2):
        if L >= 2:
            out.add(" ".join(w[:L]))
            out.add(" ".join(w[-L:]))
    return [x for x in out if x.strip()]


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--cache", default="artifacts/evcond_val_spans.json")
    a_.add_argument("--runs", default="runs/evcond5_cond_f0")
    a_.add_argument("--min-conf", type=float, default=0.35)
    a_.add_argument("--votes", type=int, default=2)
    a_.add_argument("--show", type=int, default=14)
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

    def pool_at(mc):
        P = []
        for i, r in enumerate(rows):
            lv = pred_risk.get(r, "indicator")
            if lv == "indicator":
                P.append([])
                continue
            agg = {}
            for s, cf in S[r][lv]:
                if cf < mc:
                    continue
                n = normalize_span(s)
                if n:
                    agg.setdefault(n, s)
            P.append(list(agg.values()))
        return P

    POOL = pool_at(args.min_conf)
    SHIP = []
    for i, r in enumerate(rows):
        lv = pred_risk.get(r, "indicator")
        if lv == "indicator":
            SHIP.append([])
            continue
        c = {}
        for s, cf in S[r][lv]:
            if cf < args.min_conf:
                continue
            e = c.setdefault(normalize_span(s), [0, s])
            e[0] += 1
        SHIP.append(dedupe_nested([v[1] for v in c.values()
                                   if v[0] >= args.votes]))
    base = float(np.mean([f1_of(g, p) for g, p in zip(gold, SHIP)]))
    print(f"{len(rows)} val posts   shipped MACRO {base:.4f}"
          + ("" if abs(base - 0.8824) <= 0.01 else "   !! baseline drift"))

    # ---- reach of the tokenizer, for the TRUNCATED test -------------------
    reach = {}
    try:
        from transformers import AutoTokenizer
        from train_evidence import RISK_PREFIX
        cfgp = os.path.join(args.runs, "run_config.json")
        cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
        ml = int(cfg.get("max_len", 512))
        tok = AutoTokenizer.from_pretrained(
            cfg.get("model", "microsoft/deberta-v3-large"))
        for i, r in enumerate(rows):
            lv = pred_risk.get(r, "indicator")
            enc = tok(RISK_PREFIX.format(lv), posts[i],
                      truncation="only_second", max_length=ml,
                      return_offsets_mapping=True)
            sq = enc.sequence_ids()
            reach[i] = max((b for (a, b), s in
                            zip(enc["offset_mapping"], sq) if s == 1),
                           default=0)
        print(f"  tokenizer reach computed at max_len {ml}")
    except Exception as e:                                  # noqa: BLE001
        print(f"  (tokenizer unavailable: {e}) -- TRUNCATED will read 0")

    print("\n" + "=" * 76)
    print("1. EVERY UNMATCHED GOLD SPAN, CATEGORISED")
    print("=" * 76)
    cat, ex = Counter(), {k: [] for k in
                          ("TRUNCATED", "CAP", "PARTIAL", "ABSENT")}
    for i in range(len(rows)):
        for g in gold[i]:
            if any(_span_pair_matches(p, g) for p in SHIP[i]):
                continue
            pos = posts[i].find(g)
            if pos < 0:
                pos = posts[i].lower().find(g.lower())
            if reach and pos >= 0 and pos + len(g) > reach.get(i, 10 ** 9):
                k = "TRUNCATED"
                note = f"char {pos} > reach {reach.get(i)}"
            else:
                ng = normalize_span(g)
                capfail = [p for p in POOL[i] if ng in normalize_span(p)]
                overlap = [p for p in POOL[i]
                           if set(ng.split()) & set(normalize_span(p).split())]
                if capfail:
                    k, note = "CAP", f"{capfail[0][:50]!r}"
                elif overlap:
                    k, note = "PARTIAL", f"{overlap[0][:50]!r}"
                else:
                    k, note = "ABSENT", ""
            cat[k] += 1
            if len(ex[k]) < args.show:
                ex[k].append((rows[i], g[:60], note))
    tot = sum(cat.values())
    for k in ("TRUNCATED", "CAP", "PARTIAL", "ABSENT"):
        print(f"  {k:<11}{cat[k]:>4}{cat[k]/max(tot,1):>8.1%}  "
              f"{'#' * int(36 * cat[k] / max(tot, 1))}")
    print(f"\n  {tot} gold spans unmatched out of "
          f"{sum(len(g) for g in gold)}")
    print("  TRUNCATED -> longer context.  CAP -> pool augmentation (free).")
    print("  PARTIAL -> boundary quality.  ABSENT -> a genuinely better model.")
    for k in ("TRUNCATED", "CAP", "PARTIAL", "ABSENT"):
        if not ex[k]:
            continue
        print(f"\n  -- {k} --")
        for r, g, note in ex[k]:
            print(f"    [{r}] {g!r}" + (f"   near {note}" if note else ""))

    print("\n" + "=" * 76)
    print("2. DOES min_conf STARVE THE POOL?")
    print("=" * 76)
    print(f"  {'min_conf':>9}{'cands/post':>12}{'best-subset ORACLE':>21}")
    for mc in (0.05, 0.15, 0.25, 0.35, 0.50, 0.65):
        P = pool_at(mc)
        nz = [i for i in range(len(rows)) if gold[i]]
        o = float(np.mean([f1_of(gold[i], best_subset(P[i], gold[i]))
                           if gold[i] else 1.0 for i in range(len(rows))]))
        print(f"  {mc:>9.2f}{np.mean([len(P[i]) for i in nz]):>12.2f}"
              f"{o:>21.4f}")
    print("\n  Rising oracle = the confidence floor is throwing away reachable")
    print("  spans, so the measured ceiling is an artefact of the gate.")
    print("  Flat = the floor is generous and lowering it only adds noise.")

    print("\n" + "=" * 76)
    print("3. POOL AUGMENTATION WITH TRIMMED VARIANTS (no GPU)")
    print("=" * 76)
    AUG = [list({v for s in POOL[i] for v in variants(s)}) if POOL[i] else []
           for i in range(len(rows))]
    o0 = float(np.mean([f1_of(gold[i], best_subset(POOL[i], gold[i]))
                        if gold[i] else 1.0 for i in range(len(rows))]))
    o1 = float(np.mean([f1_of(gold[i], best_subset(AUG[i], gold[i]))
                        if gold[i] else 1.0 for i in range(len(rows))]))
    nz = [i for i in range(len(rows)) if gold[i]]
    print(f"  pool          {np.mean([len(POOL[i]) for i in nz]):>6.2f} "
          f"cands/post   oracle {o0:.4f}")
    print(f"  + variants    {np.mean([len(AUG[i]) for i in nz]):>6.2f} "
          f"cands/post   oracle {o1:.4f}   ({o1-o0:+.4f})")
    print("\n  A substring of a gold span ALWAYS matches -- the 3x cap cannot")
    print("  reject it -- so trimmed variants can only add reachable spans.")
    print("  If this lifts the oracle materially the CAP failures are fixable")
    print("  without a GPU; if not, the pool's problem is coverage, not form.")


if __name__ == "__main__":
    main()
