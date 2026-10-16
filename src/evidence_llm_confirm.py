"""
evidence_llm_confirm.py

Use the LLM's spans to CONFIRM tagger candidates, not to propose new ones.

WHY THE PROPOSER FRAMING FAILED. evidence_llm_propose.py measured, on 216 val
posts with gold spans:

    gold spans total             358
    already in the tagger pool   334  (93.3%)
    MISSED by every tagger        24  (6.7%)
    of those the LLM proposes      5  (0.208)

The bar was "recover >= 20% of missed spans", which sounded reasonable until the
denominator was measured: 20% of 6.7% is FIVE spans out of 358. Buying those
would mean injecting 447 LLM proposals at precision 0.523 into the vote. The
pool was never the constraint.

WHY CONFIRMATION IS A DIFFERENT MECHANISM. The closed selection attempts
all trained or prompted a model to SCORE candidates, and all failed because gold
spans are annotator conventions a zero-shot model has no access to. This asks
nothing of the LLM's judgement -- only whether it independently quoted the same
clause. That is agreement counting, which is exactly what made cluster voting
work: a span two independent systems both emit is likelier real than one a
single tagger found.

The shipped rule needs 2 of 8 taggers. Here an LLM quote can substitute for the
second tagger. Recall has room: the pool holds 93.3% of gold and the shipped
vote converts ~70% of it.

RULES, all zero-parameter:
    k2            shipped: >= 2 taggers agree (baseline)
    k2_or_llm     >= 2 taggers, OR 1 tagger + an LLM quote
    k3_or_k2llm   >= 3 taggers, OR 2 taggers + an LLM quote  (stricter)
    llm_only_add  k2, plus LLM quotes matching NO tagger candidate (the
                  proposer framing, kept as a control -- expected to lose)

Agreement uses the SCORER'S containment+3x equivalence, not string equality --
the same relation cluster voting uses. Scored MACRO with the shipped suppression
rule, paired author bootstrap.

BAR: a positive macro gain with the 95% CI excluding zero. Evidence is 0.30 of
the composite, so that maps to a small composite gain -- worth a slot only
because it is a decision rule with nothing fitted.

Usage:
  python src/evidence_llm_confirm.py
  python src/evidence_llm_confirm.py --llm artifacts/evllm_qwen2.5-72b-instruct_gold.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_span_pair_matches, normalize_label,  # noqa: E402
                     normalize_span, parse_evidence, phrase_f1, token_len)


def equiv(a: str, b: str) -> bool:
    return _span_pair_matches(a, b) or _span_pair_matches(b, a)


def cluster(items: list[tuple[str, str]]):
    """Group (source, span) pairs under the scorer's equivalence.

    Mirrors evidence_vote_test.py's cluster_kept: longest-first then
    lexicographic, because containment is not transitive so grouping is
    order-dependent and must not depend on dict iteration order.
    """
    items = sorted(items, key=lambda x: (-token_len(x[1]),
                                         normalize_span(x[1])))
    groups: list[list[tuple[str, str]]] = []
    for src, s in items:
        for g in groups:
            if any(equiv(s, s2) for _, s2 in g):
                g.append((src, s))
                break
        else:
            groups.append([(src, s)])
    return groups


def modal(g) -> str:
    cnt: dict[str, int] = {}
    for _, s in g:
        cnt[s] = cnt.get(s, 0) + 1
    return max(cnt.items(), key=lambda kv: (kv[1], -token_len(kv[0])))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-dir", default="artifacts/evidence_runs")
    ap.add_argument("--llm",
                    default="artifacts/evllm_qwen2.5-72b-instruct_pred.csv")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--risk-from", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from data_audit import load
    va = load(args.val)
    va["row_id"] = va["row_id"].astype(str)
    rows = list(va["row_id"])
    gold = [parse_evidence(e) for e in va["evidence"]]
    print(f"{len(rows)} val posts, "
          f"{sum(1 for g in gold if not g)} with empty gold")

    # tagger candidates, at each run's own stored min_conf (the shipped view)
    import json
    per_run: dict[str, dict[str, list[str]]] = {}
    for f in sorted(glob.glob(os.path.join(args.cand_dir,
                                           "*_val_candidates.csv"))):
        run = os.path.basename(f).replace("_val_candidates.csv", "")
        cfg = os.path.join("runs", run, "run_config.json")
        mc = (float(json.load(open(cfg)).get("evidence_min_conf", 0.0))
              if os.path.exists(cfg) else 0.0)
        d = pd.read_csv(f)
        d = d[d["conf"] >= mc]
        m: dict[str, list[str]] = {r: [] for r in rows}
        for r, s in zip(d["row_id"].astype(str), d["span"]):
            if r in m:
                m[r].append(str(s))
        per_run[run] = m
    if not per_run:
        raise SystemExit(f"no *_val_candidates.csv in {args.cand_dir}")
    print(f"  {len(per_run)} tagger runs pooled")

    lc = pd.read_csv(args.llm, keep_default_na=False)
    lc["row_id"] = lc["row_id"].astype(str)
    col = "llm_verified" if "llm_verified" in lc.columns else "llm_raw"
    llm = {r: parse_evidence(v) for r, v in zip(lc["row_id"], lc[col])}
    n_cov = sum(1 for r in rows if llm.get(r))
    print(f"  LLM quotes on {n_cov} of {len(rows)} posts (from {args.llm})")

    rk = pd.read_csv(args.risk_from, keep_default_na=False)
    rcol = ("risk_level" if "risk_level" in rk.columns else "pred_risk_level")
    risk = {str(a): normalize_label(b) for a, b in zip(rk["row_id"], rk[rcol])}
    miss = [r for r in rows if r not in risk]
    if miss:
        raise SystemExit(f"{args.risk_from} lacks risk for {len(miss)} rows")

    def build(rule: str) -> list[list[str]]:
        out = []
        for r in rows:
            tag_items = [(run, s) for run in per_run for s in per_run[run][r]]
            lspans = llm.get(r, [])
            groups = cluster(tag_items + [("__LLM__", s) for s in lspans])
            kept = []
            for g in groups:
                srcs = {src for src, _ in g}
                n_tag = len(srcs - {"__LLM__"})
                has_llm = "__LLM__" in srcs
                if rule == "k2":
                    ok = n_tag >= 2
                elif rule == "k2_or_llm":
                    ok = n_tag >= 2 or (n_tag >= 1 and has_llm)
                elif rule == "k3_or_k2llm":
                    ok = n_tag >= 3 or (n_tag >= 2 and has_llm)
                elif rule == "llm_only_add":
                    ok = n_tag >= 2 or (n_tag == 0 and has_llm)
                else:
                    raise ValueError(rule)
                if ok:
                    # never emit a span only the LLM produced under the
                    # confirmation rules -- it must be a tagger's text
                    cand = [s for src, s in g if src != "__LLM__"] or \
                           [s for _, s in g]
                    kept.append(modal([("x", s) for s in cand]))
            # shipped consistency rule
            if risk[r] == "indicator":
                kept = []
            seen, ded = set(), []
            for s in kept:
                k = normalize_span(s)
                if k not in seen:
                    seen.add(k)
                    ded.append(s)
            out.append(ded)
        return out

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    amap = dict(zip(lab["row_id"], lab["anon_user_id"].astype(str)))
    author = np.array([amap.get(r, r) for r in rows])
    rng = np.random.default_rng(args.seed)
    auths = sorted(set(author))
    aidx = {a: np.where(author == a)[0] for a in auths}
    print(f"  {len(auths)} authors")

    rules = ["k2", "k2_or_llm", "k3_or_k2llm", "llm_only_add"]
    preds = {r: build(r) for r in rules}
    pp = {r: np.array([x["f1"] for x in
                       phrase_f1(gold, preds[r], average="macro").per_post])
          for r in rules}

    print(f"\n{'rule':<16}{'macro':>9}{'delta':>9}{'spans':>8}"
          f"{'95% CI':>22}")
    print("-" * 64)
    base = "k2"
    for r in rules:
        mac = pp[r].mean()
        n = sum(len(p) for p in preds[r])
        if r == base:
            print(f"{r:<16}{mac:>9.4f}{'':>9}{n:>8}")
            continue
        diff = pp[r] - pp[base]
        draws = []
        for _ in range(args.boot):
            pick = rng.choice(len(auths), len(auths), replace=True)
            sel = np.concatenate([aidx[auths[i]] for i in pick])
            draws.append(float(diff[sel].mean()))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"{r:<16}{mac:>9.4f}{mac - pp[base].mean():>+9.4f}{n:>8}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}")

    print("\nBAR: a macro gain with the CI excluding zero, mapped to composite")
    print("at the 0.30 evidence weight. `llm_only_add` is the PROPOSER framing")
    print("kept as a control -- if it wins and the confirmation rules do not,")
    print("the earlier pool-recall reading was wrong somewhere.")


if __name__ == "__main__":
    main()
