"""
apply_span_reranker.py

Scores the deployed candidate pool with the span re-ranker and compares
re-ranked selection against the `vote>=2` baseline.

`train_span_reranker.py` trains a cross-encoder on conditioned candidates over
the OOF posts, author-folded. This applies it to the seven deployed taggers'
val candidates.

PROVENANCE RULE. Val authors appear in the re-ranker's own training folds, so
every candidate must be scored by the fold model that EXCLUDED its author --
read from `artifacts/span_rerank_oof_folds.csv`. Scoring a post with a model
that trained on its author leaks the label into the score and inflates the
measured gain, so the fold map is required rather than optional.

WHAT IT SWEEPS, all cross-fitted by author against `vote>=2`:

  rerank >= t        pure re-ranking, ignore the vote entirely
  top-N by rerank    keep the N best-scoring spans per post
  votes>=2 OR rerank>=t   rescue single-tagger spans the re-ranker likes --
                     candidates only one tagger emitted, which a vote can
                     never admit however good they are.
  votes>=2 AND rerank>=t  tighten instead: drop voted spans the re-ranker
                     distrusts.

`votes OR rerank` is the only rule here that can reach spans the vote
structurally cannot.

Usage (GPU preferred, CPU works -- roughly 800 pairs):
  python src/apply_span_reranker.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     normalize_label, normalize_span, parse_evidence,
                     token_len)

RISK_PREFIX = "The suicide risk level of this post is {}."
RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]


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


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--dir", default="artifacts/evidence_runs")
    a_.add_argument("--runs", default=",".join(RC7))
    a_.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    a_.add_argument("--ckpt", default="runs/span_rerank_f{f}")
    a_.add_argument("--folds", default="artifacts/span_rerank_oof_folds.csv")
    a_.add_argument("--cache", default="artifacts/span_rerank_val.csv")
    a_.add_argument("--max-len", type=int, default=384)
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold = [parse_evidence(v) for v in va["evidence"]]
    posts = dict(zip(rows, va["post"].astype(str)))
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    amap = dict(zip(rows, auth))
    rk = pd.read_csv(args.risk, keep_default_na=False)
    rk["row_id"] = rk["row_id"].astype(str)
    col = "pred_risk_level" if "pred_risk_level" in rk.columns else "risk_level"
    risk = {r: normalize_label(v) for r, v in zip(rk["row_id"], rk[col])}

    # candidates with vote counts
    agg = {}
    for run in [x.strip() for x in args.runs.split(",")]:
        p = os.path.join(args.dir, f"{run}_val_candidates.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        for r, s in zip(d["row_id"], d["span"]):
            n = normalize_span(str(s))
            if not n:
                continue
            e = agg.setdefault((r, n), {"row_id": r, "span": str(s), "n": 0})
            e["n"] += 1
    cand = pd.DataFrame(list(agg.values()))
    cand = cand[cand.row_id.isin(rows)].reset_index(drop=True)
    print(f"{len(rows)} val posts, {len(cand)} distinct candidates "
          f"({len(cand)/len(rows):.2f}/post)")

    # ---- score with the fold model that EXCLUDED each author -------------
    if os.path.exists(args.cache):
        sc = pd.read_csv(args.cache)
        sc["row_id"] = sc["row_id"].astype(str)
        cand = cand.merge(sc[["row_id", "span", "rerank"]],
                          on=["row_id", "span"], how="left")
        print(f"  reusing {args.cache}")
    else:
        fd = pd.read_csv(args.folds)
        fmap = dict(zip(fd["author"].astype(str), fd["fold"].astype(int)))
        miss = {a for a in auth if a not in fmap}
        if miss:
            raise SystemExit(
                f"{len(miss)} val authors absent from {args.folds}. Scoring "
                f"them with an arbitrary fold could use a model that trained "
                f"on that author -- refusing.")
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  scoring on {dev}")
        cand["fold"] = [fmap[amap[r]] for r in cand.row_id]
        out = np.full(len(cand), np.nan)
        for f in sorted(cand.fold.unique()):
            d = args.ckpt.format(f=f)
            if not os.path.isdir(d):
                raise SystemExit(f"missing {d}")
            tok = AutoTokenizer.from_pretrained(d)
            m = AutoModelForSequenceClassification.from_pretrained(d).to(dev)
            m.eval()
            ix = np.flatnonzero((cand.fold == f).to_numpy())
            with torch.no_grad():
                for a in range(0, len(ix), 32):
                    b = ix[a:a + 32]
                    enc = tok([RISK_PREFIX.format(risk.get(cand.row_id[i],
                                                           "ideation"))
                               + " " + posts[cand.row_id[i]] for i in b],
                              [cand.span[i] for i in b],
                              truncation="only_first", max_length=args.max_len,
                              padding=True, return_tensors="pt").to(dev)
                    out[b] = torch.sigmoid(
                        m(**enc).logits.squeeze(-1)).float().cpu().numpy()
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"    fold {f}: {len(ix)} candidates", flush=True)
        cand["rerank"] = out
        cand[["row_id", "span", "n", "rerank"]].to_csv(args.cache, index=False)
        print(f"  wrote {args.cache}")

    y = np.array([int(any(_span_pair_matches(s, g)
                          for g in gold[rows.index(r)]))
                  for r, s in zip(cand.row_id, cand.span)])

    def auc(s, p):
        p = np.asarray(p, bool)
        if p.sum() == 0 or (~p).sum() == 0:
            return float("nan")
        rr = np.argsort(np.argsort(s)) + 1.0
        return ((rr[p].sum() - p.sum() * (p.sum() + 1) / 2)
                / (p.sum() * (~p).sum()))
    print(f"\n  AUC   rerank {auc(cand.rerank.to_numpy(), y):.3f}"
          f"   n_votes {auc(cand.n.to_numpy(), y):.3f}")

    by_post = {r: [] for r in rows}
    for r, s, nv, sc_ in zip(cand.row_id, cand.span, cand.n, cand.rerank):
        by_post[r].append((s, int(nv), float(sc_)))

    def build(rule, t, N=None):
        out = []
        for r in rows:
            if risk.get(r) == "indicator":
                out.append([])
                continue
            c = by_post[r]
            if rule == "vote":
                k = [s for s, nv, _ in c if nv >= 2]
            elif rule == "rerank":
                k = [s for s, _, sc_ in c if sc_ >= t]
            elif rule == "topN":
                k = [s for s, _, _ in sorted(c, key=lambda x: -x[2])[:N]
                     if dict((a, b) for a, _, b in c)[s] >= t]
            elif rule == "or":
                k = [s for s, nv, sc_ in c if nv >= 2 or sc_ >= t]
            else:
                k = [s for s, nv, sc_ in c if nv >= 2 and sc_ >= t]
            out.append(dedupe_nested(k))
        return out

    def macro(pred, ix=None):
        ix = range(len(rows)) if ix is None else ix
        return float(np.mean([f1_of(gold[i], pred[i]) for i in ix]))

    base = build("vote", 0)
    print(f"\n  BASELINE vote>=2: macro {macro(base):.4f}")

    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    subs = []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        n = max(int(len(o) * 0.25), 2)
        ho = set(o[:n])
        hm = np.array([a in ho for a in auth])
        subs.append((np.flatnonzero(~hm), np.flatnonzero(hm)))

    GRID = np.round(np.arange(0.1, 0.95, 0.05), 2)
    print("\n" + "=" * 72)
    print("SELECTION RULES -- threshold chosen on FIT, scored on HELD-OUT")
    print("=" * 72)
    print(f"  {'rule':<16}{'d evidence':>12}{'d comp':>9}{'win':>6}"
          f"{'LOSS':>6}{'sd':>9}{'t chosen':>13}")
    cache = {}
    for rule in ("rerank", "or", "and"):
        for t in GRID:
            cache[(rule, t)] = build(rule, t)
        ds, ts = [], []
        for fi, hi in subs:
            bt = max(GRID, key=lambda t: macro(cache[(rule, t)], fi))
            ds.append(macro(cache[(rule, bt)], hi) - macro(base, hi))
            ts.append(bt)
        d = np.array(ds)
        print(f"  {rule:<16}{d.mean():>+12.4f}{0.3*d.mean():>+9.4f}"
              f"{np.mean(d > 0):>5.0%}{np.mean(d < 0):>6.0%}"
              f"{0.3*d.std():>9.4f}"
              f"{f'{np.mean(ts):.2f}+-{np.std(ts):.2f}':>13}")
    print("\n  `or` is the only rule that can admit spans only ONE tagger")
    print("  proposed; a vote>=2 rule can never reach them, however good")
    print("  the re-ranker judges them to be.")


if __name__ == "__main__":
    main()
