"""
evidence_tagger_rank.py

THE ONE EVIDENCE LANE STILL SIZED TO THE GAP.

WHERE THIS COMES FROM. Three measurements:

  * a model-free in-sentence n-gram pool reaches oracle macro-F1 **0.984** at
    ~219 candidates/post, against ~0.89 for the argmax decode's ~1.1/post.
  * the shipped system achieves 0.776. So SELECTION loss is 0.152 and
    GENERATION loss is 0.056 -- selection is 2.7x the bigger prize.
  * a model-free ranker (TF-IDF + 9 structural features) on that pool reaches
    0.540, or 0.610 with oracle empty-detection -- and oracle-attainment
    degrades only 8.5 points across a 54x pool growth.

**The bar is therefore 0.7767 / 0.982 = 79.1% of oracle.** The model-free floor
is 55-62%. The open question is whether the seven trained taggers, used as
SPAN SCORERS rather than span generators, close that 17-24 point gap.

WHAT MAKES THIS POSSIBLE NOW. The dump pipeline records raw token posteriors
(`{run}_{split}_tokprobs.npz`) from the same forward pass that produced the
shipped candidates -- so any span can be scored by the mean P(not-O) over its
tokens without a single extra GPU second. Previously nothing in the system
could score a span the BIO decode had not already chosen.

THE KNOWN HANDICAP, STATED UP FRONT. Culotta & McCallum (2004) measured span
confidence estimators directly: an exact CRF span marginal reaches avg
precision .976, a PRODUCT/MEAN of token marginals only .912, and a Viterbi
ratio .842. The taggers here have no CRF layer, so mean P(not-O) is the weaker
estimator by construction. If this lands just short of the bar, a trained span
scorer over the same pool is the follow-up -- not a reason to abandon the lane.

WHAT IT MEASURES, author-disjoint throughout:
  1. the pool's own oracle, recomputed here so the comparison cannot drift
  2. achieved macro phrase-F1 with threshold and k tuned on FIT authors
  3. the same with the shipped suppression rule (emit nothing when predicted
     risk is indicator) -- which is how the champion actually behaves and is
     worth ~0.07 on the model-free floor
  4. the incumbent argmax pool scored by the SAME ranker, as a control: if the
     big pool does not beat the small one under one scorer, pool size is not
     the win and the whole lane closes.

Usage (CPU, a few minutes):
  python src/evidence_tagger_rank.py
  python src/evidence_tagger_rank.py --lo 2 --hi 14 --max-words 24
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span,  # noqa: E402
                     parse_evidence, phrase_f1, token_len)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]
O_IDX = 0          # train_evidence.py: O, B, I = 0, 1, 2


def matches(pred: str, gold: str) -> bool:
    p, g = normalize_span(pred), normalize_span(gold)
    if not p or not g:
        return False
    if p not in g and g not in p:
        return False
    return token_len(p) <= 3 * token_len(g)


def sent_id_of(post: str, starts: np.ndarray) -> np.ndarray:
    """Sentence index for each token, by its start character."""
    bounds, pos = [], 0
    for piece in SENT_SPLIT.split(post):
        i = post.find(piece, pos)
        if i < 0:
            i = pos
        bounds.append((i, i + len(piece)))
        pos = i + len(piece)
    sid = np.zeros(len(starts), dtype=np.int32)
    for k, s in enumerate(starts):
        sid[k] = next((j for j, (a, b) in enumerate(bounds) if a <= s < b),
                      len(bounds))
    return sid


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--dir", default="artifacts/evidence_runs_rc")
    a_.add_argument("--runs", default=",".join(RC7))
    a_.add_argument("--risk", default="artifacts/nligaur/oof_base.csv",
                    help="PREDICTED risk for the val twin. The shipped chain "
                         "suppresses evidence on indicator, so a run without "
                         "this measures a system that is not shipped.")
    a_.add_argument("--lo", type=int, default=2, help="min subword tokens")
    a_.add_argument("--hi", type=int, default=16, help="max subword tokens")
    a_.add_argument("--max-words", type=int, default=12,
                    help="whitespace-word cap on a candidate. 12 matches the "
                         "shipped decode; 56%% of gold spans have a 3x "
                         "allowance at or below it, so raising it buys little")
    a_.add_argument("--pool-cap", type=int, default=40,
                    help="top-N candidates per post kept for selection")
    a_.add_argument("--max-emit", type=int, default=4)
    a_.add_argument("--splits", type=int, default=5)
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    from data_audit import load
    va = load(args.val)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    posts = [str(p) for p in va["post"]]
    golds = [parse_evidence(v) for v in va["evidence"]]
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    n = len(rows)
    runs = [r.strip() for r in args.runs.split(",")]

    r = pd.read_csv(args.risk, keep_default_na=False)
    r["row_id"] = r["row_id"].astype(str)
    col = "risk_level" if "risk_level" in r.columns else "pred_risk_level"
    rmap = {str(a): normalize_label(b) for a, b in zip(r["row_id"], r[col])}
    is_ind = np.array([rmap.get(x, "") == "indicator" for x in rows])
    print(f"{n} val posts / {len(pd.unique(auth))} authors, "
          f"{sum(1 for g in golds if g)} with non-empty gold, "
          f"{is_ind.sum()} predicted indicator")

    # ---- token posteriors ------------------------------------------------
    PN, OFF, IDX = [], None, None
    for run in runs:
        p = os.path.join(args.dir, f"{run}_val_tokprobs.npz")
        if not os.path.exists(p):
            raise SystemExit(
                f"{p} missing. These are written by src/evidence_sweep_rc.py "
                f"(onejob Part 1). Without them nothing here can score a span "
                f"the BIO decode did not already choose, which is the entire "
                f"point of this script.")
        d = np.load(p, allow_pickle=True)
        pr = d["probs"].astype(np.float32)
        PN.append(1.0 - pr[:, O_IDX])
        if OFF is None:
            OFF, IDX = d["offsets"].astype(int), d["index"].astype(int)
            rid = [str(x) for x in d["row_id"]]
            if rid != rows:
                raise SystemExit(
                    "row order in the npz does not match the val split -- "
                    "the offsets would be applied to the wrong posts")
        elif d["index"].astype(int).tolist() != IDX.tolist():
            raise SystemExit(f"{run} has a different tokenisation index")
    PN = np.stack(PN, 0)                       # (7, total_tokens)
    print(f"  token posteriors: {PN.shape[1]} tokens over {len(runs)} taggers "
          f"({PN.shape[1] / n:.0f}/post)")

    # ---- enumerate and score ---------------------------------------------
    # Score of a span = mean over taggers of the mean P(not-O) across its
    # tokens. Cumulative sums make every span O(1).
    cs = np.concatenate([np.zeros((len(runs), 1)), np.cumsum(PN, axis=1)], 1)
    cand = []          # per post: list of (score, text)
    for i in range(n):
        lo_t, hi_t = IDX[i], IDX[i + 1]
        off = OFF[lo_t:hi_t]
        if len(off) == 0:
            cand.append([])
            continue
        sid = sent_id_of(posts[i], off[:, 0])
        best = {}
        for a in range(len(off)):
            for L in range(args.lo, args.hi + 1):
                b = a + L
                if b > len(off) or sid[b - 1] != sid[a]:
                    break
                text = posts[i][off[a, 0]:off[b - 1, 1]].strip()
                if not text or len(text.split()) > args.max_words:
                    continue
                k = normalize_span(text)
                if not k:
                    continue
                s = float(np.mean((cs[:, lo_t + b] - cs[:, lo_t + a]) / L))
                if k not in best or s > best[k][0]:
                    best[k] = (s, text)
        cand.append(sorted(best.values(), reverse=True)[:args.pool_cap])
    print(f"  enumerated {np.mean([len(c) for c in cand]):.1f} scored "
          f"candidates/post (capped at {args.pool_cap})")

    orc = []
    for i in range(n):
        g = golds[i]
        if not g:
            orc.append(1.0)
            continue
        k = sum(any(matches(t, s) for _, t in cand[i]) for s in g)
        orc.append(2.0 * k / (k + len(g)))
    pool_oracle = float(np.mean(orc))
    print(f"  pool oracle macro phrase-F1: {pool_oracle:.4f}")

    # ---- select, author-disjoint -----------------------------------------
    def run_sel(idx, thr, kmax, suppress):
        gl, pl = [], []
        for i in idx:
            if suppress and is_ind[i]:
                keep = []
            else:
                keep = [t for s, t in cand[i][:kmax] if s >= thr]
            gl.append(golds[i])
            pl.append(keep)
        return phrase_f1(gl, pl, average="macro").f1

    ua = np.array(sorted(set(auth)))
    rng = np.random.default_rng(args.seed)
    grid = np.round(np.arange(0.10, 0.96, 0.05), 2)
    res = {"plain": [], "suppressed": [], "oracle": []}
    hdr = "=" * 76
    print("\n" + hdr)
    print("TAGGER-SCORED RANKING ON THE ENUMERATED POOL")
    print(hdr)
    for sp in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        ho = set(o[:max(int(len(o) * 0.25), 2)])
        pm = np.array([a in ho for a in auth])
        fi, hi = np.flatnonzero(~pm), np.flatnonzero(pm)
        best = (-1.0, 0.5, 1)
        for thr in grid:
            for kk in range(1, args.max_emit + 1):
                v = run_sel(fi, thr, kk, True)
                if v > best[0]:
                    best = (v, float(thr), kk)
        _, thr, kk = best
        a1 = run_sel(hi, thr, kk, False)
        a2 = run_sel(hi, thr, kk, True)
        res["plain"].append(a1)
        res["suppressed"].append(a2)
        res["oracle"].append(float(np.mean([orc[i] for i in hi])))
        print(f"  split {sp}: fit thr {thr:.2f} k {kk}  ->  "
              f"held-out {a2:.4f}   (no suppression {a1:.4f}, "
              f"pool oracle {res['oracle'][-1]:.4f})")

    P = float(np.mean(res["suppressed"]))
    Oc = float(np.mean(res["oracle"]))
    print("\n" + hdr)
    print("RESULT")
    print(hdr)
    print(f"  achieved (with shipped suppression) {P:.4f} "
          f"+- {np.std(res['suppressed']):.4f}")
    print(f"  without suppression                 "
          f"{np.mean(res['plain']):.4f}")
    print(f"  pool oracle                         {Oc:.4f}")
    print(f"  fraction of oracle attained         {P / max(Oc, 1e-9):.1%}")
    print(f"\n  INCUMBENT 0.7767   BAR TO BEAT IT {0.7767 / max(Oc, 1e-9):.1%}"
          f" of this pool's oracle")
    print(f"  model-free floor on a comparable pool: 55-62%")
    print("\n  If attained% clears the bar, the enumerate-then-rank rebuild is")
    print("  worth building properly and is the largest lane left. If it lands")
    print("  between the model-free floor and the bar, the taggers ARE better")
    print("  span scorers than bag-of-words but not enough -- and the next")
    print("  step is a trained span scorer, not a bigger pool. If it lands at")
    print("  the floor, mean P(not-O) carries no more signal than surface")
    print("  features and evidence is finished.")


if __name__ == "__main__":
    main()
