"""
evidence_suppression_decouple.py

UNWELD "IS THE EVIDENCE EMPTY?" FROM "IS THE RISK LABEL INDICATOR?"

`evidence_match_quality.py` located the entire evidence deficit:

    31 posts score exactly 0
      10   gold EMPTY, spans predicted        <- risk said non-indicator
      17   gold NON-EMPTY, no spans predicted <- risk said indicator
       4   both non-empty, nothing matched     <- the tagger's fault

**27 of 31 are risk errors.** The tagger itself is excellent: 89.7% of its
spans match, and 69% match the annotator's boundaries EXACTLY. A perfect
tagger buys 4 posts, 1.1%. Making the tagger better is not the lane.

THE ARCHITECTURE IS. The pipeline hard-wires

    evidence is empty  <=>  predicted risk == indicator

Those are two different questions. In the GOLD labels they agree 98.6%
(gold-empty IS the indicator label), which is why welding them looked free.
But the PREDICTED risk is wrong on ~17% of posts, and the weld makes every
such error cost twice -- once in risk, once in evidence. Unwelding lets a
dedicated gold-empty predictor decide suppression while the risk label stays
exactly as it is.

That matters because the risk label is NOT the best available predictor of
gold-emptiness. Two independent signals exist:

  cond_indicator  the severity meter: best-span confidence when the tagger is
                  TOLD the post is only an indicator. By gold class: 0.029 /
                  0.268 / 0.630 / 0.757. It is a poor RISK classifier, but
                  gold-emptiness is the one question it was accidentally
                  trained on -- indicator posts have no gold spans, so the
                  tagger learned to emit nothing for them.
  tagger evidence how many spans clear threshold, and how confident they are --
                  shown in earlier testing to beat the risk model at
                  predicting gold-empty, on an instrument that has since
                  changed and needs re-verification.

ONLY THE EVIDENCE COLUMN MOVES. The risk label is untouched, so risk wF1 and
factors are bit-identical and composite delta = 0.3 * d(evidence macro). Clean
attribution, unlike anything that changes a label.

Everything cross-fitted by author: the rule's threshold/coefficients come from
FIT authors and are scored on HELD-OUT authors. Judge by WIN RATE and the share
of resamples that LOSE.

Usage:
  python src/evidence_suppression_decouple.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_max_bipartite_matching, _span_pair_matches,  # noqa: E402
                     normalize_label, normalize_span, parse_evidence,
                     token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def post_f1(gold, pred):
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    adj = [[j for j, g in enumerate(gold) if _span_pair_matches(p, g)]
           for p in pred]
    m = _max_bipartite_matching(adj, len(gold))
    pr, rc = m / len(pred), m / len(gold)
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)


def auc(score, pos):
    pos = np.asarray(pos, bool)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--cache", default="artifacts/evcond_val_spans.json")
    a_.add_argument("--probe", default="artifacts/risk_probe.csv")
    a_.add_argument("--min-conf", type=float, default=0.35)
    a_.add_argument("--votes", type=int, default=2)
    a_.add_argument("--splits", type=int, default=300)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    S = json.load(open(args.cache))
    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    va = va[va["row_id"].isin(S)].reset_index(drop=True)
    rows = va["row_id"].tolist()
    gold = [parse_evidence(v) for v in va["evidence"]]
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    gold_empty = np.array([not g for g in gold])

    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pv = pv.set_index("row_id")
    pred_risk = np.array([normalize_label(v)
                          for v in pv.loc[rows, "pred_risk_level"]])
    p_ind = pv.loc[rows, "prob_indicator"].astype(float).to_numpy()

    # spans the tagger WOULD emit, conditioned on the predicted risk. For posts
    # the champion suppresses, they are still computed -- that is the whole
    # point: suppression must become a separate decision, so the spans have
    # to exist.
    def spans_for(r, lv):
        c = {}
        for s, cf in S[r][lv]:
            if cf < args.min_conf:
                continue
            e = c.setdefault(normalize_span(s), [0, s, cf])
            e[0] += 1
            e[2] = max(e[2], cf)
        keep = sorted([v for v in c.values() if v[0] >= args.votes],
                      key=lambda x: -token_len(x[1]))
        out = []
        for _, s, _cf in keep:
            n = normalize_span(s)
            if n and not any(n in normalize_span(k) for k in out):
                out.append(s)
        return out, (max([v[2] for v in c.values()], default=0.0),
                     len([v for v in c.values() if v[0] >= args.votes]))

    SP, maxconf, nspan = [], [], []
    for i, r in enumerate(rows):
        lv = pred_risk[i] if pred_risk[i] != "indicator" else "ideation"
        # for suppressed posts, ask what it WOULD say at the next level up
        sp, (mc, n) = spans_for(r, lv)
        SP.append(sp)
        maxconf.append(mc)
        nspan.append(n)
    maxconf, nspan = np.array(maxconf), np.array(nspan)

    ci = np.full(len(rows), np.nan)
    if os.path.exists(args.probe):
        pr = pd.read_csv(args.probe)
        pr["row_id"] = pr["row_id"].astype(str)
        pr = pr.drop_duplicates(subset=["row_id"]).set_index("row_id")
        if "conf_indicator" in pr.columns:
            have = [r for r in rows if r in pr.index]
            if len(have) == len(rows):
                ci = pr.loc[rows, "conf_indicator"].astype(float).to_numpy()
    have_ci = not np.isnan(ci).any()

    def build(supp):
        return [[] if supp[i] else SP[i] for i in range(len(rows))]

    cur_supp = pred_risk == "indicator"
    cur = build(cur_supp)
    base = float(np.mean([post_f1(g, p) for g, p in zip(gold, cur)]))
    orc = float(np.mean([post_f1(g, p)
                         for g, p in zip(gold, build(gold_empty))]))
    print(f"{len(rows)} val posts   gold-empty {int(gold_empty.sum())}")
    print(f"  current  (suppress iff predicted indicator)   {base:.4f}")
    print(f"  ORACLE   (suppress iff gold is empty)         {orc:.4f}")
    print(f"  the weld costs                                {orc-base:+.4f}"
          f"  ({0.3*(orc-base):+.4f} composite)\n")

    print("=" * 74)
    print("WHO PREDICTS GOLD-EMPTY BEST? (AUC, higher = better)")
    print("=" * 74)
    cands = {"risk prob_indicator": p_ind,
             "tagger max conf (neg)": -maxconf,
             "tagger span count (neg)": -nspan.astype(float)}
    if have_ci:
        cands["cond_indicator (neg)"] = -ci
    for nm, s in cands.items():
        print(f"  {nm:<26}{auc(s, gold_empty):>7.3f}")
    print(f"  {'current hard rule':<26}"
          f"{float((cur_supp == gold_empty).mean()):>7.3f}   (accuracy, not AUC)")
    print("\n  The hard rule is what ships. Any signal with a materially higher")
    print("  AUC is deciding emptiness better than the risk label does, and")
    print("  the weld is throwing that away.")

    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    print("\n" + "=" * 74)
    print("CROSS-FITTED SUPPRESSION RULES (threshold from FIT authors)")
    print("=" * 74)
    print(f"  {'rule':<30}{'d evidence':>12}{'d comp':>9}{'win':>6}"
          f"{'LOSS':>6}{'sd':>9}{'flipped':>9}")

    rules = {}
    for nm, s in cands.items():
        rules[nm] = s
    rules["risk OR tagger (both say empty)"] = None
    rules["risk AND tagger (either says)"] = None

    for nm in list(cands) + ["risk OR tagger (both say empty)",
                             "risk AND tagger (either says)"]:
        ds, nf = [], []
        for _ in range(args.splits):
            o = list(ua)
            rng.shuffle(o)
            k = max(int(len(o) * 0.25), 2)
            ho = set(o[:k])
            hm = np.array([a in ho for a in auth])
            fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
            if len(fi) < 40 or len(hi) < 15:
                continue
            if nm in cands:
                s = cands[nm]
                grid = np.quantile(s[fi], np.linspace(0.05, 0.95, 19))
                best, bt = -1e9, grid[0]
                for t in grid:
                    sup = s[fi] >= t
                    v = np.mean([post_f1(gold[j], ([] if sup[q] else SP[j]))
                                 for q, j in enumerate(fi)])
                    if v > best:
                        best, bt = v, t
                sup_h = s[hi] >= bt
            else:
                tag = -maxconf
                grid = np.quantile(tag[fi], np.linspace(0.05, 0.95, 19))
                best, bt = -1e9, grid[0]
                for t in grid:
                    a_ = cur_supp[fi]
                    b_ = tag[fi] >= t
                    sup = (a_ & b_) if nm.startswith("risk OR") else (a_ | b_)
                    v = np.mean([post_f1(gold[j], ([] if sup[q] else SP[j]))
                                 for q, j in enumerate(fi)])
                    if v > best:
                        best, bt = v, t
                a_, b_ = cur_supp[hi], tag[hi] >= bt
                sup_h = (a_ & b_) if nm.startswith("risk OR") else (a_ | b_)
            new = [([] if sup_h[q] else SP[j]) for q, j in enumerate(hi)]
            ref = [cur[j] for j in hi]
            gh = [gold[j] for j in hi]
            ds.append(np.mean([post_f1(g, p) for g, p in zip(gh, new)])
                      - np.mean([post_f1(g, p) for g, p in zip(gh, ref)]))
            nf.append(int((sup_h != cur_supp[hi]).sum()))
        d = np.array(ds)
        print(f"  {nm:<30}{d.mean():>+12.4f}{0.3*d.mean():>+9.4f}"
              f"{np.mean(d > 0):>5.0%}{np.mean(d < 0):>6.0%}"
              f"{0.3*d.std():>9.4f}{np.mean(nf):>9.1f}")

    print("\n  `flipped` = posts whose suppression changed per split. A rule")
    print("  that wins while flipping very few posts is the safe kind.")
    print("  Only the EVIDENCE column moves here -- risk and factors stay")
    print("  bit-identical, so d comp = 0.3 * d evidence exactly.")


if __name__ == "__main__":
    main()
