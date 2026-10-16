"""
risk_probe_analyze.py

Analyzes a signal in the evidence tagger's confidence that runs opposite to
the naive expectation.

`risk_tagger_probe.py` conditioned every held-out post on all four risk levels
and recorded the best-span confidence. The argmax was useless (0.574 vs the
risk model's 0.786) because the `ideation` column is high for everything. But
the FIRST column is a near-perfect ordinal severity score:

    gold          cond-indicator  cond-ideation  cond-behavior  cond-attempt
    indicator          0.029          0.246          0.162         0.126
    ideation           0.268          0.897          0.762         0.645
    behavior           0.630          0.931          0.909         0.880
    attempt            0.757          0.932          0.906         0.918

26x from indicator to attempt, monotone, from a model whose training target is
evidence spans and never the risk label.

WHY IT WORKS. Gold evidence is empty iff gold risk is `indicator` (0.986). So
the tagger learned "conditioned on indicator -> emit nothing", and only breaks
that habit when the post is explicit. `cond_indicator` is a trained meter for
how loud the suicidal content is, which is the C-SSRS severity axis itself.

WHAT IS ACTUALLY NEW. Unconditioned tagger confidence already beats the risk
model at predicting gold-empty. This is the same family. The new
part is that the signal is FOUR-LEVEL ORDINAL rather than binary, and that
conditioning on `indicator` specifically sharpens it. So the question this
script must answer is not "is there signal" -- it is "is there signal the risk
model does not ALREADY have", measured WITHIN each predicted class, which is
the only place an override can pay.

THE MEASUREMENT DISCIPLINE:
  * win rate and per-split standard deviation, not the confidence interval of
    the mean delta
  * many author-disjoint splits, not one
  * an out-of-fold estimate sits 0.02-0.04 below the true held-out evaluation
    value on every component, so a small positive here is not by itself a
    submission
  * any threshold is DERIVED ON FIT AUTHORS and SCORED ON HELD-OUT AUTHORS

Usage (CPU, seconds -- reuses the cached GPU dump):
  python src/risk_probe_analyze.py
  python src/risk_probe_analyze.py --splits 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
SEV = {c: i for i, c in enumerate(LEVELS)}


def wf1(gold, pred):
    tot = 0.0
    for c in LEVELS:
        g, p = (gold == c), (pred == c)
        tp = float((g & p).sum())
        pr = tp / max(p.sum(), 1e-9)
        rc = tp / max(g.sum(), 1e-9)
        tot += 2 * pr * rc / max(pr + rc, 1e-9) * g.sum()
    return tot / max(len(gold), 1)


def auc(score, pos):
    """P(score of a positive > score of a negative), ties at 0.5."""
    pos = np.asarray(pos, bool)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--probe", default="artifacts/risk_probe.csv")
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    df = pd.read_csv(args.probe)
    df["row_id"] = df["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    oof = pd.read_csv(args.oof, keep_default_na=False)
    oof["row_id"] = oof["row_id"].astype(str)
    ixo = oof.set_index("row_id")
    df = df[df["row_id"].isin(ixo.index) & df["row_id"].isin(lab.index)]
    df = df.reset_index(drop=True)
    rid = df["row_id"].tolist()

    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rid])
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rid])
    P = np.stack([ixo.loc[rid, f"prob_{c}"].astype(float).to_numpy()
                  for c in LEVELS], axis=1)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    pred = np.array(LEVELS)[P.argmax(1)]
    C = df[[f"conf_{c}" for c in LEVELS]].to_numpy()
    ci = C[:, 0]                                    # the severity meter
    gi = np.array([SEV[g] for g in gold])
    pi = np.array([SEV[p] for p in pred])
    print(f"{len(df)} posts, {len(pd.unique(auth))} authors, "
          f"risk wF1 {wf1(gold, pred):.4f}\n")

    print("=" * 86)
    print("A. cond_indicator BY (gold x predicted). THE COLUMNS ARE THE TEST.")
    print("=" * 86)
    hdr = "gold v / pred >"          # py3.10: no backslash inside an f-string
    print(f"  {hdr:<14}" + "".join(f"{c[:9]:>13}" for c in LEVELS))
    for g in LEVELS:
        cells = []
        for p in LEVELS:
            m = (gold == g) & (pred == p)
            cells.append(f"{ci[m].mean():.3f}({int(m.sum())})" if m.sum()
                         else "  -")
        print(f"  {g:<14}" + "".join(f"{c:>13}" for c in cells))
    print("\n  Reading DOWN a column = posts the risk model called the same")
    print("  thing. If the numbers differ down a column, the probe separates")
    print("  golds the risk model has already merged -- that is the only kind")
    print("  of signal an override can spend. If a column is flat, the risk")
    print("  model already knows everything the probe knows.")

    print("\n" + "=" * 86)
    print("B. WITHIN A PREDICTED CLASS, DOES cond_indicator RANK THE GOLD?")
    print("=" * 86)
    print(f"  {'predicted':<12}{'n':>6}{'contrast':<28}{'AUC':>7}"
          f"{'risk-prob AUC':>15}")
    contrasts = [
        ("indicator", "gold is NOT indicator"),
        ("ideation", "gold is behavior/attempt"),
        ("ideation", "gold IS indicator"),
        ("behavior", "gold is attempt"),
    ]
    for pc, desc in contrasts:
        m = pred == pc
        if m.sum() < 20:
            continue
        if desc == "gold is NOT indicator":
            y = gold[m] != "indicator"
            rk = 1.0 - P[m, 0]
        elif desc == "gold is behavior/attempt":
            y = np.isin(gold[m], ["behavior", "attempt"])
            rk = P[m, 2] + P[m, 3]
        elif desc == "gold IS indicator":
            y = gold[m] == "indicator"
            rk = P[m, 0]
        else:
            y = gold[m] == "attempt"
            rk = P[m, 3]
        s = ci[m] if desc != "gold IS indicator" else -ci[m]
        print(f"  {pc:<12}{int(m.sum()):>6}{desc:<28}{auc(s, y):>7.3f}"
              f"{auc(rk, y):>15.3f}")
    print("\n  AUC 0.500 = no information. The right-hand column is the risk")
    print("  model's OWN probability on the same contrast, which is the bar:")
    print("  the probe only helps where it beats it, or where it is close but")
    print("  wrong on DIFFERENT posts.")

    print("\n" + "=" * 86)
    print("C. IS THE PROBE INDEPENDENT, OR THE SAME MISTAKES TWICE?")
    print("=" * 86)
    sev_p = ci                                       # probe severity
    sev_r = (P * np.arange(4)).sum(1)                # risk expected severity
    err_r = sev_r - gi
    print(f"  spearman(probe severity, gold severity)  "
          f"{np.corrcoef(np.argsort(np.argsort(sev_p)), gi)[0,1]:>6.3f}")
    print(f"  spearman(risk  severity, gold severity)  "
          f"{np.corrcoef(np.argsort(np.argsort(sev_r)), gi)[0,1]:>6.3f}")
    print(f"  corr(probe severity, RISK MODEL ERROR)   "
          f"{np.corrcoef(sev_p, err_r)[0,1]:>6.3f}")
    print("\n  The last line is the one that matters. If the probe correlates")
    print("  with the risk model's ERROR, it carries the correction. If it is")
    print("  near zero, it agrees with the risk model wherever the risk model")
    print("  is wrong and there is nothing to extract.")

    # ------------------------------------------------------------- D
    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    print("\n" + "=" * 86)
    print(f"D. RULES, THRESHOLD FITTED ON FIT AUTHORS AND SCORED ON HELD-OUT")
    print(f"   AUTHORS, {args.splits} splits. Held-out share matches the test "
          f"set size.")
    print("=" * 86)

    def rule_demote(t, ix):
        """predicted ideation but the tagger finds almost nothing when told
        `indicator` -> the post is an indicator."""
        n = pred[ix].copy()
        n[(pred[ix] == "ideation") & (ci[ix] < t)] = "indicator"
        return n

    def rule_promote(t, ix):
        """predicted indicator but the tagger finds strong evidence even when
        told `indicator` -> the post is explicit."""
        n = pred[ix].copy()
        n[(pred[ix] == "indicator") & (ci[ix] > t)] = "ideation"
        return n

    def rule_both(t, ix):
        n = pred[ix].copy()
        n[(pred[ix] == "ideation") & (ci[ix] < t[0])] = "indicator"
        n[(pred[ix] == "indicator") & (ci[ix] > t[1])] = "ideation"
        return n

    grid = np.round(np.arange(0.02, 0.96, 0.02), 3)
    frac = 378.0 / (378.0 + len(df))
    specs = [("ide->ind  (ci < t)", rule_demote, 1),
             ("ind->ide  (ci > t)", rule_promote, 1),
             ("both directions", rule_both, 2)]
    print(f"  {'rule':<22}{'d comp':>9}{'win':>6}{'sd':>9}{'t chosen':>18}"
          f"{'fires/split':>13}")
    for nm, fn, nt in specs:
        ds, ts, fires = [], [], []
        for s in range(args.splits):
            o = list(ua)
            rng.shuffle(o)
            k = max(int(len(o) * frac), 5)
            ho = set(o[:k])
            hm = np.array([a in ho for a in auth])
            fm = ~hm
            if fm.sum() < 50 or hm.sum() < 20:
                continue
            fi = np.flatnonzero(fm)
            if nt == 1:
                best, bt = -1e9, grid[0]
                for t in grid:
                    v = wf1(gold[fi], fn(t, fi))
                    if v > best:
                        best, bt = v, t
                use = bt
            else:
                best, bt = -1e9, (grid[0], grid[-1])
                for t1 in grid[::3]:
                    for t2 in grid[::3]:
                        v = wf1(gold[fi], fn((t1, t2), fi))
                        if v > best:
                            best, bt = v, (t1, t2)
                use = bt
            hi = np.flatnonzero(hm)
            nw = fn(use, hi)
            ds.append(wf1(gold[hi], nw) - wf1(gold[hi], pred[hi]))
            fires.append(int((nw != pred[hi]).sum()))
            ts.append(use if nt == 1 else use[0])
        ds = np.array(ds)
        print(f"  {nm:<22}{0.4*ds.mean():>+9.4f}{np.mean(ds>0):>5.0%}"
              f"{0.4*ds.std():>9.4f}"
              f"{f'{np.mean(ts):.2f}+-{np.std(ts):.2f}':>18}"
              f"{np.mean(fires):>13.1f}")
    print("\n  d comp = 0.4 x d(risk wF1) on HELD-OUT authors only.")
    print("  A STABLE t across splits means the threshold is a property of the")
    print("  signal; a t that wanders means every split fits its own sample")
    print("  and the rule will not transfer. sd is the ONE-SPLIT spread --")
    print("  compare the mean against it, never against sd/sqrt(N).")

    # ------------------------------------------------------------- E
    print("\n" + "=" * 86)
    print("E. AS A FEATURE, NOT AN OVERRIDE -- add w x cond_indicator to the")
    print("   severity direction of the risk log-probs. ZERO fitted params.")
    print("=" * 86)
    lp = np.log(np.clip(P, 1e-9, None))
    # HELD-OUT AUTHOR SUBSETS SIZED LIKE THE TEST SET, not bootstraps over all
    # 1590 posts. A bootstrap of the full sample estimates the spread of the
    # MEAN delta; the held-out evaluation draws ONE 378-post sample, whose
    # spread is ~2x larger, so quoting the full-sample bootstrap spread
    # understates the real risk of a regression. No threshold is fitted here
    # -- w is a fixed grid, so the subsets need not be split into fit/score.
    subs = []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        k = max(int(len(o) * frac), 5)
        ho = set(o[:k])
        subs.append(np.flatnonzero([a in ho for a in auth]))
    print(f"  scored on {args.splits} held-out author subsets of "
          f"~{int(np.mean([len(s) for s in subs]))} posts (test set is 378)")
    print(f"  {'w':>6}{'changed':>9}{'right':>7}{'wrong':>7}{'wF1':>9}"
          f"{'d comp':>9}{'win':>6}{'sd':>9}{'z':>6}")
    for w_ in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        nw = np.array(LEVELS)[(lp + w_ * np.outer(ci, np.arange(4))
                               / 3.0).argmax(1)]
        ch = nw != pred
        if not ch.sum():
            print(f"  {w_:>6.2f}{0:>9}")
            continue
        d = np.array([wf1(gold[b], nw[b]) - wf1(gold[b], pred[b])
                      for b in subs])
        sd = d.std()
        print(f"  {w_:>6.2f}{int(ch.sum()):>9}"
              f"{int(((nw == gold) & ~(pred == gold) & ch).sum()):>7}"
              f"{int(((pred == gold) & ~(nw == gold) & ch).sum()):>7}"
              f"{wf1(gold, nw):>9.4f}{0.4*d.mean():>+9.4f}"
              f"{np.mean(d>0):>5.0%}{0.4*sd:>9.4f}"
              f"{d.mean()/max(sd,1e-9):>6.2f}")
    print("\n  w tilts every post toward higher severity in proportion to how")
    print("  much evidence the tagger finds. w=0 is the champion, so the row")
    print("  spacing shows whether this is a plateau (real) or a spike (fit).")
    print("  z = mean/sd on ONE test-sized draw. z below ~1.0 means a losing")
    print("  held-out evaluation result is unsurprising even if the effect is")
    print("  real, and an out-of-fold estimate adds a systematic gap on top")
    print("  of that.")


if __name__ == "__main__":
    main()
