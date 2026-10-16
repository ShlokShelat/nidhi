"""
factor_free_diagnostics.py

Two cheap checks that can kill two factor ideas before any GPU is spent.

A. TRUNCATION. `score_all` tokenizes with max_len=320 and
   truncation="only_first", so after the hypothesis (~40-60 tokens) the premise
   budget is roughly 200 words. Posts longer than that have their tail
   invisible to EVERY arm. The fix would be windowed inference with max-pooling
   -- but only if truncated posts actually score worse. This measures that on
   the 1635 OOF posts, using the shipped bag's own predictions.

   KILL CONDITION: if long posts do not score measurably worse per-post, or if
   their per-label RECALL is not depressed, the lane is dead and no windowed
   re-dump is warranted.

B. `suicide means (with access)` VETO. The annotation scheme almost never
   assigns that label to `indicator` posts. The risk column is known at
   inference and is 0.8103 wF1, so demoting that one label on predicted-
   indicator posts is a candidate 0-parameter rule. It is only worth anything
   if it would actually fire, and only safe if it does not destroy true
   positives.

   KILL CONDITIONS: fires on fewer than ~3 posts (inert), or removes any OOF
   TRUE positive (not one-sided in practice, whatever the gold counts say).

Both read existing artifacts. No GPU, no training, seconds.

Usage:
  python src/factor_free_diagnostics.py --arms mnli dmnli cmine prot d4 qnli
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_bag_ship import combine, f1_at  # noqa: E402
from scoring import FACTOR_TAXONOMY, RISK_LEVELS, normalize_label  # noqa: E402
from scoring import parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)
MEANS = "suicide means (with access)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["mnli", "dmnli", "cmine", "prot", "d4", "qnli"])
    ap.add_argument("--rule", default="logit")
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--risk-oof", default="v2_rebuild/runs/risk_ctrl_f{f}")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--words", type=int, default=200)
    args = ap.parse_args()

    O, rid = [], None
    for a in args.arms:
        pr, ids = [], []
        for f in range(args.k):
            z = np.load(f"{args.dir}/{a}_cv_oof_f{f}.npz", allow_pickle=True)
            pr.append(z["probs"].astype(float))
            ids.extend(str(x) for x in z["row_id"])
        pr = np.concatenate(pr)
        if rid is None:
            rid = ids
        elif ids != rid:
            o = {k: i for i, k in enumerate(ids)}
            pr = pr[[o[k] for k in rid]]
        O.append(pr)
    P = combine(np.stack(O), args.rule, args.arms)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id").loc[rid]
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(rid), N), dtype=int)
    for i, c in enumerate(lab["factors"]):
        for x in parse_factors(c):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1

    # fire-rate decision, exactly as shipped
    on = np.zeros_like(Y)
    for j in range(N):
        best, bt = -1.0, 0.5
        for t in GRID:
            v = f1_at((P[:, j] >= t).astype(int), Y[:, j])
            if v > best:
                best, bt = v, t
        on[:, j] = (P[:, j] >= bt).astype(int)

    # ---------------------------------------------------- A. truncation
    nw = lab["post"].astype(str).str.split().str.len().to_numpy()
    long = nw > args.words
    print("=" * 72)
    print(f"A. TRUNCATION  (premise budget ~{args.words} words at max_len=320)")
    print("=" * 72)
    print(f"{int(long.sum())} of {len(rid)} OOF posts are longer "
          f"({100*long.mean():.1f}%)   median {int(np.median(nw))} words, "
          f"p99 {int(np.percentile(nw, 99))}")

    def per_post_f1(m):
        v = []
        for i in np.flatnonzero(m):
            tp = int((on[i] & Y[i]).sum())
            v.append(0.0 if tp == 0 else
                     2 * tp / (2 * tp + int((on[i] & (1 - Y[i])).sum())
                               + int(((1 - on[i]) & Y[i]).sum())))
        return float(np.mean(v)) if v else float("nan")

    fl, fs = per_post_f1(long), per_post_f1(~long)
    rl = on[long][Y[long] == 1].mean() if Y[long].sum() else float("nan")
    rs = on[~long][Y[~long] == 1].mean()
    print(f"\n{'subset':<16}{'posts':>7}{'perpostF1':>11}{'recall':>9}"
          f"{'gold/post':>11}")
    print("-" * 54)
    print(f"{'long':<16}{int(long.sum()):>7}{fl:>11.4f}{rl:>9.4f}"
          f"{Y[long].sum(1).mean():>11.2f}")
    print(f"{'short':<16}{int((~long).sum()):>7}{fs:>11.4f}{rs:>9.4f}"
          f"{Y[~long].sum(1).mean():>11.2f}")
    print(f"\ndelta long-short: F1 {fl-fs:+.4f}   recall {rl-rs:+.4f}")
    if fl - fs > -0.02 and rl - rs > -0.02:
        print("VERDICT: no truncation deficit -> windowed inference is NOT")
        print("warranted. Lane dead, no GPU spent.")
    else:
        print("VERDICT: long posts ARE worse. Windowed inference may be worth")
        print("it -- but the OOF dumps must be re-scored windowed too, never")
        print("mixing regimes between the fire-rate fit and test.")

    # ---------------------------------------------------- B. means veto
    print("\n" + "=" * 72)
    print(f"B. `{MEANS}` VETO on predicted-indicator posts")
    print("=" * 72)
    j = idx[MEANS]
    gr = np.array([normalize_label(x) == "indicator"
                   for x in lab["risk_level"]])
    print(f"gold: {MEANS} appears on {int((Y[:, j] & gr).sum())} of "
          f"{int(gr.sum())} gold-indicator posts "
          f"({100*(Y[:, j] & gr).mean():.2f}% of all posts)")

    rr = []
    for f in range(args.k):
        p = os.path.join(args.risk_oof.format(f=f), "oof_predictions.csv")
        if not os.path.exists(p):
            print(f"\n(no risk OOF at {p} -- using GOLD risk as an upper bound "
                  f"on how often the veto could fire)")
            rr = None
            break
        rr.append(pd.read_csv(p))
    if rr:
        risk = pd.concat(rr, ignore_index=True)
        risk["row_id"] = risk["row_id"].astype(str)
        risk = risk.set_index("row_id").loc[rid]
        cols = [f"prob_{c}" for c in RISK_LEVELS]
        pind = risk[cols].to_numpy().argmax(1) == 0
    else:
        pind = gr

    fires = on[:, j].astype(bool) & pind
    kills_tp = fires & (Y[:, j] == 1)
    print(f"\nveto would fire on {int(fires.sum())} posts")
    print(f"  of which TRUE positives destroyed: {int(kills_tp.sum())}")
    base = f1_at(on[:, j], Y[:, j])
    vetoed = on[:, j].copy()
    vetoed[fires] = 0
    print(f"  label F1 {base:.4f} -> {f1_at(vetoed, Y[:, j]):.4f} "
          f"({f1_at(vetoed, Y[:, j]) - base:+.4f} on 1 of 24 labels "
          f"= {(f1_at(vetoed, Y[:, j]) - base)/24*0.3:+.5f} composite)")
    if int(fires.sum()) < 3:
        print("VERDICT: inert -- fires on too few posts to matter. Dead.")
    elif int(kills_tp.sum()) > 0:
        print("VERDICT: destroys true positives, so the rule is not one-sided")
        print("in practice whatever the gold counts suggest. Dead.")
    else:
        print("VERDICT: fires, costs no true positive. Worth riding along with")
        print("the next factor submission -- never worth its own slot.")


if __name__ == "__main__":
    main()
