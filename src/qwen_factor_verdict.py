"""
qwen_factor_verdict.py

Test whether an average-precision gain from the Qwen factor arm survives
contact with the metric Subtask 2 actually scores.

An AP reading can overstate a real gain in three separable ways, all checkable
here without a GPU.

  1. TINY LABELS DOMINATE THE MEAN. AP is averaged over all 24 labels, so a
     label with four positives counts as much as one with 187, and with n=4
     promoting a single post a few ranks moves that label's AP by ~0.3. The
     --min-n sweep below shows what remains as the unmeasurably small labels
     are dropped.

  2. AP IS NOT THE SCORED METRIC. Subtask 2 is macro-F1 at a threshold. F1
     tracks AP but does not equal it: a ranking gain becomes F1 only if a
     cutoff can be placed to exploit it, and on a label with four positives
     usually none can.

  3. THRESHOLDS MUST BE CROSS-FITTED. Tuning a cutoff on the same posts it is
     scored on hands an arm roughly 0.01 of free F1, enough to flip the sign of
     a small delta. The xfit-half column fits thresholds on one author half and
     applies them to the other.

Complementary per-label win/loss patterns between the two models make blending
worth scoring too, so two zero-parameter blends are included:

    prob mean   0.5*qwen + 0.5*deberta
    rank mean   average of within-label rank percentiles

Rank averaging is included because the two models are calibrated differently --
a LoRA head on a 14B decoder and a linear head on a 435M encoder have no reason
to share a probability scale, and rank averaging is invariant to that. Both fit
ZERO parameters, so neither can overfit the data they are read on.

The weight sweep is printed FOR INFORMATION ONLY: choosing a weight from it
would fit one parameter on the same validation data. Read the 0.5 row.

Usage:
  python src/qwen_factor_verdict.py
  python src/qwen_factor_verdict.py --tag qnli --folds 0 3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

N = len(FACTOR_TAXONOMY)
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
PROT = {"social support", "coping strategy", "psychological capital",
        "sense of responsibility", "meaning in life"}


def macro(P, Y):
    out = []
    for j in range(Y.shape[1]):
        tp = int((P[:, j] & Y[:, j]).sum())
        out.append(0.0 if tp == 0 else 2 * tp /
                   (2 * tp + int((P[:, j] & (1 - Y[:, j])).sum())
                    + int(((1 - P[:, j]) & Y[:, j]).sum())))
    return float(np.mean(out)), out


def fit_th(PR, Y, rows):
    th = np.full(PR.shape[1], 0.5)
    for j in range(PR.shape[1]):
        best, bt = -1.0, 0.5
        for t in GRID:
            p = (PR[rows, j] >= t).astype(int)
            tp = int((p & Y[rows, j]).sum())
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + int((p & (1 - Y[rows, j])).sum())
                           + int(((1 - p) & Y[rows, j]).sum()))
            if f1 > best:
                best, bt = f1, t
        th[j] = bt
    return th


def author_halves(auth):
    s = pd.Series(auth)
    counts = s.value_counts()
    half, size = {}, {"A": 0, "B": 0}
    for a in sorted(counts.index, key=lambda x: (-counts[x], str(x))):
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


def scores(PR, Y, A, B):
    """(pooled macro-F1, cross-fitted macro-F1, per-label xfit F1)."""
    pooled, _ = macro((PR >= fit_th(PR, Y, np.ones(len(Y), bool))).astype(int), Y)
    Px = np.zeros_like(Y)
    Px[A] = (PR[A] >= fit_th(PR, Y, B)).astype(int)
    Px[B] = (PR[B] >= fit_th(PR, Y, A)).astype(int)
    xf, per = macro(Px, Y)
    return pooled, xf, per


def ranks(P):
    """Within-label rank percentile, so differently-calibrated models combine."""
    R = np.zeros_like(P, dtype=np.float64)
    for j in range(P.shape[1]):
        order = P[:, j].argsort().argsort()
        R[:, j] = order / max(1, len(order) - 1)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qnli")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

    rid, Q, D = [], [], []
    for f in args.folds:
        qp = f"runs/factors_{args.tag}_f{f}/val_probs.npz"
        ip = f"artifacts/fac_oof_f{f}.npz"
        for p in (qp, ip):
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
        q = np.load(qp, allow_pickle=True)
        d = np.load(ip, allow_pickle=True)
        qr, dr = list(q["row_id"]), list(d["row_id"])
        if set(qr) != set(dr):
            raise SystemExit(f"fold {f}: row sets differ")
        o = {r: i for i, r in enumerate(dr)}
        rid.extend(qr)
        Q.append(q["probs"])
        D.append(d["probs"][[o[r] for r in qr]])
    Q, D = np.concatenate(Q), np.concatenate(D)

    Y = np.zeros((len(rid), N), dtype=int)
    for i, r in enumerate(rid):
        for x in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1
    n = Y.sum(0)

    half = author_halves([lab.loc[r, "anon_user_id"] for r in rid])
    A, B = half == "A", half == "B"
    print(f"{len(rid)} posts, folds {args.folds}, author halves "
          f"{int(A.sum())}/{int(B.sum())}\n")

    arms = {
        "DeBERTa (incumbent)": D,
        "Qwen": Q,
        "blend 0.5 prob": 0.5 * Q + 0.5 * D,
        "blend 0.5 rank": 0.5 * ranks(Q) + 0.5 * ranks(D),
    }

    print("=" * 78)
    print("MACRO-F1 -- the metric you are actually scored on")
    print("=" * 78)
    print(f"{'arm':<24}{'pooled':>10}{'xfit-half':>12}{'d xfit':>10}")
    print("-" * 78)
    res = {}
    for k, P in arms.items():
        po, xf, per = scores(P, Y, A, B)
        res[k] = (po, xf, per)
        d = "" if k.startswith("DeBERTa") else f"{xf - res['DeBERTa (incumbent)'][1]:>+10.4f}"
        print(f"{k:<24}{po:>10.4f}{xf:>12.4f}{d}")
    base_xf = res["DeBERTa (incumbent)"][1]
    print("\nxfit-half is the honest column. pooled is optimistic for everyone")
    print("equally, so it is readable but inflated.")

    print("\n" + "=" * 78)
    print("HOW MUCH OF THE AP GAIN IS THE FOUR-POSITIVE LABELS?")
    print("=" * 78)
    apd = np.array([average_precision_score(Y[:, j], D[:, j]) if n[j] else np.nan
                    for j in range(N)])
    apq = np.array([average_precision_score(Y[:, j], Q[:, j]) if n[j] else np.nan
                    for j in range(N)])
    print(f"{'kept labels':<28}{'count':>7}{'AP inc':>9}{'AP qwen':>9}{'delta':>9}")
    print("-" * 78)
    for min_n in (0, 5, 10, 20, 40):
        m = (n >= min_n) & ~np.isnan(apd)
        if m.sum() == 0:
            continue
        lbl = "all labels" if min_n == 0 else f"n >= {min_n}"
        print(f"{lbl:<28}{int(m.sum()):>7}{apd[m].mean():>9.4f}"
              f"{apq[m].mean():>9.4f}{apq[m].mean()-apd[m].mean():>+9.4f}")
    print("\nIf the delta collapses as rare labels drop out, the headline was")
    print("driven by a handful of positives and will not survive five folds.")

    print("\n" + "=" * 78)
    print("PER-LABEL, cross-fitted F1  (movers only)")
    print("=" * 78)
    pd_, pq = res["DeBERTa (incumbent)"][2], res["Qwen"][2]
    pb = res["blend 0.5 prob"][2]
    print(f"{'factor':<44}{'n':>4}{'inc':>8}{'qwen':>8}{'blend':>8}{'dQ':>8}")
    print("-" * 84)
    for j, t in enumerate(FACTOR_TAXONOMY):
        if abs(pq[j] - pd_[j]) > 0.02:
            print(f"{t:<44}{int(n[j]):>4}{pd_[j]:>8.3f}{pq[j]:>8.3f}"
                  f"{pb[j]:>8.3f}{pq[j]-pd_[j]:>+8.3f}"
                  f"{'  prot' if t in PROT else ''}")

    print("\n" + "=" * 78)
    print("THE SAME QUESTION IN F1 -- the metric that counts")
    print("=" * 78)
    print("The AP sweep above tests whether the RANKING gain is carried by the")
    print("four-positive labels. This tests the same thing for cross-fitted F1,")
    print("which is what Subtask 2 scores. The two can disagree: a ranking gain")
    print("only becomes F1 if a threshold can be placed to exploit it, and on a")
    print("label with four positives it usually cannot.")
    print()
    pdq = np.array(pd_), np.array(pq)
    print(f"{'kept labels':<28}{'count':>7}{'F1 inc':>9}{'F1 qwen':>9}{'delta':>9}")
    print("-" * 78)
    for min_n in (0, 5, 10, 20, 40):
        m = n >= min_n
        if m.sum() == 0:
            continue
        lbl = "all labels" if min_n == 0 else f"n >= {min_n}"
        print(f"{lbl:<28}{int(m.sum()):>7}{pdq[0][m].mean():>9.4f}"
              f"{pdq[1][m].mean():>9.4f}"
              f"{pdq[1][m].mean()-pdq[0][m].mean():>+9.4f}")
    print("\nIf AP stays positive here while F1 turns negative, Qwen ranks")
    print("better but its scores cannot be thresholded -- which is exactly what")
    print("the pooled-vs-xfit gap below measures.")

    print("\n" + "=" * 78)
    print("THRESHOLD FRAGILITY -- how much each arm loses when thresholds are")
    print("fitted honestly instead of on the posts being scored")
    print("=" * 78)
    for k in arms:
        po, xf, _ = res[k]
        print(f"  {k:<24} pooled {po:.4f} -> xfit {xf:.4f}   drop {po-xf:.4f}")
    print("\n  A bigger drop means the probabilities do not transfer across")
    print("  authors, so any gain read with self-fitted thresholds is a")
    print("  property of those posts rather than of the model.")

    print("\n" + "=" * 78)
    print("PER FOLD -- is the gain consistent, or one lucky fold?")
    print("=" * 78)
    off = 0
    for f in args.folds:
        k = f"runs/factors_{args.tag}_f{f}/val_probs.npz"
        m = len(np.load(k, allow_pickle=True)["row_id"])
        sl = slice(off, off + m)
        off += m
        Yf = Y[sl]
        hf = author_halves([lab.loc[r, "anon_user_id"] for r in rid[sl]])
        Af, Bf = hf == "A", hf == "B"
        _, xd, _ = scores(D[sl], Yf, Af, Bf)
        _, xq, _ = scores(Q[sl], Yf, Af, Bf)
        print(f"  fold {f}  n={m:<5} inc {xd:.4f}   qwen {xq:.4f}   {xq-xd:+.4f}")
    print("\n  Both folds positive -> the effect is consistent, if small.")
    print("  One fold carrying it -> this many posts cannot resolve it.")

    print("\n" + "=" * 78)
    print("BLEND WEIGHT SWEEP -- INFORMATION ONLY, DO NOT PICK FROM IT")
    print("=" * 78)
    for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        _, xf, _ = scores(w * Q + (1 - w) * D, Y, A, B)
        mark = "   <- the zero-parameter choice" if abs(w - 0.5) < 1e-9 else ""
        print(f"  w_qwen {w:>4.2f}   xfit macro-F1 {xf:.4f}"
              f"   {xf - base_xf:>+7.4f}{mark}")
    print("\nPicking the best w here fits one parameter on the same validation")
    print("data it is read on, so its apparent gain is not transferable.")
    print("Read the 0.50 row and nothing else.")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    best = max(("Qwen", "blend 0.5 prob", "blend 0.5 rank"),
               key=lambda k: res[k][1])
    print(f"  best zero-parameter arm : {best}  "
          f"({res[best][1]:.4f}, {res[best][1]-base_xf:+.4f})")

    # The headline delta is over ALL 24 labels, so a label with four positives
    # counts as much as one with 187. Extending to five folds is only justified
    # if the gain survives dropping the labels too small to measure.
    m5 = n >= 5
    d_all = pdq[1].mean() - pdq[0].mean()
    d_5 = pdq[1][m5].mean() - pdq[0][m5].mean()
    print(f"\n  cross-fitted F1 delta, all 24 labels : {d_all:+.4f}")
    print(f"  cross-fitted F1 delta, n >= 5        : {d_5:+.4f}"
          f"   ({int(m5.sum())} labels)")
    if d_5 > 0.010:
        print("\n  -> EXTEND. The gain is not an artefact of the rare labels.")
    elif d_5 > 0.0:
        print("\n  -> MARGINAL. Positive but inside the noise on 653 posts.")
    else:
        print("\n  -> STOP. The all-label figure is carried by labels with a")
        print("     handful of positives; on everything measurable Qwen is")
        print("     behind. A ranking gain that will not threshold is not a")
        print("     score gain, and three more folds will not change that.")


if __name__ == "__main__":
    main()
