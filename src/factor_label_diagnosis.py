"""
factor_label_diagnosis.py

Full 24-label breakdown of the factor model on OUT-OF-FOLD predictions over
all 1635 labelled posts -- the diagnostic that should precede any per-label
intervention.

Why this is needed before "fix the six rare labels". Those six measured
0.125 / 0.000 / 0.222 / 0.571 / 0.500 / 0.444 on the 372-post val, which is
where the "rare labels are the problem" story came from. On 1635 OOF posts
the same labels measure 0.600 / 0.598 / 0.416 / 0.625 / 0.623 / 0.331 against
a macro of 0.6575. They are below average but not catastrophic, and val was
understating them badly because they have 0-5 positives there.

So the real question -- where does the missing macro-F1 actually live? -- is
still unanswered. This answers it, ranking every label by how much it costs
the macro average, and splitting each into a precision problem or a recall
problem, because those need opposite fixes:

    low precision, high recall  -> the model over-predicts: raise its bar
    high precision, low recall  -> the model misses: lower its bar or give it
                                   more signal
    both low                    -> the label is not learnable from this data
                                   and no thresholding will save it

Uses the CV dumps from factors_oof_cv.py, so every prediction is honest.

Usage:
  python src/factor_label_diagnosis.py
  python src/factor_label_diagnosis.py --arm perlabel   # compare a treatment
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

ARMS = {
    "ctrl": "artifacts/fac_oof_f{f}.npz",
    "perlabel": "v2_rebuild/artifacts/fac_perlabel_oof_f{f}.npz",
    "asl": "v2_rebuild/artifacts/fac_asl_oof_f{f}.npz",
}


def load_probs(patt, k, gmap, tax):
    """Raw probabilities + deployed thresholds + fold id per row."""
    # thresholds are PER FOLD -- each fold's rows must be scored with that
    # fold's own thresholds. Keeping a single vector applied fold 4's cut-offs
    # to all 1635 rows and understated the current F1.
    rids, PR, TH, FD = [], [], [], []
    for f in range(k):
        p = patt.format(f=f)
        if not os.path.exists(p):
            return None, None, None, None, None
        d = np.load(p, allow_pickle=True)
        rids.extend(list(d["row_id"]))
        PR.append(d["probs"])
        TH.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
        FD.extend([f] * len(d["row_id"]))
    PR = np.concatenate(PR, axis=0)
    TH = np.concatenate(TH, axis=0)          # (n_rows, n_labels)
    idx = {t: i for i, t in enumerate(tax)}
    Y = np.zeros_like(PR, dtype=int)
    for i, r in enumerate(rids):
        for fa in parse_factors(gmap.get(r, "")):
            if str(fa).strip().lower() in idx:
                Y[i, idx[str(fa).strip().lower()]] = 1
    return rids, Y, PR, TH, np.array(FD)


def load_arm(patt, k, gmap, tax):
    rids, P = [], []
    for f in range(k):
        p = patt.format(f=f)
        if not os.path.exists(p):
            return None, None, None
        d = np.load(p, allow_pickle=True)
        rids.extend(list(d["row_id"]))
        P.append((d["probs"] >= d["thresholds"]).astype(int))
    P = np.concatenate(P, axis=0)
    idx = {t: i for i, t in enumerate(tax)}
    Y = np.zeros_like(P)
    for i, r in enumerate(rids):
        for f in parse_factors(gmap.get(r, "")):
            if str(f).strip().lower() in idx:
                Y[i, idx[str(f).strip().lower()]] = 1
    return rids, Y, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--arm", default=None,
                    help="also show this treatment arm alongside ctrl")
    ap.add_argument("--sweep", action="store_true",
                    help="DIAGNOSTIC: can a threshold change fix each label? "
                         "Reports current / oracle / CROSS-FITTED F1. Only the "
                         "cross-fitted number is meaningful -- an oracle sweep "
                         "always shows a gain, and per-label tuning was already "
                         "measured to overfit by 0.0487 macro-F1 even on 1635 "
                         "posts. Do NOT ship thresholds from this.")
    ap.add_argument("--by-risk", action="store_true",
                    help="DIAGNOSTIC: does knowing the risk level make a factor "
                         "easier? Per-label F1 split by GOLD risk level, plus "
                         "P(factor|risk). Uses gold risk, so it is a ceiling: "
                         "deployment would use predicted risk (~80% accurate).")
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled)
    gmap = dict(zip(lab["row_id"], lab["factors"]))
    tax = FACTOR_TAXONOMY

    rids, Y, P = load_arm(ARMS["ctrl"], args.k, gmap, tax)
    if rids is None:
        raise SystemExit("missing artifacts/fac_oof_f*.npz -- run factors_oof_cv.py")
    n = len(rids)
    macro = f1_score(Y, P, average=None, zero_division=0)
    print(f"out-of-fold over {n} posts; macro-F1 = {macro.mean():.4f}\n")

    rows = []
    for j, t in enumerate(tax):
        y, p = Y[:, j], P[:, j]
        rows.append({
            "label": t,
            "gold": int(y.sum()),
            "pred": int(p.sum()),
            "ratio": p.sum() / max(y.sum(), 1),
            "P": precision_score(y, p, zero_division=0),
            "R": recall_score(y, p, zero_division=0),
            "F1": f1_score(y, p, zero_division=0),
        })
    df = pd.DataFrame(rows)
    df["cost"] = (1 - df["F1"]) / len(tax)   # macro-F1 points this label loses
    df = df.sort_values("cost", ascending=False)

    print(f"{'label':<44}{'gold':>6}{'pred':>6}{'p/g':>6}"
          f"{'P':>7}{'R':>7}{'F1':>7}{'cost':>8}  diagnosis")
    for _, r in df.iterrows():
        if r.F1 >= 0.75:
            dx = "fine"
        elif r.P < 0.45 and r.R >= 0.55:
            dx = "OVER-predicts -> raise bar"
        elif r.R < 0.45 and r.P >= 0.55:
            dx = "MISSES -> lower bar / more signal"
        elif r.P < 0.45 and r.R < 0.45:
            dx = "NOT LEARNABLE here"
        else:
            dx = "balanced, just weak"
        print(f"{r.label:<44}{r.gold:>6}{r.pred:>6}{r.ratio:>6.1f}"
              f"{r.P:>7.3f}{r.R:>7.3f}{r.F1:>7.3f}{r.cost:>8.4f}  {dx}")

    top = df.head(6)
    print(f"\nthe 6 costliest labels account for {top['cost'].sum():.4f} of the "
          f"{df['cost'].sum():.4f} macro-F1 deficit "
          f"({top['cost'].sum()/df['cost'].sum():.0%})")
    print(f"  -> {list(top['label'])}")

    print(f"\ndiagnosis counts: "
          f"{dict(pd.Series([('fine' if r.F1>=0.75 else 'over' if (r.P<0.45 and r.R>=0.55) else 'miss' if (r.R<0.45 and r.P>=0.55) else 'unlearnable' if (r.P<0.45 and r.R<0.45) else 'weak') for _, r in df.iterrows()]).value_counts())}")

    if args.sweep:
        r2, Y2, PR, TH, FD = load_probs(ARMS["ctrl"], args.k, gmap, tax)
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
        print(f"\n{'=' * 96}")
        print("THRESHOLD SWEEP -- is the signal already in the probabilities?")
        print(f"{'=' * 96}")
        print(f"{'label':<44}{'cur':>7}{'oracle':>8}{'CROSSFIT':>10}"
              f"{'curThr':>8}{'orcThr':>8}  verdict")
        gain_o = gain_c = 0.0
        for _, row in df.head(10).iterrows():
            j = tax.index(row["label"])
            y, p, th = Y2[:, j], PR[:, j], TH[:, j]
            cur = f1_score(y, (p >= th).astype(int), zero_division=0)
            f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
            orc_t = grid[int(np.argmax(f1s))]
            orc = max(f1s)
            # cross-fitted: tune on the other folds, apply to this one
            pred = np.zeros_like(y)
            for f in range(args.k):
                tr, te = FD != f, FD == f
                if y[tr].sum() == 0:
                    pred[te] = (p[te] >= th[te]).astype(int)
                    continue
                sc = [f1_score(y[tr], (p[tr] >= t).astype(int), zero_division=0)
                      for t in grid]
                pred[te] = (p[te] >= grid[int(np.argmax(sc))]).astype(int)
            cf = f1_score(y, pred, zero_division=0)
            gain_o += orc - cur
            gain_c += cf - cur
            v = ("THRESHOLD FIXABLE" if cf - cur > 0.03 else
                 "oracle only -- not recoverable" if orc - cur > 0.05 else
                 "already near-optimal")
            # thresholds differ per fold, so show their mean
            print(f"{row['label']:<44}{cur:>7.3f}{orc:>8.3f}{cf:>10.3f}"
                  f"{th.mean():>8.2f}{orc_t:>8.2f}  {v}")
        print(f"\nmacro-F1 effect over these labels:")
        print(f"  oracle      {gain_o/len(tax):+.4f}   <- CIRCULAR, ignore")
        print(f"  cross-fitted{gain_c/len(tax):+.4f}   <- the honest number")
        print("\nA label is only threshold-fixable if CROSSFIT beats cur. Where")
        print("oracle gains but crossfit does not, the 'signal' is fold-specific")
        print("noise and re-thresholding would not transfer.")
        return

    if args.by_risk:
        lab2 = pd.read_csv(args.labeled).set_index("row_id")
        from scoring import normalize_label
        LV = ["indicator", "ideation", "behavior", "attempt"]
        risks = np.array([LV.index(normalize_label(lab2.loc[r, "risk_level"]))
                          for r in rids])
        print(f"\n{'=' * 96}")
        print("PER-RISK BREAKDOWN for the costliest labels (GOLD risk = ceiling)")
        print(f"{'=' * 96}")
        print(f"{'label':<40}" + "".join(f"{l[:9]:>22}" for l in LV))
        print(f"{'':<40}" + "".join(f"{'P(f|r)  F1':>22}" for _ in LV))
        for _, row in df.head(6).iterrows():
            j = tax.index(row["label"])
            cells = ""
            for li in range(4):
                m = risks == li
                pr = Y[m, j].mean() if m.sum() else 0.0
                f = (f1_score(Y[m, j], P[m, j], zero_division=0)
                     if m.sum() and Y[m, j].sum() else float("nan"))
                cells += f"{pr:>10.3f}{f:>12.3f}"
            print(f"{row['label']:<40}{cells}")
        print("\nA factor whose P(f|risk) is flat across levels gains nothing")
        print("from risk conditioning. A steep gradient means the risk head")
        print("carries information the factor model is not using.")
        return

    if args.arm and args.arm in ARMS:
        r2, Y2, P2 = load_arm(ARMS[args.arm], args.k, gmap, tax)
        if r2 is None:
            print(f"\n({args.arm} dumps not found)")
            return
        m2 = f1_score(Y2, P2, average=None, zero_division=0)
        print(f"\n{args.arm} vs ctrl, per label (only where they differ by >0.02):")
        for j, t in enumerate(tax):
            if abs(m2[j] - macro[j]) > 0.02:
                print(f"  {t:<44}{macro[j]:>7.3f} -> {m2[j]:>7.3f}"
                      f"{m2[j]-macro[j]:>+8.3f}")


if __name__ == "__main__":
    main()
