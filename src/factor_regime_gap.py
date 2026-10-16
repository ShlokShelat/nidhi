"""
factor_regime_gap.py

THE OOF AND TEST PROBABILITIES COME FROM DIFFERENT KINDS OF MODEL.

    OOF   one fold model per post, each trained on 1308 posts
    TEST  the AVERAGE of five fold models   (`bagged 5 folds -> *_cv_test.npz`)

Averaging five models pulls probabilities toward the middle. So the test
distribution is NARROWER than the OOF distribution, and `tfit` fitted its 24
thresholds on the wide one and applies them to the narrow one.

The shipped build already shows the signature: **3.02 factors/post on test**
against the ~3.15 those thresholds produce on OOF. A threshold above the median
fires on fewer posts once the tail is compressed.

This is the same OOF/test regime-shift mechanism documented on the risk side
-- one fold model per post in OOF vs a 5-fold average in test -- which has
previously cost measurable score there. Nobody had checked whether factors
had it too.

WHAT THIS MEASURES, per label, in the units that matter:
  * mean and SD of the OOF vs test probability, in LOGIT space where the
    shrinkage actually lives
  * the fire rate a tfit threshold produces on OOF vs on test. If test fires
    systematically less, the decode is being throttled by the regime gap and
    not by anything about the test posts.
  * the same comparison after a SD-MATCHING correction: rescale test logits
    about their own mean so their spread matches OOF's. That undoes the
    ensembling shrinkage WITHOUT forcing the fire rate to a training estimate
    (which is the quantile decode, and it lost to thresholds by 0.0183).

NOTHING HERE IS FITTED TO A SCORE. The correction uses only the two probability
distributions, never a label, so it costs zero fitted parameters.

READ IT AS A GO/NO-GO. If the SDs match, the regime gap is imaginary and this
line closes for free. If test is materially narrower, there are two fixes and
they are worth different things:
  cheap   SD-match the test logits (CPU, minutes, unvalidatable offline)
  clean   retrain the 8 arms on ALL 1635 posts and score test with the SINGLE
          full-data model, so test and OOF are the same kind of object AND the
          rare labels get 25% more gold. ~8 GPU-hours, and it is exactly what
          `risk_lora_fulldata.sh` does for risk -- factors never got it.

Usage:
  python src/factor_regime_gap.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
PROTECTIVE = {"social support", "coping strategy", "psychological capital",
              "sense of responsibility", "meaning in life"}
TGRID = np.round(np.arange(0.02, 0.985, 0.005), 3)


def load_oof(d, arms, k):
    O, rid = [], None
    for a in arms:
        pr, ids = [], []
        for f in range(k):
            p = f"{d}/{a}_cv_oof_f{f}.npz"
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
            z = np.load(p, allow_pickle=True)
            pr.append(z["probs"].astype(float))
            ids.extend(str(x) for x in z["row_id"])
        pr = np.concatenate(pr)
        if rid is None:
            rid = ids
        elif ids != rid:
            o = {x: i for i, x in enumerate(ids)}
            pr = pr[[o[x] for x in rid]]
        O.append(pr)
    return rid, np.stack(O)


def resolve_test(d, a):
    for c in (f"{d}/{a}_cv_test.npz", f"{d}/{a}_test.npz",
              f"{d}/nli_ensemble_{a}_test.npz"):
        if os.path.exists(c):
            return c
    return None


def logit_mean(S):
    X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def best_thr(p, y):
    npos = int(y.sum())
    if npos < 2:
        return 0.5
    o = np.argsort(-p, kind="stable")
    ps, ys = p[o], y[o]
    cum = np.cumsum(ys)
    ks = np.searchsorted(-ps, -TGRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    f1 = 2.0 * tp / np.maximum(ks + npos, 1)
    return float(TGRID[int(np.argmax(f1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, 5)
    P = logit_mean(S)
    n = len(rid)

    T = []
    for a in args.arms:
        p = resolve_test(args.dir, a)
        if p is None:
            raise SystemExit(f"no test dump for {a}")
        z = np.load(p, allow_pickle=True)
        T.append(z["probs"].astype(float))
    Pt = logit_mean(np.stack(T))
    nT = len(Pt)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = np.zeros((n, N), dtype=np.int8)
    for i, r in enumerate(rid):
        for x in set(parse_factors(str(lab.loc[r, "factors"]))):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1

    thr = np.array([best_thr(P[:, j], Y[:, j]) for j in range(N)])

    Lo, Lt = lg(P), lg(Pt)
    print(f"OOF {n} posts (single fold model each)   "
          f"TEST {nT} posts (average of 5 fold models)\n")
    print("=" * 104)
    print("LOGIT-SPACE DISTRIBUTIONS, and what the tfit threshold actually "
          "fires")
    print("=" * 104)
    print(f"{'label':<42}{'oof sd':>8}{'test sd':>9}{'ratio':>7}"
          f"{'oof mu':>8}{'test mu':>9}{'fire oof':>10}{'fire test':>10}"
          f"{'diff':>8}")
    ratios, dfire = [], []
    for j in range(N):
        so, st = Lo[:, j].std(), Lt[:, j].std()
        fo = float((P[:, j] >= thr[j]).mean())
        ft = float((Pt[:, j] >= thr[j]).mean())
        ratios.append(st / max(so, 1e-9))
        dfire.append(ft - fo)
        star = "* " if FACTOR_TAXONOMY[j] in PROTECTIVE else "  "
        print(f"{star}{FACTOR_TAXONOMY[j]:<40}{so:>8.2f}{st:>9.2f}"
              f"{st/max(so,1e-9):>7.2f}{Lo[:, j].mean():>8.2f}"
              f"{Lt[:, j].mean():>9.2f}{100*fo:>9.1f}%{100*ft:>9.1f}%"
              f"{100*(ft-fo):>+8.1f}")
    ratios = np.array(ratios)
    dfire = np.array(dfire)
    print(f"\n  median SD ratio (test/oof)  {np.median(ratios):.3f}"
          f"   labels below 0.95: {(ratios < 0.95).sum()} of {N}")
    print(f"  mean fire-rate shift        {100*dfire.mean():+.2f} pp"
          f"   total: OOF {100*(P >= thr).mean()*N/N:.2f} vs "
          f"TEST {100*(Pt >= thr).mean():.2f} per label")
    print(f"  factors/post   OOF {(P >= thr).sum()/n:.2f}   "
          f"TEST {(Pt >= thr).sum()/nT:.2f}")

    # ---- what an SD-match would do ----------------------------------------
    Ladj = np.empty_like(Lt)
    for j in range(N):
        so, st = Lo[:, j].std(), Lt[:, j].std()
        Ladj[:, j] = Lt[:, j].mean() + (Lt[:, j] - Lt[:, j].mean()) * (
            so / max(st, 1e-9))
    Padj = 1.0 / (1.0 + np.exp(-Ladj))
    on0 = (Pt >= thr[None, :]).astype(np.int8)
    on1 = (Padj >= thr[None, :]).astype(np.int8)
    print("\n" + "=" * 104)
    print("IF WE SD-MATCHED THE TEST LOGITS TO OOF (0 fitted parameters)")
    print("=" * 104)
    print(f"  factors/post   now {on0.sum()/nT:.2f}  ->  "
          f"corrected {on1.sum()/nT:.2f}   (OOF target "
          f"{(P >= thr).sum()/n:.2f}, gold 2.92)")
    print(f"  cells changed  {int((on0 != on1).sum())} of {nT*N}   "
          f"posts changed {int((on0 != on1).any(1).sum())} of {nT}")
    print(f"  0-factor posts {int((on0.sum(1) == 0).sum())} -> "
          f"{int((on1.sum(1) == 0).sum())}   (gold rate 7.8%)")

    print("\n" + "=" * 104)
    print("VERDICT")
    print("=" * 104)
    if np.median(ratios) > 0.95:
        print("  SDs match. The 5-fold average is NOT materially narrower and")
        print("  this whole line is closed for free -- no correction, no")
        print("  retraining. tfit is applying its thresholds to the")
        print("  distribution it was fitted on.")
    else:
        print(f"  Test logits are {100*(1-np.median(ratios)):.0f}% narrower. "
              f"The regime gap is REAL and")
        print("  it is throttling the decode. Two fixes:")
        print("    cheap  SD-match the test logits -- 0 fitted parameters, but")
        print("           NOT validatable offline, so it is a faith change on")
        print("           top of a decode that is already earning real gains.")
        print("    clean  retrain the 8 arms on ALL 1635 posts and score test")
        print("           with the SINGLE full-data model. Then OOF and test")
        print("           are the same kind of object AND rare labels get 25%")
        print("           more gold. ~8 GPU-hours. This is what")
        print("           risk_lora_fulldata.sh does for risk; factors never")
        print("           got it.")
        print("  Prefer the clean one. The cheap one trades a known distortion")
        print("  for an unmeasured correction.")


if __name__ == "__main__":
    main()
