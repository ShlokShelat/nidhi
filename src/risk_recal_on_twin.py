"""
risk_recal_on_twin.py

Tests whether a linear recalibration of the risk model's own four logits
generalizes, evaluated on a twin that reproduces the deployed model rather
than an out-of-fold proxy.

An out-of-fold proxy (`champion_oof.csv`) scores 0.786 where the deployed
model scores 0.826, because it lacks the rules+route layer (0.7896 -> 0.8294
with that layer applied). A recalibration fitted on that proxy risks partly
re-deriving corrections that the rules and route_attempt already apply,
which can look valuable against the proxy while adding nothing once the
same rules run on the real model's output.

The twin used here reproduces the deployed model's weighted-F1 (0.8294
against the deployed 0.826). The honest test applies rules + route_attempt
ON TOP of the recalibrated probabilities, and asks whether anything survives
the symbolic layer.

THREE MODELS, INCREASING CAPACITY -- because 372 posts is not much to fit on:

    temperature   1 param   logits / T. Fixes over/under-confidence only.
    per-class     4 params  logits + b_c. Fixes class prior drift.
    logistic     20 params  the full recalibration.

If only the 20-param version wins, that is capacity fitting noise, not a real
miscalibration -- a genuine calibration error shows up in the 1-param fix too.

Everything is cross-fitted by AUTHOR: coefficients from FIT authors, scored on
HELD-OUT authors, never both. Read fix/break and the share of resamples that
LOSE, not the confidence interval of the mean delta, which understates the
per-draw variance a single held-out evaluation draw actually samples from.

BAR: >=90% win AND >=2:1 fix/break.

Usage:
  python src/risk_recal_on_twin.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rules_to_test import apply_rules  # noqa: E402
from scoring import normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def wf1(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    tot = 0.0
    for c in LEVELS:
        g, p = (gold == c), (pred == c)
        tp = float((g & p).sum())
        pr = tp / max(p.sum(), 1e-9)
        rc = tp / max(g.sum(), 1e-9)
        tot += 2 * pr * rc / max(pr + rc, 1e-9) * g.sum()
    return tot / max(len(gold), 1)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--route", default="artifacts/cot_soft_oof.csv")
    a_.add_argument("--splits", type=int, default=300)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold = np.array([normalize_label(v) for v in va["risk_level"]])
    posts = list(va["post"].astype(str))
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    yi = np.array([LEVELS.index(g) for g in gold])

    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pv = pv.set_index("row_id")
    M = pv.loc[rows, [f"prob_{c}" for c in LEVELS]].to_numpy(float)
    L = np.log(np.clip(M, 1e-9, None))

    route = {}
    if os.path.exists(args.route):
        d = pd.read_csv(args.route, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        c = "cot" if "cot" in d.columns else "pred_risk_level"
        route = {r: normalize_label(v) for r, v in zip(d["row_id"], d[c])}

    def finalize(X, idx):
        lab = [LEVELS[i] for i in X.argmax(1)]
        lab = [normalize_label(apply_rules(posts[i], l)[0])
               for l, i in zip(lab, idx)]
        return np.array([("attempt" if (route.get(rows[i]) == "attempt"
                                        and x != "indicator") else x)
                         for x, i in zip(lab, idx)])

    allix = np.arange(len(rows))
    base = finalize(M, allix)
    print(f"twin {len(rows)} posts / {len(pd.unique(auth))} authors")
    print(f"  base (argmax + rules + route)  {wf1(gold, base):.4f}")
    print(f"  argmax only                    "
          f"{wf1(gold, np.array(LEVELS)[M.argmax(1)]):.4f}\n")

    from sklearn.linear_model import LogisticRegression

    def fit_apply(kind, fi, hi):
        if kind == "temperature":
            best, bt = -1e9, 1.0
            for t in np.arange(0.4, 3.01, 0.05):
                v = wf1(gold[fi], finalize(np.exp(L[fi] / t), fi))
                if v > best:
                    best, bt = v, t
            return np.exp(L[hi] / bt), f"T={bt:.2f}"
        if kind == "per-class":
            b = np.zeros(4)
            for _ in range(3):                    # coordinate ascent, 4 params
                for c in range(4):
                    best, bb = -1e9, b[c]
                    for v in np.arange(-1.5, 1.51, 0.1):
                        bb_ = b.copy()
                        bb_[c] = v
                        s = wf1(gold[fi], finalize(np.exp(L[fi] + bb_), fi))
                        if s > best:
                            best, bb = s, v
                    b[c] = bb
            return np.exp(L[hi] + b), "b=" + ",".join(f"{x:+.1f}" for x in b)
        m = LogisticRegression(C=1.0, max_iter=500, multi_class="multinomial")
        m.fit(L[fi], yi[fi])
        return m.predict_proba(L[hi]), f"{m.coef_.size + m.intercept_.size}p"

    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    print(f"  {'model':<14}{'moved':>7}{'fix':>5}{'brk':>5}{'f/b':>7}"
          f"{'d comp':>9}{'win':>6}{'LOSS':>6}{'sd':>9}   params")
    out = []
    for kind in ("temperature", "per-class", "logistic"):
        ds, mv, fx, bk, notes = [], [], [], [], []
        for _ in range(args.splits):
            o = list(ua)
            rng.shuffle(o)
            k = max(int(len(o) * 0.25), 2)
            ho = set(o[:k])
            hm = np.array([a in ho for a in auth])
            fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
            if len(fi) < 40 or len(hi) < 15 or len(np.unique(yi[fi])) < 4:
                continue
            try:
                X, note = fit_apply(kind, fi, hi)
            except Exception:                       # noqa: BLE001
                continue
            new, ref = finalize(X, hi), base[hi]
            ds.append(wf1(gold[hi], new) - wf1(gold[hi], ref))
            ch = new != ref
            mv.append(int(ch.sum()))
            fx.append(int(((new == gold[hi]) & (ref != gold[hi]) & ch).sum()))
            bk.append(int(((ref == gold[hi]) & (new != gold[hi]) & ch).sum()))
            notes.append(note)
        if not ds:
            print(f"  {kind:<14}   (no valid splits)")
            continue
        d = np.array(ds)
        f_, b_ = float(np.mean(fx)), float(np.mean(bk))
        print(f"  {kind:<14}{np.mean(mv):>7.1f}{f_:>5.1f}{b_:>5.1f}"
              f"{f_/max(b_,1e-9):>7.2f}{0.4*d.mean():>+9.4f}"
              f"{np.mean(d > 0):>5.0%}{np.mean(d < 0):>6.0%}"
              f"{0.4*d.std():>9.4f}   {notes[0]}")
        out.append((float(np.mean(d > 0)), 0.4 * d.mean(), f_, b_, kind))

    print("\n  If ONLY `logistic` wins, that is 20 parameters fitting 372")
    print("  posts -- a real miscalibration also shows in the 1-param fix.")
    print("  A recalibration measured against champion_oof is measured against")
    print("  a base with no rules+route layer, which this base does carry --")
    print("  a result there does not necessarily hold here.")
    if out:
        out.sort(reverse=True)
        w, dc, f_, b_, kind = out[0]
        ok = w >= 0.90 and f_ >= 2 * max(b_, 1e-9)
        print(f"\n  BEST: {kind}  {dc:+.4f} at {w:.0%} win, "
              f"{f_:.1f} fix / {b_:.1f} break")
        print(f"  {'CLEARS' if ok else 'does NOT clear'} the bar "
              f"(>=90% win AND >=2:1 fix/break).")


if __name__ == "__main__":
    main()
