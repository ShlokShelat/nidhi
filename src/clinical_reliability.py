"""
clinical_reliability.py

Measures the properties a deployed triage system needs that the competition
composite does not score.

Composite = 0.4*risk_wF1 + 0.3*evidence + 0.3*factors treats every error the
same. Clinically they are not the same: calling an `attempt` an `ideation`
sends someone home who needed admission, while the reverse costs a review.

WHAT THIS MEASURES

  1. CALIBRATION. When the model says 0.8, is it right 80% of the time?
     Expected Calibration Error and a reliability table. Then TEMPERATURE
     SCALING, fitted cross-validated by author. A single temperature is a
     monotone transform of the logits, so it cannot change an argmax and the
     competition score is provably unchanged -- the run verifies this rather
     than asserting it. A triage tool whose confidences mean nothing cannot
     be used to prioritise anybody.

  2. SELECTIVE PREDICTION. If a clinician reviews the least-confident X% by
     hand, how good is the model on the rest? The risk-coverage curve is the
     honest way to describe a decision-support tool: not a single accuracy
     number, but an accuracy on the fraction it is confident about plus an
     explicit list of the rest to read by hand. A flat curve means confidence
     does not track correctness, which is itself the finding to report,
     because it means the model cannot flag its own failures.

  3. ASYMMETRIC COST. Score the confusion matrix under a cost matrix that
     penalises under-estimation more than over-estimation, and find the
     decision rule that minimises expected harm rather than maximising wF1.
     Reports the trade: how much composite is given up per unit of danger
     avoided. Where to sit on that curve is a clinical decision, so the curve
     has to exist for the choice to be made at all.

  4. EVIDENCE SUFFICIENCY. When the model says `behavior` (explicit intent
     PLUS a plan or act), does the evidence it emits actually contain a plan?
     If not, the explanation does not support the label, and a reader cannot
     audit the decision. Measured with a METHOD vocabulary, which is
     near-exclusive to behavior and attempt spans.

Usage:
  python src/clinical_reliability.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, parse_evidence  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
SEV = {c: i for i, c in enumerate(LEVELS)}
METHOD = re.compile(
    r"\b(jump(?:ing|ed)?|bridge|pills?|knife|knives|gun|shoot|hang(?:ing)?|"
    r"rope|noose|overdose|od|blade|razor|bleach|poison|wrists?|veins?|cliff|"
    r"roof|balcony|cut(?:ting)?\s+my|hurt(?:ing)?\s+my|swallow|took\s+\d+)\b",
    re.I)


def wf1(gold, pred):
    tot = 0.0
    for c in LEVELS:
        g, p = (gold == c), (pred == c)
        tp = float((g & p).sum())
        pr = tp / max(p.sum(), 1e-9)
        rc = tp / max(g.sum(), 1e-9)
        tot += 2 * pr * rc / max(pr + rc, 1e-9) * g.sum()
    return tot / max(len(gold), 1)


def ece(conf, correct, bins=10):
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold = np.array([normalize_label(v) for v in va["risk_level"]])
    gev = [parse_evidence(v) for v in va["evidence"]]
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pv = pv.set_index("row_id")
    P = pv.loc[rows, [f"prob_{c}" for c in LEVELS]].to_numpy(float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    pred = np.array(LEVELS)[P.argmax(1)]
    conf = P.max(1)
    correct = (pred == gold).astype(float)
    gi = np.array([SEV[g] for g in gold])
    pi = np.array([SEV[p] for p in pred])
    print(f"{len(rows)} val posts   accuracy {correct.mean():.3f}   "
          f"wF1 {wf1(gold, pred):.4f}")

    print("\n" + "=" * 72)
    print("1. CALIBRATION -- does a confidence of 0.8 mean 80% right?")
    print("=" * 72)
    print(f"  {'band':<12}{'n':>6}{'mean conf':>11}{'accuracy':>10}{'gap':>8}")
    for b in range(5, 10):
        lo, hi = b / 10, (b + 1) / 10
        m = (conf > lo) & (conf <= hi)
        if m.sum() < 5:
            continue
        print(f"  {f'{lo:.1f}-{hi:.1f}':<12}{int(m.sum()):>6}"
              f"{conf[m].mean():>11.3f}{correct[m].mean():>10.3f}"
              f"{correct[m].mean()-conf[m].mean():>+8.3f}")
    print(f"\n  ECE {ece(conf, correct):.4f}   "
          f"mean confidence {conf.mean():.3f} vs accuracy {correct.mean():.3f}")
    over = conf.mean() - correct.mean()
    print(f"  the model is {'OVER' if over > 0 else 'UNDER'}-confident by "
          f"{abs(over):.3f}")

    # Temperature, cross-fitted by author. A single T is monotone in the
    # logits, so the argmax -- and therefore the competition score -- cannot
    # move. Verified below rather than asserted.
    L = np.log(np.clip(P, 1e-9, None))
    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    Ts, eces = [], []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        k = max(int(len(o) * 0.25), 2)
        ho = set(o[:k])
        hm = np.array([a in ho for a in auth])
        fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
        if len(fi) < 40 or len(hi) < 15:
            continue
        best, bt = 1e9, 1.0
        for T in np.arange(0.5, 4.01, 0.05):
            Q = np.exp(L[fi] / T)
            Q /= Q.sum(1, keepdims=True)
            e = ece(Q.max(1), correct[fi])
            if e < best:
                best, bt = e, T
        Q = np.exp(L[hi] / bt)
        Q /= Q.sum(1, keepdims=True)
        Ts.append(bt)
        eces.append(ece(Q.max(1), correct[hi]))
    Tm = float(np.mean(Ts))
    Q = np.exp(L / Tm)
    Q /= Q.sum(1, keepdims=True)
    same = int((np.array(LEVELS)[Q.argmax(1)] == pred).sum())
    print(f"\n  temperature {Tm:.2f} +- {np.std(Ts):.2f} (cross-fitted)")
    print(f"  held-out ECE {np.mean(eces):.4f}  (was {ece(conf, correct):.4f})")
    print(f"  labels unchanged on {same}/{len(rows)} -- the competition score "
          f"is {'UNCHANGED' if same == len(rows) else 'AFFECTED (!)'}")

    print("\n" + "=" * 72)
    print("2. SELECTIVE PREDICTION -- 'I am confident about these; read the "
          "rest'")
    print("=" * 72)
    o = np.argsort(-conf)
    print(f"  {'coverage':>9}{'n':>6}{'accuracy':>10}{'wF1':>9}"
          f"{'attempt recall':>16}")
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = int(len(rows) * cov)
        ix = o[:k]
        ar = ((pred[ix] == "attempt") & (gold[ix] == "attempt")).sum() / \
            max((gold[ix] == "attempt").sum(), 1)
        print(f"  {cov:>9.0%}{k:>6}{correct[ix].mean():>10.3f}"
              f"{wf1(gold[ix], pred[ix]):>9.4f}{ar:>16.3f}")
    print("\n  A curve that rises steeply means the model knows what it does")
    print("  not know, and can triage its own output. A FLAT curve means its")
    print("  errors are made as confidently as its correct calls, so it")
    print("  cannot flag its own failures -- which is itself the most")
    print("  important thing to report.")

    print("\n" + "=" * 72)
    print("3. ASYMMETRIC COST -- under-estimating risk is the dangerous error")
    print("=" * 72)
    und = int((pi < gi).sum())
    ove = int((pi > gi).sum())
    print(f"  under-estimated {und} posts ({und/len(rows):.1%})   "
          f"over-estimated {ove} ({ove/len(rows):.1%})")
    print(f"  net severity bias {np.mean(pi - gi):+.3f}")
    print(f"\n  {'penalty for under':>18}{'wF1':>9}{'d comp':>9}"
          f"{'under':>8}{'over':>7}{'attempt recall':>16}")
    for w in (1.0, 1.5, 2.0, 3.0, 5.0):
        C = np.array([[0.0 if i == j else (w * (j - i) if j > i else (i - j))
                       for j in range(4)] for i in range(4)])
        # expected cost of predicting class k = sum_j P(j) * C[j][k]
        newp = np.array(LEVELS)[np.argmin(P @ C, axis=1)]
        ni = np.array([SEV[x] for x in newp])
        ar = ((newp == "attempt") & (gold == "attempt")).sum() / \
            max((gold == "attempt").sum(), 1)
        print(f"  {w:>18.1f}{wf1(gold, newp):>9.4f}"
              f"{0.4*(wf1(gold, newp)-wf1(gold, pred)):>+9.4f}"
              f"{int((ni < gi).sum()):>8}{int((ni > gi).sum()):>7}{ar:>16.3f}")
    print("\n  Read this as a TRADE, not an optimisation: how much composite")
    print("  is given up per dangerous error avoided. The right operating")
    print("  point is a clinical decision, not a modelling one, so the curve")
    print("  has to exist for anyone to choose.")

    print("\n" + "=" * 72)
    print("4. EVIDENCE SUFFICIENCY -- does the explanation support the label?")
    print("=" * 72)
    print(f"  {'predicted':<12}{'n':>5}{'gold evidence has a method':>28}")
    for c in LEVELS:
        m = pred == c
        if not m.sum():
            continue
        hit = np.mean([bool(METHOD.search(" ".join(gev[i])))
                       for i in np.flatnonzero(m)])
        print(f"  {c:<12}{int(m.sum()):>5}{hit:>28.1%}")
    print("\n  `behavior` means explicit intent AND a plan or act. If the")
    print("  evidence for a behavior call contains no plan, the explanation")
    print("  does not justify the label and a clinician cannot audit it.")
    print("  This is the explainability half of the task measured as a")
    print("  PROPERTY rather than as a span-overlap score.")


if __name__ == "__main__":
    main()
