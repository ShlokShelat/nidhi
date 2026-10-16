"""
factor_span_count.py

Measures whether the span multiplicity discarded by the binary factor target
explains where the factor models lose macro-F1.

The gold `factors` column repeats a label once per supporting piece of
evidence, e.g. a post whose `psychological capital` label is backed by four
spans lists that label four times. The arms train on a de-duplicated 0/1
target, so a factor backed by four spans is indistinguishable from one
mentioned in passing.

THE HYPOTHESIS. If the missed factors are overwhelmingly SINGLE-span, part of
the measured error sits on borderline annotations rather than on real misses,
and the practical ceiling for factor macro-F1 is below 1.0 by a knowable
amount.

FOUR THINGS IT MEASURES
  1. is the count real -- does a post's total factor-count match its number of
     evidence spans? If yes, the repetition IS span provenance and not a
     formatting quirk.
  2. per label: how many golds are single-span, and which labels rest mostly
     on single-span annotations.
  3. recall by span count at the fire rates of the --champion submission. A
     steep gradient -- 3-span golds caught, 1-span golds missed -- means the
     model already tracks annotation STRENGTH, the signal the binary target
     discards.
  4. the counterfactual macro-F1 with single-span golds excluded. Not a
     submission -- the real scorer counts every gold -- but it sizes how much
     of the gap is weak annotation rather than genuine error.

The output decides whether a future factor model should train on counts (as
sample weights or an ordinal target) instead of on flat binaries.

Usage (cluster, CPU, <1 min):
  python src/factor_span_count.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_evidence, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
PROTECTIVE = {
    "social support", "coping strategy", "psychological capital",
    "sense of responsibility", "meaning in life",
}


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


def logit_mean(S):
    X = np.clip(np.asarray(S, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.mean(np.log(X / (1 - X)), axis=0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=BAG8)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--champion", default="submissions/NIDHI_routeatt.csv")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    rid, S = load_oof(args.dir, args.arms, args.k)
    P = logit_mean(S)
    n = len(rid)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    C = np.zeros((n, N), dtype=np.int16)       # span count per (post, label)
    nev = np.zeros(n, dtype=np.int16)          # evidence spans per post
    for i, r in enumerate(rid):
        for x in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                C[i, j] += 1
        nev[i] = len(parse_evidence(lab.loc[r, "evidence"]))
    Y = (C > 0).astype(np.int8)

    # ---------------- 1. is the count real? ---------------------------------
    tot = C.sum(1)
    both = (nev > 0)
    r = np.corrcoef(tot[both], nev[both])[0, 1] if both.sum() > 2 else float("nan")
    exact = int((tot == nev).sum())
    print("=" * 92)
    print("1. IS THE REPETITION REAL SPAN PROVENANCE?")
    print("=" * 92)
    print(f"  posts                                  {n}")
    print(f"  total factor mentions (with repeats)   {int(tot.sum())}")
    print(f"  distinct factors (de-duplicated)       {int(Y.sum())}")
    print(f"  total evidence spans                   {int(nev.sum())}")
    print(f"  posts where factor-count == span-count {exact} "
          f"({exact / n:.1%})")
    print(f"  correlation(factor mentions, spans)    {r:.3f}")
    if r > 0.8:
        print("  -> the repetition tracks evidence spans. The count is a real "
              "strength signal.")
    else:
        print("  -> weak correlation. The repetition is NOT simply one entry "
              "per span; treat")
        print("     the count as an annotator-agreement proxy at best.")

    print(f"\n  mentions per (post,label) pair:")
    dist = Counter(C[C > 0].ravel().tolist())
    for c in sorted(dist):
        print(f"    {c:>2} span(s): {dist[c]:>6}  ({dist[c] / Y.sum():>5.1%})")

    # ---------------- 2. per label ------------------------------------------
    print("\n" + "=" * 92)
    print("2. PER LABEL -- how flimsy is each label's evidence?")
    print("=" * 92)
    print(f"{'label':<44}{'gold':>6}{'mean':>7}{'%1span':>8}{'%3+':>7}")
    rows = []
    for j in range(N):
        m = C[:, j][C[:, j] > 0]
        if len(m) == 0:
            continue
        rows.append((j, len(m), m.mean(), (m == 1).mean(), (m >= 3).mean()))
    for j, g, mu, p1, p3 in sorted(rows, key=lambda x: -x[3]):
        tag = FACTOR_TAXONOMY[j]
        star = "* " if tag in PROTECTIVE else "  "
        print(f"{star}{tag:<42}{g:>6}{mu:>7.2f}{p1:>7.0%}{p3:>7.0%}")
    print("  * = protective   sorted by %1span, flimsiest first")

    # ---------------- 3. recall by span count -------------------------------
    ch = pd.read_csv(args.champion, keep_default_na=False)
    cc = np.zeros(N)
    for cell in ch["factors"]:
        for x in set(parse_factors(cell)):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                cc[j] += 1
    rates = cc / len(ch)
    pred = np.zeros((n, N), dtype=np.int8)
    for j in range(N):
        k = min(max(int(round(rates[j] * n)), 1), n - 1)
        cut = np.sort(P[:, j])[-k]
        pred[:, j] = (P[:, j] >= cut).astype(np.int8)

    print("\n" + "=" * 92)
    print("3. ARE THE FLIMSY GOLDS THE MISSED ONES? recall at --champion rates")
    print("=" * 92)
    print(f"  {'span count':<14}{'golds':>8}{'recall':>9}")
    for lo, hi, name in ((1, 1, "1 span"), (2, 2, "2 spans"),
                         (3, 99, "3+ spans")):
        m = (C >= lo) & (C <= hi)
        g = int(m.sum())
        if g:
            print(f"  {name:<14}{g:>8}{float(pred[m].mean()):>9.1%}")
    print("\n  a steep gradient means the model already tracks annotation "
          "STRENGTH --")
    print("  the exact signal the binary target throws away.")

    # ---------------- 4. counterfactual ceiling -----------------------------
    def macro(Pm, Ym, mask=None):
        v = []
        for j in range(N):
            p = Pm[:, j].astype(np.int64)
            y = Ym[:, j].astype(np.int64)
            if mask is not None:
                keep = ~mask[:, j]
                p, y = p[keep], y[keep]
            tp = int((p * y).sum())
            den = 2 * tp + int((p * (1 - y)).sum()) + int(((1 - p) * y).sum())
            v.append(0.0 if den == 0 else 2 * tp / den)
        return float(np.mean(v)), np.array(v)

    m_all, v_all = macro(pred, Y)
    weak = (C == 1)                       # single-span golds only
    m_str, v_str = macro(pred, Y, mask=weak)

    print("\n" + "=" * 92)
    print("4. HOW MUCH OF THE GAP IS WEAK ANNOTATION?")
    print("=" * 92)
    print(f"  macro-F1, all golds counted            {m_all:.4f}")
    print(f"  macro-F1, single-span golds ignored    {m_str:.4f}   "
          f"({m_str - m_all:+.4f})")
    print("\n  NOT a submission -- the real scorer counts every gold. This is "
          "the size of the")
    print("  penalty charged by borderline annotations, i.e. how much of the "
          "remaining gap is")
    print("  not actually winnable by a better model.")
    print(f"\n  {'label':<44}{'all':>8}{'strong':>8}{'delta':>8}")
    for j in np.argsort(-(v_str - v_all)):
        if abs(v_str[j] - v_all[j]) < 0.01:
            continue
        tag = FACTOR_TAXONOMY[j]
        star = "* " if tag in PROTECTIVE else "  "
        print(f"{star}{tag:<42}{v_all[j]:>8.3f}{v_str[j]:>8.3f}"
              f"{v_str[j] - v_all[j]:>+8.3f}")

    print("\n" + "=" * 92)
    print("WHAT THIS DECIDES")
    print("=" * 92)
    print("If the recall gradient in (3) is steep and (4) is large, the next "
          "factor model should")
    print("train on span COUNTS -- sample weights or an ordinal target -- "
          "rather than flat 0/1,")
    print("because the binaries are telling it that a passing mention and a "
          "four-span factor are")
    print("the same thing. If both are flat, the count carries nothing and "
          "the binary target is")
    print("correct as it stands.")


if __name__ == "__main__":
    main()
