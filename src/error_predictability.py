"""
error_predictability.py

IS "THE MODEL WILL GET THIS WRONG" PREDICTABLE FROM THE TEXT?

THE PRIZE IS MEASURED. `evidence_gated_hedge.py` established that hedging the
evidence conditioning pays only where P(risk wrong) > ~0.55 in the flagged
slice, because adding a span costs 0.33 on a correct post and gains ~0.30 on a
wrong one. The best confidence-based gate reaches **38.7%**, so the lane is
shut on confidence alone. The oracle upper bound for this lane is documented
in [[evidence-conditioning-oracle]].

WHY A DETECTOR MIGHT SUCCEED WHERE A CORRECTOR CANNOT. Every attempt to FIX
these posts failed -- the factor model separates the behavior/ideation cell
only 27%, the conditioned tagger reaches AUC 0.57-0.67 against the risk model's
0.80-0.84, and four LoRA'd LLM families all lost. But "classify this ambiguous
post correctly" and "notice that this post is ambiguous" are different
problems, and the second is usually easier: annotator difficulty is often
predictable even when the correct label is not.

[[risk-architecture-audit]] found errors are CONFIDENTLY wrong, which kills
confidence as a detector. It says nothing about TEXT as a detector.

THIS SCRIPT IS THE CHEAP PRE-CHECK, deliberately run before any GPU:
  * characterise wrong-risk posts against right ones on interpretable features
  * fit a logistic on those features, cross-fitted by author
  * report AUC and -- the number that matters -- PRECISION AT EACH COVERAGE,
    against the 38.7% bar and the 21% base rate
  * add the risk model's own confidence as a feature, to see whether text adds
    anything BEYOND what confidence already says

If simple features cannot beat 38.7%, a cross-encoder probably cannot either,
so this precheck saves the GPU spend. If they get close, a cross-encoder is
very likely to clear it and the oracle-bounded lane opens.

Uses the val twin, whose predictions ARE the deployed chain's
([[champion-risk-chain-reproduced]]) -- unlike the OOF artefacts, which
[[gate-validates-operating-point-not-components]] showed do not match what
ships.

Usage:
  python src/error_predictability.py
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
HEDGE = re.compile(r"\b(maybe|might|guess|sort of|kind of|i think|not sure|"
                   r"probably|perhaps|idk|i dunno|somewhat)\b", re.I)
EXPLICIT = re.compile(r"\b(kill myself|suicide|suicidal|end my life|end it|"
                      r"take my life|kms)\b", re.I)
PLAN = re.compile(r"\b(plan|method|tonight|tomorrow|when i|going to|"
                  r"about to|ready|decided)\b", re.I)
PAST = re.compile(r"\b(tried|attempted|last time|before|used to|years ago|"
                  r"when i was)\b", re.I)
NEG = re.compile(r"\b(not|never|don'?t|won'?t|wouldn'?t|no longer|"
                 r"can'?t)\b", re.I)
QUESTION = re.compile(r"\?")


def auc(score, pos):
    pos = np.asarray(pos, bool)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    return ((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
            / (pos.sum() * (~pos).sum()))


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
    posts = list(va["post"].astype(str))
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    gev = [parse_evidence(v) for v in va["evidence"]]
    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pv = pv.set_index("row_id")
    P = pv.loc[rows, [f"prob_{c}" for c in LEVELS]].to_numpy(float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    pred = np.array([normalize_label(v) for v in pv.loc[rows,
                                                        "pred_risk_level"]])
    wrong = (pred != gold).astype(int)
    print(f"{len(rows)} val posts / {len(pd.unique(auth))} authors")
    print(f"  the deployed chain is wrong on {wrong.mean():.1%} "
          f"({wrong.sum()} posts) -- that is the base rate any detector must "
          f"beat")
    print(f"  the bar for the evidence hedge to pay: 55% precision in the "
          f"flagged slice")

    srt = np.sort(P, axis=1)
    F, names = [], []

    def add(nm, v):
        F.append(np.asarray(v, float))
        names.append(nm)

    add("log_words", [np.log1p(len(p.split())) for p in posts])
    add("hedge_rate", [len(HEDGE.findall(p)) / max(len(p.split()), 1) * 100
                       for p in posts])
    add("explicit", [len(EXPLICIT.findall(p)) for p in posts])
    add("plan", [len(PLAN.findall(p)) for p in posts])
    add("past", [len(PAST.findall(p)) for p in posts])
    add("negation", [len(NEG.findall(p)) / max(len(p.split()), 1) * 100
                     for p in posts])
    add("questions", [len(QUESTION.findall(p)) for p in posts])
    add("n_gold_spans", [len(g) for g in gev])
    X_text = np.stack(F, axis=1)
    conf = srt[:, -1]
    margin = srt[:, -1] - srt[:, -2]
    entropy = -(P * np.log(np.clip(P, 1e-9, None))).sum(1)

    print("\n" + "=" * 74)
    print("1. WHAT DISTINGUISHES A WRONG POST FROM A RIGHT ONE?")
    print("=" * 74)
    print(f"  {'feature':<16}{'right':>10}{'WRONG':>10}{'ratio':>8}{'AUC':>8}")
    for j, nm in enumerate(names):
        r = X_text[wrong == 0, j].mean()
        w = X_text[wrong == 1, j].mean()
        print(f"  {nm:<16}{r:>10.2f}{w:>10.2f}"
              f"{(w / r if r else float('nan')):>8.2f}"
              f"{auc(X_text[:, j], wrong):>8.3f}")
    for nm, v in (("confidence", -conf), ("margin", -margin),
                  ("entropy", entropy)):
        print(f"  {nm:<16}{'':>10}{'':>10}{'':>8}{auc(v, wrong):>8.3f}")
    print("\n  AUC 0.500 = no information. Confidence is the incumbent gate")
    print("  and reaches 38.7% precision in its best slice.")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    sets = {"text only": X_text,
            "confidence only": np.stack([conf, margin, entropy], 1),
            "text + confidence": np.hstack(
                [X_text, np.stack([conf, margin, entropy], 1)])}
    print("\n" + "=" * 74)
    print("2. CROSS-FITTED DETECTOR -- can it beat the 55% bar?")
    print("=" * 74)
    for nm, Xs in sets.items():
        oof = np.full(len(rows), np.nan)
        for _ in range(args.splits):
            o = list(ua)
            rng.shuffle(o)
            k = max(int(len(o) * 0.25), 2)
            ho = set(o[:k])
            hm = np.array([a in ho for a in auth])
            fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
            if len(fi) < 40 or len(np.unique(wrong[fi])) < 2:
                continue
            sc = StandardScaler().fit(Xs[fi])
            m = LogisticRegression(C=0.5, max_iter=500).fit(sc.transform(Xs[fi]),
                                                            wrong[fi])
            p = m.predict_proba(sc.transform(Xs[hi]))[:, 1]
            oof[hi] = np.where(np.isnan(oof[hi]), p,
                               (oof[hi] + p) / 2)
        ok = ~np.isnan(oof)
        a = auc(oof[ok], wrong[ok])
        line = f"  {nm:<20}AUC {a:.3f}   precision@coverage:"
        for cov in (0.05, 0.10, 0.20, 0.30):
            k = max(int(ok.sum() * cov), 3)
            idx = np.argsort(-oof[ok])[:k]
            line += f"  {cov:.0%}:{wrong[ok][idx].mean():.0%}"
        print(line)
    print(f"\n  base rate {wrong.mean():.0%}   confidence-gate best 39%   "
          f"BAR 55%")
    print("  If `text + confidence` beats `confidence only` materially, the")
    print("  text carries information about difficulty that confidence does")
    print("  not -- and a cross-encoder reading the post properly would carry")
    print("  much more. If they are equal, the text adds nothing and a GPU")
    print("  detector will fail the same way.")

    print("\n" + "=" * 74)
    print("3. WHICH ERRORS WOULD A DETECTOR EVEN HELP WITH?")
    print("=" * 74)
    print("  the hedge only helps posts where the RIGHT span exists under a")
    print("  DIFFERENT conditioning, so count the cells that matter:")
    from collections import Counter
    cc = Counter((g, p) for g, p in zip(gold[wrong == 1], pred[wrong == 1]))
    for (g, p), n in cc.most_common(8):
        print(f"    gold {g:<10} pred {p:<10} {n:>4}")


if __name__ == "__main__":
    main()
