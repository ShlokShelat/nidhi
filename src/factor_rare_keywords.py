"""
factor_rare_keywords.py

THE RAREST LABELS ARE WORTH THE MOST AND ARE THE MOST LEXICALLY CLOSED.

Factor macro-F1 weights all 24 labels equally, so ONE label is 1/24 = 0.0417
of S2 = **0.0125 of composite**. And several have almost no gold on 378 test
posts:

    sexual orientation related issues    8 gold / 1635  ->  ~1.9 on test
    exposure to others' suicide         14              ->  ~3.2
    poor school performance             16              ->  ~3.7
    substance use / cognitive deficits  33              ->  ~7.6

A label with 2 gold instances is decided by two decisions. Moving
`sexual orientation` from F1 0.5 to 1.0 is worth more composite than most
measured gains elsewhere in this project, from two posts.

WHY LEXICAL SHOULD WORK HERE AND FAILED BEFORE. `factor_lexical_knn.py` lost
on all 24 labels because whole-post TF-IDF cosine measures TOPICAL overlap,
and every post in this corpus is about depression. But these particular labels
are lexically CLOSED in a way suicidality is not: "gay", "trans", "coming out",
"my brother killed himself", "failing my exams" are unambiguous surface forms.
The encoders have 8 training examples; a keyword has none to learn from.

WHAT THIS MEASURES -- nothing is shipped from it:
  1. precision/recall of a hand-written pattern per rare label, against gold
  2. the same, RESTRICTED to the model's top-ranked posts, since a keyword is
     only useful where it disagrees with the pool
  3. whether a keyword-gated re-rank beats the pool's own ranking, per label,
     by AP -- the honest question

The patterns are written from the label DEFINITIONS, not by inspecting which
posts are gold. That distinction is the difference between a prior and a fit.

Usage:
  python src/factor_rare_keywords.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]

# Written from the taxonomy definitions. No post was inspected to build these.
KW = {
    "sexual orientation related issues": r"""
        \b(gay|lesbian|bisexual|bi-?sexual|trans|transgender|transitioning|
        queer|lgbt\w*|homosexual|homophobi\w*|coming out|came out|closeted|
        in the closet|my sexuality|same.?sex|asexual|nonbinary|non-?binary|
        pansexual|deadname\w*|pronouns)\b""",
    "exposure to others' suicide": r"""
        ((friend|brother|sister|mother|father|mom|dad|mum|cousin|uncle|aunt|
        classmate|teacher|neighbou?r|grandma|grandpa|boyfriend|girlfriend|
        someone\s+i\s+knew?)[^.!?\n]{0,60}
        (killed\s+(him|her|them)self|committed\s+suicide|took\s+(his|her|their)
        \s+own\s+life|hung\s+(him|her)self|hanged|overdosed|died\s+by\s+suicide)
        )|((killed\s+(him|her|them)self|committed\s+suicide|died\s+by\s+suicide)
        [^.!?\n]{0,40}(friend|brother|sister|mother|father|mom|dad|cousin))""",
    "poor school performance": r"""
        \b(failing|failed\s+(my|the|a)?\s*(class|exam|test|semester|year|
        course)|bad\s+grades|my\s+grades|gpa|dropped\s+out|drop\s+out|
        dropping\s+out|flunk\w*|held\s+back|expelled|suspended\s+from\s+school|
        academic\s+probation|behind\s+in\s+school|failing\s+school)\b""",
    "substance use": r"""
        \b(drunk|alcohol\w*|drinking|booze|weed|marijuana|cannabis|cocaine|
        heroin|meth|opioid|xanax|benzo\w*|getting\s+high|stoned|sober|
        relapse[d]?|addict\w*|substance\s+abuse|overdos\w*\s+on|
        drug\s+problem)\b""",
    "cognitive deficits": r"""
        \b(adhd|add\b|autis\w*|aspergers?|dyslex\w*|learning\s+disab\w*|
        memory\s+(problems|loss|issues)|can'?t\s+(concentrate|focus|remember)|
        brain\s+fog|cognitive|iq\b|special\s+ed\w*|slow\s+learner)\b""",
    "low socio-economic status": r"""
        \b(broke|poverty|homeless|evicted|can'?t\s+afford|no\s+money|
        food\s+stamps|welfare|benefits|minimum\s+wage|unemployed|jobless|
        laid\s+off|fired\s+from|debt|bills?\s+i\s+can'?t|rent)\b""",
}
RX = {k: re.compile(v, re.I | re.X) for k, v in KW.items()}


def cells(col, n):
    M = np.zeros((n, N), dtype=np.int8)
    for i, c in enumerate(col):
        for x in set(parse_factors(str(c))):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


def ap(p, y):
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum()
                 / ys.sum())


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--boost", type=float, default=2.0,
                    help="logit added to the pool when the keyword fires")
    args = a_.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    posts = lab["post"].astype(str).tolist()
    Y = cells(lab["factors"], len(lab))
    rid = lab["row_id"].tolist()

    pool = None
    try:
        O, r0 = [], None
        for a in BAG8:
            pr, ids = [], []
            for f in range(5):
                z = np.load(f"{args.dir}/{a}_cv_oof_f{f}.npz",
                            allow_pickle=True)
                pr.append(z["probs"].astype(float))
                ids.extend(str(x) for x in z["row_id"])
            pr = np.concatenate(pr)
            if r0 is None:
                r0 = ids
            elif ids != r0:
                o = {x: i for i, x in enumerate(ids)}
                pr = pr[[o[x] for x in r0]]
            O.append(pr)
        o = {x: i for i, x in enumerate(r0)}
        idx = [o[r] for r in rid if r in o]
        keep = [i for i, r in enumerate(rid) if r in o]
        pool = 1.0 / (1.0 + np.exp(-np.mean(logit(np.stack(O)), 0)))[idx]
        posts = [posts[i] for i in keep]
        Y = Y[keep]
        print(f"pool loaded for {len(idx)} of {len(rid)} posts")
    except Exception as e:                      # noqa: BLE001
        print(f"pool unavailable ({e}) -- keyword stats only")

    print("\n" + "=" * 100)
    print("HAND-WRITTEN KEYWORD vs GOLD, on the labels macro-F1 overpays for")
    print("=" * 100)
    print(f"{'label':<38}{'gold':>5}{'fires':>7}{'TP':>5}{'P':>7}{'R':>7}"
          f"{'F1':>7}{'poolAP':>8}{'kwAP':>7}{'mixAP':>7}")
    for name, rx in RX.items():
        j = IDX[name]
        y = Y[:, j]
        hit = np.array([bool(rx.search(t)) for t in posts])
        tp = int((hit & (y == 1)).sum())
        p = tp / max(int(hit.sum()), 1)
        r = tp / max(int(y.sum()), 1)
        f = 2 * p * r / max(p + r, 1e-9)
        pa = ka = ma = float("nan")
        if pool is not None:
            pa = ap(pool[:, j], y)
            ka = ap(hit.astype(float) + 1e-6 * pool[:, j], y)
            mixed = logit(pool[:, j]) + args.boost * hit
            ma = ap(mixed, y)
        print(f"{name:<38}{int(y.sum()):>5}{int(hit.sum()):>7}{tp:>5}"
              f"{p:>7.3f}{r:>7.3f}{f:>7.3f}{pa:>8.3f}{ka:>7.3f}{ma:>7.3f}"
              + ("  <- MIX WINS" if pool is not None and ma > pa else ""))

    print("\n  poolAP = the 8-arm ensemble alone")
    print("  kwAP   = the keyword alone as a ranker (ties broken by the pool)")
    print(f"  mixAP  = pool logit + {args.boost} where the keyword fires")
    print("\n  A keyword that fires on 40 posts to catch 8 golds is still")
    print("  useful IF it reorders the pool's top -- AP is what says so, and")
    print("  the decode only ever fires a handful of these labels anyway.")
    print("\n  One label moved from F1 0.5 to 1.0 is +0.0125 * 0.5 = +0.006")
    print("  composite. That is why these six are worth more attention than")
    print("  the eighteen common ones combined.")


if __name__ == "__main__":
    main()
