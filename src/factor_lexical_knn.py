"""
factor_lexical_knn.py

A NON-NEURAL FACTOR ARM. Every existing arm is a cross-encoder: a transformer
scoring (post, hypothesis). Those arms correlate 0.72-0.96 with the pool, so
adding another one leaves almost no independent variance for the average to
exploit.

A LEXICAL RETRIEVAL ARM IS A DIFFERENT KIND OF MODEL, aimed at the labels the
metric makes expensive. Macro-F1 weights all 24 factors equally, so the five
rarest are 21% of the factor score, and they are where the encoders are
weakest:

    sexual orientation related issues    8 gold   pool AP 0.664
    exposure to others' suicide         14 gold   pool AP 0.467
    poor school performance             16 gold   pool AP 0.801
    meaning in life                     45 gold   pool AP 0.232
    substance use                       33 gold   pool AP 0.257 (author-only)

With 8 positives a classifier has nothing to fit. But those labels carry
DISTINCTIVE VOCABULARY -- identity terms, bereavement-by-suicide phrasing,
school-failure wording. Nearest-neighbour over character and word n-grams can
work precisely where gradient descent cannot, and it shares no inductive bias
with a cross-encoder.

TWO SCORERS, both fit on FIT authors only and scored on held-out authors:
  knn    cosine similarity to the k nearest FIT posts carrying the label,
         which handles multi-modal labels (several distinct ways to express
         the same factor)
  proto  cosine to the centroid of all FIT positives, which is steadier when
         positives are very few

Reported as per-label AP against the 8-arm pool, because a rare-label arm is
judged on the labels it targets, not on a mean the common labels dominate.
Then tested as a 9th pool member -- both globally and RESTRICTED to the labels
where it actually beats the pool, which is the only form with a mechanism
behind it.

Usage:
  python src/factor_lexical_knn.py
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
TGRID = np.round(np.arange(0.02, 0.985, 0.005), 3)


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


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


def f1_curve(p, y):
    npos = int(y.sum())
    if npos == 0:
        return np.zeros(len(TGRID))
    o = np.argsort(-p, kind="stable")
    ys, ps = y[o], p[o]
    cum = np.cumsum(ys)
    ks = np.searchsorted(-ps, -TGRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    return 2.0 * tp / np.maximum(ks + npos, 1)


def f1pl(P, Y):
    P = P.astype(np.int64)
    Y = Y.astype(np.int64)
    tp = (P * Y).sum(0)
    den = 2 * tp + (P * (1 - Y)).sum(0) + ((1 - P) * Y).sum(0)
    return np.where(den == 0, 0.0, 2.0 * tp / np.maximum(den, 1))


def load_oof(d, arms, k=5):
    O, rid = [], None
    for a in arms:
        pr, ids = [], []
        for f in range(k):
            z = np.load(f"{d}/{a}_cv_oof_f{f}.npz", allow_pickle=True)
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


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--splits", type=int, default=60)
    a_.add_argument("--knn", type=int, default=5)
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    rid, S = load_oof(args.dir, BAG8)
    pool = sig(logit(S).mean(0))
    n = len(rid)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)
    posts = [str(lab.loc[r, "post"]) for r in rid]
    auth = np.asarray([str(lab.loc[r, "anon_user_id"]) for r in rid])
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    frac = 378.0 / (378.0 + n)

    print(f"{n} posts / {len(ua)} authors, {args.splits} author partitions")
    print("word 1-2 grams + char 3-5 grams, cosine; fit on FIT authors only\n")

    KN = np.zeros((n, N))
    PR = np.zeros((n, N))
    cnt = np.zeros(n)
    rng = np.random.default_rng(args.seed)
    for _ in range(args.splits):
        order = list(ua)
        rng.shuffle(order)
        ev, got = [], 0
        for a in order:
            if got >= frac * n:
                break
            ev.append(a)
            got += len(by[a])
        E = np.concatenate([by[a] for a in ev])
        F = np.setdiff1d(np.arange(n), E)

        vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                             max_features=60000, strip_accents="unicode")
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                             sublinear_tf=True, max_features=60000)
        tf = [posts[i] for i in F]
        te = [posts[i] for i in E]
        Xf = normalize(np.hstack([vw.fit_transform(tf).toarray(),
                                  vc.fit_transform(tf).toarray()]))
        Xe = normalize(np.hstack([vw.transform(te).toarray(),
                                  vc.transform(te).toarray()]))
        Sim = Xe @ Xf.T
        for j in range(N):
            pos = np.flatnonzero(Y[F, j] == 1)
            if len(pos) == 0:
                continue
            sub = Sim[:, pos]
            k = min(args.knn, len(pos))
            KN[E, j] += np.sort(sub, axis=1)[:, -k:].mean(1)
            PR[E, j] += sub.mean(1)
        cnt[E] += 1

    ok = cnt > 0
    KN[ok] /= cnt[ok, None]
    PR[ok] /= cnt[ok, None]
    print(f"scored {int(ok.sum())} of {n} posts as held-out at least once\n")

    print("=" * 96)
    print("PER-LABEL AP -- the lexical arm against the 8-arm neural pool")
    print("=" * 96)
    print(f"{'factor':<44}{'gold':>6}{'POOL':>8}{'knn':>8}{'proto':>8}"
          f"{'best-pool':>11}")
    wins = []
    order_j = np.argsort(Y.sum(0))
    for j in order_j:
        yy = Y[ok, j]
        if yy.sum() == 0:
            continue
        a_p = ap(pool[ok, j], yy)
        a_k = ap(KN[ok, j], yy)
        a_r = ap(PR[ok, j], yy)
        d = max(a_k, a_r) - a_p
        if d > 0:
            wins.append(j)
        print(f"{FACTOR_TAXONOMY[j]:<44}{int(Y[:, j].sum()):>6}{a_p:>8.3f}"
              f"{a_k:>8.3f}{a_r:>8.3f}{d:>+11.3f}"
              + ("  <-" if d > 0 else ""))
    print(f"\n  the lexical arm beats the neural pool on "
          f"{len(wins)} of {N} labels")
    if wins:
        print("  " + ", ".join(FACTOR_TAXONOMY[j] for j in wins))
    print("\n  A rare-label arm is judged on the labels it targets. Beating")
    print("  the pool on a handful of starved labels is the whole point --")
    print("  the mean is dominated by common labels it cannot help.")

    # ---- does it add to the pool? ----------------------------------------
    print("\n" + "=" * 96)
    print("AS A 9TH POOL MEMBER, cross-fitted, tfit decode")
    print("=" * 96)
    ZP = logit(S)
    variants = {"bag8": None, "bag8+knn": KN, "bag8+proto": PR}
    res = {k: [] for k in variants}
    res["bag8+knn(rare only)"] = []
    rng = np.random.default_rng(args.seed + 1)
    rare = np.array([j for j in range(N) if Y[:, j].sum() <= 60])
    for _ in range(args.splits):
        order = list(ua)
        rng.shuffle(order)
        ev, got = [], 0
        for a in order:
            if got >= frac * n:
                break
            ev.append(a)
            got += len(by[a])
        E = np.concatenate([by[a] for a in ev])
        F = np.setdiff1d(np.arange(n), E)
        for k, X in variants.items():
            if X is None:
                Z = ZP
            else:
                Z = np.concatenate([ZP, logit(np.clip(X, 1e-6, 1-1e-6))[None]])
            PF, PE = sig(Z[:, F].mean(0)), sig(Z[:, E].mean(0))
            th = np.array([float(TGRID[int(np.argmax(f1_curve(PF[:, j],
                                                              Y[F, j])))])
                           for j in range(N)])
            res[k].append(f1pl((PE >= th[None, :]).astype(np.int8),
                               Y[E]).mean())
        # rare-only: the lexical arm joins the pool for starved labels alone
        Z = np.concatenate([ZP, logit(np.clip(KN, 1e-6, 1-1e-6))[None]])
        Pm = sig(Z[:, F].mean(0))
        Pb = sig(ZP[:, F].mean(0))
        Pm[:, ~np.isin(np.arange(N), rare)] = Pb[:, ~np.isin(np.arange(N),
                                                             rare)]
        Pe = sig(Z[:, E].mean(0))
        Pbe = sig(ZP[:, E].mean(0))
        Pe[:, ~np.isin(np.arange(N), rare)] = Pbe[:, ~np.isin(np.arange(N),
                                                              rare)]
        th = np.array([float(TGRID[int(np.argmax(f1_curve(Pm[:, j],
                                                          Y[F, j])))])
                       for j in range(N)])
        res["bag8+knn(rare only)"].append(
            f1pl((Pe >= th[None, :]).astype(np.int8), Y[E]).mean())

    base = np.array(res["bag8"])
    print(f"{'variant':<24}{'macro':>9}{'delta':>10}{'win':>7}{'sd':>9}"
          f"{'one-draw 95%':>24}")
    for k in list(variants) + ["bag8+knn(rare only)"]:
        v = np.array(res[k])
        if k == "bag8":
            print(f"{k:<24}{v.mean():>9.4f}{'--':>10}{'--':>7}{'--':>9}"
                  f"{'--':>24}")
            continue
        d = v - base
        sd = d.std(ddof=1)
        print(f"{k:<24}{v.mean():>9.4f}{d.mean():>+10.4f}"
              f"{(d > 0).mean():>7.0%}{sd:>9.4f}"
              f"{f'[{d.mean()-1.96*sd:+.4f},{d.mean()+1.96*sd:+.4f}]':>24}")
    print(f"\n  'rare only' = the lexical arm joins the pool for the "
          f"{len(rare)} labels with <=60 gold\n  and is ignored elsewhere. "
          f"That is the version with a mechanism.")


if __name__ == "__main__":
    main()
