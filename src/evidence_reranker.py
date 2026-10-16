"""
evidence_reranker.py

Candidate generation + reranking for evidence spans.

Motivation, measured by headroom_analysis.py on val:

    best single tagger            F1 0.6969
    union of 4 taggers (as-is)    F1 0.6491   (precision collapses)
    PERFECT selector over union   F1 0.8617   <- the achievable ceiling

The candidate pool already contains 75.7% of all gold spans. Candidates are
not the shortage; selection among them is. `vote>=2` is a one-feature
reranker (count the models that agree) and it already beat every single
tagger. This replaces that single feature with a proper learned scorer.

Design choices, all aimed at NOT repeating this campaign's overfitting history:

  * LOGISTIC REGRESSION on a handful of cheap features, not a neural
    reranker. ~12 parameters against ~480 candidates. The factor-threshold
    experiment measured a 0.068 macro-F1 overfitting gap from fitting 24
    parameters to 186 posts; parameter count is the thing to control here.
  * Everything is evaluated AUTHOR-DISJOINT cross-fitted: fit on one half's
    authors, score the other, both directions, then measure with the official
    phrase metric over the combined out-of-fold predictions.
  * The decode threshold is fitted on the training half too, so it is inside
    the cross-fit rather than tuned on the scored data.

Features are all available with no new model: how many taggers proposed the
span, which ones, its length, where it sits in the post, and how it nests with
other candidates. No gold information leaks into them.

Usage:
  python src/evidence_reranker.py
  python src/evidence_reranker.py --cache-dir artifacts/evidence_per_run
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (parse_evidence, phrase_f1, token_len, normalize_span,
                     _span_pair_matches)
from rule_ablation_v2 import author_split

RUN_ORDER = ["evidence_v2", "evidence_v1", "evidence_s14", "evidence_s15"]


def load_runs(val, cache_dir, tag):
    runs = {}
    if cache_dir and os.path.isdir(cache_dir):
        for f in sorted(glob.glob(os.path.join(cache_dir, f"*__{tag}.csv"))):
            name = os.path.basename(f).split("__")[0]
            d = pd.read_csv(f, keep_default_na=False).set_index("row_id")["evidence"]
            runs[name] = [parse_evidence(d.get(r, "")) for r in val["row_id"]]
    if not runs:
        for r in RUN_ORDER:
            p = f"runs/{r}/val_predictions.csv"
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            col = next(c for c in d.columns if "pred" in c.lower())
            m = val[["row_id"]].merge(d[["row_id", col]], on="row_id", how="left")
            runs[r] = [parse_evidence(x) for x in m[col]]
    return runs


def build_candidates(val, runs, gold):
    """One row per (post, candidate span) with features and a match label."""
    names = list(runs)
    rows = []
    for i, post in enumerate(val["post"]):
        post_s = str(post)
        plen = max(len(post_s), 1)
        cands = []
        for n in names:
            for s in runs[n][i]:
                hit = next((c for c in cands if _span_pair_matches(s, c["span"])), None)
                if hit is None:
                    cands.append({"span": s, "by": {n}})
                else:
                    hit["by"].add(n)
                    # prefer the shorter surface form: the 3x cap punishes
                    # over-long spans and containment favours the tighter one
                    if token_len(s) < token_len(hit["span"]):
                        hit["span"] = s

        # greedy one-to-one gold assignment, mirroring the official matcher
        used = set()
        for c in cands:
            c["y"] = 0
        for gi, gspan in enumerate(gold[i]):
            for c in cands:
                if c["y"] or gi in used:
                    continue
                if _span_pair_matches(c["span"], gspan):
                    c["y"] = 1
                    used.add(gi)
                    break

        for c in cands:
            s = c["span"]
            pos = post_s.lower().find(s[:40].lower())
            nested_in = sum(1 for o in cands
                            if o is not c and normalize_span(s) in normalize_span(o["span"]))
            contains = sum(1 for o in cands
                           if o is not c and normalize_span(o["span"]) in normalize_span(s))
            rows.append({
                "post_idx": i, "span": s, "y": c["y"],
                "n_models": len(c["by"]),
                **{f"by_{n}": int(n in c["by"]) for n in names},
                "tok_len": token_len(s),
                "char_frac": len(s) / plen,
                "rel_pos": (pos / plen) if pos >= 0 else 0.5,
                "n_cands_in_post": len(cands),
                "nested_in": nested_in,
                "contains": contains,
            })
    return pd.DataFrame(rows), names


def decode(df, scores, thr):
    """Per post, keep candidates scoring >= thr."""
    out = {}
    for (pi, s, sc) in zip(df["post_idx"], df["span"], scores):
        if sc >= thr:
            out.setdefault(pi, []).append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--cache-dir", default="artifacts/evidence_per_run")
    ap.add_argument("--tag", default="val_split")
    args = ap.parse_args()

    val = pd.read_csv(args.val)
    gold = [parse_evidence(x) for x in val["evidence"]]
    runs = load_runs(val, args.cache_dir, args.tag)
    if not runs:
        raise SystemExit("no evidence predictions found")
    print(f"taggers: {list(runs)}")

    df, names = build_candidates(val, runs, gold)
    feats = (["n_models"] + [f"by_{n}" for n in names] +
             ["tok_len", "char_frac", "rel_pos", "n_cands_in_post",
              "nested_in", "contains"])
    X = df[feats].values.astype(float)
    y = df["y"].values
    print(f"candidates {len(df)}   positives {y.sum()} ({y.mean():.1%})   "
          f"features {len(feats)}")

    half = author_split(val)[0].values
    post_half = np.array([half[i] for i in df["post_idx"]])
    idxA = np.where(post_half == "A")[0]
    idxB = np.where(post_half == "B")[0]

    # ---- baselines -------------------------------------------------------
    print(f"\n{'baseline':<34}{'P':>8}{'R':>8}{'F1':>8}")
    best_single, best_name = 0.0, ""
    for n, spans in runs.items():
        s = phrase_f1(gold, spans)
        if s.f1 > best_single:
            best_single, best_name = s.f1, n
        print(f"{n:<34}{s.precision:>8.4f}{s.recall:>8.4f}{s.f1:>8.4f}")
    for k in (2, 3):
        sel = decode(df, (df["n_models"] >= k).astype(float).values, 0.5)
        spans = [sel.get(i, []) for i in range(len(val))]
        s = phrase_f1(gold, spans)
        print(f"{f'vote>={k} (1 feature)':<34}{s.precision:>8.4f}"
              f"{s.recall:>8.4f}{s.f1:>8.4f}")

    # ---- learned reranker, author cross-fitted ---------------------------
    scores = np.zeros(len(df))
    thr_used = []
    for tr, te in ((idxA, idxB), (idxB, idxA)):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        # threshold fitted on the TRAINING half only
        p_tr = clf.predict_proba(X[tr])[:, 1]
        tr_posts = sorted(set(df["post_idx"].values[tr]))
        g_tr = [gold[i] for i in tr_posts]
        best_t, best_f = 0.5, -1.0
        for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
            sel = decode(df.iloc[tr], p_tr, t)
            sp = [sel.get(i, []) for i in tr_posts]
            f = phrase_f1(g_tr, sp).f1
            if f > best_f:
                best_t, best_f = t, f
        thr_used.append(best_t)
        scores[te] = clf.predict_proba(X[te])[:, 1]

    thr = float(np.mean(thr_used))
    sel = decode(df, scores, thr)
    spans = [sel.get(i, []) for i in range(len(val))]
    s = phrase_f1(gold, spans)
    print(f"\n{'LEARNED reranker (cross-fitted)':<34}{s.precision:>8.4f}"
          f"{s.recall:>8.4f}{s.f1:>8.4f}   thresholds {thr_used}")

    # per-half, so a one-sided win is visible
    for h, idx in (("A", idxA), ("B", idxB)):
        posts = sorted(set(df["post_idx"].values[idx]))
        sub = decode(df.iloc[idx], scores[idx], thr)
        sp = [sub.get(i, []) for i in posts]
        print(f"    half {h}: F1 {phrase_f1([gold[i] for i in posts], sp).f1:.4f}")

    delta = s.f1 - best_single
    print(f"\nvs best single tagger ({best_name} {best_single:.4f}): {delta:+.4f} "
          f"phrase-F1 = {0.3 * delta:+.4f} composite")

    clf_all = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    print(f"\nfeature weights (fitted on all of val, for inspection only):")
    for f_, w in sorted(zip(feats, clf_all.coef_[0]), key=lambda z: -abs(z[1])):
        print(f"    {f_:<22}{w:>+8.3f}")


if __name__ == "__main__":
    main()
