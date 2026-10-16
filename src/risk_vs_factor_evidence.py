"""
risk_vs_factor_evidence.py

IS THE EVIDENCE THAT EXPLAINS RISK THE SAME EVIDENCE THAT IDENTIFIES FACTORS?

The corpus supplies one set of gold evidence spans, and those spans were
annotated to justify the RISK LEVEL. If a system presents them as its
explanation, it is implicitly claiming they also account for the 24 factors it
predicts. This script tests that claim directly by measuring how much factor
information the risk-evidence spans actually carry.

THREE INPUTS, ONE MODEL, ONE PROTOCOL. The same classifier is trained and
evaluated on three views of each post:

    POST        the whole post
    SPAN        only the gold risk-evidence spans
    COMPLEMENT  the post with those spans deleted

SPAN against POST answers "how much factor signal is inside the risk
evidence". COMPLEMENT against POST answers the sharper question: if the factor
signal survives after the risk evidence is removed, then the two kinds of
evidence are largely disjoint, and one rationale cannot serve both purposes.

WHY A DELIBERATELY SIMPLE MODEL. The question is about the INFORMATION in the
text, not about the capacity of an encoder. A bag-of-words classifier holds
model capacity fixed across the three views, so a difference between them is
attributable to the text and nothing else. A large pretrained model would
confound the comparison with what it already knows.

THE CONTROL THAT MATTERS. Gold evidence is empty for every `indicator` post,
by the task's own definition, so the SPAN view is empty on roughly a third of
the corpus. That alone would depress SPAN and prove nothing. Every comparison
is therefore reported twice: over all posts, and over the subset that HAS a
gold span. Only the second supports a claim about information content.

Splits are grouped by author, because posts by one author are near-duplicates
and a row-level split would leak them across the partition.

SECURITY: reads local files only and prints aggregate statistics. No post text
and no span text is emitted.

Usage:
  python src/risk_vs_factor_evidence.py
  python src/risk_vs_factor_evidence.py --csv artifacts/risk_vs_factor.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_evidence, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}


def strip_spans(post, spans):
    """The post with every gold evidence span deleted, case-insensitively."""
    out = post
    for s in spans:
        s = str(s).strip()
        if not s:
            continue
        low, sl = out.lower(), s.lower()
        i = low.find(sl)
        while i >= 0:
            out = out[:i] + " " + out[i + len(s):]
            low = out.lower()
            i = low.find(sl)
    return " ".join(out.split())


def oracle_windows(posts, spans, Y, groups, folds, seed, stride=2):
    """The best window of the same length as the gold span, per post.

    THE CONTROL THIS EXISTS FOR. If the gold risk span and a random window of
    equal length score the same, there are two possible explanations: the risk
    span genuinely carries no factor signal, or a window that short carries no
    signal for anyone. Those are very different conclusions and the experiment
    cannot separate them without an upper bound.

    So we take, for each post, the window that a fold model scores as matching
    that post's gold factors best. This uses the gold label to SELECT the
    window, which makes it an oracle and therefore unattainable, exactly like
    the oracle thresholds elsewhere in this work. Its only job is to answer
    whether any window of this length can be informative.
    """
    P = np.zeros_like(Y)
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(posts, groups=groups):
        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                              max_features=40000, strip_accents="unicode")
        Xtr = vec.fit_transform([posts[i] for i in tr])
        models = {}
        for j in range(N):
            if Y[tr, j].sum() < 3:
                continue
            m = LogisticRegression(max_iter=2000, C=4.0,
                                   class_weight="balanced", random_state=seed)
            m.fit(Xtr, Y[tr, j])
            models[j] = m
        for i in te:
            w = posts[i].split()
            k = max(1, len(" ".join(spans[i]).split()))
            if k >= len(w):
                cands = [posts[i]]
            else:
                cands = [" ".join(w[s:s + k])
                         for s in range(0, len(w) - k + 1, stride)]
            if not cands:
                cands = [posts[i]]
            Xc = vec.transform(cands)
            pred = np.zeros((len(cands), N), dtype=int)
            for j, m in models.items():
                pred[:, j] = m.predict(Xc)
            # pick the window whose predicted label set best matches gold
            g = Y[i]
            best, bs = 0, -1.0
            for c in range(len(cands)):
                tp = int((pred[c] & g).sum())
                den = int(pred[c].sum() + g.sum())
                sc = 0.0 if den == 0 else 2.0 * tp / den
                if sc > bs:
                    best, bs = c, sc
            P[i] = pred[best]
    return np.array([f1_score(Y[:, j], P[:, j], zero_division=0)
                     for j in range(N)])


def label_matrix(cells):
    Y = np.zeros((len(cells), N), dtype=int)
    for i, c in enumerate(cells):
        for f in parse_factors(str(c)):
            j = IDX.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    return Y


def run_view(texts, Y, groups, folds, seed):
    """Author-grouped CV; returns per-label F1 and the out-of-fold matrix."""
    P = np.zeros_like(Y)
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(texts, groups=groups):
        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                              max_features=40000, strip_accents="unicode")
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        for j in range(N):
            if Y[tr, j].sum() < 3:
                continue
            clf = LogisticRegression(max_iter=2000, C=4.0,
                                     class_weight="balanced",
                                     random_state=seed)
            clf.fit(Xtr, Y[tr, j])
            P[te, j] = clf.predict(Xte)
    per = np.array([f1_score(Y[:, j], P[:, j], zero_division=0)
                    for j in range(N)])
    return per, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default="artifacts/full_train.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.labelled, keep_default_na=False)
    posts = [str(x) for x in df["post"]]
    spans = [parse_evidence(str(c)) for c in df["evidence"]]
    acol = "anon_user_id" if "anon_user_id" in df.columns else "author"
    groups = np.array([str(a) for a in df[acol]])
    Y = label_matrix(df["factors"])

    span_txt = [" ".join(s) for s in spans]
    comp_txt = [strip_spans(p, s) for p, s in zip(posts, spans)]
    has = np.array([bool(t.strip()) for t in span_txt])

    # THE CONTROL THAT MAKES THE RESULT MEAN ANYTHING. Deleting the risk
    # evidence removes about a tenth of the text, and losing a tenth of the
    # words might cost a little performance whatever those words are. So we
    # also delete a RANDOM contiguous window of the SAME length from the same
    # post. If random deletion hurts as little as targeted deletion, the
    # comparison is uninformative. If random deletion hurts MORE, then the
    # risk evidence carries less factor signal than ordinary text does.
    rng = np.random.default_rng(args.seed)
    rand_txt, randspan_txt = [], []
    for p, s in zip(posts, spans):
        w = p.split()
        k = len(" ".join(s).split())
        if k == 0 or k >= len(w):
            rand_txt.append(p)
            randspan_txt.append("")
            continue
        i = int(rng.integers(0, len(w) - k + 1))
        rand_txt.append(" ".join(w[:i] + w[i + k:]))
        randspan_txt.append(" ".join(w[i:i + k]))

    print(f"{len(df)} posts, {len(set(groups))} authors, {N} factors")
    print(f"posts WITH a gold risk-evidence span: {has.sum()} "
          f"({100*has.mean():.0f}%)")
    wp = np.mean([len(p.split()) for p in posts])
    ws = np.mean([len(t.split()) for t in span_txt if t.strip()])
    print(f"mean words: post {wp:.1f}   risk-evidence span {ws:.1f} "
          f"({100*ws/wp:.0f}% of the post)\n")

    views = {"POST": posts, "SPAN": span_txt, "COMPLEMENT": comp_txt,
             "RAND SPAN": randspan_txt, "RAND COMPL": rand_txt}
    res = {}
    for name, txt in views.items():
        per, _ = run_view(txt, Y, groups, args.folds, args.seed)
        res[name] = per
        print(f"  {name:<11} fitted", flush=True)

    # Refit restricted to posts that HAVE a gold span. Without this control
    # the SPAN view is handed an empty string on every indicator post and
    # loses for a reason that has nothing to do with information content.
    subY, subg = Y[has], groups[has]
    for name, txt in views.items():
        t = [txt[i] for i in range(len(txt)) if has[i]]
        per, _ = run_view(t, subY, subg, args.folds, args.seed)
        res[name + "_sub"] = per

    # the upper bound on what ANY window of this length could achieve
    sub_posts = [posts[i] for i in range(len(posts)) if has[i]]
    sub_spans = [spans[i] for i in range(len(spans)) if has[i]]
    orc = oracle_windows(sub_posts, sub_spans, subY, subg, args.folds,
                         args.seed)
    res["ORACLE WIN_sub"] = orc

    print("\n" + "=" * 74)
    print("MACRO-F1 OVER 24 FACTORS, SAME MODEL, DIFFERENT INPUT TEXT")
    print("=" * 74)
    print(f"{'view':<14}{'all posts':>12}{'posts with a span':>20}")
    for name in views:
        print(f"{name:<14}{res[name].mean():>12.4f}"
              f"{res[name + '_sub'].mean():>20.4f}")
    print(f"{'ORACLE WIN':<14}{'':>12}{orc.mean():>20.4f}"
          f"   <- upper bound, uses gold to pick the window")

    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    p, s, c = (res["POST_sub"].mean(), res["SPAN_sub"].mean(),
               res["COMPLEMENT_sub"].mean())
    print(f"  SPAN reaches {100*s/max(p,1e-9):.0f}% of what the full post")
    print(f"  reaches. The risk-evidence spans are {100*ws/wp:.0f}% of the")
    print(f"  text, so compare those two shares before concluding anything.")
    print()
    rs, rc = res["RAND SPAN_sub"].mean(), res["RAND COMPL_sub"].mean()
    print(f"  COMPLEMENT reaches {100*c/max(p,1e-9):.0f}% of the full post,")
    print(f"  against {100*rc/max(p,1e-9):.0f}% when a RANDOM window of the")
    print("  same length is deleted instead.")
    print()
    print("  Those two numbers are the experiment. Deleting the risk evidence")
    print("  should hurt MORE than deleting arbitrary text if the two kinds of")
    print("  evidence coincide. If it hurts the same or less, the factor")
    print("  signal does not live in the risk rationale.")
    print()
    print(f"  Read the spans the same way: the gold risk span scores {s:.4f}")
    print(f"  and a random window of equal length scores {rs:.4f}.")

    if args.csv:
        out = pd.DataFrame({"factor": FACTOR_TAXONOMY,
                            "n_pos": Y.sum(0),
                            "post": res["POST_sub"],
                            "span": res["SPAN_sub"],
                            "complement": res["COMPLEMENT_sub"]})
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        out.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
