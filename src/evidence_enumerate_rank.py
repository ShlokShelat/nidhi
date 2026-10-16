"""
evidence_enumerate_rank.py

IF THE POOL IS SATURATED, HOW HARD IS THE RANKING?

`evidence_generator_ceiling.py` established, on gold data alone, that a dumb
mechanical generator saturates the candidate pool:

    generator              cands/post   gold cov   posts full   oracle macroF1
    argmax decode (ours)          1.1         --         58%          ~0.89
    in-sentence 3-8 grams       218.8      96.9%       94.9%         0.9836
    in-sentence 2-10 grams      317.6      98.4%       97.3%         0.9913

So generation was never the wall. The wall is that nothing in the system can
SCORE an arbitrary span -- the BIO decode only scores the spans it already
chose. That reframes evidence from "the pool is starved" to "rank 219
candidates", which is the regime the two-stage IE literature says rankers work
in, and it is the regime the seven taggers' token posteriors could serve.

BUT A BIGGER POOL IS NOT A FREE WIN, AND THIS SCRIPT IS THE HONEST TEST OF
THAT. Phrase-F1 punishes precision, and going from 1.1 to 219 candidates means
a selector must reject 218 of them per post. The shipped decode currently
reaches 0.774 against a 0.89 oracle -- 87% of ceiling. If a selector on the
big pool reaches only 80%
of 0.98, that is 0.784 and the whole exercise buys 0.001.

So: fit a deliberately WEAK, model-free ranker on the saturated pool and see
what it achieves. This is a FLOOR, not a proposal. Its features are structural
plus a TF-IDF bag of words over the span -- no tagger, no transformer, nothing
that has seen a suicide corpus. The seven taggers are enormously stronger than
this. If even this floor lands near the incumbent, the lane is real. If it
lands at 0.4, ranking 219 candidates is genuinely hard and the soft-decode
sweep (a smaller, smarter pool) is the better route.

TWO HAZARDS THIS SCRIPT IS BUILT TO AVOID.

  * PROVENANCE. Hand-written lexical rules mined from the posts they are
    scored on tend to show 0 novel fixes, because the pattern is circular.
    So there are NO hand-picked lexical cues here. The only lexical signal is
    TF-IDF fitted on the FIT authors and applied to HELD-OUT authors.
  * SUPPRESSION LEAK. An earlier harness (evidence_vote_oof.py) inflated its
    score by suppressing on GOLD risk. 36% of posts have empty
    gold and score 1.0 for predicting nothing, so who decides "emit nothing"
    dominates the metric. Both numbers are reported and the oracle one is
    labelled as an upper bound, never as the result.

Everything is author-disjoint: authors are split, TF-IDF and the ranker and the
emit threshold are all fitted on FIT authors only, and scored on HELD-OUT ones.

Usage:
  python src/evidence_enumerate_rank.py
  python src/evidence_enumerate_rank.py --lo 2 --hi 10 --splits 8
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_span, parse_evidence,  # noqa: E402
                     phrase_f1, token_len)
from text_norm import clean_text  # noqa: E402

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def matches(pred: str, gold: str) -> bool:
    p, g = normalize_span(pred), normalize_span(gold)
    if not p or not g:
        return False
    if p not in g and g not in p:
        return False
    return token_len(p) <= 3 * token_len(g)


def _match_size(adj, left, right):
    """Max one-to-one matching, Kuhn. left/right are index lists into the
    shared candidate array; adj[i][j] is the official pair-match predicate."""
    pair = {}
    for u in left:
        seen = set()
        stack = [(u, iter(right))]
        # iterative augmenting path, small graphs so clarity over speed
        def try_u(x):
            for v in right:
                if adj[x][v] and v not in seen:
                    seen.add(v)
                    if v not in pair or try_u(pair[v]):
                        pair[v] = x
                        return True
            return False
        try_u(u)
    return len(pair)


def mbr_subset(scores, adj, idx, n_refs, rng):
    """Exact MBR over all 2^K subsets of the top-K candidates.

    References are sampled from the scorer's own Bernoullis, so this is
    argmax_S E_h[F1(S,h)] under the model -- the decision-theoretically
    correct object, against which threshold+k is a heuristic approximation.

    The empty subset is included and is NOT a special case: Waegeman et al.
    (JMLR 2014) Thm 8 gives E[F1(empty)] = P(gold empty) exactly, which here is
    the fraction of sampled references that came out empty. That is also the
    known TRAP -- a P(empty) derived from the candidate scores is not
    calibrated to "this post has no evidence at all", and with few references
    it can be pinned at zero so MBR can never abstain. Reported separately.
    """
    K = len(idx)
    ps = np.asarray(scores, dtype=float)
    draws = rng.random((n_refs, K)) < ps
    refs = [[idx[j] for j in range(K) if d[j]] for d in draws]
    best_u, best_S = -1.0, []
    for mask in range(1 << K):
        S = [idx[j] for j in range(K) if (mask >> j) & 1]
        tot = 0.0
        for h in refs:
            if not S and not h:
                tot += 1.0
                continue
            if not S or not h:
                continue
            m = _match_size(adj, S, h)
            tot += 2.0 * m / (len(S) + len(h))
        u = tot / n_refs
        if u > best_u:
            best_u, best_S = u, S
    return best_S


def enumerate_spans(post: str, lo: int, hi: int):
    """In-sentence word n-grams, with the structural features that come free
    from the enumeration itself. Returns (text, feats) pairs."""
    out = []
    sents = [s for s in SENT_SPLIT.split(post) if s.strip()]
    ns = max(len(sents), 1)
    total = max(len(post.split()), 1)
    seen = set()
    for si, s in enumerate(sents):
        w = s.split()
        for n in range(lo, hi + 1):
            for i in range(len(w) - n + 1):
                t = " ".join(w[i:i + n])
                k = normalize_span(t)
                if not k or k in seen:
                    continue
                seen.add(k)
                out.append((t, [
                    n,                          # span length
                    si / ns,                    # which sentence, relative
                    i / max(len(w), 1),         # position inside sentence
                    n / max(len(w), 1),         # fraction of sentence covered
                    len(w) / total,             # sentence size, relative
                    float(i == 0),              # starts the sentence
                    float(i + n == len(w)),     # ends the sentence
                    float(si == 0),             # first sentence
                    float(si == ns - 1),        # last sentence
                ]))
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--data", default="artifacts/full_train.csv")
    a_.add_argument("--lo", type=int, default=3)
    a_.add_argument("--hi", type=int, default=8)
    a_.add_argument("--splits", type=int, default=5)
    a_.add_argument("--seed", type=int, default=13)
    a_.add_argument("--max-emit", type=int, default=4)
    a_.add_argument("--mbr", type=int, default=0, metavar="K",
                    help="ALSO run exact MBR subset selection over the top-K "
                         "candidates (K<=8; 2^K subsets, no greedy needed -- "
                         "the whole fast-MBR literature is about k=1000 "
                         "hypotheses and is irrelevant here). This isolates "
                         "the DECISION RULE from the scorer: MBR and the "
                         "threshold+k heuristic see identical probabilities, "
                         "so any difference is decision theory alone. The "
                         "shipped 'span in >=2 of 7 taggers' rule is a "
                         "PER-ELEMENT vote that never scores a set as a "
                         "whole, which is the one thing in the selection "
                         "family not yet tried.")
    a_.add_argument("--mbr-refs", type=int, default=32)
    a_.add_argument("--gen", default="ngram",
                    choices=["ngram", "clause", "sentence"],
                    help="THE CONTROL. Running the SAME weak ranker on a "
                         "small pool separates 'the pool is too big to rank' "
                         "from 'the features are too weak'. clause gives "
                         "~7.6 candidates/post at oracle 0.827; ngram gives "
                         "~275 at oracle 0.984. If the small pool scores "
                         "HIGHER, pool size is the cost and enumerate-then-"
                         "rank needs a strong scorer to be worth it.")
    args = a_.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import hstack, csr_matrix

    df = pd.read_csv(args.data, keep_default_na=False)
    posts = [clean_text(str(p)) for p in df["post"]]
    golds = [parse_evidence(v) for v in df["evidence"]]
    auth = np.asarray([str(x) for x in df["anon_user_id"]])
    n = len(df)

    print(f"{args.data}: {n} posts, {len(set(auth))} authors, "
          f"{sum(1 for g in golds if g)} with non-empty gold")
    print(f"enumerating in-sentence {args.lo}-{args.hi} grams ...", flush=True)

    CLAUSE_SPLIT = re.compile(r"(?<=[.!?,;:])\s+|\n+|\s+(?:and|but|because|"
                              r"so|though|although|while|since|then|or)\s+",
                              re.I)

    def units(post, rx):
        """Same feature vector as the n-gram path so the control is clean:
        only the candidate SET differs, not the representation."""
        out = []
        sents = [s for s in SENT_SPLIT.split(post) if s.strip()]
        ns = max(len(sents), 1)
        total = max(len(post.split()), 1)
        seen = set()
        for si, s in enumerate(sents):
            us = [u.strip() for u in rx.split(s) if u.strip()] if rx else [s]
            for ui, u in enumerate(us):
                k = normalize_span(u)
                if not k or k in seen:
                    continue
                seen.add(k)
                w = u.split()
                out.append((u, [len(w), si / ns, ui / max(len(us), 1),
                                len(w) / max(len(s.split()), 1),
                                len(s.split()) / total,
                                1.0, 1.0, float(si == 0),
                                float(si == ns - 1)]))
        return out

    def gen(p):
        if args.gen == "ngram":
            return enumerate_spans(p, args.lo, args.hi)
        if args.gen == "clause":
            return units(p, CLAUSE_SPLIT)
        return units(p, None)

    texts, feats, owner, ylab = [], [], [], []
    for pi, (p, g) in enumerate(zip(posts, golds)):
        for t, f in gen(p):
            texts.append(t)
            feats.append(f)
            owner.append(pi)
            ylab.append(int(any(matches(t, s) for s in g)))
    owner = np.asarray(owner)
    y = np.asarray(ylab)
    F = np.asarray(feats, dtype=np.float32)
    print(f"  {len(texts)} candidates ({len(texts)/n:.1f}/post), "
          f"{y.sum()} positive ({y.mean():.2%}) -- the class imbalance a "
          f"ranker on this pool has to survive")

    # per-post oracle: the ceiling this pool allows, recomputed here so the
    # comparison below cannot drift from the pool actually being ranked
    orc = []
    for pi in range(n):
        g = golds[pi]
        if not g:
            orc.append(1.0)
            continue
        sel = np.flatnonzero(owner == pi)
        k = sum(any(matches(texts[j], s) for j in sel) for s in g)
        orc.append(2.0 * k / (k + len(g)))
    print(f"  pool oracle macro phrase-F1: {np.mean(orc):.4f}")

    ua = np.array(sorted(set(auth)))
    rng = np.random.default_rng(args.seed)
    hdr = "=" * 78
    print("\n" + hdr)
    print("AUTHOR-DISJOINT RANKING, WEAK FEATURES ONLY")
    print(hdr)

    rows = []
    for sp in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        ho = set(o[:max(int(len(o) * 0.25), 2)])
        pm = np.array([a in ho for a in auth])
        fit_p = np.flatnonzero(~pm)
        ho_p = np.flatnonzero(pm)
        fit_c = np.isin(owner, fit_p)
        ho_c = np.isin(owner, ho_p)
        if y[fit_c].sum() < 20:
            continue

        tf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                             min_df=3, max_features=40000,
                             sublinear_tf=True)
        Xf = tf.fit_transform([texts[i] for i in np.flatnonzero(fit_c)])
        Xh = tf.transform([texts[i] for i in np.flatnonzero(ho_c)])
        Xf = hstack([Xf, csr_matrix(F[fit_c])]).tocsr()
        Xh = hstack([Xh, csr_matrix(F[ho_c])]).tocsr()
        clf = LogisticRegression(C=1.0, max_iter=300, solver="liblinear")
        clf.fit(Xf, y[fit_c])
        sh = clf.predict_proba(Xh)[:, 1]
        sf = clf.predict_proba(Xf)[:, 1]

        # score-per-post, both halves, so the emit threshold is chosen on FIT
        def by_post(idx_mask, score, pool):
            d = {}
            for loc, gi in enumerate(np.flatnonzero(idx_mask)):
                d.setdefault(owner[gi], []).append((score[loc], texts[gi]))
            return {p: sorted(v, reverse=True)[:pool] for p, v in d.items()}

        Pf = by_post(fit_c, sf, 40)
        Ph = by_post(ho_c, sh, 40)

        def run(bypost, post_idx, thr, kmax, oracle_empty):
            gl, pl = [], []
            for pi in post_idx:
                cand = bypost.get(pi, [])
                if oracle_empty and not golds[pi]:
                    keep = []
                else:
                    keep = [t for s, t in cand[:kmax] if s >= thr]
                gl.append(golds[pi])
                pl.append(keep)
            # average="macro" IS MANDATORY. phrase_f1 defaults to MICRO, and
            # under micro a single empty-gold post returns f1 0.0 rather than
            # the 1.0 the official scorer gives it -- 36% of posts here. Called
            # per post in a mean() that mistake is invisible and it silently
            # rewrites the objective: emitting a span on an empty-gold post
            # looks free, so the threshold search collapses to "always emit".
            # This is the same defect already recorded for evidence_vote_oof.py.
            return phrase_f1(gl, pl, average="macro").f1

        best_thr, best_k, best_v = 0.5, 1, -1.0
        for thr in np.arange(0.05, 0.96, 0.05):
            for k in range(1, args.max_emit + 1):
                v = run(Pf, fit_p, thr, k, False)
                if v > best_v:
                    best_thr, best_k, best_v = float(thr), k, v
        honest = run(Ph, ho_p, best_thr, best_k, False)
        upper = run(Ph, ho_p, best_thr, best_k, True)
        orc_h = float(np.mean([orc[i] for i in ho_p]))

        mbr_v = float("nan")
        if args.mbr:
            K = min(args.mbr, 8)
            mrng = np.random.default_rng(args.seed + sp)
            gl, pl = [], []
            for pi in ho_p:
                cand = Ph.get(pi, [])[:K]
                gl.append(golds[pi])
                if not cand:
                    pl.append([])
                    continue
                txt = [t for _, t in cand]
                sc = [s for s, _ in cand]
                n = len(txt)
                adj = [[matches(txt[i], txt[j]) for j in range(n)]
                       for i in range(n)]
                keep = mbr_subset(sc, adj, list(range(n)), args.mbr_refs, mrng)
                pl.append([txt[j] for j in keep])
            mbr_v = phrase_f1(gl, pl, average="macro").f1

        rows.append((honest, upper, orc_h, best_thr, best_k, mbr_v))
        print(f"  split {sp}: fit-tuned thr {best_thr:.2f} k {best_k}  ->  "
              f"held-out macro {honest:.4f}   "
              f"(oracle-empty variant {upper:.4f}, pool oracle {orc_h:.4f})"
              + (f"   MBR {mbr_v:.4f}" if args.mbr else ""))

    if not rows:
        print("  no usable split")
        return
    R = np.asarray([r[:3] for r in rows], dtype=float)
    if args.mbr:
        M = np.asarray([r[5] for r in rows], dtype=float)
        d = M - R[:, 0]
        print(f"\n  MBR (exact, top-{min(args.mbr,8)}, "
              f"{args.mbr_refs} refs)      {M.mean():.4f}   "
              f"vs threshold+k {d.mean():+.4f}   win {np.mean(d>0):.0%}")
        print("  Same scores, different decision rule -- so this delta is")
        print("  decision theory alone, and it transfers to any better scorer.")
    print("\n" + hdr)
    print("RESULT")
    print(hdr)
    print(f"  honest held-out macro phrase-F1   {R[:, 0].mean():.4f} "
          f"+- {R[:, 0].std():.4f}")
    print(f"  with oracle empty-detection       {R[:, 1].mean():.4f}  "
          f"(UPPER BOUND, not a result)")
    print(f"  pool oracle                       {R[:, 2].mean():.4f}")
    print(f"  fraction of pool oracle attained  "
          f"{R[:, 0].mean() / max(R[:, 2].mean(), 1e-9):.1%}")
    print("\n  BENCHMARKS")
    print("    incumbent shipped evidence, honest val instrument   0.7767")
    print("    incumbent's own pool oracle                         ~0.89")
    print("    incumbent attains                                   ~87% of it")
    print("\n  HOW TO READ THIS. The ranker here has NO tagger, no transformer")
    print("  and no suicide-domain knowledge -- it is a bag of words and nine")
    print("  structural numbers. Whatever it reaches is a FLOOR on what the")
    print("  seven taggers could reach on the same pool, since scoring a span")
    print("  by mean P(not-O) over its tokens is strictly more informed.")
    print("  If the floor is already near 0.7767, the enumerate-then-rank")
    print("  rebuild is the biggest lane in the system. If it is far below,")
    print("  a 219-candidate pool is too hard to select from and the smaller")
    print("  soft-decode pool is the right target instead.")


if __name__ == "__main__":
    main()
