"""
evidence_vote_oof.py

Scores the evidence span vote over the full out-of-fold set rather than a
single held-out split.

scripts/evidence_5fold.sh trains five diverse taggers on author-disjoint
folds, so every labelled post has a prediction from every tagger that was
never trained on it. Scoring the vote over all of them, instead of over one
small split, keeps a vote threshold from being chosen on a sample too small to
separate a real effect from split-specific noise.

The baseline arm is the single reference tagger on exactly the same posts,
with the consistency rule applied to baseline and vote alike, so the vote is
the only difference between arms. Each arm is also scored on two
author-disjoint halves, so an arm that wins only on one half is visible as
such.

Usage:
  python src/evidence_vote_oof.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_span_pair_matches, normalize_label, normalize_span,
                     parse_evidence, phrase_f1, token_len)

TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]


def author_halves(auth):
    s = pd.Series(auth)
    counts = s.value_counts()
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    half, size = {}, {"A": 0, "B": 0}
    for a in order:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


def dedupe(spans):
    out = []
    # Character-length tie-break, matching evidence_vote_test.py. Two spans can
    # share a token count while one still contains the other, so with a
    # token-only sort key the survivor depends on arbitrary order and both can
    # be kept. Since the scorer matches one-to-one, the extra span is then a
    # guaranteed false positive.
    for s in sorted(spans, key=lambda x: (-token_len(x), -len(normalize_span(x)))):
        n = normalize_span(s)
        if not n or any(n in normalize_span(k) for k in out):
            continue
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--dir", default="artifacts/evidence_oof")
    ap.add_argument("--tags", default=",".join(TAGS))
    ap.add_argument("--average", default="micro", choices=["micro", "macro"],
                    help="MICRO pools every span then computes one P/R. MACRO "
                         "scores each post separately and averages, so a post "
                         "with empty gold AND empty prediction scores 1.0. The "
                         "two disagree sharply on this task because a large "
                         "share of posts have gold 'none': under macro those "
                         "posts are scorable and reward correct abstention, "
                         "under micro they contribute nothing. The official "
                         "scorer averages per post, so macro is the setting "
                         "that reflects the reported metric.")
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled).set_index("row_id")
    tags = [t.strip() for t in args.tags.split(",")]

    per_tag = {}
    for t in tags:
        p = os.path.join(args.dir, f"{t}_oof_candidates.csv")
        if not os.path.exists(p):
            print(f"  skip {t}: no {p}")
            continue
        d = pd.read_csv(p)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        d = d[d["conf"] >= mc]
        m = {}
        for r, s in zip(d["row_id"], d["span"]):
            m.setdefault(r, []).append(str(s))
        per_tag[t] = {r: dedupe(v) for r, v in m.items()}
        print(f"  {t}: min_conf={mc}  "
              f"{sum(len(v) for v in per_tag[t].values())} spans")
    if len(per_tag) < 2:
        raise SystemExit("need at least two taggers")

    rows = sorted(set(lab.index) & set().union(*[set(v) for v in per_tag.values()]))
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    risk = [normalize_label(lab.loc[r, "risk_level"]) for r in rows]
    half = author_halves([lab.loc[r, "anon_user_id"] for r in rows])
    print(f"\n{len(rows)} OOF posts   halves "
          f"{int((half=='A').sum())}/{int((half=='B').sum())}")

    base_tag = "v2" if "v2" in per_tag else tags[0]

    def consistency(spans, r, i):
        """The consistency rule, applied identically to every arm."""
        if risk[i] == "indicator":
            return []
        if spans:
            return spans
        pool = per_tag[base_tag].get(r, []) or [
            s for t in per_tag for s in per_tag[t].get(r, [])]
        return sorted(pool, key=token_len)[:1]

    def build(pick):
        return [consistency(dedupe(pick(r)), r, i) for i, r in enumerate(rows)]

    def sc(P, mask=None):
        if mask is None:
            return phrase_f1(G, P, average=args.average)
        return phrase_f1([g for g, m in zip(G, mask) if m],
                         [p for p, m in zip(P, mask) if m],
                         average=args.average)

    def votes(r, k):
        c = {}
        for t in per_tag:
            for s in per_tag[t].get(r, []):
                key = normalize_span(s)
                c.setdefault(key, [0, s])
                c[key][0] += 1
        return [s for n, s in c.values() if n >= k]

    def cluster_votes(r, k, rep="modal"):
        """Vote using the SCORER'S equivalence relation instead of exact
        string equality.

        WHY CONTAINMENT RATHER THAN STRING EQUALITY. `votes()` above keys
        agreement on `normalize_span(s)`, so two taggers must produce
        byte-identical text to count as agreeing. The official scorer
        (`_span_pair_matches`) is looser: it calls two spans equivalent when
        EITHER CONTAINS the other and the prediction is within 3x the gold
        token length. Under exact-string voting, taggers that agree on WHICH
        clause matters but disagree on where it starts or ends split their
        votes, neither variant reaches the threshold, and BOTH are discarded --
        even though either one would have matched gold.

        Containment is substring-based, so it covers EDGE extensions only; a
        word inserted in the middle breaks containment in both directions.
        Those candidates keep splitting, which is correct, because under the
        scorer they really are different spans.

        Here candidates are grouped under that same containment relation,
        DISTINCT TAGGERS per group are counted, and the group's modal exact
        string is emitted. `rep` selects the representative: `modal` (the
        most-agreed variant), `short` or `long`.

        The relation is not transitive, so greedy grouping is order-dependent;
        candidates are sorted longest-first then lexicographically to make the
        result deterministic rather than dependent on dict iteration order.
        """
        items = [(t, s) for t in per_tag for s in per_tag[t].get(r, [])]
        items.sort(key=lambda x: (-token_len(x[1]), normalize_span(x[1])))
        groups = []
        for t, s in items:
            for g in groups:
                if any(_span_pair_matches(s, s2) or _span_pair_matches(s2, s)
                       for _, s2 in g):
                    g.append((t, s))
                    break
            else:
                groups.append([(t, s)])
        out = []
        for g in groups:
            if len({t for t, _ in g}) < k:
                continue
            if rep == "short":
                out.append(min((s for _, s in g), key=token_len))
            elif rep == "long":
                out.append(max((s for _, s in g), key=token_len))
            elif rep == "median":
                ss = sorted((s for _, s in g), key=token_len)
                out.append(ss[len(ss) // 2])
            elif rep == "capsafe":
                # CAP-AWARE representative. `_span_pair_matches` rejects a
                # prediction longer than 3x the gold it contains, so a cluster
                # whose modal variant is much longer than its shortest member
                # can lose a match that the shortest member would have won. The
                # asymmetry is one-sided: a prediction CONTAINED IN gold always
                # passes the cap, so erring short can never cost a match the
                # long variant would have made.
                #
                # So: take the modal variant, but if it exceeds 3x the shortest
                # candidate in the same cluster, fall back to the shortest --
                # against any gold near that shorter length the modal variant
                # is guaranteed to fail the cap while the short one cannot.
                #
                # The 3 is the SCORER'S OWN CONSTANT, not a fitted threshold,
                # so this stays a 0-parameter rule.
                cnt = {}
                for _, s in g:
                    cnt[s] = cnt.get(s, 0) + 1
                modal = max(cnt.items(),
                            key=lambda kv: (kv[1], -token_len(kv[0])))[0]
                shortest = min((s for _, s in g), key=token_len)
                out.append(shortest
                           if token_len(modal) > 3 * max(token_len(shortest), 1)
                           else modal)
            else:
                cnt = {}
                for _, s in g:
                    cnt[s] = cnt.get(s, 0) + 1
                out.append(max(cnt.items(),
                               key=lambda kv: (kv[1], -token_len(kv[0])))[0])
        return out

    arms = {f"{base_tag} alone": build(lambda r: per_tag[base_tag].get(r, [])),
            "union all": build(
                lambda r: [s for t in per_tag for s in per_tag[t].get(r, [])])}
    for k in range(2, len(per_tag) + 1):
        arms[f"vote>={k}"] = build(lambda r, k=k: votes(r, k))
    for k in range(2, len(per_tag) + 1):
        # `capsafe` targets the scorer's 3x token cap, which discards a
        # containment match when the prediction is more than three times the
        # gold length. A prediction CONTAINED IN gold always passes the cap, so
        # erring short is one-sided-safe. `median` is the neutral middle of the
        # short/long axis, included for comparison.
        for rp in ("modal", "capsafe", "median", "short", "long"):
            arms[f"cluster>={k} {rp}"] = build(
                lambda r, k=k, rp=rp: cluster_votes(r, k, rp))

    b = sc(arms[f"{base_tag} alone"])
    bA = sc(arms[f"{base_tag} alone"], half == "A")
    bB = sc(arms[f"{base_tag} alone"], half == "B")

    print(f"\n{'arm':<18}{'F1':>9}{'P':>8}{'R':>8}{'pred':>7}"
          f"{'delta':>9}{'halfA':>9}{'halfB':>9}")
    print("-" * 76)
    best = None
    for n, P in arms.items():
        r = sc(P)
        a = sc(P, half == "A").f1 - bA.f1
        bb = sc(P, half == "B").f1 - bB.f1
        d = r.f1 - b.f1
        ok = n != f"{base_tag} alone" and d > 0 and a > 0 and bb > 0
        if ok and (best is None or d > best[1]):
            best = (n, d)
        print(f"{n:<18}{r.f1:>9.4f}{r.precision:>8.4f}{r.recall:>8.4f}"
              f"{r.n_pred:>7}{d:>+9.4f}{a:>+9.4f}{bb:>+9.4f}"
              f"{'  <- both halves' if ok else ''}")

    print("\n" + "=" * 76)
    if best:
        print(f"On {len(rows)} OOF posts the best arm is {best[0]} at "
              f"{best[1]:+.4f}  ({0.3 * best[1]:+.4f} composite).")
    else:
        print(f"No arm beats {base_tag} alone on both author-disjoint halves.")
    print("=" * 76)


if __name__ == "__main__":
    main()
