"""
evidence_matchprob.py

SELECT SPANS BY A CALIBRATED MATCH PROBABILITY, NOT BY A VOTE COUNT.

Per-post phrase F1 is 2m / (n_pred + n_gold). Adding one more span with match
probability q is worth it exactly when

    2(m+q)/(n_pred+1+n_gold) > 2m/(n_pred+n_gold)   <=>   q > m/(n_pred+n_gold)

i.e. **emit a span iff P(it matches) > F1/2.** That is the multi-label
plug-in result of Lipton, Elkan & Naryanaswamy (arXiv 1402.1892): for
well-calibrated probabilities the F1-optimal threshold is half the optimal F1.
At an evidence level near 0.78 that puts the cut near **0.39**.

THE ALTERNATIVE IT REPLACES. `cluster>=k modal` keeps a span when k distinct
TAGGERS agree. A vote count is a crude, uncalibrated proxy for P(match), and it
throws away everything else that predicts a match:

    the tagger confidences themselves (`conf`, otherwise used only as a
    min_conf gate), the span's token length (the scorer's 3x cap is
    one-sided -- short spans inside gold pass automatically), how many
    other candidates compete in the post, and the post's own length.

So this fits P(match | features) on out-of-fold data and selects by that. The
theory's precondition is calibrated probabilities, which is exactly what
`vote >= k` does not supply; the per-post SET-SELECTION problem here is the
setting the plug-in rule is stated for.

TARGET. A candidate matches if `_span_pair_matches` fires against any gold span
of its post: containment either way plus len(pred) <= 3*len(gold). That
slightly over-counts versus the scorer's one-to-one bipartite matching, so the
calibration target is approximate; final scoring below uses the REAL
`phrase_f1`, so the evaluation is not.

REGIME. Suppression uses PREDICTED risk and averaging is MACRO, matching how
the submission is scored. Suppressing on GOLD risk leaks the label and
inflates every arm by roughly the same amount, which makes arm-to-arm
comparison meaningless.

JUDGEMENT. Judge by win rate and PER-SPLIT sd, never by the CI of the mean:
the CI of the mean shrinks with the number of resamples and says nothing about
what a single 378-post test set returns. sd ~ mean / z(win rate).

Usage (cluster, CPU, a few minutes):
  python src/evidence_matchprob.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_span_pair_matches, normalize_label, normalize_span,  # noqa: E402
                     parse_evidence, phrase_f1, token_len)

TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]


def dedupe(spans):
    out = []
    for s in sorted(spans, key=lambda x: (-token_len(x),
                                          -len(normalize_span(x)))):
        n = normalize_span(s)
        if not n or any(n in normalize_span(k) for k in out):
            continue
        out.append(s)
    return out


def clusters_for(items):
    """Group candidates by the SCORER'S containment relation (the same
    grouping cluster_votes uses), returning one record per cluster."""
    items = sorted(items, key=lambda x: (-token_len(x[1]),
                                         normalize_span(x[1])))
    groups = []
    for t, s, c in items:
        for g in groups:
            if any(_span_pair_matches(s, s2) or _span_pair_matches(s2, s)
                   for _, s2, _ in g):
                g.append((t, s, c))
                break
        else:
            groups.append([(t, s, c)])
    out = []
    for g in groups:
        cnt = {}
        for _, s, _ in g:
            cnt[s] = cnt.get(s, 0) + 1
        modal = max(cnt.items(), key=lambda kv: (kv[1], -token_len(kv[0])))[0]
        confs = [c for _, _, c in g]
        lens = [token_len(s) for _, s, _ in g]
        out.append({
            "span": modal,
            "n_tag": len({t for t, _, _ in g}),
            "conf_mean": float(np.mean(confs)),
            "conf_max": float(np.max(confs)),
            "conf_min": float(np.min(confs)),
            "toks": token_len(modal),
            "toks_min": int(min(lens)),
            "spread": int(max(lens) - min(lens)),
        })
    return out


def featurise(recs, n_clu, n_words):
    X = []
    for r in recs:
        X.append([r["n_tag"], r["conf_mean"], r["conf_max"], r["conf_min"],
                  r["toks"], r["toks_min"], r["spread"], n_clu,
                  np.log1p(n_words), r["toks"] / max(n_words, 1)])
    return np.asarray(X, dtype=float)


FEATNAMES = ["n_taggers", "conf_mean", "conf_max", "conf_min", "toks",
             "toks_min", "len_spread", "n_clusters", "log_postlen",
             "toks/postlen"]


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--dir", default="artifacts/evidence_oof")
    a_.add_argument("--tags", default=",".join(TAGS))
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=13)
    a_.add_argument("--min-conf", type=float, default=None,
                    help="override the per-tagger min_conf gate. When the "
                         "candidate pool holds barely more clusters per post "
                         "than there are gold spans, selection has nothing to "
                         "choose between and the binding constraint is "
                         "GENERATION. Lowering this widens the pool at zero "
                         "GPU cost IF the dumps kept sub-threshold "
                         "candidates.")
    a_.add_argument("--suffix", default="",
                    help="'_long' reads the re-dumped candidate files that "
                         "were decoded WITHOUT the hardcoded 12-token cap")
    a_.add_argument("--max-tokens", type=int, default=None,
                    help="downstream length cap. The dumps hardcoded 12; the "
                         "scorer accepts up to 3x gold, i.e. 24 against an "
                         "8-token gold. Only meaningful with --suffix _long.")
    a_.add_argument("--vote-k", type=int, default=2,
                    help="distinct taggers that must agree. MUST be 1 for the "
                         "risk-conditioned family, which is a single model -- "
                         "with k=2 nothing ever fires and the FORCE rule "
                         "silently supplies every span.")
    a_.add_argument("--pool-only", action="store_true",
                    help="just report the candidate pool at a range of conf "
                         "cuts and exit -- seconds, no cross-fit")
    args = a_.parse_args()

    from sklearn.linear_model import LogisticRegression

    oof = pd.read_csv(args.oof, keep_default_na=False)
    oof["row_id"] = oof["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    per_tag, per_tag_ship = {}, {}
    for t in [x.strip() for x in args.tags.split(",")]:
        p = os.path.join(args.dir, f"{t}{args.suffix}_oof_candidates.csv")
        if not os.path.exists(p):
            print(f"  skip {t}: no {p}")
            continue
        d = pd.read_csv(p)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        below = int((d["conf"] < mc).sum())
        cut = mc if args.min_conf is None else args.min_conf
        if args.max_tokens is not None:
            tl = (d["toks"] if "toks" in d.columns
                  else d["span"].astype(str).str.split().str.len())
            nlong = int((tl > args.max_tokens).sum())
            d = d[tl <= args.max_tokens]
            print(f"  {t}: length cap {args.max_tokens} drops {nlong} spans")
        print(f"  {t}: min_conf={mc:.2f}  {len(d)} rows, {below} below the "
              f"gate ({below/max(len(d),1):.0%})  -> keeping conf>={cut:.2f}")
        # THE CONTROL MUST BE THE TRUE SHIPPED POOL: the ORIGINAL argmax-only
        # dump at its ORIGINAL gate. Reading the _long file here would put
        # soft-decode candidates into the baseline and quietly move it, which
        # is the same error as letting --min-conf shift the control.
        bp = os.path.join(args.dir, f"{t}_oof_candidates.csv")
        db = pd.read_csv(bp) if os.path.exists(bp) else d
        mcb = (float(db["min_conf"].iloc[0])
               if "min_conf" in db.columns else 0.0)
        db = db[db["conf"] >= mcb]
        ms = {}
        for r, s, c in zip(db["row_id"].astype(str), db["span"], db["conf"]):
            ms.setdefault(r, []).append((str(s), float(c)))
        per_tag_ship[t] = ms
        d = d[d["conf"] >= cut]
        m = {}
        for r, s, c in zip(d["row_id"].astype(str), d["span"], d["conf"]):
            m.setdefault(r, []).append((str(s), float(c)))
        per_tag[t] = m
    if not per_tag:
        raise SystemExit("no tagger candidate files found")
    if len(per_tag) < 2:
        print("  NOTE: single tagger -- vote/cluster arms degenerate to that "
              "tagger's own spans. The RISK-CONDITIONED family (evcond5_cond)\n"
              "  is one model, not five, so this is expected for --tags v2_rc.")
    base_tag = "v2" if "v2" in per_tag else list(per_tag)[0]

    rows = [r for r in oof["row_id"] if r in lab.index]
    n = len(rows)
    ix = oof.set_index("row_id")
    yship = np.array([normalize_label(x)
                      for x in ix.loc[rows, "pred_risk_level"]])
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])

    # per-post candidate clusters, features, and match targets
    REC, RECS, X, Y, POST = [], [], [], [], []
    for i, r in enumerate(rows):
        items = [(t, s, c) for t in per_tag for s, c in per_tag[t].get(r, [])]
        recs = clusters_for(items) if items else []
        si = [(t, s, c) for t in per_tag_ship
              for s, c in per_tag_ship[t].get(r, [])]
        RECS.append(clusters_for(si) if si else [])
        nw = len(str(lab.loc[r, "post"]).split())
        REC.append(recs)
        if not recs:
            continue
        F = featurise(recs, len(recs), nw)
        for k, rc in enumerate(recs):
            X.append(F[k])
            Y.append(int(any(_span_pair_matches(rc["span"], g) for g in G[i])))
            POST.append(i)
    X = np.asarray(X)
    Y = np.asarray(Y)
    POST = np.asarray(POST)
    ngold = sum(len(g) for g in G)
    nmatch = int(Y.sum())
    print(f"{n} OOF posts, {len(X)} candidate clusters, "
          f"{Y.mean():.1%} of them match a gold span")
    print(f"gold spans/post {np.mean([len(g) for g in G]):.2f}   "
          f"candidate clusters/post {len(X)/n:.2f}")
    print(f"\nORACLE CEILING OF THIS CANDIDATE POOL -- keep every matching")
    print(f"candidate, drop every non-matching one, which no rule can beat:")
    print(f"  matching candidates {nmatch}   gold spans {ngold}   "
          f"micro-F1 {2*nmatch/max(nmatch+ngold,1):.4f}")
    print(f"  posts where NO candidate matches any gold: "
          f"{sum(1 for i in range(n) if len(G[i]) and not any(_span_pair_matches(rc['span'], g) for rc in REC[i] for g in G[i]))}"
          f" of {sum(1 for g in G if g)} with non-empty gold")
    print("  Those posts are UNREACHABLE by any decode -- the right span was")
    print("  never proposed. That is the generation bottleneck, and it is what")
    print("  a lower min_conf is meant to attack.\n")
    if args.pool_only:
        return

    pool_of = [sorted([x for t in per_tag_ship
                       for x, _ in per_tag_ship[t].get(r, [])],
                      key=token_len)[:1] for r in rows]

    def finish(i, picked):
        """Suppression on PREDICTED risk, then the confirmed FORCE rule."""
        if yship[i] == "indicator":
            return []
        s = dedupe(picked)
        return s if s else pool_of[i]

    def mac(P, idx=None):
        if idx is None:
            return phrase_f1(G, P, average="macro").f1
        return phrase_f1([G[i] for i in idx], [P[i] for i in idx],
                         average="macro").f1

    # THE CONTROL IS THE SHIPPED CONFIG -- cluster>=2 modal on the ORIGINAL
    # per-tagger gates, never on a widened pool. Otherwise --min-conf moves
    # the baseline and every delta below is measured against a different
    # system than the one being replaced.
    kk = args.vote_k
    inc = [finish(i, [rc["span"] for rc in RECS[i] if rc["n_tag"] >= kk])
           for i in range(n)]
    print(f"SHIPPED control cluster>={kk} modal at the original gates : "
          f"{mac(inc):.4f}   spans/post {np.mean([len(p) for p in inc]):.2f}")
    wide = [finish(i, [rc["span"] for rc in REC[i] if rc["n_tag"] >= kk])
            for i in range(n)]
    print(f"same rule on the WIDENED pool                           : "
          f"{mac(wide):.4f}   spans/post {np.mean([len(p) for p in wide]):.2f}")

    # ---- cross-fitted: calibrate on FIT authors, select on EVAL -----------
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    frac = 378.0 / (378.0 + n)
    grid = [0.25, 0.30, 0.35, 0.39, 0.45, 0.50, 0.55, 0.60]
    rng = np.random.default_rng(args.seed)
    res = {("inc", 0): [], ("theory", 0): [], ("wide", 0): []}
    for t in grid:
        res[("thr", t)] = []
    coefs = []

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
        Emask = np.zeros(n, bool)
        Emask[E] = True
        trm = ~Emask[POST]
        if Y[trm].sum() < 20:
            continue
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(X[trm], Y[trm])
        coefs.append(lr.coef_[0])
        p_all = np.zeros(len(X))
        p_all[~trm] = lr.predict_proba(X[~trm])[:, 1]

        # probabilities aligned to REC[i] order -- the global index k is
        # REC-major within a post, so sorting by k restores that order
        acc = {}
        for k in np.flatnonzero(~trm):
            acc.setdefault(int(POST[k]), []).append((int(k), float(p_all[k])))
        prob_of = {i: [q for _, q in sorted(v)] for i, v in acc.items()}

        Eidx = list(E)
        gE = [G[i] for i in Eidx]
        res[("inc", 0)].append(mac(inc, Eidx))
        res[("wide", 0)].append(mac(wide, Eidx))

        def score_at(t):
            P = []
            for i in Eidx:
                q = prob_of.get(i)
                P.append(finish(i, [] if not q else
                                [rc["span"] for rc, qq in zip(REC[i], q)
                                 if qq > t]))
            return phrase_f1(gE, P, average="macro").f1

        for t in grid:
            res[("thr", t)].append(score_at(t))

        # the self-consistent theory rule: t = F1/2, F1 from the FIT half only
        res[("theory", 0)].append(
            score_at(mac(inc, list(np.flatnonzero(~Emask))) / 2.0))

    base = np.array(res[("inc", 0)])
    print("\n" + "=" * 84)
    print(f"CALIBRATED MATCH-PROBABILITY SELECTION -- {len(base)} author "
          f"partitions")
    print("=" * 84)
    print(f"{'rule':<22}{'macro':>9}{'delta':>10}{'win':>7}{'per-split sd':>14}"
          f"{'1-draw 95%':>22}")

    def row(name, v):
        d = v - base
        w = (d > 0).mean()
        sd = d.std(ddof=1)
        lo, hi = d.mean() - 1.96 * sd, d.mean() + 1.96 * sd
        print(f"{name:<22}{v.mean():>9.4f}{d.mean():>+10.4f}{w:>7.0%}"
              f"{sd:>14.4f}{f'[{lo:+.4f},{hi:+.4f}]':>22}")

    print(f"{'SHIPPED (control)':<22}{base.mean():>9.4f}{'--':>10}"
          f"{'--':>7}{'--':>14}{'--':>22}")
    row("cluster>=2 wide pool", np.array(res[("wide", 0)]))
    for t in grid:
        row(f"P(match) > {t:.2f}", np.array(res[("thr", t)]))
    row("theory  F1/2", np.array(res[("theory", 0)]))

    C = np.mean(coefs, axis=0)
    print(f"\nmean logistic coefficients (what actually predicts a match):")
    for nm, c in sorted(zip(FEATNAMES, C), key=lambda z: -abs(z[1])):
        print(f"   {nm:<14}{c:>+8.3f}")

    print("\n  The '1-draw 95%' column is mean +- 1.96*PER-SPLIT sd -- what one")
    print("  378-post test set can actually return. The CI of the mean shrinks")
    print("  with the resample count and does not bound a single draw. Accept")
    print("  only if the win rate is high AND the 1-draw interval is mostly")
    print("  positive.")
    print("\n  If the best threshold lands near F1/2 ~ 0.39, the plug-in theory")
    print("  holds on this task and the rule has 0 FITTED PARAMETERS.")


if __name__ == "__main__":
    main()
