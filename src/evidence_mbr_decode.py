#!/usr/bin/env python
"""SET-LEVEL MBR for evidence: enumerate subsets, score with the real metric.

WHY A SET-LEVEL DECODER AND NOT ANOTHER PER-CANDIDATE RANKER. A large share of
the candidate pool matches gold, and several matching candidates typically
point at the SAME gold span. Under one-to-one bipartite matching, emitting
three correct spans for one gold scores ONE match and TWO false positives. No
per-candidate score can express that constraint, because the cost of keeping a
candidate depends on which OTHER candidates are kept. A pool oracle scores far
above any per-candidate rule precisely because it picks one representative per
gold span.

Modelling clusters as INDEPENDENT Bernoullis has the same defect one level up:
it lets the gold count vary freely, when in fact a large share of posts have
exactly zero gold spans and most of the rest have one or two. This decoder puts
an EMPIRICAL COUNT PRIOR P(g) in front, estimated on FIT authors and
conditioned on predicted risk level, then enumerates every subset of clusters
and scores it with the actual metric:

    F1(S, g) = 1                      if |S| = 0 and g = 0
             = 0                      if exactly one of them is 0
             = 2*E[TP] / (|S| + g)    otherwise

    E[F1(S)] = sum_g P(g | risk) * F1(S, g)

Two consequences worth stating. Suppression becomes EMERGENT rather than a hard
rule -- if P(g=0 | indicator) is high, the argmax subset is empty on its own,
with no threshold. And the count prior is 4x~5 numbers, not a fitted scorer, so
there is very little to overfit.

E[TP | g] uses the cluster scores as a soft assignment: with g real spans among
the clusters, cluster i is real with probability proportional to its score
(`--assign soft`), or the top-g clusters are (`--assign hard`). Both are swept.

Clusters come from the SHIPPED equivalence (containment + 3x cap), so at most
one member of a cluster can match a given gold -- that is what makes "one
representative per cluster" the right hypothesis space.

Everything is cross-fitted by author: prior, assignment rule and any threshold
are fitted on FIT authors and scored on HELD-OUT. Val gains attenuate on test,
so the reported composite estimate divides the val delta by 2.8.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_span_pair_matches, normalize_label,  # noqa: E402
                     normalize_span, parse_evidence, phrase_f1, token_len)

RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]
LEVELS = ["indicator", "ideation", "behavior", "attempt"]
GMAX = 4          # gold counts are capped here; >=GMAX is pooled into GMAX


def cluster_post(items):
    """Shipped equivalence: containment either way within the 3x token cap.
    items: [(span, votes)] -> [(rep_span, votes_of_cluster)].
    Longest-first then lexicographic, because containment is not transitive and
    grouping would otherwise depend on dict order."""
    order = sorted(items, key=lambda t: (-token_len(t[0]), normalize_span(t[0])))
    groups = []
    for s, n in order:
        for g in groups:
            if any(_span_pair_matches(s, s2) or _span_pair_matches(s2, s)
                   for s2, _ in g):
                g.append((s, n))
                break
        else:
            groups.append([(s, n)])
    out = []
    for g in groups:
        cnt = {}
        for s, _ in g:
            cnt[s] = cnt.get(s, 0) + 1
        rep = max(cnt.items(), key=lambda kv: (kv[1], -token_len(kv[0])))[0]
        out.append((rep, max(n for _, n in g)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RC7))
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--split", default="val")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--baseline-k", type=int, default=2)
    ap.add_argument("--max-clusters", type=int, default=8,
                    help="enumerate 2^m subsets; posts with more clusters keep "
                         "only the top m by vote (m=8 -> 256 subsets)")
    args = ap.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = [str(x) for x in va["row_id"]]
    gold = [parse_evidence(e) for e in va["evidence"]]

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    acol = "anon_user_id" if "anon_user_id" in lab.columns else "author"
    amap = dict(zip(lab["row_id"], lab[acol].astype(str)))
    authors = np.array([amap.get(r, r) for r in rows])

    rk = pd.read_csv(args.risk, keep_default_na=False)
    rk["row_id"] = rk["row_id"].astype(str)
    rc = "pred_risk_level" if "pred_risk_level" in rk.columns else "risk_level"
    risk = {r: normalize_label(v) for r, v in zip(rk["row_id"], rk[rc])}

    # ---- candidates -> clusters -------------------------------------------
    votes, M = {}, 0
    for run in [x.strip() for x in args.runs.split(",") if x.strip()]:
        p = os.path.join(args.dir, f"{run}_{args.split}_candidates.csv")
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        M += 1
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        if "conf" in d.columns:
            d = d[d["conf"] >= mc]
        for r, s in zip(d["row_id"], d["span"]):
            n = normalize_span(str(s))
            if n:
                e = votes.setdefault(r, {}).setdefault(n, [0, str(s)])
                e[0] += 1

    CL = []
    for r in rows:
        c = cluster_post([(v[1], v[0]) for v in votes.get(r, {}).values()])
        c.sort(key=lambda t: -t[1])
        CL.append(c[:args.max_clusters])
    ncl = np.array([len(c) for c in CL])
    ng = np.array([len(g) for g in gold])
    print(f"{len(rows)} posts, {M} taggers")
    print(f"clusters/post mean {ncl.mean():.2f} max {ncl.max()}   "
          f"gold/post mean {ng.mean():.2f}")
    print(f"gold-count distribution: "
          f"{ {int(g): int((ng == g).sum()) for g in range(0, ng.max()+1)} }")

    subsets = {m: [tuple(s) for k in range(m + 1)
                   for s in itertools.combinations(range(m), k)]
               for m in range(0, args.max_clusters + 1)}

    def count_prior(fit_mask, alpha=1.0):
        """P(g | predicted risk level), Beta/Dirichlet-smoothed, FIT only."""
        tab = {}
        gi = np.minimum(ng, GMAX)
        for lv in LEVELS:
            m = fit_mask & np.array([risk.get(r) == lv for r in rows])
            c = np.full(GMAX + 1, alpha)
            for g in gi[m]:
                c[g] += 1
            tab[lv] = c / c.sum()
        return tab

    def decode(prior, assign, idxs):
        pred = {}
        for i in idxs:
            cl = CL[i]
            m = len(cl)
            if m == 0:
                pred[i] = []
                continue
            P = prior.get(risk.get(rows[i]), prior[LEVELS[0]])
            s = np.array([c[1] for c in cl], float)
            s = s / max(s.sum(), 1e-9)
            best, bv = (), -1.0
            for S in subsets[m]:
                k = len(S)
                v = 0.0
                for g in range(GMAX + 1):
                    pg = P[g]
                    if pg <= 0:
                        continue
                    if g == 0:
                        v += pg * (1.0 if k == 0 else 0.0)
                        continue
                    if k == 0:
                        continue
                    if assign == "hard":
                        etp = float(len(set(S) & set(range(min(g, m)))))
                    else:
                        etp = float(sum(min(1.0, g * s[j]) for j in S))
                    v += pg * 2.0 * etp / (k + g)
                if v > bv:
                    bv, best = v, S
            pred[i] = [cl[j][0] for j in best]
        return pred

    def sc(pred, idxs):
        return np.array([phrase_f1([gold[i]], [pred.get(i, [])],
                                   average="macro").f1 for i in idxs])

    # ---- baseline ----------------------------------------------------------
    base = {}
    for i, r in enumerate(rows):
        if risk.get(r) == "indicator":
            base[i] = []
            continue
        base[i] = [s for s, n in CL[i] if n >= args.baseline_k]
    allidx = np.arange(len(rows))
    bp = sc(base, allidx)
    print(f"\nBASELINE vote>={args.baseline_k} on clusters + suppress: "
          f"MACRO {bp.mean():.4f}   "
          f"{np.mean([len(base[i]) for i in allidx]):.2f} spans/post")

    fullprior = count_prior(np.ones(len(rows), bool))
    for assign in ("soft", "hard"):
        pr = decode(fullprior, assign, allidx)
        v = sc(pr, allidx)
        print(f"MBR {assign:<5} (in-sample prior): MACRO {v.mean():.4f}  "
              f"{v.mean()-bp.mean():+.4f}   "
              f"{np.mean([len(pr[i]) for i in allidx]):.2f} spans/post   "
              f"empty {sum(1 for i in allidx if not pr[i])}")

    # ---- cross-fit: prior AND assignment rule chosen on FIT ---------------
    rng = np.random.default_rng(args.seed)
    ua = np.array(sorted(set(authors)))
    held, chosen = [], {}
    for _ in range(args.splits):
        perm = rng.permutation(len(ua))
        fa = set(ua[perm[:len(ua) // 2]])
        fm = np.array([a in fa for a in authors])
        if fm.all() or not fm.any():
            continue
        pri = count_prior(fm)
        fi, hi = np.where(fm)[0], np.where(~fm)[0]
        best, bv = None, -1.0
        for assign in ("soft", "hard"):
            v = sc(decode(pri, assign, fi), fi).mean()
            if v > bv:
                bv, best = v, assign
        held.append(sc(decode(pri, best, hi), hi).mean() - bp[hi].mean())
        chosen[best] = chosen.get(best, 0) + 1
    held = np.array(held)
    win = float((held > 0).mean())

    print("\n" + "=" * 70)
    print("CROSS-FITTED (count prior + rule on FIT, scored on HELD-OUT)")
    print(f"  d evidence {held.mean():+.4f}  sd {held.std(ddof=1):.4f}  "
          f"win {100*win:.0f}%  over {len(held)} splits")
    print(f"  d composite {0.3*held.mean():+.4f}   "
          f"test estimate {0.3*held.mean()/2.8:+.4f} after 2.8x")
    print(f"  modal assignment: {max(chosen, key=chosen.get)}")
    if win >= 0.90 and held.mean() > 0:
        print("\n  CLEARS the 90% win-rate bar. To apply: fit the prior on ALL")
        print("  val authors, decode the test clusters, and leave every other")
        print("  component byte-identical so the delta is attributable.")
    else:
        print(f"\n  {100*win:.0f}% does not clear the 90% win-rate bar. This")
        print("  decoder's form matches the pool oracle -- it chooses subsets")
        print("  with the real metric and a count prior rather than a")
        print("  per-candidate score -- so a loss here indicates the oracle is")
        print("  not reachable from these cluster scores and the remaining")
        print("  headroom is in candidate generation, not selection.")
    print("=" * 70)


if __name__ == "__main__":
    main()
