#!/usr/bin/env python
"""PLUG-IN EXPECTED-F1 DECODING for evidence. Training-free, CPU, no new model.

WHY THIS DIFFERS FROM A THRESHOLD-BASED EVIDENCE DECODE.
`k` is an AGREEMENT threshold and a cap is a COUNT ceiling; both commit every
post to one global operating point. The metric is per-post F1 averaged over
posts, and the decision-theoretically correct decoder for that is not a
threshold at all -- it maximises EXPECTED F1 per post, choosing the CARDINALITY
post by post. Ye, Chai, Lee & Chieu (ICML 2012) and Waegeman et al. (JMLR 2014)
report **+3.9 to +10.1 F-points over marginal-mode decoding at 1,123-1,500
training examples**, which matches this regime exactly.

THE ABSTAIN SLOT IS NATIVE AND EXACT. MBR-style decoding otherwise needs a
CALIBRATED P(empty) or it can never abstain. Here k=0 is just another
candidate cardinality and its expected score is
    E[F(Y, empty)] = P(no span is correct) = prod_i (1 - p_i)
computed from the same marginals as every other k. Nothing is fitted. Given
that ~40% of posts have empty gold and score a free 1.0, this is the part most
likely to pay.

THE MATH, exact rather than the independence approximation.
For a selected set A of size k and its complement B:
    F = 2*TP / (S + k),  TP = #correct in A,  S = TP + FN = total gold
Under independent marginals TP ~ PoissonBinomial(p_i, i in A) and
FN ~ PoissonBinomial(p_j, j in B) are independent, so
    E[F] = sum_tp sum_fn P(TP=tp) P(FN=fn) * 2tp/(tp+fn+k)
Both Poisson-binomials are exact DPs, and m is small (a few clusters per post),
so the decoder enumerates k = 0..m exactly and takes the argmax. No sampling,
no search.

WHY CLUSTERS AND NOT RAW SPANS. The scorer matches one-to-one under containment,
so two overlapping candidates cannot both match the same gold -- they are not
independent events and treating them as such would inflate every E[F]. The
shipped decode already collapses overlaps into clusters; this decodes over
CLUSTER representatives, which restores the independence the math assumes.

THE MARGINAL IS THE HARD PART, AND A NAIVE ESTIMATE GETS IT WRONG.
Using the raw vote fraction n/M as the marginal is a badly BIASED estimate: a
2-of-7 cluster gets p = 0.286, so E[F|k=0] = 0.71 beats E[F|k=1] = 0.29 and the
decoder would abstain far too often, because the taggers actually hit gold at
75-81%, not 28.6%. A temperature rescales a logit rather than shifting it, so
it cannot fix a biased estimate. This version instead estimates P(correct | v
votes) EMPIRICALLY from FIT authors, Beta-smoothed, which is a bias correction
rather than a scale one.

HONEST LIMITS. (i) The calibration table is M+1 numbers fitted on FIT authors
and scored on HELD-OUT ones -- honest, but it is still fitted. (ii)
Independence across clusters is an approximation even after collapsing
overlaps. (iii) Divide any val gain by ~2.8 for val->test attenuation before it
earns a submission slot.
"""
import argparse
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


def poisson_binomial(ps):
    """Exact P(exactly j successes) for independent Bernoulli(ps)."""
    d = np.zeros(len(ps) + 1)
    d[0] = 1.0
    for p in ps:
        d[1:] = d[1:] * (1 - p) + d[:-1] * p
        d[0] *= (1 - p)
    return d


def expected_f1(sel_p, rest_p, k):
    """E[2*TP/(S+k)] with TP over the selected, FN over the rest."""
    if k == 0:
        # F = 1 exactly when there is no gold at all, else 0
        return float(np.prod([1 - p for p in rest_p])) if rest_p else 1.0
    a = poisson_binomial(sel_p)
    b = poisson_binomial(rest_p) if rest_p else np.array([1.0])
    tot = 0.0
    for tp, pa in enumerate(a):
        if pa <= 0:
            continue
        for fn, pb in enumerate(b):
            if pb <= 0:
                continue
            tot += pa * pb * (2.0 * tp / (tp + fn + k))
    return float(tot)


def cluster(items):
    """Collapse overlapping spans under the SCORER's own equivalence.

    items: list of (span, n_votes). Returns [(rep_span, n_votes_of_cluster)].
    Longest-first then lexicographic so grouping is deterministic (containment
    is not transitive, so order would otherwise decide the result).
    """
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
        # representative = modal span, tie-broken short (the shipped `modal`)
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
    supp = {r: normalize_label(v) == "indicator"
            for r, v in zip(rk["row_id"], rk[rc])}

    runs = [x.strip() for x in args.runs.split(",") if x.strip()]
    per = {}
    for run in runs:
        p = os.path.join(args.dir, f"{run}_{args.split}_candidates.csv")
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        if "conf" in d.columns:
            d = d[d["conf"] >= mc]
        m = {}
        for r, s in zip(d["row_id"], d["span"]):
            m.setdefault(r, set()).add(str(s))
        per[run] = m
    M = len(per)
    if M < 3:
        raise SystemExit("need at least 3 taggers with candidate dumps")
    print(f"{len(rows)} posts, {M} taggers")

    # per-post clusters with a vote-fraction marginal
    cl = []
    for r in rows:
        cnt = {}
        for run in per:
            for s in per[run].get(r, ()):
                n = normalize_span(s)
                if n:
                    e = cnt.setdefault(n, [0, s])
                    e[0] += 1
        cl.append(cluster([(v[1], v[0]) for v in cnt.values()]))
    print(f"mean clusters/post {np.mean([len(c) for c in cl]):.2f}   "
          f"mean gold/post {np.mean([len(g) for g in gold]):.2f}")

    # ---- THE MARGINAL HAS TO BE CALIBRATED ---------------------------------
    # The first version of this used the raw vote fraction n/M and abstained
    # on 45% of posts. The arithmetic was faithful and the input was wrong: a
    # 2-of-7 cluster gets p = 0.286, so E[F|k=0] = 0.71 beats E[F|k=1] = 0.29
    # and abstaining wins -- while the taggers actually hit gold at 75-81%.
    # Vote fraction is a badly BIASED estimate of P(this span matches gold),
    # and temperature cannot fix a bias, only a scale. So estimate
    # P(correct | v votes) empirically, on FIT authors only.
    hit = [[bool(any(_span_pair_matches(s, g) or _span_pair_matches(g, s)
                     for g in gold[i])) for s, _ in cl[i]]
           for i in range(len(rows))]
    votes = [[int(n) for _, n in cl[i]] for i in range(len(rows))]
    allv = np.array([v for vv in votes for v in vv])
    allh = np.array([h for hh in hit for h in hh])
    print("\nempirical P(span matches gold | vote count), whole val:")
    for v in range(1, M + 1):
        m_ = allv == v
        if m_.sum():
            print(f"  {v}/{M} votes: {allh[m_].mean():.3f}   n={int(m_.sum())}")
    print("  (the raw vote FRACTION would have said "
          + ", ".join(f"{v/M:.2f}" for v in range(1, M + 1)) + ")")

    def calib_table(fit_mask, prior=0.5, strength=4.0):
        """P(correct | v) from FIT authors only, Beta-smoothed toward `prior`
        so a vote count seen twice does not produce 0.0 or 1.0."""
        tab = np.full(M + 1, prior)
        for v in range(0, M + 1):
            num = den = 0.0
            for i in np.where(fit_mask)[0]:
                for vv, hh in zip(votes[i], hit[i]):
                    if vv == v:
                        den += 1
                        num += float(hh)
            tab[v] = (num + prior * strength) / (den + strength)
        return np.clip(tab, 1e-4, 1 - 1e-4)

    def decode(tab, idx=None, use_supp=True):
        """tab: P(correct | vote count). idx: posts to decode (None = all)."""
        want = range(len(rows)) if idx is None else idx
        pred = {}
        for i in want:
            r = rows[i]
            if use_supp and supp.get(r, False):
                pred[i] = []
                continue
            c = cl[i]
            if not c:
                pred[i] = []
                continue
            p = np.array([tab[int(n)] for _, n in c])
            o = np.argsort(-p)
            sp = [c[j][0] for j in o]
            pp = [float(p[j]) for j in o]
            best_k, best_v = 0, expected_f1([], pp, 0)
            for k in range(1, len(sp) + 1):
                v = expected_f1(pp[:k], pp[k:], k)
                if v > best_v:
                    best_k, best_v = k, v
            pred[i] = sp[:best_k]
        return pred

    def macro(pred, idx=None):
        if idx is None:
            return phrase_f1(gold, pred, average="macro").f1
        g = [gold[i] for i in idx]
        p = [pred[i] for i in idx]
        return phrase_f1(g, p, average="macro").f1

    # baseline: the shipped vote>=k over the same clusters
    base = []
    for i, r in enumerate(rows):
        if supp.get(r, False):
            base.append([])
            continue
        base.append([s for s, n in cl[i] if n >= args.baseline_k])
    b = macro(base)
    print(f"\nBASELINE vote>={args.baseline_k} on clusters: MACRO {b:.4f}   "
          f"{np.mean([len(p) for p in base]):.2f} spans/post")

    # in-sample sanity: what the decoder does with a table fitted on everything
    full_tab = calib_table(np.ones(len(rows), bool))
    pr = decode(full_tab)
    print(f"\ncalibrated table (whole val, IN-SAMPLE -- direction only):")
    print("  " + "  ".join(f"{v}:{full_tab[v]:.2f}" for v in range(1, M + 1)))
    print(f"  MACRO {macro([pr[i] for i in range(len(rows))]):.4f}   "
          f"{np.mean([len(pr[i]) for i in range(len(rows))]):.2f} spans/post   "
          f"empty {sum(1 for i in range(len(rows)) if not pr[i])}")

    # ---- honest cross-fit: the CALIBRATION is fitted on FIT authors only ----
    rng = np.random.default_rng(args.seed)
    ua = np.array(sorted(set(authors)))
    bp = np.array([phrase_f1([gold[i]], [base[i]], average="macro").f1
                   for i in range(len(rows))])
    held = []
    for _ in range(args.splits):
        perm = rng.permutation(len(ua))
        fa = set(ua[perm[:len(ua) // 2]])
        fm = np.array([a in fa for a in authors])
        if fm.all() or not fm.any():
            continue
        tab = calib_table(fm)                       # FIT authors only
        hi = np.where(~fm)[0]
        pr_h = decode(tab, idx=hi)
        v = float(np.mean([phrase_f1([gold[i]], [pr_h[i]],
                                     average="macro").f1 for i in hi]))
        held.append(v - bp[hi].mean())
    held = np.array(held)
    win = float((held > 0).mean())
    print("\n" + "=" * 68)
    print("CROSS-FITTED (calibration table fitted on FIT, scored on HELD-OUT)")
    print(f"  d evidence {held.mean():+.4f}  sd {held.std(ddof=1):.4f}  "
          f"win {100*win:.0f}%  over {len(held)} splits")
    print(f"  d composite {0.3*held.mean():+.4f}   "
          f"test estimate {0.3*held.mean()/2.8:+.4f} after 2.8x")
    if win >= 0.90 and held.mean() > 0:
        print("\n  CLEARS. Build the test column with a table fitted on ALL")
        print("  val authors, hold factors byte-identical, S2 must be 0.6908.")
    else:
        print(f"\n  {100*win:.0f}% does not clear 90%.")
        print("  Diagnosis: compare the calibrated table above against the raw")
        print("  vote fractions. If calibration moved the marginals a lot and")
        print("  the score barely moved, the loss is not in the CARDINALITY")
        print("  choice -- vote>=2 was already picking the right number of")
        print("  spans -- and per-post cardinality is closed for evidence.")
    print("=" * 68)


if __name__ == "__main__":
    main()
