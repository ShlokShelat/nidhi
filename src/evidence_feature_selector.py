#!/usr/bin/env python
"""A FEATURE-BASED candidate scorer for RC7. CPU, no GPU, no new model.

THE MEASUREMENT THAT JUSTIFIES THIS (evidence_pool_oracle_compare.py):

    pool   cands/post   gold covered   ORACLE
    RC7        3.0         86.6%       0.9457
    shipped decode (vote>=2 + dedupe)              ~0.806

**+0.14 of selection headroom inside a pool of THREE candidates per post.**
The enum pool adds only +0.05 on top. So the evidence gap is not candidate
coverage -- it is choosing among three things, and `vote>=2` is leaving a lot
on the table.

WHY EVERY PREVIOUS SELECTOR FAILED, in one table of AUCs on RC7's own pool:

    raw vote count                 0.610
    reranker trained on ENUM       0.686   (out of distribution)
    reranker trained on RC7 (3.7k) 0.61    (data-starved, below confidence)
    HAND-CRAFTED FEATURES          0.752   <- best, and NEVER USED IN A DECODE

That 0.752 only ever appeared as a printed comparator string. This script
actually decodes with it.

WHY FEATURES AND NOT A CROSS-ENCODER. The whole training set is ~1.1k
candidates. A 435M cross-encoder starves on that (measured: AUC 0.61, below the
raw confidence baseline). A dozen features with logistic regression does not.
The literature agrees for this regime -- BERT needs ~100 instances per class to
predict it at all.

HONESTY. Features are fitted on FIT authors and every number is scored on
HELD-OUT authors, over many splits; the baseline is re-scored identically. The
label is "this span matches some gold" under the official matcher, which is a
per-candidate proxy for a set-level metric -- fine as a ranking signal, and the
decode handles the set. Divide any val gain by ~2.8 for val->test.
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
LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def matches(a, b):
    return _span_pair_matches(a, b) or _span_pair_matches(b, a)


def dedupe_nested(spans):
    """Keep the longest of any containment chain -- the shipped rule."""
    out = []
    for s in sorted(spans, key=lambda x: -token_len(x)):
        n = normalize_span(s)
        if not any(n in normalize_span(k) or normalize_span(k) in n
                   for k in out):
            out.append(s)
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
    post = {r: str(p) for r, p in zip(va["row_id"], va["post"])}

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    acol = "anon_user_id" if "anon_user_id" in lab.columns else "author"
    amap = dict(zip(lab["row_id"], lab[acol].astype(str)))
    authors = np.array([amap.get(r, r) for r in rows])

    rk = pd.read_csv(args.risk, keep_default_na=False)
    rk["row_id"] = rk["row_id"].astype(str)
    rc = "pred_risk_level" if "pred_risk_level" in rk.columns else "risk_level"
    risk = {r: normalize_label(v) for r, v in zip(rk["row_id"], rk[rc])}

    # ---- gather candidates with per-tagger confidence ----------------------
    cand = {}
    ntag = 0
    for run in [x.strip() for x in args.runs.split(",") if x.strip()]:
        p = os.path.join(args.dir, f"{run}_{args.split}_candidates.csv")
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        ntag += 1
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        cf = d["conf"] if "conf" in d.columns else pd.Series([1.0] * len(d))
        if "conf" in d.columns:
            keep = d["conf"] >= mc
            d, cf = d[keep], cf[keep]
        for r, s, c in zip(d["row_id"], d["span"], cf):
            n = normalize_span(str(s))
            if not n:
                continue
            e = cand.setdefault(r, {}).setdefault(
                n, {"span": str(s), "confs": []})
            e["confs"].append(float(c))
    print(f"{len(rows)} posts, {ntag} taggers, "
          f"{sum(len(v) for v in cand.values())} distinct candidates "
          f"({sum(len(v) for v in cand.values())/len(rows):.2f}/post)")

    # ---- features ----------------------------------------------------------
    X, y, pidx, spans = [], [], [], []
    for i, r in enumerate(rows):
        cs = list(cand.get(r, {}).values())
        if not cs:
            continue
        plen = max(len(post.get(r, "").split()), 1)
        norms = [normalize_span(c["span"]) for c in cs]
        lens = [token_len(c["span"]) for c in cs]
        for j, c in enumerate(cs):
            nv = len(c["confs"])
            tl = lens[j]
            nested = sum(1 for k2, n2 in enumerate(norms)
                         if k2 != j and norms[j] in n2)
            contains = sum(1 for k2, n2 in enumerate(norms)
                           if k2 != j and n2 in norms[j])
            posn = post.get(r, "").lower().find(c["span"].lower()[:40])
            lv = risk.get(r, "indicator")
            X.append([
                nv, nv / max(ntag, 1),
                tl, tl / plen, np.log1p(tl),
                float(np.mean(c["confs"])), float(np.max(c["confs"])),
                float(np.min(c["confs"])),
                len(cs), nested, contains,
                tl - float(np.mean(lens)),
                1.0 if tl == max(lens) else 0.0,
                1.0 if tl == min(lens) else 0.0,
                (posn / max(len(post.get(r, "")), 1)) if posn >= 0 else -1.0,
                np.log1p(plen),
            ] + [1.0 if lv == L else 0.0 for L in LEVELS])
            y.append(1 if any(matches(c["span"], g) for g in gold[i]) else 0)
            pidx.append(i)
            spans.append(c["span"])
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    pidx = np.asarray(pidx, int)
    print(f"{len(X)} candidates, {y.mean()*100:.1f}% match gold, "
          f"{X.shape[1]} features")

    # ---- baseline: the shipped decode --------------------------------------
    def score_pred(pred_by_post):
        pred = [pred_by_post.get(i, []) for i in range(len(rows))]
        return np.array([phrase_f1([gold[i]], [pred[i]],
                                   average="macro").f1
                         for i in range(len(rows))])

    base_by = {}
    for i, r in enumerate(rows):
        if risk.get(r) == "indicator":
            base_by[i] = []
            continue
        keep = [c["span"] for c in cand.get(r, {}).values()
                if len(c["confs"]) >= args.baseline_k]
        base_by[i] = dedupe_nested(keep) if keep else []
    bp = score_pred(base_by)
    print(f"\nBASELINE vote>={args.baseline_k} + dedupe + suppress: "
          f"MACRO {bp.mean():.4f}")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise SystemExit("needs scikit-learn")

    RULES = [("thr", t) for t in np.arange(0.20, 0.85, 0.05)] + \
            [("top1", 0.0), ("top2", 0.0)] + \
            [("top1t", t) for t in (0.3, 0.4, 0.5, 0.6)] + \
            [("top2t", t) for t in (0.3, 0.4, 0.5, 0.6)]

    def decode(prob, idxs, rule, t):
        out = {}
        for i in idxs:
            if risk.get(rows[i]) == "indicator":
                out[i] = []
                continue
            m = np.where(pidx == i)[0]
            if len(m) == 0:
                out[i] = []
                continue
            o = m[np.argsort(-prob[m])]
            if rule == "thr":
                sel = [k for k in o if prob[k] >= t]
            elif rule == "top1":
                sel = list(o[:1])
            elif rule == "top2":
                sel = list(o[:2])
            elif rule == "top1t":
                sel = [k for k in o[:1] if prob[k] >= t]
            else:
                sel = [k for k in o[:2] if prob[k] >= t]
            out[i] = dedupe_nested([spans[k] for k in sel]) if sel else []
        return out

    rng = np.random.default_rng(args.seed)
    ua = np.array(sorted(set(authors)))
    held, aucs, chosen = [], [], {}
    for _ in range(args.splits):
        perm = rng.permutation(len(ua))
        fa = set(ua[perm[:len(ua) // 2]])
        fm_post = np.array([a in fa for a in authors])
        fm = fm_post[pidx]
        if fm.all() or not fm.any() or y[fm].sum() < 10:
            continue
        sc = StandardScaler().fit(X[fm])
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[fm]),
                                                           y[fm])
        prob = clf.predict_proba(sc.transform(X))[:, 1]
        if len(set(y[~fm])) > 1:
            aucs.append(roc_auc_score(y[~fm], prob[~fm]))
        fi = np.where(fm_post)[0]
        hi = np.where(~fm_post)[0]
        best, bv = None, -1.0
        for rule, t in RULES:
            v = score_pred(decode(prob, fi, rule, t))[fi].mean()
            if v > bv:
                bv, best = v, (rule, t)
        v = score_pred(decode(prob, hi, best[0], best[1]))[hi].mean()
        held.append(v - bp[hi].mean())
        chosen[best] = chosen.get(best, 0) + 1

    held = np.array(held)
    win = float((held > 0).mean())
    bk = max(chosen, key=chosen.get)
    print(f"\nheld-out AUC of the feature scorer: {np.mean(aucs):.3f}")
    print(f"  (vote count alone 0.610, enum reranker 0.686, "
          f"hand features 0.752)")
    print("\n" + "=" * 70)
    print("CROSS-FITTED (model AND rule fitted on FIT, scored on HELD-OUT)")
    print(f"  d evidence {held.mean():+.4f}  sd {held.std(ddof=1):.4f}  "
          f"win {100*win:.0f}%  over {len(held)} splits")
    print(f"  d composite {0.3*held.mean():+.4f}   "
          f"test estimate {0.3*held.mean()/2.8:+.4f} after 2.8x")
    print(f"  modal rule {bk[0]} t={bk[1]:.2f}  {chosen[bk]}/{len(held)}")
    if win >= 0.90 and held.mean() > 0:
        print("\n  CLEARS. Fit on ALL val authors, apply to the test")
        print("  candidates, hold factors byte-identical, S2 must be 0.6908.")
    else:
        print(f"\n  {100*win:.0f}% does not clear 90%. If the AUC is well above")
        print("  0.610 and the decode still loses, ranking is not the binding")
        print("  constraint -- the oracle's 0.9457 needs the RIGHT SUBSET, and")
        print("  per-candidate scoring cannot express that.")
    print("=" * 70)


if __name__ == "__main__":
    main()
