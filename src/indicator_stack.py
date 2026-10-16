"""
indicator_stack.py

Fit P(gold evidence is empty) from risk logits PLUS evidence-tagger features,
and use it to decide indicator suppression.

THE ONE DECISION THE SCORER CHARGES TWICE. Gold evidence is empty iff gold risk
is `indicator` (0.986 of posts). Evidence is macro-per-post with empty-vs-empty
scoring 1.0. So a wrong indicator call costs the risk credit AND the whole
post's evidence, and the two error directions are:

    predicted indicator, gold empty       score 1.0
    predicted indicator, gold NON-empty   score 0.0  (real spans suppressed)
    predicted OTHER,     gold empty       score 0.0  (spans on empty gold)
    predicted OTHER,     gold NON-empty   scored on span quality

WHY THE RISK MODEL'S OWN p_indicator CANNOT FIX IT. Risk errors on this task are
CONFIDENTLY wrong, so re-decoding the same model's probabilities -- shifting a
threshold, biasing the argmax -- cannot recover a post the model is certain
about. Any improvement has to come from a signal that model does not contain.

THE SIGNAL IT ADDS. The evidence tagger's candidate count separates empty from
non-empty gold far more sharply than chance, and it is independent of every risk
model: a tagger trained to find risk spans going silent on a post is information
about that post which the risk classifier's logits do not carry.

WHAT THIS FITS. A small logistic model for P(gold evidence empty) on:

    the four risk logits
    n candidate clusters, and whether there are none at all
    max / mean candidate confidence
    post length
    lexical flags (see below)

cross-fitted on AUTHOR partitions, and scored on the REAL joint objective
0.4*risk_wF1 + 0.3*evidence_macro with the actual tagger spans, so the number
reported is the one the scorer would produce rather than a proxy.

TWO WAYS TO USE IT, AND THEY ARE NOT THE SAME:
  couple    the label becomes `indicator`, changing risk AND evidence
  decouple  keep the argmax risk label, use the stack ONLY to decide
            suppression. risk_level and evidence are INDEPENDENT COLUMNS, so
            this buys the evidence without paying any risk penalty.

JUDGED ON WIN RATE AND PER-DRAW SD, never the CI of the mean: the CI of the
mean shrinks with the number of resampled splits, which is a free parameter, so
it can be made to exclude zero for an effect that loses on a single draw. Bar:
win >= 65% AND a one-draw interval that is mostly positive.

Usage (cluster, CPU, ~2 min):
  python src/indicator_stack.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span, parse_evidence,  # noqa: E402
                     phrase_f1, token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
PCOL = {c: f"prob_{c}" for c in LEVELS}
TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]


def wf1(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    tot = 0.0
    for c in LEVELS:
        g, p = gold == c, pred == c
        tp = int((g & p).sum())
        den = 2 * tp + int((~g & p).sum()) + int((g & ~p).sum())
        tot += (0.0 if den == 0 else 2.0 * tp / den) * g.sum()
    return float(tot / max(len(gold), 1))


def dedupe(spans):
    out = []
    for s in sorted(spans, key=lambda x: (-token_len(x),
                                          -len(normalize_span(x)))):
        n = normalize_span(s)
        if not n or any(n in normalize_span(k) for k in out):
            continue
        out.append(s)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--dir", default="artifacts/evidence_oof")
    a_.add_argument("--tags", default=",".join(TAGS))
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=13)
    args = a_.parse_args()

    from sklearn.linear_model import LogisticRegression

    oof = pd.read_csv(args.oof, keep_default_na=False)
    oof["row_id"] = oof["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rows = [r for r in oof["row_id"] if r in lab.index]
    n = len(rows)
    ix = oof.set_index("row_id")

    per_tag = {}
    for t in [x.strip() for x in args.tags.split(",")]:
        p = os.path.join(args.dir, f"{t}_oof_candidates.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        d = d[d["conf"] >= mc]
        m = {}
        for r, s, c in zip(d["row_id"].astype(str), d["span"], d["conf"]):
            m.setdefault(r, []).append((str(s), float(c)))
        per_tag[t] = m
    if len(per_tag) < 2:
        raise SystemExit("need at least two taggers")
    base_tag = "v2" if "v2" in per_tag else list(per_tag)[0]

    P = np.stack([ix.loc[rows, PCOL[c]].astype(float).to_numpy()
                  for c in LEVELS], axis=1)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    Z = np.log(np.clip(P, 1e-6, 1)) - np.log(np.clip(1 - P, 1e-6, 1))
    ygold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    yship = np.array([normalize_label(x)
                      for x in ix.loc[rows, "pred_risk_level"]])
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    empty = np.array([len(g) == 0 for g in G])

    # tagger features + the shipped vote spans
    nclu, cmax, cmean, U = [], [], [], []
    for r in rows:
        allc = [(t, s, c) for t in per_tag for s, c in per_tag[t].get(r, [])]
        cnt = {}
        for t, s, c in allc:
            k = normalize_span(s)
            cnt.setdefault(k, [set(), s, []])
            cnt[k][0].add(t)
            cnt[k][2].append(c)
        nclu.append(len(cnt))
        cs = [c for _, _, c in allc]
        cmax.append(max(cs) if cs else 0.0)
        cmean.append(float(np.mean(cs)) if cs else 0.0)
        keep = dedupe([v[1] for v in cnt.values() if len(v[0]) >= 2])
        if not keep:
            pool = [s for t in per_tag for s, _ in per_tag[t].get(r, [])]
            keep = sorted(pool, key=token_len)[:1]
        U.append(keep)
    nclu = np.array(nclu, float)
    nw = np.array([len(str(lab.loc[r, "post"]).split()) for r in rows], float)
    # LEXICAL FLAGS. "Indicator = no explicit mention of suicide" is the
    # organizers' own definition, so an explicit-mention regex is a direct read
    # on the label. Alone it has poor recall -- suicidality is expressed in an
    # open vocabulary, and a phrasing like "I'm going to die, goodbye" carries
    # no keyword -- but when it does fire the post is non-indicator at higher
    # precision than the risk model reaches. It is therefore a low-recall,
    # high-precision signal pointing the OPPOSITE way to the tagger-silence
    # signal. A hand-written rule would have to pick one of the two whenever
    # they conflict; a fitted stack weighs them instead.
    from risk_symbolic_fences import DONE, EXPLICIT, PLAN, explicit_unnegated
    posts = [str(lab.loc[r, "post"]) for r in rows]
    rx = np.column_stack([
        np.array([bool(EXPLICIT.search(t)) for t in posts], float),
        np.array([explicit_unnegated(t) for t in posts], float),
        np.array([bool(PLAN.search(t)) for t in posts], float),
        np.array([bool(DONE.search(t)) for t in posts], float)])
    X = np.column_stack([Z, nclu, (nclu == 0).astype(float),
                         np.array(cmax), np.array(cmean), np.log1p(nw), rx])
    XR = Z                                    # risk-only baseline features
    XT = np.column_stack([Z, nclu, (nclu == 0).astype(float),
                          np.array(cmax), np.array(cmean), np.log1p(nw)])
    FN = ["z_ind", "z_ide", "z_beh", "z_att", "n_clusters", "no_candidates",
          "conf_max", "conf_mean", "log_postlen",
          "rx_explicit", "rx_expl_unneg", "rx_plan", "rx_done"]

    print(f"{n} OOF posts   gold-empty {int(empty.sum())} "
          f"({empty.mean():.1%})")
    z0, z1 = empty[nclu == 0], empty[nclu > 0]
    print(f"  posts with NO tagger candidate: {int((nclu==0).sum())}, "
          f"{z0.mean():.1%} of them gold-empty")
    print(f"  posts WITH candidates:          {int((nclu>0).sum())}, "
          f"{z1.mean():.1%} of them gold-empty")
    pi = yship == "indicator"
    print(f"  shipped indicator recall {(pi & empty).sum()/max(empty.sum(),1):.3f}"
          f"  precision {(pi & empty).sum()/max(pi.sum(),1):.3f}\n")

    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    frac = 378.0 / (378.0 + n)
    rng = np.random.default_rng(args.seed)
    # The grid runs well below the plausible operating point so that the
    # selected t is interior. An optimum sitting on the edge of the grid is not
    # an optimum, only the end of the search.
    grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55, 0.65]
    keys = [("ship", 0), ("decfit", 0), ("cplfit", 0),
            ("addfit", 0), ("rmfit", 0)] + \
           [("dec", t) for t in grid] + \
           [("cpl", t) for t in grid] + [("risk", t) for t in grid] + \
           [("notx", t) for t in grid]
    res = {k: [] for k in keys}
    picks = {"dec": [], "cpl": [], "add": [], "rm": []}
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
        M = np.ones(n, bool)
        M[E] = False
        if empty[M].sum() < 20:
            continue
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(X[M], empty[M])
        coefs.append(lr.coef_[0])
        q = lr.predict_proba(X[E])[:, 1]
        lr2 = LogisticRegression(max_iter=2000, C=1.0)
        lr2.fit(XR[M], empty[M])
        q2 = lr2.predict_proba(XR[E])[:, 1]
        # second control: tagger features but NO regex, so the marginal value
        # of the lexical flags is isolated from the tagger's contribution
        lr3 = LogisticRegression(max_iter=2000, C=1.0)
        lr3.fit(XT[M], empty[M])
        q3 = lr3.predict_proba(XT[E])[:, 1]

        gE = [G[i] for i in E]
        rk_ship = wf1(ygold[E], yship[E])
        base = 0.4 * rk_ship + 0.3 * phrase_f1(
            gE, [[] if yship[i] == "indicator" else U[i] for i in E],
            average="macro").f1
        res[("ship", 0)].append(base)

        other = np.array(LEVELS[1:])[P[E][:, 1:].argmax(1)]
        for t in grid:
            for tag, qq in (("dec", q), ("risk", q2), ("notx", q3)):
                sup = qq >= t
                res[(tag, t)].append(
                    0.4 * rk_ship + 0.3 * phrase_f1(
                        gE, [[] if s else U[i] for s, i in zip(sup, E)],
                        average="macro").f1)
            sup = q >= t
            pr = np.where(sup, "indicator", other)
            res[("cpl", t)].append(
                0.4 * wf1(ygold[E], pr) + 0.3 * phrase_f1(
                    gE, [[] if s else U[i] for s, i in zip(sup, E)],
                    average="macro").f1)

        # ---- THE ONLY NUMBER THAT CAN SHIP -------------------------------
        # t chosen on the FIT authors, then applied to held-out. Reading the
        # best t off the eval column would be selecting a parameter on the very
        # data used to score it, which reports the maximum of noise as a gain.
        Mi = np.flatnonzero(M)
        qf = lr.predict_proba(X[Mi])[:, 1]
        gF = [G[i] for i in Mi]
        otherF = np.array(LEVELS[1:])[P[Mi][:, 1:].argmax(1)]

        def jf(t, couple):
            s = qf >= t
            e = phrase_f1(gF, [[] if ss else U[i] for ss, i in zip(s, Mi)],
                          average="macro").f1
            r = (wf1(ygold[Mi], np.where(s, "indicator", otherF)) if couple
                 else wf1(ygold[Mi], yship[Mi]))
            return 0.4 * r + 0.3 * e

        # ONE-DIRECTIONAL variants. The shipped evidence column comes from a
        # RISK-CONDITIONED tagger, so the spans it would have emitted on a post
        # it suppressed cannot be reconstructed offline. Emptying a post needs
        # no reconstruction at all, which makes `add` the only variant
        # buildable from the shipped column alone.
        shipsupF = yship[Mi] == "indicator"

        def jf1(t, mode):
            s = ((shipsupF | (qf >= t)) if mode == "add"
                 else (shipsupF & (qf >= t)))
            e = phrase_f1(gF, [[] if ss else U[i] for ss, i in zip(s, Mi)],
                          average="macro").f1
            return 0.4 * wf1(ygold[Mi], yship[Mi]) + 0.3 * e

        ta = max(grid, key=lambda t: jf1(t, "add"))
        trm = max(grid, key=lambda t: jf1(t, "rm"))
        shipsupE = yship[E] == "indicator"
        for nm, mask in (("addfit", shipsupE | (q >= ta)),
                         ("rmfit", shipsupE & (q >= trm))):
            res[(nm, 0)].append(
                0.4 * rk_ship + 0.3 * phrase_f1(
                    gE, [[] if s else U[i] for s, i in zip(mask, E)],
                    average="macro").f1)
        picks["add"].append(ta)
        picks["rm"].append(trm)

        td = max(grid, key=lambda t: jf(t, False))
        tc = max(grid, key=lambda t: jf(t, True))
        picks["dec"].append(td)
        picks["cpl"].append(tc)
        sd_ = q >= td
        res[("decfit", 0)].append(
            0.4 * rk_ship + 0.3 * phrase_f1(
                gE, [[] if s else U[i] for s, i in zip(sd_, E)],
                average="macro").f1)
        sc_ = q >= tc
        res[("cplfit", 0)].append(
            0.4 * wf1(ygold[E], np.where(sc_, "indicator", other))
            + 0.3 * phrase_f1(
                gE, [[] if s else U[i] for s, i in zip(sc_, E)],
                average="macro").f1)

    base = np.array(res[("ship", 0)])
    print("=" * 92)
    print(f"JOINT 0.4*risk + 0.3*evidence -- {len(base)} author partitions")
    print("=" * 92)
    print(f"{'rule':<28}{'joint':>9}{'delta':>10}{'win':>7}"
          f"{'sd':>9}{'one-draw 95%':>24}")
    print(f"{'SHIPPED (argmax suppress)':<28}{base.mean():>9.4f}{'--':>10}"
          f"{'--':>7}{'--':>9}{'--':>24}")

    def row(nm, v):
        d = v - base
        sd = d.std(ddof=1)
        print(f"{nm:<28}{v.mean():>9.4f}{d.mean():>+10.4f}"
              f"{(d > 0).mean():>7.0%}{sd:>9.4f}"
              f"{f'[{d.mean()-1.96*sd:+.4f},{d.mean()+1.96*sd:+.4f}]':>24}")

    for t in grid:
        row(f"risk-only stack t={t:.2f}", np.array(res[("risk", t)]))
    print()
    for t in grid:
        row(f"tagger,NO regex t={t:.2f}", np.array(res[("notx", t)]))
    print()
    for t in grid:
        row(f"+TAGGER decouple t={t:.2f}", np.array(res[("dec", t)]))
    print()
    for t in grid:
        row(f"+TAGGER couple  t={t:.2f}", np.array(res[("cpl", t)]))

    print("\n" + "-" * 92)
    print("t CHOSEN ON FIT AUTHORS, SCORED HELD-OUT -- the only shippable row")
    print("-" * 92)
    row("FITTED decouple", np.array(res[("decfit", 0)]))
    row("FITTED couple", np.array(res[("cplfit", 0)]))
    print("  -- and the only variants buildable from the shipped column --")
    row("FITTED add-only", np.array(res[("addfit", 0)]))
    row("FITTED remove-only", np.array(res[("rmfit", 0)]))
    for k in ("dec", "cpl", "add", "rm"):
        v, c = np.unique(picks[k], return_counts=True)
        print(f"  {k} chose: " + "  ".join(f"{a:.2f}x{b}"
                                           for a, b in zip(v, c)))
    print("  An unstable choice of t means the parameter is fitted to noise")
    print("  however good the mean looks.")

    C = np.mean(coefs, axis=0)
    print("\nmean logistic coefficients for P(gold evidence empty):")
    for nm, c in sorted(zip(FN, C), key=lambda z: -abs(z[1])):
        print(f"   {nm:<16}{c:>+8.3f}")
    print("\n  The 'risk-only' block is the control: the SAME stack without")
    print("  the tagger features. If +TAGGER beats risk-only, the evidence")
    print("  tagger carries indicator information the risk model does not,")
    print("  which is the whole claim. If they match, it does not.")
    print("  DECOUPLE changes only the evidence column, so S1's risk term is")
    print("  untouched and the change is one variable.")


if __name__ == "__main__":
    main()
