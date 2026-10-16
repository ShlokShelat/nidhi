"""
risk_crossmodal_stack.py

RISK CARRIES 0.4 AND IS PREDICTED FROM ONE MODEL'S OPINION OF THE TEXT.

indicator_stack.py established that the EVIDENCE TAGGER knows things the risk
model does not: adding its confidence features moved the joint objective
substantially more than a risk-only control did, and most of that gain traces
back to the risk label rather than the evidence column it was allowed to
touch directly. So a cross-modal feature improved RISK while only being
allowed to touch ONE class. Risk is the heaviest component in the composite
and it has never been given anything except its own logits.

TWO INDEPENDENT SIGNALS ARE SITTING UNUSED:

  the evidence tagger   span count and confidence. A tagger going silent is
                        evidence of no explicit content (91.9% of no-candidate
                        posts are gold-empty, hence indicator).

  the FACTOR model      the C-SSRS definitions separate the classes by exactly
                        the things already predicted as factors:
                          Behavior = explicit expression AND a plan/self-harm
                          Attempt  = a recent/past attempt
                        `suicide means (with access)` (AP 0.84) and `prior
                        self-harm or suicidal thought/attempt` (AP 0.75) are
                        the literal criteria, produced by an 8-arm ensemble
                        that is NOT the risk model.

A prior audit closed risk decoding with "errors are confidently wrong so
decoding cannot fix them" -- but every one of those attempts
re-decoded the risk model's OWN probabilities. A confidently wrong classifier
is exactly what INDEPENDENT evidence is for, and the indicator result is the
existence proof.

MODELS COMPARED, all multinomial over the 4 classes, cross-fitted on AUTHOR
partitions, scored on the REAL joint 0.4*risk_wF1 + 0.3*evidence_macro with
actual tagger spans and suppression driven by the predicted label:

    argmax      the risk model alone (control)
    stack-r     restacking the risk logits ONLY -- the second control, and the
                thing the audit already closed. Must be flat.
    +tag        risk logits + tagger features
    +fac3       risk logits + the three C-SSRS-relevant factor probabilities
    +all        risk logits + tagger + all 24 factors (L2-regularised; the
                honest test of whether the extra 20 columns are signal)

Judged on WIN RATE and PER-DRAW sd -- [[ci-of-the-mean-error]].

Usage (cluster, CPU, ~3 min):
  python src/risk_crossmodal_stack.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (FACTOR_TAXONOMY, normalize_label, normalize_span,  # noqa: E402
                     parse_evidence, phrase_f1, token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
PCOL = {c: f"prob_{c}" for c in LEVELS}
TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
KEY3 = ["suicide means (with access)",
        "prior self-harm or suicidal thought/attempt",
        "hopelessness"]


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


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--evdir", default="artifacts/evidence_oof")
    a_.add_argument("--facdir", default="artifacts/factor_runs")
    a_.add_argument("--tags", default=",".join(TAGS))
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=13)
    a_.add_argument("--C", type=float, default=0.5)
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

    P = np.stack([ix.loc[rows, PCOL[c]].astype(float).to_numpy()
                  for c in LEVELS], axis=1)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    Z = logit(P)
    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    yship = np.array([normalize_label(x)
                      for x in ix.loc[rows, "pred_risk_level"]])
    G = [parse_evidence(lab.loc[r, "evidence"]) for r in rows]
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])

    # ---- tagger features + the shipped vote spans -------------------------
    per = {}
    for t in [x.strip() for x in args.tags.split(",")]:
        p = os.path.join(args.evdir, f"{t}_oof_candidates.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        d = d[d["conf"] >= mc]
        m = {}
        for r, s, c in zip(d["row_id"].astype(str), d["span"], d["conf"]):
            m.setdefault(r, []).append((str(s), float(c)))
        per[t] = m
    if len(per) < 2:
        raise SystemExit("need at least two taggers")

    TF, U = [], []
    for r in rows:
        allc = [(t, s, c) for t in per for s, c in per[t].get(r, [])]
        cnt = {}
        for t, s, c in allc:
            k = normalize_span(s)
            cnt.setdefault(k, [set(), s])
            cnt[k][0].add(t)
        cs = [c for _, _, c in allc]
        TF.append([len(cnt), 1.0 if not cnt else 0.0,
                   max(cs) if cs else 0.0,
                   float(np.mean(cs)) if cs else 0.0,
                   np.log1p(len(str(lab.loc[r, "post"]).split()))])
        keep = dedupe([v[1] for v in cnt.values() if len(v[0]) >= 2])
        if not keep:
            pool = [s for t in per for s, _ in per[t].get(r, [])]
            keep = sorted(pool, key=token_len)[:1]
        U.append(keep)
    TF = np.asarray(TF, float)

    # ---- factor probabilities (the 8-arm pool) ----------------------------
    FP = None
    try:
        O, rid = [], None
        for a in BAG8:
            pr, ids = [], []
            for f in range(5):
                z = np.load(f"{args.facdir}/{a}_cv_oof_f{f}.npz",
                            allow_pickle=True)
                pr.append(z["probs"].astype(float))
                ids.extend(str(x) for x in z["row_id"])
            pr = np.concatenate(pr)
            if rid is None:
                rid = ids
            elif ids != rid:
                o = {x: i for i, x in enumerate(ids)}
                pr = pr[[o[x] for x in rid]]
            O.append(pr)
        pool = 1.0 / (1.0 + np.exp(-np.mean(logit(np.stack(O)), axis=0)))
        o = {x: i for i, x in enumerate(rid)}
        miss = [r for r in rows if r not in o]
        if miss:
            raise KeyError(f"{len(miss)} rows absent from factor OOF")
        FP = logit(pool[[o[r] for r in rows]])
        print(f"factor pool loaded: {FP.shape}")
    except Exception as e:                       # noqa: BLE001
        print(f"factor features unavailable ({e}) -- +fac3/+all skipped")

    ki = [FACTOR_TAXONOMY.index(k) for k in KEY3] if FP is not None else []
    FEATS = {"stack-r": Z, "+tag": np.column_stack([Z, TF])}
    if FP is not None:
        FEATS["+fac3"] = np.column_stack([Z, FP[:, ki]])
        FEATS["+all"] = np.column_stack([Z, TF, FP])

    print(f"{n} OOF posts / {len(pd.unique(auth))} authors")
    print(f"argmax risk wF1 {wf1(gold, np.array(LEVELS)[P.argmax(1)]):.4f}   "
          f"shipped {wf1(gold, yship):.4f}\n")

    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    frac = 378.0 / (378.0 + n)
    rng = np.random.default_rng(args.seed)
    names = ["argmax"] + list(FEATS)
    res = {k: [] for k in names}
    rkonly = {k: [] for k in names}

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
        gE = [G[i] for i in E]

        def joint(pred):
            return (0.4 * wf1(gold[E], pred) + 0.3 * phrase_f1(
                gE, [[] if p == "indicator" else U[i]
                     for p, i in zip(pred, E)], average="macro").f1,
                wf1(gold[E], pred))

        j, r = joint(np.array(LEVELS)[P[E].argmax(1)])
        res["argmax"].append(j)
        rkonly["argmax"].append(r)
        for k, X in FEATS.items():
            lr = LogisticRegression(max_iter=3000, C=args.C,
                                    multi_class="multinomial")
            lr.fit(X[M], gold[M])
            pr = lr.predict(X[E])
            j, r = joint(pr)
            res[k].append(j)
            rkonly[k].append(r)

    base = np.array(res["argmax"])
    brk = np.array(rkonly["argmax"])
    print("=" * 96)
    print(f"CROSS-MODAL RISK STACK -- {args.splits} author partitions")
    print("=" * 96)
    print(f"{'features':<12}{'joint':>9}{'delta':>10}{'win':>7}{'sd':>9}"
          f"{'one-draw 95%':>24}{'risk wF1':>10}")
    print(f"{'argmax':<12}{base.mean():>9.4f}{'--':>10}{'--':>7}{'--':>9}"
          f"{'--':>24}{brk.mean():>10.4f}")
    for k in FEATS:
        v = np.array(res[k])
        d = v - base
        sd = d.std(ddof=1)
        print(f"{k:<12}{v.mean():>9.4f}{d.mean():>+10.4f}"
              f"{(d > 0).mean():>7.0%}{sd:>9.4f}"
              f"{f'[{d.mean()-1.96*sd:+.4f},{d.mean()+1.96*sd:+.4f}]':>24}"
              f"{np.mean(rkonly[k]):>10.4f}")

    print("\n  'stack-r' is the control the audit already closed -- restacking")
    print("  the risk model's own logits. It MUST be flat. Anything above it")
    print("  is what the independent signals are worth.")
    print("  Bar: win >= 65% AND a mostly-positive one-draw interval.")


if __name__ == "__main__":
    main()
