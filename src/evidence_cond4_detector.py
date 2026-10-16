"""
evidence_cond4_detector.py

Test whether the evidence taggers can flag posts where the predicted risk
label is wrong.

WHY A DETECTOR IS NEEDED. Conditioning the evidence taggers on the wrong risk
level costs a large share of evidence F1 on the posts where the risk model
errs. Hedging those posts (emitting spans for more than one conditioning) only
pays where P(risk wrong) exceeds about 0.55 in the flagged slice, because
adding a span costs roughly 0.33 macro-F1 on a post whose label was already
correct and gains roughly 0.30 on a post whose label was wrong. That
break-even sets the acceptance bar below.

THE INDEPENDENT CHANNEL. The `cond4` dumps run all seven taggers over both
splits with each of the four risk levels FORCED as the conditioning prefix.
That is the EVIDENCE model's opinion about the risk level, not the risk
model's, so it is not a restatement of risk-model confidence. The taggers do
hold such an opinion: forced conditioning changes spans-per-post by more than
a factor of two on identical text.

THE HYPOTHESIS. If a post really is `ideation`, conditioning on `ideation`
should produce the most COHERENT tagger output -- the seven taggers should
agree with each other about which spans matter. Conditioning on the wrong
level should scatter them. Per post, then, the evidence models have a
"preferred" level, and disagreement with the deployed prediction is an error
signal that owes nothing to the risk model's confidence.

FEATURES, all per post, all derived only from the four cond4 dumps:
  * n_spans[c]      mean spans per tagger under conditioning c
  * agree[c]        share of distinct spans proposed by >= 2 of 7 taggers
  * conf[c]         mean candidate confidence
  * the argmax over c of each, and the MARGIN between the preferred level and
    the deployed one -- the actual disagreement measure
plus the risk model's own confidence, as a control: the evidence channel is
only useful if it adds something BEYOND confidence rather than restating it.

THE ACCEPTANCE BAR, fixed in advance:
    precision in the flagged slice must exceed 55%
A detector at 40% does not "nearly work" -- below break-even it loses macro-F1
on every post it flags. Precision is reported at several coverages so the best
slice can be read directly.

A CHEAPER PRIOR QUESTION, reported first because it is nearly free: does the
evidence-preferred level predict the GOLD risk level at all? If the four-way
argmax is at chance, the premise fails and no feature engineering on top of it
can help.

Usage (CPU, a few minutes):
  python src/evidence_cond4_detector.py
  python src/evidence_cond4_detector.py --probs artifacts/champion_val_probs.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, normalize_span  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]


def auc(score, pos):
    pos = np.asarray(pos, bool)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.asarray(score, float))) + 1.0
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * (~pos).sum()))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--dir", default="artifacts/evidence_cond4_{lvl}")
    a_.add_argument("--runs", default=",".join(RC7))
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv",
                    help="the DEPLOYED chain's val predictions. Must be the "
                         "chain that ships -- an OOF artefact from a different "
                         "model has a different error set, so a detector "
                         "measured on it is measuring a different object.")
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=17)
    a_.add_argument("--allow-partial", action="store_true",
                    help="screen on the rows --probs actually covers instead "
                         "of refusing. The uncovered rows are where the friend "
                         "member is absent, so they are not a random sample; "
                         "use for a go/no-go read, never for a reported "
                         "number.")
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold = np.array([normalize_label(v) for v in va["risk_level"]])
    auth = np.asarray([str(x) for x in va["anon_user_id"]])
    runs = [r.strip() for r in args.runs.split(",")]
    n = len(rows)

    if not os.path.exists(args.probs):
        raise SystemExit(
            f"{args.probs} missing. Build it with src/champion_oof_twin.py -- "
            f"the detector must be measured against the DEPLOYED chain's "
            f"predictions, not an OOF artefact of a different model.")
    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pv = pv.set_index("row_id")
    miss = [r for r in rows if r not in pv.index]
    if miss and not args.allow_partial:
        raise SystemExit(
            f"{args.probs} lacks {len(miss)} val rows. These are the rows the "
            f"friend member does not cover (ensemble_risk.py falls back per "
            f"row), so the twin never scored them. Pass --allow-partial to "
            f"screen on the {len(rows) - len(miss)} covered rows instead -- "
            f"acceptable for a go/no-go read, NOT for a reported number, "
            f"because the dropped rows are not a random sample.")
    if miss:
        keep = [i for i, r in enumerate(rows) if r in pv.index]
        print(f"--allow-partial: dropping {len(miss)} rows the twin does not "
              f"cover ({len(keep)} remain). The dropped rows are exactly where "
              f"the friend member is absent, so they are NOT random -- read "
              f"this as a screen only.")
        rows = [rows[i] for i in keep]
        gold = gold[keep]
        auth = auth[keep]
        n = len(rows)
    P = pv.loc[rows, [f"prob_{c}" for c in LEVELS]].to_numpy(float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    pred = np.array([normalize_label(v)
                     for v in pv.loc[rows, "pred_risk_level"]])
    wrong = (pred != gold).astype(int)

    print(f"{n} val posts / {len(pd.unique(auth))} authors")
    print(f"  deployed chain is wrong on {wrong.mean():.1%} "
          f"({wrong.sum()} posts) -- the base rate to beat")
    print(f"  BAR FOR THE HEDGE: 55% precision in the flagged slice")

    # ---- load the four conditioning dumps -------------------------------
    # per (level, post): span count, >=2-tagger agreement share, mean conf
    NS = np.zeros((4, n))
    AG = np.zeros((4, n))
    CF = np.zeros((4, n))
    for li, lvl in enumerate(LEVELS):
        d = args.dir.format(lvl=lvl)
        if not os.path.isdir(d):
            raise SystemExit(
                f"{d} missing -- the cond4 dump writes "
                f"artifacts/evidence_cond4_<level>/. Without all four this "
                f"probe has nothing independent to measure.")
        per_post = {r: [] for r in rows}
        for run in runs:
            p = os.path.join(d, f"{run}_val_candidates.csv")
            if not os.path.exists(p):
                print(f"  WARNING: {p} missing, skipping this tagger for "
                      f"{lvl} -- counts across levels will not be comparable")
                continue
            c = pd.read_csv(p, keep_default_na=False)
            c["row_id"] = c["row_id"].astype(str)
            for r, s, cf in zip(c["row_id"], c["span"], c["conf"]):
                if r in per_post:
                    per_post[r].append((run, normalize_span(str(s)),
                                        float(cf)))
        for i, r in enumerate(rows):
            recs = per_post[r]
            NS[li, i] = len(recs) / max(len(runs), 1)
            CF[li, i] = float(np.mean([x[2] for x in recs])) if recs else 0.0
            by_span = Counter()
            for run, s, _ in recs:
                if s:
                    by_span[s] += 1
            if by_span:
                AG[li, i] = sum(1 for v in by_span.values() if v >= 2) \
                    / len(by_span)
        print(f"  {lvl:<10} {NS[li].mean():.2f} spans/tagger/post   "
              f"agree>=2 {AG[li].mean():.2f}   conf {CF[li].mean():.3f}")

    hdr = "=" * 76
    print("\n" + hdr)
    print("1. DOES THE EVIDENCE-PREFERRED LEVEL PREDICT GOLD RISK AT ALL?")
    print(hdr)
    print("  If this is at chance the premise is dead and no feature")
    print("  engineering on top of it will help.\n")
    gi = np.array([LEVELS.index(g) if g in LEVELS else 0 for g in gold])
    pi = np.array([LEVELS.index(p) if p in LEVELS else 0 for p in pred])
    for nm, M in (("span count", NS), ("agreement", AG), ("confidence", CF)):
        am = M.argmax(0)
        print(f"  argmax {nm:<12} matches GOLD {np.mean(am == gi):>6.1%}   "
              f"matches DEPLOYED PRED {np.mean(am == pi):>6.1%}   "
              f"distribution {dict(Counter(LEVELS[j] for j in am))}")
    print(f"\n  for reference: the deployed chain matches gold "
          f"{np.mean(pi == gi):.1%}, and always-guessing the most common level "
          f"would score {max(Counter(gold).values()) / n:.1%}")

    # ---- 2. the detector ------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    srt = np.sort(P, axis=1)
    conf = srt[:, -1]
    margin = srt[:, -1] - srt[:, -2]
    entropy = -(P * np.log(np.clip(P, 1e-9, None))).sum(1)
    CONF = np.stack([conf, margin, entropy], 1)

    # the disagreement measures: how much better does the evidence channel
    # like some OTHER level than the one actually predicted?
    def gap(M):
        own = M[pi, np.arange(n)]
        best = M.max(0)
        return np.stack([own, best, best - own], 1)

    EV = np.hstack([NS.T, AG.T, CF.T, gap(NS), gap(AG), gap(CF),
                    (NS.argmax(0) != pi).astype(float)[:, None],
                    (AG.argmax(0) != pi).astype(float)[:, None]])

    print("\n" + hdr)
    print("2. CROSS-FITTED DETECTOR -- does the evidence channel beat 55%?")
    print(hdr)
    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    sets = {"confidence only": CONF,
            "cond4 evidence only": EV,
            "cond4 + confidence": np.hstack([EV, CONF])}
    for nm, Xs in sets.items():
        oof = np.full(n, np.nan)
        for _ in range(args.splits):
            o = list(ua)
            rng.shuffle(o)
            ho = set(o[:max(int(len(o) * 0.25), 2)])
            hm = np.array([a in ho for a in auth])
            fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
            if len(fi) < 40 or len(np.unique(wrong[fi])) < 2:
                continue
            sc = StandardScaler().fit(Xs[fi])
            m = LogisticRegression(C=0.5, max_iter=1000).fit(
                sc.transform(Xs[fi]), wrong[fi])
            p = m.predict_proba(sc.transform(Xs[hi]))[:, 1]
            oof[hi] = np.where(np.isnan(oof[hi]), p, (oof[hi] + p) / 2)
        ok = ~np.isnan(oof)
        line = f"  {nm:<22}AUC {auc(oof[ok], wrong[ok]):.3f}  precision@cov:"
        for cov in (0.05, 0.10, 0.20, 0.30):
            k = max(int(ok.sum() * cov), 3)
            idx = np.argsort(-oof[ok])[:k]
            line += f"  {cov:.0%}:{wrong[ok][idx].mean():.0%}"
        print(line)

    print(f"\n  base rate {wrong.mean():.0%}   BAR 55%")
    print("  The row that matters is `cond4 + confidence` against")
    print("  `confidence only`. Equal rows mean the evidence channel is only")
    print("  restating the risk model's confidence and the hedge stays shut.")
    print("  The hedge is viable only if cond4 adds over confidence alone AND")
    print("  the best slice clears the 55% break-even.")


if __name__ == "__main__":
    main()
