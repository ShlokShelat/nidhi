#!/usr/bin/env python
"""PER-POST SPAN CAP on the shipped RC7 decode. CPU, seconds.

THE OBSERVATION. RC7 at k=2 emits **1.65 spans/post** while gold averages
**1.13**. It over-emits by ~46%, and macro phrase-F1 punishes that hard: a post
with one gold span and two predictions, one correct, scores 2*1/(2+1) = 0.67.
Cap it at one and pick right, it scores 1.0.

WHY THE k-SWEEP CANNOT FIX THIS. `k` is an AGREEMENT threshold, not a COUNT.
Raising it to 3 removes spans that fewer than three taggers proposed --
uniformly, across all posts -- and lands at 1.43 spans/post scoring 0.7953,
WORSE than k=2's 0.8045. That is a different operation from "drop the weakest
span on each post". Every evidence sweep in this project has moved k and left
the per-post count unbounded; grep finds no cap anywhere in the evidence path.

WHAT THIS SWEEPS. The shipped decode, unchanged, plus one knob:

    k    vote agreement, as now
    cap  maximum spans emitted per post, ranked by vote count

The vote/suppression/dedupe logic is IMPORTED from evidence_member_loo rather
than reimplemented, and cap=inf must reproduce that script's number exactly --
printed as a gate. A harness that cannot reproduce the baseline cannot be
trusted about a delta from it.

HONEST LIMITS. This is chosen on FIT authors and scored on HELD-OUT authors
over 200 splits; judge by WIN RATE and per-split sd, never the CI of the mean.
Any val win divides by ~2.8 for val->test attenuation before it earns a
submission slot.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, normalize_span, parse_evidence  # noqa: E402
from evidence_member_loo import dedupe_nested, f1_of  # noqa: E402

RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RC7))
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--add-run", default=None,
                    help="name of an extra member to test ADDING to the "
                         "shipped set; needs {name}_val_candidates.csv in "
                         "--dir. Runs a paired author-split test at k=2/cap=3.")
    ap.add_argument("--loo", action="store_true",
                    help="leave-one-out over the members AT the shipped "
                         "0.65/k2/cap3 operating point; the set was chosen "
                         "under the old per-run gates.")
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
    col = "pred_risk_level" if "pred_risk_level" in rk.columns else "risk_level"
    supp = {r: normalize_label(v) == "indicator"
            for r, v in zip(rk["row_id"], rk[col])}

    per = {}
    raw = {}      # unfiltered frames, so the conf gate can be SWEPT
    mc_of = {}    # each run's own training-time evidence_min_conf
    for run in [x.strip() for x in args.runs.split(",") if x.strip()]:
        p = os.path.join(args.dir, f"{run}_val_candidates.csv")
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
        raw[run] = d
        mc_of[run] = mc
        if "conf" in d.columns:
            d = d[d["conf"] >= mc]
        m = {}
        for r, s in zip(d["row_id"], d["span"]):
            m.setdefault(r, set()).add(str(s))
        per[run] = m

    # THE REAL SHIPPED GATES. They live in runs/{run}/run_config.json, NOT in
    # the candidate dumps (which have no min_conf column at all -- that is why
    # this harness defaulted to no gate and why an offset sweep measured
    # nothing). evidence_vote_test.py reads them from exactly here.
    import json as _json
    real_gate = {}
    for run in raw:
        cfg = os.path.join("runs", run, "run_config.json")
        real_gate[run] = (float(_json.load(open(cfg)).get("evidence_min_conf", 0.0))
                          if os.path.exists(cfg) else None)
    _known = {k: v for k, v in real_gate.items() if v is not None}
    print(f"  shipped per-run gates: {_known}")
    if len(_known) != len(raw):
        print(f"  WARNING: {len(raw)-len(_known)} runs have no run_config.json;"
              f" their shipped gate is unknown and treated as 0.0")

    def build_per(gate):
        """gate=None -> each run's OWN shipped gate (the deployed baseline).
        A float -> that uniform gate on every run."""
        if gate is None:
            out = {}
            for run, d in raw.items():
                g = real_gate.get(run) or 0.0
                dd = d[d["conf"] >= g] if "conf" in d.columns else d
                m = {}
                for r, s in zip(dd["row_id"], dd["span"]):
                    m.setdefault(r, set()).add(str(s))
                out[run] = m
            return out
        return _build_per_uniform(gate)

    def _build_per_uniform(gate):
        """Rebuild the per-run candidate sets at an ABSOLUTE confidence gate.

        NOT an offset. The dumps carry no `min_conf` column (it reads nan), so
        this harness has always defaulted mc to 0.0 and applied NO gate --
        while the shipped evidence_vote_test.py reads the real gates from
        runs/*/run_config.json (0.70/0.70/0.60/0.55/0.70/0.00/0.65) and drops
        ~17% of candidates. Both score 0.8045, which says the per-run gate is
        very nearly REDUNDANT with the k>=2 vote: what it removes is what no
        second tagger corroborates anyway.

        `conf` lives in [0.51, 0.998] with median ~0.92, so any gate below 0.51
        is a no-op -- which is why an earlier offset sweep over 0.00-0.20
        returned byte-identical macro at every setting and measured nothing.
        The untested direction is STRICTER, up where the mass actually is."""
        out = {}
        for run, d in raw.items():
            dd = d[d["conf"] >= gate] if "conf" in d.columns else d
            m = {}
            for r, s in zip(dd["row_id"], dd["span"]):
                m.setdefault(r, set()).add(str(s))
            out[run] = m
        return out
    def build_per_quantile(q):
        """Drop the WEAKEST q fraction of EACH RUN's own candidates.

        A uniform absolute gate assumes `conf` is comparable across taggers.
        It is not: s17_rc's conf median is 0.76 against ~0.92 for the rest, so
        the shipped uniform 0.65 removes 23% of s17_rc and ~5% of the others.
        The per-run ABSOLUTE gates tried to respect that but underperformed a
        simpler uniform gate, which ignores per-tagger scale entirely. Neither
        equalises the ADMISSION RATE, which is the quantity that actually
        normalises for calibration differences.

        q=0 keeps everything and must reproduce the ungated pool exactly."""
        out = {}
        for run, d in raw.items():
            if q > 0 and "conf" in d.columns and len(d):
                thr = float(np.quantile(d["conf"].astype(float), q))
                dd = d[d["conf"] >= thr]
            else:
                dd = d
            m = {}
            for r, s in zip(dd["row_id"], dd["span"]):
                m.setdefault(r, set()).add(str(s))
            out[run] = m
        return out

    if len(per) < 3:
        raise SystemExit("need at least 3 runs with val candidate dumps")
    members = list(per)
    print(f"{len(rows)} val posts   {len(per)} taggers")

    def vote(k, cap, rel=None):
        """The shipped decode, plus a per-post cap ranked by vote count.

        `rel` adds a PER-POST RELATIVE agreement floor: keep a span only if its
        agreement is >= rel * the best agreement on its own post. The cap can
        only count; this can tell apart a post whose spans all agree 6/7 (keep
        them all) from one with a 7/7 and two 2/7 stragglers (drop the
        stragglers). Applied before the cap so the two compose.
        """
        out = []
        for r in rows:
            if supp.get(r, False):
                out.append([])
                continue
            cnt = {}
            for run in members:
                for s in per[run].get(r, ()):
                    n = normalize_span(s)
                    if n:
                        e = cnt.setdefault(n, [0, s])
                        e[0] += 1
            kept = [v for v in cnt.values() if v[0] >= k]
            if rel is not None and kept:
                best = max(v[0] for v in kept)
                if best > 0:
                    kept = [v for v in kept if v[0] >= rel * best]
            if cap is not None:
                # rank by agreement so the cap drops the WEAKEST span on this
                # post -- the thing raising k cannot do. Stable within ties, so
                # cap=inf is byte-identical to the shipped ordering.
                kept = sorted(kept, key=lambda v: -v[0])
            spans = dedupe_nested([v[1] for v in kept])
            out.append(spans[:cap] if cap is not None else spans)
        return out

    def macro(pred, idx=None):
        idx = range(len(rows)) if idx is None else idx
        return float(np.mean([f1_of(gold[i], pred[i]) for i in idx]))

    # ---- GATE: cap=None must reproduce evidence_member_loo's number --------
    base = vote(args.baseline_k, None)
    b = macro(base)
    spp = np.mean([len(p) for p in base])
    gpp = np.mean([len(g) for g in gold])
    print(f"\nBASELINE k={args.baseline_k} uncapped: MACRO {b:.4f}   "
          f"{spp:.2f} spans/post   gold {gpp:.2f}")
    print(f"  over-emission {100*(spp/max(gpp,1e-9)-1):+.0f}%")
    print("  (evidence_member_loo reports 0.8045 for RC7 at k=2 -- if this "
          "differs, STOP: the harness is not the shipped decode)")

    # ---- grid ---------------------------------------------------------------
    # cap grid (as before) PLUS the per-post relative-agreement rule, alone and
    # composed with the shipped cap=3. rel is the continuous version of a cap.
    grid = [(k, c, None) for k in (1, 2, 3, 4) for c in (1, 2, 3, 4, None)]
    grid += [(2, None, rl) for rl in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    grid += [(2, 3, rl) for rl in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    F = np.zeros((len(grid), len(rows)))
    for i, (k, c, rl) in enumerate(grid):
        pred = vote(k, c, rl)
        F[i] = [f1_of(gold[j], pred[j]) for j in range(len(rows))]

    print("\n  k  cap  rel    MACRO        d   spans/post")
    for i, (k, c, rl) in enumerate(grid):
        pred = vote(k, c, rl)
        sp = np.mean([len(p) for p in pred])
        tag = "inf" if c is None else str(c)
        rtag = "-" if rl is None else f"{rl:.1f}"
        print(f"  {k}  {tag:>3}  {rtag:>3}   {F[i].mean():.4f}  "
              f"{F[i].mean()-b:+.4f}   {sp:.2f}")

    # ---- CONFIDENCE-GATE OFFSET, never swept before --------------------------
    # Each tagger filters its candidates at its OWN evidence_min_conf, fixed
    # when that tagger was trained in isolation. Nothing ever asked whether the
    # POOL wants a different operating point. This is a decode parameter -- it
    # only re-reads fixed candidate dumps -- so val macro * 0.30 is the
    # composite delta directly, with no attenuation (decode-only changes do
    # not suffer the fitted-parameter transfer gap that trained ones do).
    print("\n" + "=" * 70)
    print("ABSOLUTE CONFIDENCE GATE x (k, cap)   conf lives in [0.51, 0.998]")
    print("  gate <= 0.50 is a no-op; the shipped per-run gates are 0.55-0.70")
    print("  and are nearly redundant with k>=2, so look STRICTER")
    print("=" * 70)
    print("  gate   k  cap    MACRO        d   spans/post   cands kept")
    ref = None
    best_off = None
    # `None` = the SHIPPED per-run gates. That is the only correct reference:
    # a uniform gate is LOOSER than the shipped 0.70s and STRICTER than the
    # shipped 0.00 (s17_rc), so it is a two-directional change and measuring it
    # against an ungated pool scores a different edit from the one that would
    # actually be built.
    for gate in (None, 0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85):
        per = build_per(gate)
        members[:] = list(per)
        nc = sum(len(v) for m in per.values() for v in m.values())
        # k=2 and cap=3 were BOTH chosen under the old per-run gates. Uniform
        # 0.65 shipped and won, so the pool has changed and the joint optimum
        # may have moved with it.
        #
        # k=1 IS THE CELL THE 48-GRID MISSED. The gate and k do the same job by
        # different means: k>=2 demands corroboration, the gate demands
        # certainty. k=1 was hopeless on the UNGATED pool (0.7739) because one
        # tagger would emit anything -- but the gate deletes exactly that
        # population. On a cleaned pool, demanding a second vote may now be
        # discarding correct spans only one tagger found.
        for (k, c) in ((1, 1), (1, 2), (1, 3), (1, None),
                       (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (2, None)):
            pred = vote(k, c)
            mm = macro(pred)
            sp = np.mean([len(p) for p in pred])
            tag = "inf" if c is None else str(c)
            if gate is None and c == 3:
                ref = mm          # THE SHIPPED CONFIGURATION
            mark = ""
            if gate is not None and (best_off is None or mm > best_off[0]):
                best_off = (mm, gate, k, c)
                mark = "  <--"
            gtag = "SHIP" if gate is None else f"{gate:.2f}"
            print(f"  {gtag:>5}  {k}  {tag:>3}   {mm:.4f}  {mm-b:+.4f}"
                  f"   {sp:.2f}   {nc:>9}{mark}")
    mm, gate, k, c = best_off
    print(f"\n  best uniform {mm:.4f} at gate {gate:.2f}, k={k}, cap={c}")
    print(f"  vs the SHIPPED per-run gates ({ref:.4f}): {mm - ref:+.4f} macro"
          f" = {0.30 * (mm - ref):+.5f} composite")
    # The CURRENT CHAMPION is uniform 0.65 / k=2 / cap=3 = 0.8098 val. Any
    # further change must beat THAT, not an old per-run baseline -- the same
    # stale-baseline trap that made a starved pool look like a large,
    # high-win-rate gain purely because the baseline itself was weak.
    print(f"  vs the CURRENT CHAMPION (0.65/k2/cap3 = 0.8098): "
          f"{mm - 0.8098:+.4f} macro = {0.30 * (mm - 0.8098):+.5f} composite")
    if mm - 0.8098 <= 0.0005:
        print("  -> nothing beats the shipped champion by a usable margin.")
    print("  Decode-only change, so val macro * 0.30 IS the composite delta.")
    print("  But a single val point is not enough on its own -- get a WIN")
    print("  RATE before shipping.")

    # ---- PER-RUN QUANTILE GATE, the untested third option -------------------
    # The champion (uniform 0.65 / k2 / cap3) scores 0.8098. Beat THAT.
    print("\n" + "=" * 70)
    print("PER-RUN QUANTILE GATE x cap   (champion: uniform 0.65 = 0.8098)")
    print("=" * 70)
    print("     q   k  cap    MACRO    d-vs-champ   spans/post   cands kept")
    CH = 0.8098
    bestq = None
    for q in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        per = build_per_quantile(q)
        members[:] = list(per)
        nc = sum(len(v) for m in per.values() for v in m.values())
        for (k, c) in ((2, 3), (2, 4)):
            pred = vote(k, c)
            mm = macro(pred)
            sp = np.mean([len(p) for p in pred])
            mark = ""
            if bestq is None or mm > bestq[0]:
                bestq = (mm, q, k, c)
                mark = "  <--"
            print(f"  {q:.2f}   {k}  {c:>3}   {mm:.4f}   {mm-CH:+.4f}"
                  f"      {sp:.2f}   {nc:>9}{mark}")
    mm, q, k, c = bestq
    print(f"\n  best {mm:.4f} at q={q:.2f}, k={k}, cap={c}")
    print(f"  vs the CHAMPION (0.8098): {mm-CH:+.4f} macro"
          f" = {0.30*(mm-CH):+.5f} composite")
    if mm - CH < 0.0010:
        print("  -> under +0.0010 macro. Not worth a slot; the uniform gate")
        print("     already captures whatever the quantile version would.")
    else:
        print("  -> CANDIDATE. Needs a paired win rate before it ships.")

    # ---- LEAVE-ONE-OUT AT THE NEW OPERATING POINT ---------------------------
    # The member set was chosen under the OLD per-run gates, when dropping any
    # one member showed essentially nothing. Uniform 0.65 changed the pool,
    # and it changed s17_rc most of all: its gate went 0.00 -> 0.65, costing 91
    # of 390 spans, and its conf median is 0.76 against ~0.92 for the others.
    # A member is not the same member at a different gate.
    if args.loo:
        print("\n" + "=" * 70)
        print("LEAVE-ONE-OUT at gate 0.65 / k=2 / cap=3 (the shipped champion)")
        print("=" * 70)
        per = build_per(0.65)
        members[:] = list(per)
        base_pred = vote(2, 3)
        fbase = np.array([f1_of(gold[i], base_pred[i])
                          for i in range(len(rows))])
        print(f"  all {len(members)} members: {fbase.mean():.4f}")
        rngL = np.random.default_rng(args.seed)
        uaL = np.array(sorted(set(authors)))
        holds = []
        for _ in range(args.splits):
            h = set(rngL.choice(uaL, size=max(1, len(uaL) // 3), replace=False))
            mk = np.array([a in h for a in authors])
            if mk.sum() >= 5:
                holds.append(mk)
        print(f"  {'drop':<22}{'MACRO':>9}{'d':>10}{'win':>7}{'sd':>9}")
        for drop in list(members):
            keep = [m for m in members if m != drop]
            saved = per.pop(drop)
            members.remove(drop)
            pr = vote(2, 3)
            fd = np.array([f1_of(gold[i], pr[i]) for i in range(len(rows))])
            dl = np.array([fd[mk].mean() - fbase[mk].mean() for mk in holds])
            w = float((dl > 0).mean())
            print(f"  {drop:<22}{fd.mean():>9.4f}{fd.mean()-fbase.mean():>+10.4f}"
                  f"{100*w:>6.0f}%{dl.std():>9.4f}"
                  + ("   <-- clears 90%" if w >= 0.90 else ""))
            per[drop] = saved
            members.append(drop)
        members[:] = list(per)
        print("\n  A drop must clear 90% win AND a positive macro gain to be")
        print("  worth a slot.")

    # ---- OPTIONAL: paired test for ADDING A MEMBER --------------------------
    # A = the shipped member set, B = the same set plus --add-run. Both at
    # k=2/cap=3, both pre-specified, so there is no winner's curse. Member LOO
    # on the shipped 7 shows dropping any one member is essentially costless,
    # so an 8th member has to clear a real bar to be worth the slot.
    if args.add_run:
        # RESTORE FIRST. The gate loop above leaves `per` at its LAST setting
        # (0.95, a deliberately starved pool at 0.55 spans/post). Testing a new
        # member against that inflates the apparent gain purely because
        # anything helps a starved pool -- the baseline itself (0.7376) is far
        # below the true operating point (0.8072). Same class of error as
        # measuring an arm against a legacy model that is below every arm in
        # the pool.
        per = build_per(None)      # the SHIPPED per-run gates, not ungated
        members[:] = list(per)
        p_ = os.path.join(args.dir, f"{args.add_run}_val_candidates.csv")
        if not os.path.exists(p_):
            raise SystemExit(f"--add-run given but {p_} is missing")
        d_ = pd.read_csv(p_, keep_default_na=False)
        d_["row_id"] = d_["row_id"].astype(str)
        m_ = {}
        for r_, s_ in zip(d_["row_id"], d_["span"]):
            m_.setdefault(r_, set()).add(str(s_))
        print("\n" + "=" * 70)
        print(f"PAIRED MEMBER TEST: {len(members)} members vs "
              f"+{args.add_run}   both k=2 cap=3")
        print("=" * 70)
        cov = sum(1 for r_ in rows if r_ in m_)
        print(f"  new member covers {cov} of {len(rows)} val rows, "
              f"{sum(len(v) for v in m_.values())} spans")
        predA = vote(2, 3)
        fA = np.array([f1_of(gold[i], predA[i]) for i in range(len(rows))])
        if ref is not None and abs(fA.mean() - ref) > 0.0005:
            raise SystemExit(
                f"REFUSING: baseline A is {fA.mean():.4f}, but the shipped "
                f"config scored {ref:.4f} in the sweep above. The pool is not "
                f"the shipped one, so any delta against it is meaningless.")
        per[args.add_run] = m_
        members.append(args.add_run)
        predB = vote(2, 3)
        fB = np.array([f1_of(gold[i], predB[i]) for i in range(len(rows))])
        rng3 = np.random.default_rng(args.seed)
        ua3 = np.array(sorted(set(authors)))
        dd = []
        for _ in range(args.splits):
            hold = set(rng3.choice(ua3, size=max(1, len(ua3) // 3),
                                   replace=False))
            mm_ = np.array([a in hold for a in authors])
            if mm_.sum() < 5:
                continue
            dd.append(fB[mm_].mean() - fA[mm_].mean())
        dd = np.array(dd)
        w = float((dd > 0).mean())
        print(f"  full-sample  A {fA.mean():.4f}   B {fB.mean():.4f}   "
              f"d {fB.mean()-fA.mean():+.4f} macro")
        print(f"  held-out     mean d {dd.mean():+.4f}   sd {dd.std():.4f}"
              f"   WIN {100*w:.0f}%   over {len(dd)} splits")
        print(f"  posts changed {int((fA != fB).sum())} of {len(rows)}")
        print(f"  composite estimate {0.30*(fB.mean()-fA.mean()):+.5f}")
        print("\n  BAR >=90% win. " + ("CLEARS -- build the test column."
                                       if w >= 0.90 else
                                       "DOES NOT CLEAR -- do not ship."))
        del per[args.add_run]
        members.remove(args.add_run)

    # ---- PAIRED WIN RATE for the winning gate, both configs PRE-SPECIFIED ----
    # No fitting: A = shipped-equivalent (no uniform gate, cap 3), B = the peak.
    # A sharp argmax on one val sample is not a reliable signal on its own --
    # it can read as a solid gain during offline validation and fail to hold
    # up once deployed. Win rate and per-split sd decide, never the CI of the
    # mean (sd ~ mean / z(win)).
    print("\n" + "=" * 70)
    print(f"PAIRED AUTHOR-SPLIT TEST: gate {gate:.2f} vs no gate, both k=2 cap=3")
    print("=" * 70)
    per = build_per(None)      # A = the SHIPPED per-run gates
    members[:] = list(per)
    predA = vote(2, 3)
    fA = np.array([f1_of(gold[i], predA[i]) for i in range(len(rows))])
    per = build_per(gate)
    members[:] = list(per)
    predB = vote(2, 3)
    fB = np.array([f1_of(gold[i], predB[i]) for i in range(len(rows))])

    rng2 = np.random.default_rng(args.seed)
    ua2 = np.array(sorted(set(authors)))
    deltas = []
    for _ in range(args.splits):
        hold = set(rng2.choice(ua2, size=max(1, len(ua2) // 3), replace=False))
        m = np.array([a in hold for a in authors])
        if m.sum() < 5:
            continue
        deltas.append(fB[m].mean() - fA[m].mean())
    deltas = np.array(deltas)
    win = float((deltas > 0).mean())
    print(f"  full-sample  A {fA.mean():.4f}   B {fB.mean():.4f}   "
          f"d {fB.mean()-fA.mean():+.4f} macro")
    print(f"  held-out     mean d {deltas.mean():+.4f}   sd {deltas.std():.4f}"
          f"   WIN {100*win:.0f}%   over {len(deltas)} splits")
    print(f"  posts changed {int((fA != fB).sum())} of {len(rows)}")
    print(f"  composite estimate {0.30*(fB.mean()-fA.mean()):+.5f}")
    print("\n  BAR: >=90% win. " + ("CLEARS IT -- build and ship."
                                    if win >= 0.90 else
                                    "DOES NOT CLEAR -- do not ship. A sharp "
                                    "val argmax without a win rate is not "
                                    "enough evidence to ship."))

    per = build_per(None)         # restore the SHIPPED per-run gates
    members[:] = list(per)

    # ---- cross-fit by author ------------------------------------------------
    rng = np.random.default_rng(args.seed)
    ua = np.array(sorted(set(authors)))
    bi = grid.index((args.baseline_k, None, None))
    held, chosen = [], {}
    for _ in range(args.splits):
        perm = rng.permutation(len(ua))
        fit_a = set(ua[perm[:len(ua) // 2]])
        fm = np.array([a in fit_a for a in authors])
        if fm.all() or not fm.any():
            continue
        gi = int(np.argmax(F[:, fm].mean(1)))
        held.append(F[gi, ~fm].mean() - F[bi, ~fm].mean())
        chosen[grid[gi]] = chosen.get(grid[gi], 0) + 1
    held = np.array(held)
    win = float((held > 0).mean())
    best = max(chosen, key=chosen.get)

    print("\n" + "=" * 66)
    print("CROSS-FITTED vs the shipped decode (config on FIT, scored HELD-OUT)")
    print(f"  d evidence {held.mean():+.4f}   sd {held.std(ddof=1):.4f}   "
          f"win {100*win:.0f}%   over {len(held)} splits")
    print(f"  d composite {0.3*held.mean():+.4f}   "
          f"(test estimate {0.3*held.mean()/2.8:+.4f} after 2.8x attenuation)")
    print(f"  modal config k={best[0]} cap={best[1] or 'inf'}   "
          f"{chosen[best]}/{len(held)} splits")
    if win >= 0.90 and best[1] is not None:
        print("\n  CLEARS the 90% bar with a finite cap. Build it:")
        print("    evidence_vote_test.py needs the same cap on the TEST side,")
        print("    then patch only the evidence column and hold S2 fixed.")
    elif best[1] is None:
        print("\n  The uncapped config wins -- over-emission is NOT costing")
        print("  what the counts suggest, because the extra spans are often")
        print("  matching a second gold span. Lane closed.")
    else:
        print(f"\n  {100*win:.0f}% does NOT clear 90%. Not shippable.")
    print("=" * 66)


if __name__ == "__main__":
    main()
