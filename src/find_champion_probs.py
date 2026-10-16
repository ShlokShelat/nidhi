"""
find_champion_probs.py

WHICH artifacts/ensembled_risk_*.csv PRODUCED THE CHAMPION'S RISK COLUMN?

A prior correction was fitted on `champion_oof.csv` and applied to the
champion under the assumption that they were the same model. They were not:
0.786 vs 0.826. The resulting rule is "prove same-model before stacking
anything", and this script is that proof, run BEFORE anything is built.

Nine test-time probability dumps exist. Exactly one (possibly after the NLI
blend and the symbolic rules) reproduces `NIDHI_factfit.csv`'s risk column.
A file that reproduces 378/378 IS the champion's probability artifact; a file
at 370/378 is a near relative and is NOT usable -- the 8 disagreements are
precisely the confusable posts any correction would touch.

The champion build, from scripts/risk_fulldata_submission.sh, is:
    ensemble_risk.py -> 0.30 x NLI seeds -> apply_rules_to_test.py
    (--rules urgency,completed_act) -> patch_submission_risk.py
so a raw argmax may fall a few posts short of the shipped column even for the
RIGHT file. Each candidate is therefore scored three ways: raw argmax, argmax
after the NLI blend, and after the blend plus the rules. Reproduction at ANY
stage identifies the file; the stage identifies where to inject.

Usage (CPU, seconds):
  python src/find_champion_probs.py
  python src/find_champion_probs.py --sub submissions/NIDHI_factfit.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def load_probs(path):
    """Return (row_ids, P[n,4], own_labels|None), or None if not a risk dump.

    Column matching is CASE-INSENSITIVE: these dumps ship `p_Indicator`, and a
    case-sensitive matcher silently skipped all twelve of them.
    """
    try:
        d = pd.read_csv(path, keep_default_na=False)
    except Exception:
        return None
    low = {c.lower(): c for c in d.columns}
    rid_col = next((low[c] for c in ("row_id", "id", "post_id") if c in low),
                   None)
    if rid_col is None:
        return None
    cols = None
    for pat in ("prob_{}", "p_{}", "{}", "prob{}"):
        c = [low.get(pat.format(x)) for x in LEVELS]
        if all(c):
            cols = c
            break
    P = None
    if cols is not None:
        P = d[cols].astype(float).to_numpy()
        P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    lc = low.get("pred_risk_level") or low.get("risk_level")
    own = ([normalize_label(v) for v in d[lc]] if lc else None)
    # LABEL-ONLY files count. risk_blend_equal.csv and risk_blend_with_rules
    # .csv carry no probabilities -- apply_rules_to_test.py reads only row_id
    # and pred_risk_level from them -- and the first version of this loader
    # discarded exactly the two files the build script names as the champion's
    # own intermediates.
    if P is None and own is None:
        return None
    return d[rid_col].astype(str).to_numpy(), P, own


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--sub", default="submissions/NIDHI_factfit.csv")
    # apply_rules_to_test.py's docstring names the champion's actual input:
    # "the champion's input is artifacts/risk_blend_equal.csv ... Verified:
    #  with risk_blend_equal.csv this script reproduces
    #  artifacts/risk_blend_with_rules.csv with zero differing rows."
    # The first version of this glob only looked at ensembled_risk*, which is
    # why nothing matched.
    a_.add_argument("--glob", default="artifacts/ensembled_risk*.csv,"
                                      "artifacts/risk_*.csv")
    a_.add_argument("--extra", default="artifacts/nli_risk_test_probs.csv,"
                                       "artifacts/qwen_ord_test_probs.csv")
    a_.add_argument("--nli", default="artifacts/nli_risk_test_probs.csv")
    a_.add_argument("--nli-w", type=float, default=0.30)
    a_.add_argument("--route", default="artifacts/cot_soft_oof.csv.extra.csv",
                    help="CoT-72B TEST calls for route_attempt")
    args = a_.parse_args()

    sub = pd.read_csv(args.sub, keep_default_na=False)
    rc = "risk_level" if "risk_level" in sub.columns else "pred_risk_level"
    sub["row_id"] = sub["row_id"].astype(str)
    tgt = {r: normalize_label(v) for r, v in zip(sub["row_id"], sub[rc])}
    print(f"target: {args.sub}   {len(sub)} rows")
    print("  distribution: " + "  ".join(
        f"{c} {sum(1 for v in tgt.values() if v == c)}" for c in LEVELS))

    nli = load_probs(args.nli) if os.path.exists(args.nli) else None
    if nli:
        nli = dict(zip(nli[0], nli[1]))
        print(f"  NLI blend source: {args.nli} ({len(nli)} rows), "
              f"weight {args.nli_w}")

    # THE DEPLOYED RULE LAYER, via its single source of truth. The earlier
    # attempt called risk_symbolic_fences.apply_rule, which is the ABLATION
    # harness (it returns a tuple and takes a class->column dict), not what the
    # champion ran. apply_rules_to_test.apply_rules is what the build called.
    # Text must be CANONICAL: that module warns 113 of 378 test posts differ
    # between raw and cleaned text, and several patterns contain a literal
    # apostrophe that cannot match a curly one.
    ptxt, apply_rules = {}, None
    try:
        from apply_rules_to_test import apply_rules as _ar  # noqa: E402
        from data_audit import load as _load  # noqa: E402
        td = _load("data/test.csv")
        td["row_id"] = td["row_id"].astype(str)
        ptxt = dict(zip(td["row_id"], td["post"].astype(str)))
        apply_rules = _ar
    except Exception as e:                                  # noqa: BLE001
        print(f"  (rules unavailable: {e}) -- stages 0-2 only")

    cands = sorted({g for pat in args.glob.split(",")
                    for g in glob.glob(pat.strip())}
                   | {p.strip() for p in args.extra.split(",") if p.strip()})
    print(f"\n{len(cands)} candidate probability dumps")
    print("  stage 0 = the file's OWN pred_risk_level column\n")
    print(f"  {'file':<40}{'rows':>6}{'own':>7}{'argmax':>8}{'+NLI':>7}"
          f"{'+rules':>8}")
    best = []
    for p in cands:
        if not os.path.exists(p):
            continue
        got = load_probs(p)
        if got is None:
            print(f"  {os.path.basename(p):<40}{'-':>6}   (not a risk dump)")
            continue
        rids, P, own = got
        keep = [i for i, r in enumerate(rids) if r in tgt]
        if len(keep) < 0.9 * len(tgt):
            print(f"  {os.path.basename(p):<40}{len(rids):>6}"
                  f"   (only {len(keep)} rows overlap)")
            continue
        rids = rids[keep]
        P = P[keep] if P is not None else None
        own = [own[i] for i in keep] if own else None

        def rate(lab):
            return sum(1 for r, l in zip(rids, lab) if tgt[r] == l)

        n = len(rids)
        s0 = rate(own) if own else -1
        s1 = rate(np.array(LEVELS)[P.argmax(1)]) if P is not None else -1
        B = None
        if P is not None:
            B = (np.stack([(1 - args.nli_w) * P[i]
                           + args.nli_w * nli.get(r, P[i])
                           for i, r in enumerate(rids)]) if nli else P)
        s2 = rate(np.array(LEVELS)[B.argmax(1)]) if B is not None else -1
        s3 = -1
        if apply_rules is not None:
            txt = [ptxt.get(r, "") for r in rids]
            srcs = [own] if own else []
            if P is not None:
                srcs.append(list(np.array(LEVELS)[P.argmax(1)]))
            if B is not None:
                srcs.append(list(np.array(LEVELS)[B.argmax(1)]))
            for lab0 in srcs:
                try:
                    lab = [apply_rules(t, l)[0] for t, l in zip(txt, lab0)]
                except Exception as e:                      # noqa: BLE001
                    print(f"    (rule error: {e})")
                    break
                s3 = max(s3, rate([normalize_label(x) for x in lab]))
        top = max(s0, s1, s2, s3)
        mark = ("  <== EXACT" if top == n else
                "  (probs)" if P is not None else "  (labels only)")
        f_ = lambda v: (v if v >= 0 else "-")              # noqa: E731
        print(f"  {os.path.basename(p):<40}{n:>6}{f_(s0):>7}{f_(s1):>8}"
              f"{f_(s2):>7}{f_(s3):>8}{mark}")
        best.append((top, n, p, s0, s1, s2, s3))

    # ------------------------------------------------------------------
    # RECONSTRUCT THE CHAIN. assemble_champion_qwen.py documents it exactly:
    #     blended  = 0.70 x ensembled_risk + 0.30 x nli_risk_test_probs
    #     withqwen = (1-w) x blended       + w   x qwen_ord_test_probs
    #     final    = apply_rules(withqwen, [urgency, completed_act])
    # Finding a file that happens to match is luck; rebuilding the chain means
    # the probability vector is owned outright and a correction can be
    # injected into it. The
    # sweep over w exists because the deployed w is a judgement call (the
    # script's own docstring picks 0.30 over the 0.35-0.40 sweep peak), so the
    # shipped value is worth confirming rather than assuming.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RECONSTRUCTING THE DOCUMENTED CHAIN")
    print("=" * 78)
    try:
        ens = load_probs("artifacts/ensembled_risk.csv")
        nl = load_probs("artifacts/nli_risk_test_probs.csv")
        qw = load_probs("artifacts/qwen_ord_test_probs.csv")
        rows = [r for r in ens[0] if r in tgt]
        gi = {r: i for i, r in enumerate(ens[0])}
        ni = {r: i for i, r in enumerate(nl[0])}
        qi = {r: i for i, r in enumerate(qw[0])}
        E = np.stack([ens[1][gi[r]] for r in rows])
        N = np.stack([nl[1][ni[r]] for r in rows])
        Q = np.stack([qw[1][qi[r]] for r in rows])
        txt = [ptxt.get(r, "") for r in rows]
        # route_attempt, the LAST step. build_route_attempt.py: "take the
        # CoT-72B's ATTEMPT calls, keep the champion's risk label everywhere
        # else", and risk_conservative_override.py:20 states outright that
        # "the deployed champion's risk IS route_attempt on top of the chain".
        # The .extra.csv is the TEST split despite the `oof` in the filename.
        # INDICATOR POSTS ARE NEVER ROUTED -- promoting one would leave it with
        # suppressed (empty) evidence against non-empty gold, scoring 0 on that
        # post's evidence. That safeguard is part of the rule, not an add-on.
        route = {}
        rp = args.route
        if os.path.exists(rp):
            lm = pd.read_csv(rp, keep_default_na=False)
            lm["row_id"] = lm["row_id"].astype(str)
            rc_ = "cot" if "cot" in lm.columns else "pred_risk_level"
            route = {r: normalize_label(v)
                     for r, v in zip(lm["row_id"], lm[rc_])}
            print(f"  route source: {rp} ({len(route)} rows, col '{rc_}')")
        else:
            print(f"  route source {rp} NOT FOUND -- +route column will "
                  f"equal +rules")

        def do_route(lab):
            return [("attempt" if (route.get(r) == "attempt"
                                   and l != "indicator") else l)
                    for r, l in zip(rows, lab)]

        print(f"  {'w_nli':>7}{'w_qwen':>8}{'pre-rules':>11}{'+rules':>9}"
              f"{'+route':>9}{'of':>5}")
        hit = None
        for wn in (0.30, 0.25, 0.35, 0.20, 0.40):
            bl = (1 - wn) * E + wn * N
            for wq in (0.0, 0.30, 0.20, 0.25, 0.35, 0.40, 0.50):
                M = (1 - wq) * bl + wq * Q
                lab0 = list(np.array(LEVELS)[M.argmax(1)])
                a = sum(1 for r, l in zip(rows, lab0) if tgt[r] == l)
                lb = ([normalize_label(apply_rules(t, l)[0])
                       for t, l in zip(txt, lab0)] if apply_rules else lab0)
                b = sum(1 for r, l in zip(rows, lb) if tgt[r] == l)
                lr = do_route(lb)
                c_ = sum(1 for r, l in zip(rows, lr) if tgt[r] == l)
                mk = "  <== EXACT" if max(a, b, c_) == len(rows) else ""
                if max(a, b, c_) == len(rows) and hit is None:
                    hit = (wn, wq, "pre-rules" if a == len(rows) else
                           "+rules" if b == len(rows) else "+rules+route")
                if max(a, b, c_) >= len(rows) - 6:
                    print(f"  {wn:>7.2f}{wq:>8.2f}{a:>11}{b:>9}{c_:>9}"
                          f"{len(rows):>5}{mk}")
        if hit:
            print(f"\n  EXACT at w_nli={hit[0]}, w_qwen={hit[1]} ({hit[2]}).")
            print("  The champion's probability vector is reproducible from")
            print("  ensembled_risk.csv + nli_risk_test_probs.csv +")
            print("  qwen_ord_test_probs.csv. Corrections can be injected into")
            print("  M BEFORE the argmax, and this script re-run to prove that")
            print("  only the intended posts moved.")
        else:
            print("\n  no (w_nli, w_qwen) reproduces all 378 -- the deployed")
            print("  column had a further step. The residual is printed below:")
            # THE RESIDUAL. A global blend that fits 373/378 and cannot be
            # pushed further by ANY weight is not a mis-tuned blend -- it is
            # the right blend plus a PER-POST step: per-class routing to a
            # reasoning LLM re-decides one predicted class and therefore
            # moves exactly a handful of posts. If all the residuals share a
            # from-class, that is it.
            bl = 0.70 * E + 0.30 * N
            lab0 = list(np.array(LEVELS)[bl.argmax(1)])
            lb = ([normalize_label(apply_rules(t, l)[0])
                   for t, l in zip(txt, lab0)] if apply_rules else lab0)
            lb = do_route(lb)
            bad = [i for i, r in enumerate(rows) if tgt[r] != lb[i]]
            print(f"\n  {len(bad)} posts where rebuilt != champion\n")
            print(f"  {'row_id':<10}{'rebuilt':<12}{'CHAMPION':<12}"
                  f"{'qwen says':<12}{'p(champ)':>9}{'p(rebuilt)':>11}")
            for i in bad:
                q = LEVELS[int(Q[i].argmax())]
                jc, jr = LEVELS.index(tgt[rows[i]]), LEVELS.index(lb[i])
                print(f"  {rows[i]:<10}{lb[i]:<12}{tgt[rows[i]]:<12}{q:<12}"
                      f"{bl[i, jc]:>9.3f}{bl[i, jr]:>11.3f}")
            fr = sorted({lb[i] for i in bad})
            to = sorted({tgt[rows[i]] for i in bad})
            print(f"\n  from-classes {fr}   to-classes {to}")
            nq = sum(1 for i in bad if LEVELS[int(Q[i].argmax())]
                     == tgt[rows[i]])
            print(f"  Qwen's own argmax equals the CHAMPION on {nq}/{len(bad)}")
            print("  A single from-class, and Qwen agreeing with the champion,")
            print("  together identify per-class LLM routing as the last step.")
    except Exception as e:                                  # noqa: BLE001
        print(f"  reconstruction failed: {e}")

    if not best:
        print("\nno usable candidates")
        return
    best.sort(reverse=True)
    top, n, p, s0, s1, s2, s3 = best[0]
    print("\n" + "=" * 78)
    if top == n:
        stage = ("its own pred_risk_level column" if s0 == n else
                 "raw argmax" if s1 == n else
                 "after NLI blend" if s2 == n else "after NLI blend + rules")
        print(f"EXACT MATCH: {p}")
        print(f"  reproduces all {n} champion risk labels at: {stage}")
        print("  -> this IS the champion's probability artifact. The risk")
        print("     recalibration and probe-feature lanes are unblocked, and")
        print("     any change can be injected at that stage and verified by")
        print("     re-running this script against the new submission.")
    else:
        print(f"NO EXACT MATCH. Best is {os.path.basename(p)} at {top}/{n} "
              f"({top/n:.1%}).")
        print(f"  The {n-top} disagreements are exactly the confusable posts a")
        print("  correction would move, so this file is NOT a safe base --")
        print("  substituting it for the true artifact is a documented failure")
        print("  mode. Either the champion's dump was overwritten, or the build")
        print("  used a blend weight or rule set this script does not know.")
        print("  Next: rebuild the risk column from scratch and CONFIRM it")
        print("  reproduces the champion before changing anything.")


if __name__ == "__main__":
    main()
