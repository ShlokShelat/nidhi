"""
champion_risk_inject.py

Rebuild the deployed risk column from its component artefacts, optionally
inject a correction ahead of the argmax, and report exactly which labels move.

The deployed risk column is a deterministic function of files on disk:

    M      = 0.70 * artifacts/ensembled_risk.csv
           + 0.30 * artifacts/nli_risk_test_probs.csv
    labels = argmax(M)
           -> apply_rules(post, label, [urgency, completed_act])   canonical text
           -> route_attempt   (cot_soft_oof.csv.extra.csv says attempt,
                               and the label is not indicator)

Because `M` is recoverable, a correction can be injected BEFORE the argmax
rather than substituted for the labels afterwards, so the symbolic rules and
the routing step still operate on a coherent probability vector.

THE HARD GATE. With no correction applied, this script must reproduce every
label of the reference submission or it refuses to write anything. A correction
is only meaningful relative to the exact probability vector it is applied to:
an out-of-fold vector comes from fold-trained models and a test vector from
full-data models, and those are different classifiers with different accuracy,
so a correction estimated against one is not valid against the other. The gate
turns that assumption into an assertion.

SCOPE OF THE GATE. It proves the BASE is the deployed vector. It does not prove
that an injected CORRECTION was estimated on that same vector. A correction fitted
on out-of-fold predictions stays unvalidated until an out-of-fold vector is
rebuilt from these same members.

Usage:
  # gate only -- prove the rebuild is exact, change nothing
  python src/champion_risk_inject.py --verify

  # inject the severity nudge (needs the test-side probe dump)
  python src/champion_risk_inject.py --severity artifacts/risk_probe_test.csv \\
      --w 0.5 --out submissions/NIDHI_probenudge.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rules_to_test import apply_rules  # noqa: E402
from data_audit import load as load_canonical  # noqa: E402
from scoring import normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
ENSCOLS = ["p_Indicator", "p_Ideation", "p_Behavior", "p_Attempt"]
NLICOLS = [f"prob_{c}" for c in LEVELS]


def norm(X):
    return X / np.clip(X.sum(1, keepdims=True), 1e-9, None)


def build(args, rows):
    """The blended probability vector M, in reference-submission row order."""
    ens = pd.read_csv(args.ensembled).set_index("row_id")
    nli = pd.read_csv(args.nli).set_index("row_id")
    E = norm(ens.loc[rows, ENSCOLS].to_numpy(float))
    N = norm(nli.loc[rows, NLICOLS].to_numpy(float))
    return (1 - args.nli_w) * E + args.nli_w * N


def finalize(M, rows, posts, route, rules):
    lab = [LEVELS[i] for i in M.argmax(1)]
    lab = [normalize_label(apply_rules(p, l, rules)[0])
           for p, l in zip(posts, lab)]
    # route_attempt never touches indicator: evidence is suppressed on
    # indicator posts, so promoting one would leave empty evidence against
    # non-empty gold and score 0 on that post's evidence.
    return [("attempt" if (route.get(r) == "attempt" and l != "indicator")
             else l) for r, l in zip(rows, lab)]


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--champion", default="submissions/NIDHI_factfit.csv")
    a_.add_argument("--ensembled", default="artifacts/ensembled_risk.csv")
    a_.add_argument("--nli", default="artifacts/nli_risk_test_probs.csv")
    a_.add_argument("--route", default="artifacts/cot_soft_oof.csv.extra.csv")
    a_.add_argument("--test", default="data/test.csv")
    a_.add_argument("--nli-w", type=float, default=0.30)
    a_.add_argument("--rules", default="urgency,completed_act")
    a_.add_argument("--alt-ensembled", default=None,
                    help="rebuild M from a DIFFERENT encoder ensemble (e.g. an "
                         "ensemble with one member left out), after the gate "
                         "has proved the original reproduces the reference "
                         "column. Use src/risk_member_loo.py to quantify a "
                         "member's contribution before changing membership.")
    a_.add_argument("--out-risk", default=None,
                    help="write the rebuilt risk labels alone (row_id, "
                         "pred_risk_level) as an artefact, even when posts "
                         "cross the indicator boundary -- this is the input to "
                         "a rebuilt evidence vote")
    a_.add_argument("--alt-nli-w", type=float, default=None,
                    help="alternative NLI blend weight for the alternative "
                         "build. Vary this or --alt-ensembled, not both: with "
                         "two changes at once the resulting label diff is not "
                         "attributable to either.")
    a_.add_argument("--alt-rules", default=None,
                    help="rule set to apply INSTEAD of --rules, as the "
                         "correction. The gate still runs on --rules first, so "
                         "the base is proved to reproduce the reference column "
                         "before the alternative is built. Use `urgency` to "
                         "drop completed_act.")
    a_.add_argument("--severity", default=None,
                    help="csv with row_id + conf_indicator, from the "
                         "risk-conditioned tagger run over the TEST posts")
    a_.add_argument("--w", type=float, default=0.0, help="nudge weight")
    a_.add_argument("--out", default=None)
    a_.add_argument("--max-changes", type=int, default=40)
    a_.add_argument("--out-probs", default=None,
                    help="write the BLENDED per-class probabilities "
                         "(0.70*ensembled + nli_w*nli, after normalisation, "
                         "before argmax). The evidence suppression rule needs "
                         "p_indicator and no other file carries it post-blend.")
    a_.add_argument("--verify", action="store_true")
    a_.add_argument("--allow-missing-route", action="store_true",
                    help="proceed when the route file does not cover every "
                         "test row. route_attempt is then skipped on those "
                         "rows, so the risk column is a degraded version of "
                         "the validated one. For a deliberate degraded "
                         "diagnostic run only -- never for a submission.")
    args = a_.parse_args()

    ch = pd.read_csv(args.champion, keep_default_na=False)
    ch["row_id"] = ch["row_id"].astype(str)
    rc = "risk_level" if "risk_level" in ch.columns else "pred_risk_level"
    rows = ch["row_id"].tolist()
    tgt = [normalize_label(v) for v in ch[rc]]

    te = load_canonical(args.test)
    te["row_id"] = te["row_id"].astype(str)
    tmap = dict(zip(te["row_id"], te["post"].astype(str)))
    posts = [tmap[r] for r in rows]

    lm = pd.read_csv(args.route, keep_default_na=False)
    lm["row_id"] = lm["row_id"].astype(str)
    rcol = "cot" if "cot" in lm.columns else "pred_risk_level"
    route = {r: normalize_label(v) for r, v in zip(lm["row_id"], lm[rcol])}

    # COVERAGE GUARD. `finalize` does route.get(r), so a row absent from the
    # route file silently receives NO routing and the chain proceeds without a
    # word. `artifacts/cot_soft_oof.csv.extra.csv` is a precomputed Qwen-72B
    # dump keyed to one specific set of test row_ids; run against a different
    # test set, every lookup misses and route_attempt becomes a no-op that
    # produces no output of any kind. A missing artefact must be an error, not
    # a silently weaker model.
    #
    # Regenerate it with the script that already does this -- it accepts an
    # arbitrary CSV and writes <out>.extra.csv in exactly this format:
    #   python src/risk_cot_probe.py --model <qwen-72b-path> --soft \
    #       --extra-csv <fresh test.csv> --extra-risk-from <risk column> \
    #       --out artifacts/cot_soft_oof.csv
    _miss = [r for r in rows if r not in route]
    if _miss:
        msg = (f"{args.route} covers {len(rows)-len(_miss)} of {len(rows)} "
               f"rows; {len(_miss)} would silently skip route_attempt, the "
               f"only promotion into the rare `attempt` class. First few: "
               f"{_miss[:5]}. Regenerate with src/risk_cot_probe.py "
               f"--extra-csv (see the note at this line), or pass "
               f"--allow-missing-route to proceed WITHOUT routing and accept "
               f"the loss knowingly.")
        if not args.allow_missing_route:
            raise SystemExit("REFUSING: " + msg)
        print("WARNING: " + msg)

    rules = [x.strip() for x in args.rules.split(",")]
    M = build(args, rows)
    base = finalize(M, rows, posts, route, rules)

    # ---------------- THE GATE ----------------
    agree = sum(1 for a, b in zip(base, tgt) if a == b)
    print(f"rebuild vs champion: {agree}/{len(rows)}")
    if agree != len(rows):
        bad = [(rows[i], base[i], tgt[i])
               for i in range(len(rows)) if base[i] != tgt[i]]
        for r, a, b in bad[:15]:
            print(f"   {r}  rebuilt={a}  champion={b}")
        raise SystemExit(
            "REFUSING: the rebuild is not the deployed probability vector. "
            "Injecting into a base that is not the deployed model applies the "
            "adjustment to logits it was never calibrated against. "
            "Fix the chain before changing anything.")
    print("  GATE PASSED -- this is the champion's own probability vector.\n")
    # Export the blended probabilities. Needed because the evidence
    # suppression rule can be driven by a continuous p_indicator instead of
    # the risk ARGMAX, and no file on disk carries the post-blend probability
    # -- `ensembled_risk*.csv` is pre-NLI and the submission has labels only.
    # Gold evidence is empty exactly on indicator posts, so p_indicator is the
    # quantity the suppression decision is really about.
    if args.out_probs:
        os.makedirs(os.path.dirname(args.out_probs) or ".", exist_ok=True)
        pd.DataFrame({"row_id": rows,
                      **{f"prob_{c}": M[:, i] for i, c in enumerate(LEVELS)},
                      "pred_risk_level": [x.capitalize() for x in base]}
                     ).to_csv(args.out_probs, index=False)
        print(f"wrote {args.out_probs}  (blended 0.70*ens + "
              f"{args.nli_w}*nli, the champion's own vector)")

    if args.verify:
        print("(--verify: nothing written)")
        return

    if args.alt_ensembled is not None:
        import copy
        alt = copy.copy(args)
        alt.ensembled = args.alt_ensembled
        if args.alt_nli_w is not None:
            alt.nli_w = args.alt_nli_w
        M2 = build(alt, rows)
        new = finalize(M2, rows, posts, route, rules)
        ch_i = [i for i in range(len(rows)) if new[i] != base[i]]
        print(f"{args.ensembled} -> {args.alt_ensembled}"
              + (f"  (nli_w {args.nli_w} -> {alt.nli_w})"
                 if args.alt_nli_w is not None else ""))
        print(f"  {len(ch_i)} of {len(rows)} risk labels change\n")
        from collections import Counter
        print(f"  before: {dict(Counter(base))}")
        print(f"  after : {dict(Counter(new))}")
        # THE COUPLING THAT MATTERS. Evidence is generated by a tagger
        # CONDITIONED on the predicted risk, and suppression fires exactly on
        # `indicator`. A post crossing that boundary must gain or lose its
        # whole evidence cell, so its S1 evidence term moves too and the
        # change is no longer attributable to risk alone.
        ib = [i for i in ch_i if base[i] == "indicator"]
        ia = [i for i in ch_i if new[i] == "indicator"]
        print(f"\n  crossing the indicator boundary: "
              f"{len(ib)} leave indicator, {len(ia)} become indicator")
        if ib or ia:
            print("  -> EVIDENCE MUST BE REGENERATED for those posts; a "
                  "risk-only patch would leave them inconsistent:")
            for i in (ib + ia)[:12]:
                print(f"     {rows[i]}  {base[i]} -> {new[i]}")
        else:
            print("  -> none. The evidence column stays consistent and this "
                  "is a risk-only change.")
        print(f"\n  {'row_id':<10}{'champion':<12}{'new':<12}{'LLM says':<12}")
        for i in ch_i[:25]:
            print(f"  {rows[i]:<10}{base[i]:<12}{new[i]:<12}"
                  f"{route.get(rows[i], '-'):<12}")
        if len(ch_i) > 25:
            print(f"  ... and {len(ch_i)-25} more")
        if args.out_risk:
            # risk labels ONLY, as an artefact -- the input to a rebuilt
            # evidence vote. Written even when posts cross the indicator
            # boundary, because regenerating evidence from this file is
            # precisely how that inconsistency gets fixed.
            os.makedirs(os.path.dirname(args.out_risk) or ".", exist_ok=True)
            pd.DataFrame({"row_id": rows,
                          "pred_risk_level": [x.capitalize() for x in new]}
                         ).to_csv(args.out_risk, index=False)
            print(f"\nwrote {args.out_risk}  ({len(ch_i)} labels differ from "
                  f"the champion)")
            print("  next: rebuild evidence with --risk-from this file, then")
            print("  patch BOTH columns together -- a risk-only patch would")
            print("  leave the crossing posts inconsistent.")
        if args.out and ch_i and not (ib or ia):
            spell = {}
            for v in ch[rc]:
                spell.setdefault(normalize_label(v), str(v))
            out = ch.copy()
            out[rc] = [spell.get(x, x.capitalize()) for x in new]
            for c in ("evidence", "factors"):
                if c in ch.columns and not all(
                        str(a) == str(b) for a, b in zip(ch[c], out[c])):
                    raise SystemExit(f"REFUSING: {c} changed")
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            out.to_csv(args.out, index=False)
            print(f"\nwrote {args.out}   {len(ch_i)} risk labels changed")
        elif args.out and (ib or ia):
            raise SystemExit(
                "REFUSING to write: posts cross the indicator boundary, so "
                "evidence must be rebuilt with the new risk column before "
                "this is a valid submission.")
        elif not args.out:
            print("\n(diagnose only; pass --out to write)")
        return

    if args.alt_rules is not None:
        alt = [x.strip() for x in args.alt_rules.split(",") if x.strip()]
        new = finalize(M, rows, posts, route, alt)
        ch_i = [i for i in range(len(rows)) if new[i] != base[i]]
        print(f"rules {rules} -> {alt}: {len(ch_i)} of {len(rows)} labels "
              f"change\n")
        # WHY EACH POST MOVES. `completed_act` promotes behavior->attempt by
        # regex and `route_attempt` promotes to attempt by LLM reasoning, so
        # they overlap. A post the regex promotes that the LLM does NOT is a
        # regex-only attempt -- the redundancy claim stands or falls on these.
        print(f"  {'row_id':<10}{'champion':<12}{'without':<12}"
              f"{'LLM says':<12}{'p(attempt)':>11}")
        for i in ch_i:
            print(f"  {rows[i]:<10}{base[i]:<12}{new[i]:<12}"
                  f"{route.get(rows[i], '-'):<12}{M[i, 3]:>11.3f}")
        n_ind = sum(1 for i in ch_i
                    if "indicator" in (base[i], new[i]))
        if n_ind:
            print(f"\n  WARNING: {n_ind} cross the indicator boundary -- S2 "
                  f"moves and this is not a risk-only change.")
        if args.out and ch_i:
            spell = {}
            for v in ch[rc]:
                spell.setdefault(normalize_label(v), str(v))
            out = ch.copy()
            out[rc] = [spell.get(x, x.capitalize()) for x in new]
            for c in ("evidence", "factors"):
                if c in ch.columns and not all(
                        str(a) == str(b) for a, b in zip(ch[c], out[c])):
                    raise SystemExit(f"REFUSING: {c} changed")
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            out.to_csv(args.out, index=False)
            print(f"\nwrote {args.out}   {len(ch_i)} risk labels changed")
            print("  evidence and factors BYTE-IDENTICAL, so S2 must return "
                  "the champion's value exactly")
        elif not args.out:
            print("\n(diagnose only; pass --out to write)")
        return

    if not args.severity or args.w == 0.0:
        raise SystemExit("nothing to inject: pass --severity and --w")

    sv = pd.read_csv(args.severity)
    sv["row_id"] = sv["row_id"].astype(str)
    col = next((c for c in ("conf_indicator", "sev", "severity")
                if c in sv.columns), None)
    if col is None:
        raise SystemExit(f"{args.severity} needs a conf_indicator column")
    miss = [r for r in rows if r not in set(sv["row_id"])]
    if miss:
        raise SystemExit(f"severity file lacks {len(miss)} test rows "
                         f"(e.g. {miss[:3]}); a partial nudge ships an "
                         f"unmeasured mixture")
    smap = dict(zip(sv["row_id"], sv[col].astype(float)))
    s = np.array([smap[r] for r in rows])

    # The nudge: tilt toward higher severity in proportion to how much
    # evidence the tagger finds when told the post is only an indicator.
    # Applied in LOG space to M, before the argmax, so the rules and the route
    # still see a coherent probability vector.
    Mn = np.log(np.clip(M, 1e-9, None)) + args.w * np.outer(s, np.arange(4)) / 3.0
    new = finalize(np.exp(Mn), rows, posts, route, rules)

    ch_i = [i for i in range(len(rows)) if new[i] != base[i]]
    print(f"nudge w={args.w}: {len(ch_i)} of {len(rows)} risk labels change\n")
    print(f"  {'row_id':<10}{'from':<12}{'to':<12}{'severity':>9}")
    for i in ch_i:
        print(f"  {rows[i]:<10}{base[i]:<12}{new[i]:<12}{s[i]:>9.3f}")
    from collections import Counter
    print(f"\n  before: {dict(Counter(base))}")
    print(f"  after : {dict(Counter(new))}")

    if len(ch_i) > args.max_changes:
        raise SystemExit(f"REFUSING: {len(ch_i)} changes exceeds "
                         f"--max-changes {args.max_changes}")
    n_ind = sum(1 for i in ch_i if base[i] == "indicator"
                or new[i] == "indicator")
    if n_ind:
        print(f"\n  WARNING: {n_ind} changes cross the indicator boundary. "
              f"Evidence is suppressed exactly on indicator posts, so S2 will "
              f"MOVE and this is no longer a risk-only change.")

    if not args.out:
        print("\n(diagnose only; pass --out to write)")
        return

    spell = {}
    for v in ch[rc]:
        spell.setdefault(normalize_label(v), str(v))
    out = ch.copy()
    out[rc] = [spell.get(x, x.capitalize()) for x in new]
    for c in ("evidence", "factors"):
        if c in ch.columns and not all(str(a) == str(b)
                                       for a, b in zip(ch[c], out[c])):
            raise SystemExit(f"REFUSING: {c} changed")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}   {len(ch_i)} risk labels changed")
    print("  evidence and factors BYTE-IDENTICAL")
    print("  re-run find_champion_probs.py against this file: it should now "
          "read exactly 378 minus the changes listed above, and nothing else.")


if __name__ == "__main__":
    main()
