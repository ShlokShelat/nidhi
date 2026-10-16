"""
scope_transfer_study.py

DOES SCOPING A SYMBOLIC RULE TO THE EVIDENCE CLAUSE GENERALISE ACROSS RULES?

A symbolic rule normally reads the whole post. The evidence tagger exists to
locate the clause that carries the risk signal, and that clause is roughly an
order of magnitude shorter than the post. This measures every rule on the shelf
under three scopes and reports which ones benefit.

    POST    the whole post -- the conventional scope
    GOLD    the annotator's evidence span -- the ceiling
    PRED    the tagger's predicted span -- the operational scope

WHY ALL THREE. GOLD is a ceiling, not a result, and it is partly circular: the
annotator chose that clause BECAUSE it justifies the label, so a pattern
matching it is correlated with the label by construction. The number that
carries a claim is PRED, which uses only what the system can actually produce.
The GOLD column's job is to bound what scoping could achieve if extraction were
perfect, so the gap between PRED and GOLD says whether the extractor is the
limiting factor.

EACH RULE IS EVALUATED AS A DETECTOR of its own target label. No model is
involved: `fires` counts posts whose text matches the rule's trigger, and
precision is the share of those whose GOLD label is the target. This isolates
the rule from the risk model entirely, so a change between scopes is a property
of the scope and nothing else.

THE HYPOTHESIS BEING TESTED, stated so it can fail: positive criteria should
transfer to the clause and negative criteria should not. Evidence that an event
occurred is local -- it lives inside the clause the annotator picked. Evidence
that something is ABSENT ("names no method") is a property of the whole post,
and a rule asserting it cannot be evaluated on three words. If both kinds
improve equally, the mechanism is not what we claim.

SECURITY: reads local files only, prints AGGREGATE COUNTS ONLY. No post text
and no span text is emitted, so the output is safe to paste into a report.

Usage:
  python src/scope_transfer_study.py
  python src/scope_transfer_study.py --pred artifacts/<predicted_spans>.csv
  python src/scope_transfer_study.py --csv artifacts/scope_transfer.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rules_to_test import ALL_DEFS, trigger_fires  # noqa: E402
from scoring import normalize_label, parse_evidence  # noqa: E402

SCOPES = ["POST", "GOLD", "PRED"]


def _f1(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def evaluate(rule_name, rd, rows, scope_key):
    """Return detector stats for one rule under one scope.

    `rows` is a list of (text_for_this_scope, gold_label). A row whose text is
    empty under this scope cannot fire, which is itself informative: a rule
    that can only fire on posts WITH evidence is structurally unable to reach
    the indicator class.
    """
    target = normalize_label(rd.get("to_label") or "")
    fires = right = 0
    n_target = 0
    for text, gold in rows:
        if gold == target:
            n_target += 1
        if not text:
            continue
        try:
            if trigger_fires(text, rd):
                fires += 1
                if gold == target:
                    right += 1
        except ValueError:
            return None
    prec = right / fires if fires else 0.0
    rec = right / n_target if n_target else 0.0
    return {"rule": rule_name, "scope": scope_key, "target": target,
            "fires": fires, "right": right, "n_target": n_target,
            "precision": prec, "recall": rec, "f1": _f1(prec, rec)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default="artifacts/full_train.csv")
    ap.add_argument("--pred", nargs="+", default=None,
                    help="one or more CSVs (or a directory of them) holding "
                         "predicted spans, either wide (row_id,evidence) or "
                         "long (row_id,span,conf). Several files are POOLED, "
                         "which is what the deployed decode does. Absent, the "
                         "PRED scope is skipped.")
    ap.add_argument("--csv", default=None, help="write the full table here")
    ap.add_argument("--cap", type=int, default=3,
                    help="max predicted spans per post, matching the shipped "
                         "evidence decode")
    ap.add_argument("--gate", type=float, default=None,
                    help="confidence gate for candidate spans. Default: use "
                         "each run's own min_conf column if present.")
    args = ap.parse_args()

    df = pd.read_csv(args.labelled, keep_default_na=False)
    df["row_id"] = df["row_id"].astype(str)
    gold_lab = [normalize_label(x) for x in df["risk_level"]]
    posts = [str(x) for x in df["post"]]

    # The gold evidence cell is a list-like string of verbatim spans. Joining
    # them reproduces the text a rule would see if it were handed exactly the
    # annotator's justification and nothing else.
    gold_span = [" ... ".join(parse_evidence(str(c))) for c in df["evidence"]]

    pred_span = None
    if args.pred:
        # COVERAGE IS THE CONFOUND HERE, so the PRED scope must be built from
        # the same pool of taggers the deployed decode votes over. A single
        # tagger leaves many posts with no extracted clause at all, and a rule
        # cannot fire on a clause that does not exist -- which reads as the
        # scoping mechanism failing when it is really the extractor missing.
        # Accept several files (or a directory) and pool them.
        paths = []
        for p in args.pred:
            if os.path.isdir(p):
                paths += sorted(os.path.join(p, f) for f in os.listdir(p)
                                if f.endswith(".csv"))
            elif os.path.exists(p):
                paths.append(p)
            else:
                raise SystemExit(f"--pred {p} not found")
        if not paths:
            raise SystemExit("--pred matched no CSV files")
        frames = []
        for p in paths:
            f = pd.read_csv(p, keep_default_na=False)
            f["row_id"] = f["row_id"].astype(str)
            frames.append(f)
        pv = pd.concat(frames, ignore_index=True)
        print(f"PRED pooled from {len(paths)} tagger file(s): "
              + ", ".join(os.path.basename(p) for p in paths[:6])
              + (" ..." if len(paths) > 6 else ""))
        pv["row_id"] = pv["row_id"].astype(str)

        if "span" in pv.columns and "evidence" not in pv.columns:
            # LONG format: one row per candidate span, with a confidence and
            # optionally the gate that run was decoded at. Reproduce a decode
            # rather than dumping the raw pool -- an ungated candidate pool is
            # not what any deployed system emits, and scoping a rule to it
            # would measure the wrong object.
            if "conf" in pv.columns:
                gate = args.gate
                if gate is None and "min_conf" in pv.columns:
                    pv = pv[pv["conf"].astype(float)
                            >= pv["min_conf"].astype(float)]
                elif gate is not None:
                    pv = pv[pv["conf"].astype(float) >= float(gate)]
                pv = pv.sort_values("conf", ascending=False)
            m = {}
            for rid, grp in pv.groupby("row_id", sort=False):
                m[rid] = " ... ".join(str(s) for s in
                                      grp["span"].head(args.cap))
            print(f"PRED built from candidate spans: {len(m)} posts, "
                  f"cap {args.cap} per post"
                  + (f", gate {args.gate}" if args.gate else
                     ", each run's own gate"))
        else:
            col = next((c for c in ("evidence", "pred_evidence", "spans")
                        if c in pv.columns), pv.columns[-1])
            m = dict(zip(pv["row_id"], pv[col]))
            m = {k: " ... ".join(parse_evidence(str(v))) for k, v in m.items()}

        covered = sum(1 for r in df["row_id"] if m.get(r))
        missing = len(df) - covered
        print(f"PRED COVERAGE: {covered} of {len(df)} labelled posts "
              f"({100*covered/len(df):.0f}%) have an extracted clause.\n")
        if missing:
            print(f"  The other {missing} contribute an EMPTY pred scope. "
                  f"That is the correct treatment -- a rule cannot fire on a "
                  f"clause the system never produced -- but it means low\n"
                  f"  coverage depresses the PRED column for reasons that "
                  f"have nothing to do with scoping. Read the coverage figure "
                  f"before reading any PRED result.\n")
        pred_span = [str(m.get(r, "")) for r in df["row_id"]]

    n = len(df)
    have_gold = sum(1 for s in gold_span if s)
    print(f"{n} labelled posts   {have_gold} with a gold evidence span "
          f"({100*have_gold/n:.0f}%)")
    wp = sum(len(p.split()) for p in posts) / max(n, 1)
    wg = (sum(len(s.split()) for s in gold_span if s) / max(have_gold, 1))
    print(f"mean length: post {wp:.1f} words   gold span {wg:.1f} words   "
          f"ratio {wp/max(wg,1e-9):.1f}x\n")

    scope_rows = {"POST": list(zip(posts, gold_lab)),
                  "GOLD": list(zip(gold_span, gold_lab))}
    if pred_span is not None:
        scope_rows["PRED"] = list(zip(pred_span, gold_lab))

    out = []
    for name, rd in ALL_DEFS.items():
        if not rd.get("to_label"):
            continue
        for sc in SCOPES:
            if sc not in scope_rows:
                continue
            r = evaluate(name, rd, scope_rows[sc], sc)
            if r:
                out.append(r)
    res = pd.DataFrame(out)
    if res.empty:
        raise SystemExit("no rules evaluated")

    active = [s for s in SCOPES if s in scope_rows]
    print("=" * 78)
    print("PER-RULE DETECTOR PERFORMANCE BY SCOPE")
    print("=" * 78)
    hdr = f"{'rule':<26}{'target':<10}"
    for s in active:
        hdr += f"{s+' fires':>11}{s+' P':>8}"
    print(hdr)
    for name in res["rule"].unique():
        sub = res[res.rule == name].set_index("scope")
        line = f"{name[:25]:<26}{sub.iloc[0]['target']:<10}"
        for s in active:
            if s in sub.index:
                line += f"{int(sub.loc[s,'fires']):>11}{sub.loc[s,'precision']:>8.3f}"
            else:
                line += f"{'-':>11}{'-':>8}"
        print(line)

    print("\n" + "=" * 78)
    print("DOES SCOPING HELP? PRECISION CHANGE VS THE WHOLE POST")
    print("=" * 78)
    # A RULE THAT NEVER FIRES HAS UNDEFINED PRECISION, NOT ZERO PRECISION.
    # Scoring a silent rule as 0.000 and averaging it in makes the aggregate a
    # statement about coverage rather than about precision, and drags the mean
    # below zero even when every rule that actually fires improved. The two
    # populations are reported separately.
    for tgt in ("GOLD", "PRED"):
        if tgt not in active:
            continue
        w = l = 0
        deltas = []
        silent = []
        print(f"\n  {tgt} vs POST -- rules that FIRE under both scopes")
        print(f"    {'rule':<26}{'P post':>9}{'P '+tgt:>9}{'delta':>9}"
              f"{'fires post':>12}{'fires '+tgt:>12}")
        for name in res["rule"].unique():
            sub = res[res.rule == name].set_index("scope")
            if "POST" not in sub.index or tgt not in sub.index:
                continue
            fp, ft = sub.loc["POST", "fires"], sub.loc[tgt, "fires"]
            if fp == 0 and ft == 0:
                continue
            if ft == 0:
                silent.append((name, int(fp), sub.loc[tgt, "target"]))
                continue
            pp, pt = sub.loc["POST", "precision"], sub.loc[tgt, "precision"]
            d = pt - pp
            deltas.append(d)
            w += d > 0
            l += d < 0
            print(f"    {name[:25]:<26}{pp:>9.3f}{pt:>9.3f}{d:>+9.3f}"
                  f"{int(fp):>12}{int(ft):>12}")
        if deltas:
            mean_d = sum(deltas) / len(deltas)
            print(f"\n    improved {w}, worsened {l}, "
                  f"mean precision change {mean_d:+.3f} over {len(deltas)} "
                  f"rules that fire")
        if silent:
            print(f"\n  {tgt} -- rules that CANNOT FIRE under this scope "
                  f"({len(silent)}). Precision is undefined, not zero, so "
                  f"these are excluded from the mean above.")
            for name, fp, target in silent:
                why = ("target class has empty evidence by definition"
                       if target == "indicator" else
                       "trigger is not present in the extracted clause")
                print(f"    {name[:25]:<26} fired {fp:>3} on the post, 0 here"
                      f"   -- {why}")

    if "PRED" in active:
        print("\n" + "=" * 78)
        print("IS THE EXTRACTOR GOOD ENOUGH TO SERVE AS SCOPE?")
        print("=" * 78)
        print("  PRED close to GOLD means the tagger recovers the useful")
        print("  scope. PRED far below GOLD means extraction is the limit.")
        for name in res["rule"].unique():
            sub = res[res.rule == name].set_index("scope")
            if not {"GOLD", "PRED"} <= set(sub.index):
                continue
            g, p = sub.loc["GOLD", "precision"], sub.loc["PRED", "precision"]
            print(f"    {name[:25]:<26} gold {g:.3f}  pred {p:.3f}  "
                  f"attained {100*p/g if g else 0:.0f}%")

    print("\n" + "=" * 78)
    print("READ THIS BEFORE QUOTING ANY NUMBER ABOVE")
    print("=" * 78)
    print("  GOLD is a CEILING and is partly circular: the annotator selected")
    print("  that clause because it justifies the label, so a pattern matching")
    print("  it correlates with the label by construction. Only the PRED")
    print("  column supports a claim about a deployable system.")
    print("  A rule whose trigger is a NEGATIVE criterion (an absent-list)")
    print("  should NOT improve under scoping -- absence is a property of the")
    print("  whole post. If those improve too, the mechanism is misdiagnosed.")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        res.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}  {res.shape}")


if __name__ == "__main__":
    main()
