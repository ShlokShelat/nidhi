"""
patch_submission_risk.py

Builds a new submission that differs from a base submission in the risk_level
column ONLY, carrying evidence and factors across untouched.

Why this exists rather than re-running predict.py: re-running the full
pipeline to change a handful of risk labels also re-derives evidence and
factors. Any nondeterminism, model reload, or changed default in those
stages co-varies with the change under test, so the resulting score
difference is no longer attributable to the risk column alone. Copying the
evidence and factor columns of an already-scored submission byte-for-byte
makes the single-variable comparison exact and verifiable, and needs no
GPU.

The script refuses to write if anything other than risk_level would change,
and prints every changed row so the diff can be eyeballed before upload.

Usage:
  python src/patch_submission_risk.py \
      --base submissions/archive/NIDHI_rules_urgency_completedact.csv \
      --risk artifacts/risk_blend_methodfix.csv \
      --out submissions/NIDHI_urgency_methodfix.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="submission to patch: the exact scored CSV that is "
                         "the comparison baseline. Its two untouched columns "
                         "are copied through byte-for-byte, so this must be "
                         "the file whose score the patched one is compared "
                         "against, not a same-named rebuild of it.")
    ap.add_argument("--risk", default=None,
                    help="CSV with row_id,pred_risk_level (apply_rules_to_test.py output)")
    ap.add_argument("--evidence", default=None,
                    help="CSV with row_id,evidence to swap in INSTEAD of risk "
                         "(evidence_vote_ensemble.py output). Exactly one of "
                         "--risk / --evidence may be given: two components "
                         "changed at once produce one score difference for "
                         "two causes, which is not attributable to either.")
    ap.add_argument("--factors", default=None,
                    help="CSV with row_id,factors to swap in. Mutually "
                         "exclusive with --risk / --evidence.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-changes", type=int, default=25,
                    help="refuse to write if more rows change than this; a "
                         "single-variable rule change should touch very few")
    args = ap.parse_args()

    n_given = sum(bool(x) for x in (args.risk, args.evidence, args.factors))
    if n_given != 1:
        raise SystemExit("give exactly one of --risk / --evidence / --factors: "
                         "a single submission must change exactly one component")

    base = pd.read_csv(args.base, keep_default_na=False)
    for c in ("row_id", "risk_level", "evidence", "factors"):
        if c not in base.columns:
            raise SystemExit(f"base submission missing column {c!r}")

    # ---- evidence-only patch -------------------------------------------
    if args.evidence:
        ev = pd.read_csv(args.evidence, keep_default_na=False)
        if "evidence" not in ev.columns:
            raise SystemExit("evidence CSV needs an 'evidence' column")
        if set(base.row_id) != set(ev.row_id):
            raise SystemExit("row_id sets differ between submission and evidence CSV")
        new_ev = ev.set_index("row_id")["evidence"]
        out = base.copy()
        out["evidence"] = [new_ev[r] for r in out["row_id"]]
        for c in ("risk_level", "factors"):
            if not out[c].equals(base[c]):
                raise SystemExit(f"internal error: {c} changed; refusing to write")
        changed = int((out["evidence"] != base["evidence"]).sum())
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out.to_csv(args.out, index=False)
        n_empty_before = int((base["evidence"].astype(str).str.strip() == "").sum())
        n_empty_after = int((out["evidence"].astype(str).str.strip() == "").sum())
        print(f"base : {args.base}\nev   : {args.evidence}\nout  : {args.out}")
        print(f"\nevidence changed on {changed} of {len(out)} rows; "
              f"risk_level and factors carried over unchanged.")
        print(f"posts with no evidence: {n_empty_before} -> {n_empty_after}")
        return

    # ---- factors-only patch --------------------------------------------
    if args.factors:
        fa = pd.read_csv(args.factors, keep_default_na=False)
        if "factors" not in fa.columns:
            raise SystemExit("factors CSV needs a 'factors' column")
        if set(base.row_id) != set(fa.row_id):
            raise SystemExit("row_id sets differ between submission and factors CSV")
        new_fa = fa.set_index("row_id")["factors"]
        out = base.copy()
        out["factors"] = [new_fa[r] for r in out["row_id"]]
        for c in ("risk_level", "evidence"):
            if not out[c].equals(base[c]):
                raise SystemExit(f"internal error: {c} changed; refusing to write")
        changed = int((out["factors"] != base["factors"]).sum())
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"base : {args.base}\nfac  : {args.factors}\nout  : {args.out}")
        print(f"\nfactors changed on {changed} of {len(out)} rows; "
              f"risk_level and evidence carried over unchanged.")
        return

    # ---- risk-only patch -----------------------------------------------
    risk = pd.read_csv(args.risk)
    if "pred_risk_level" not in risk.columns:
        raise SystemExit("risk CSV needs a pred_risk_level column")
    if set(base.row_id) != set(risk.row_id):
        raise SystemExit("row_id sets differ between base submission and risk CSV")

    # Preserve the base file's label casing convention exactly.
    sample = str(base["risk_level"].iloc[0])
    to_case = (str.lower if sample.islower()
               else str.title if sample[0].isupper() else str.lower)

    new_risk = risk.set_index("row_id")["pred_risk_level"]
    out = base.copy()
    out["risk_level"] = [to_case(normalize_label(new_risk[r])) for r in out["row_id"]]

    changed = out[out["risk_level"].str.lower()
                  != base["risk_level"].str.lower()]
    if len(changed) > args.max_changes:
        raise SystemExit(f"{len(changed)} rows would change, above --max-changes "
                         f"{args.max_changes}. Refusing to write -- confirm this is "
                         f"really a single-variable change.")

    # evidence/factors must be carried over byte-identically
    for c in ("evidence", "factors"):
        if not out[c].fillna("__na__").equals(base[c].fillna("__na__")):
            raise SystemExit(f"internal error: {c} column changed; refusing to write")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"base : {args.base}")
    print(f"risk : {args.risk}")
    print(f"out  : {args.out}")
    print(f"\nrisk_level changed on {len(changed)} of {len(out)} rows; "
          f"evidence and factors carried over unchanged.")
    for rid in changed["row_id"]:
        b = base.loc[base.row_id == rid, "risk_level"].iloc[0]
        n = out.loc[out.row_id == rid, "risk_level"].iloc[0]
        print(f"    {rid}  {b} -> {n}")

    from collections import Counter
    print(f"\nbase risk distribution: {dict(Counter(base.risk_level.str.lower()))}")
    print(f"new  risk distribution: {dict(Counter(out.risk_level.str.lower()))}")


if __name__ == "__main__":
    main()
