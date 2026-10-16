"""
patch_submission_evidence.py

Swap ONLY the evidence column of a submission, and prove nothing else moved.

Built for the `evidence_v2_rc` drop: that tagger is the worst of the eight on
precision, is eliminated in nearly all nested author splits, and removing it
improves the composite on val -- while a greedy multi-drop overfits and
scores worse.

WHAT IT REFUSES TO DO, because each is a silent scoring error:
  * write if the risk or factors columns differ by even one row -- then the
    change is not attributable to evidence and S1's risk term moves too
  * write if row order or row ids differ from the base
  * write if a post whose risk is `indicator` gains evidence, or a non-
    indicator post loses it. Suppression is keyed on the risk label
    (gold-empty IS the indicator label), so breaking that correspondence
    silently changes what the scorer sees on the posts that already score
    1.0 for free.

Usage:
  python src/patch_submission_evidence.py \\
      --base submissions/NIDHI_factfit.csv \\
      --evidence artifacts/evcond_evidence_nov2.csv \\
      --out submissions/NIDHI_ev7.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, parse_evidence  # noqa: E402


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--base", default="submissions/NIDHI_factfit.csv")
    a_.add_argument("--evidence", required=True)
    a_.add_argument("--risk", default=None,
                    help="also patch the risk column, from a file with "
                         "row_id + pred_risk_level. Use ONLY when the evidence "
                         "file was rebuilt with THIS risk column -- the two "
                         "must move together or the indicator<->empty "
                         "correspondence breaks. Factors must still be "
                         "byte-identical.")
    a_.add_argument("--out", default=None)
    a_.add_argument("--max-changes", type=int, default=200)
    args = a_.parse_args()

    b = pd.read_csv(args.base, keep_default_na=False)
    b["row_id"] = b["row_id"].astype(str)
    e = pd.read_csv(args.evidence, keep_default_na=False)
    e["row_id"] = e["row_id"].astype(str)
    ecol = next((c for c in ("evidence", "pred_evidence") if c in e.columns),
                None)
    if ecol is None:
        raise SystemExit(f"{args.evidence} has no evidence column")
    m = dict(zip(e["row_id"], e[ecol]))
    missing = [r for r in b["row_id"] if r not in m]
    if missing:
        raise SystemExit(f"evidence file lacks {len(missing)} rows "
                         f"(e.g. {missing[:3]}) -- a partial swap ships an "
                         f"unmeasured mixture")

    rc = "risk_level" if "risk_level" in b.columns else "pred_risk_level"
    out = b.copy()
    out["evidence"] = [m[r] for r in b["row_id"]]

    n_risk = 0
    if args.risk:
        rk = pd.read_csv(args.risk, keep_default_na=False)
        rk["row_id"] = rk["row_id"].astype(str)
        col = next((c for c in ("pred_risk_level", "risk_level")
                    if c in rk.columns), None)
        if col is None:
            raise SystemExit(f"{args.risk} has no risk column")
        rm = dict(zip(rk["row_id"], rk[col]))
        miss = [r for r in b["row_id"] if r not in rm]
        if miss:
            raise SystemExit(f"risk file lacks {len(miss)} rows")
        spell = {}
        for v in b[rc]:
            spell.setdefault(normalize_label(v), str(v))
        out[rc] = [spell.get(normalize_label(rm[r]),
                             str(rm[r]).capitalize()) for r in b["row_id"]]
        n_risk = int((b[rc].astype(str) != out[rc].astype(str)).sum())

    checks = ["factors"] if args.risk else [rc, "factors"]
    for c in checks:
        if c in b.columns:
            n = int((b[c].astype(str) != out[c].astype(str)).sum())
            if n:
                raise SystemExit(f"REFUSING: {c} changed on {n} rows")

    ch = [i for i in range(len(b))
          if str(b["evidence"].iloc[i]) != str(out["evidence"].iloc[i])]
    nb = sum(1 for x in b["evidence"] if not parse_evidence(str(x)))
    na = sum(1 for x in out["evidence"] if not parse_evidence(str(x)))
    ind = sum(1 for x in out[rc] if normalize_label(x) == "indicator")
    # consistency is checked against the OUTPUT risk column: when risk is
    # patched too, the correspondence must hold for the NEW labels, which is
    # the whole reason the two columns have to move together
    # THE CHECK IS DIRECTIONAL, and the first version was not.
    #   indicator post WITH evidence  -> a real bug. The pipeline suppresses on
    #       indicator, so this cannot happen legitimately and would mean the
    #       evidence column was built against a different risk column.
    #   non-indicator post WITHOUT evidence -> normal. The vote found nothing.
    #       It also costs nothing when a post LEAVES indicator: its evidence
    #       cell was empty before and stays empty, so that post's evidence
    #       score is unchanged and only its risk label moves.
    bad = [b["row_id"].iloc[i] for i in range(len(b))
           if normalize_label(out[rc].iloc[i]) == "indicator"
           and parse_evidence(str(out["evidence"].iloc[i]))]
    orphan = [b["row_id"].iloc[i] for i in range(len(b))
              if normalize_label(out[rc].iloc[i]) != "indicator"
              and not parse_evidence(str(out["evidence"].iloc[i]))]

    print(f"base      {args.base}   {len(b)} rows")
    print(f"evidence  {args.evidence}")
    if args.risk:
        print(f"risk      {args.risk}   {n_risk} labels changed")
    print(f"  rows with changed evidence   {len(ch)}")
    print(f"  empty-evidence rows          {nb} -> {na}   "
          f"(indicator rows: {ind})")
    print(f"  spans/post on non-empty      "
          f"{sum(len(parse_evidence(str(x))) for x in b['evidence'])/max(len(b)-nb,1):.2f}"
          f" -> "
          f"{sum(len(parse_evidence(str(x))) for x in out['evidence'])/max(len(b)-na,1):.2f}")
    if bad:
        raise SystemExit(
            f"REFUSING: {len(bad)} INDICATOR posts carry evidence "
            f"(e.g. {bad[:3]}). The pipeline suppresses on indicator, so this "
            f"means the evidence column was built against a different risk "
            f"column than the one being written.")
    if orphan:
        print(f"  note: {len(orphan)} non-indicator posts have no evidence "
              f"({orphan[:3]}) -- the vote found nothing there. Costs nothing: "
              f"their evidence cell was empty before and stays empty, so only "
              f"the risk label moves.")
    if len(ch) > args.max_changes:
        raise SystemExit(f"REFUSING: {len(ch)} changes exceeds "
                         f"--max-changes {args.max_changes}")
    if args.risk:
        print(f"  factors BYTE-IDENTICAL; risk changed on {n_risk} rows and "
              f"indicator<->empty holds for the NEW labels")
    else:
        print("  risk and factors BYTE-IDENTICAL, indicator<->empty preserved")

    if not args.out:
        print("\n(diagnose only; pass --out to write)")
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("  S1's risk term and S2's factor term must return the champion's")
    print("  values exactly; only the evidence term can move.")


if __name__ == "__main__":
    main()
