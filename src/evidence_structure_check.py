"""
evidence_structure_check.py

WHAT IS DIFFERENT ABOUT THE TEST SPANS, WITHOUT NEEDING TEST GOLD.

Val overstates every evidence defect 3-5x (span over-emission +27% vs +5%,
optimal k=4 vs k=2, fragmentation 5-per-gold vs 3.8%), because the `_rc` taggers
are risk-conditioned and test conditions them on the champion's better risk. So
val error analysis cannot be trusted -- but the STRUCTURE of what is emitted on
test can be compared against the structure of GOLD, measured on train.

Any systematic mismatch is a defect visible ON TEST. That is the same logic that
found the factor prevalence bug (rare labels fired 2.8x too often), and it is
the only kind of evidence measurement not poisoned by the conditioning gap.

FOUR STRUCTURAL PROPERTIES, none of which needs a test label:

  1. LOCATABILITY. The scorer matches on normalised substring containment. A
     predicted span that does not appear verbatim in its post can NEVER match
     anything. The evidence trainer already logs "18 gold spans unlocatable
     after full recovery chain", so the pipeline has a known locatability
     hazard -- nobody has checked it on the OUTPUT side.

  2. SPAN LENGTH. Containment is all-or-nothing and the cap is
     len(pred) <= 3 x len(gold). If the spans run systematically LONGER than
     gold, the cap is breached on short gold spans and they are lost
     outright. If systematically SHORTER, gold contains them and they are
     safe. The asymmetry means the length distribution matters, not just the
     mean.

  3. POSITION IN POST. If the spans cluster near the opening while gold is
     spread through the post, the taggers have a lead bias -- plausible given
     512-token truncation and the way BIO taggers over-attend early tokens.

  4. SPANS PER POST. Gold runs 1.76 per non-empty post. A different shape (many
     posts with 1, a few with 6) scores differently under macro-per-post even at
     the same total.

READ IT AS: any row where TEST departs from GOLD by more than ~10% is a defect
present on the submitted artefact, unlike everything val reported.

Usage:
  python src/evidence_structure_check.py
  python src/evidence_structure_check.py --sub artifacts/evvote_test_k3.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, token_len  # noqa: E402


def norm(s):
    return " ".join(str(s).lower().split())


def locate(span, post):
    """Where the span sits in the post, on normalised text. -1 if absent."""
    return norm(post).find(norm(span))


def describe(name, spans_by_post, posts, out):
    lens, rel, per_post, missing, tot = [], [], [], 0, 0
    for r, sp in spans_by_post.items():
        post = posts.get(r, "")
        np_ = norm(post)
        per_post.append(len(sp))
        for s in sp:
            tot += 1
            lens.append(token_len(s))
            i = locate(s, post)
            if i < 0:
                missing += 1
            elif len(np_) > 0:
                rel.append(i / max(len(np_) - 1, 1))
    lens = np.array(lens) if lens else np.array([0])
    rel = np.array(rel) if rel else np.array([0.0])
    pp = np.array([p for p in per_post if p > 0]) if per_post else np.array([0])
    out[name] = {
        "spans": tot,
        "posts_nonempty": int((np.array(per_post) > 0).sum()),
        "unlocatable": missing,
        "len_mean": lens.mean(), "len_med": float(np.median(lens)),
        "len_p90": float(np.percentile(lens, 90)),
        "len_le3": float((lens <= 3).mean()),
        "rel_mean": rel.mean(), "rel_med": float(np.median(rel)),
        "rel_first_third": float((rel < 1 / 3).mean()),
        "pp_mean": pp.mean(), "pp_med": float(np.median(pp)),
        "pp_ge3": float((pp >= 3).mean()),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--sub", default="artifacts/factors_tfit_test.csv")
    args = ap.parse_args()

    tr = pd.read_csv(args.labeled, keep_default_na=False)
    tr["row_id"] = tr["row_id"].astype(str)
    gold = {r: parse_evidence(e) for r, e in zip(tr["row_id"], tr["evidence"])}
    gposts = {r: str(p) for r, p in zip(tr["row_id"], tr["post"])}

    te = pd.read_csv(args.test, keep_default_na=False)
    te["row_id"] = te["row_id"].astype(str)
    tposts = {r: str(p) for r, p in zip(te["row_id"], te["post"])}
    sub = pd.read_csv(args.sub, keep_default_na=False)
    sub["row_id"] = sub["row_id"].astype(str)
    ours = {r: parse_evidence(e)
            for r, e in zip(sub["row_id"], sub["evidence"])}

    out = {}
    describe("GOLD (train)", gold, gposts, out)
    describe("OURS (test)", ours, tposts, out)

    g, o = out["GOLD (train)"], out["OURS (test)"]
    print(f"gold: {g['spans']} spans over {g['posts_nonempty']} non-empty posts")
    print(f"ours: {o['spans']} spans over {o['posts_nonempty']} non-empty posts")

    def row(label, key, fmt="{:.2f}", pct=False):
        gv, ov = g[key], o[key]
        if pct:
            gs, os_ = f"{100*gv:.1f}%", f"{100*ov:.1f}%"
            rel = f"{100*(ov-gv):+.1f}pp"
        else:
            gs, os_ = fmt.format(gv), fmt.format(ov)
            rel = (f"{100*(ov/gv-1):+.0f}%" if gv else "--")
        flag = ""
        if gv and abs(ov / gv - 1) > 0.10 and not pct:
            flag = "  <-- OFF"
        if pct and abs(ov - gv) > 0.10:
            flag = "  <-- OFF"
        print(f"  {label:<34}{gs:>10}{os_:>10}{rel:>10}{flag}")

    print("\n" + "=" * 78)
    print("1. LOCATABILITY -- a span not verbatim in its post can never match")
    print("=" * 78)
    print(f"  gold unlocatable  {g['unlocatable']} of {g['spans']} "
          f"({100*g['unlocatable']/max(g['spans'],1):.1f}%)")
    print(f"  OURS unlocatable  {o['unlocatable']} of {o['spans']} "
          f"({100*o['unlocatable']/max(o['spans'],1):.1f}%)")
    if o["unlocatable"] > 0.01 * o["spans"]:
        print("  *** ANY non-zero number here is dead loss: those spans score")
        print("      zero by construction and cost precision on every post they")
        print("      appear in. Fix the extraction, not the model. ***")

    print("\n" + "=" * 78)
    print("2. SPAN LENGTH (tokens) -- the 3x cap is one-sided")
    print("=" * 78)
    print(f"  {'':<34}{'GOLD':>10}{'OURS':>10}{'diff':>10}")
    row("mean", "len_mean")
    row("median", "len_med")
    row("p90", "len_p90")
    row("share <= 3 tokens", "len_le3", pct=True)
    print("\n  If OURS runs LONGER than gold, len(pred) <= 3 x len(gold) is")
    print("  breached on short gold spans and they are lost outright. SHORTER")
    print("  is safe -- gold contains it. The asymmetry is why this matters")
    print("  more than the mean.")

    print("\n" + "=" * 78)
    print("3. POSITION IN POST -- lead bias?")
    print("=" * 78)
    print(f"  {'':<34}{'GOLD':>10}{'OURS':>10}{'diff':>10}")
    row("mean relative position", "rel_mean")
    row("median", "rel_med")
    row("share in first third", "rel_first_third", pct=True)

    print("\n" + "=" * 78)
    print("4. SPANS PER NON-EMPTY POST")
    print("=" * 78)
    print(f"  {'':<34}{'GOLD':>10}{'OURS':>10}{'diff':>10}")
    row("mean", "pp_mean")
    row("median", "pp_med")
    row("share with >= 3 spans", "pp_ge3", pct=True)

    print("\n" + "=" * 78)
    print("HOW TO READ IT")
    print("=" * 78)
    print("Every `<-- OFF` row is a defect on the SUBMITTED artefact, not on a")
    print("surrogate. That is the whole point: val inflated three defects in a")
    print("row because of the risk-conditioning gap, and none of these four")
    print("measurements needs a test label at all.")
    print("")
    print("Rows that come back matched are genuinely closed -- the output has")
    print("the same shape as gold on that axis, so there is nothing to fix.")


if __name__ == "__main__":
    main()
