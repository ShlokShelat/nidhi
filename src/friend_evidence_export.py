"""
friend_evidence_export.py

Export the friend's evidence spans in the candidate format the vote expects, so
it can be tested as a 9th voter.

WHY, GIVEN IT SCORES 0.5790 AGAINST THE SHIPPED 0.6840. Because vote>=2 uses AGREEMENT,
not quality, and the asymmetry favours adding members:

  * a span only this member proposes never reaches two votes and is discarded,
    so its false positives are nearly free
  * a true span it CONFIRMS lifts a 1-vote span to 2, which is pure recall

That is why the shipped 8-member vote beats the 4-member one even though the
extra four are individually weaker, and the note in
`scripts/evidence_more_models.sh` says so explicitly: "A weak member is NOT a
reason to exclude it from the vote."

WHAT MAKES THIS MEMBER DIFFERENT FROM THE OTHER EIGHT. The other eight are all
`microsoft/deberta-v3-large`, and four of them (v2, v1, s14, s15) differ only by
seed -- the bag's entire diversity is seed noise. This is mental-roberta-base,
trained multi-task alongside risk and factor heads. Its errors have real reason
to be decorrelated, which is the one property the vote can convert into recall.

SPAN CONFIDENCE is the MAX token probability inside the span, not the mean: a
span is evidence if any part of it is confidently tagged, and averaging would
punish long spans that are right about their core.

Writes `artifacts/evidence_oof/friend_oof_candidates.csv` with the same
row_id/span/conf/min_conf columns the other taggers use, so
`evidence_vote_oof.py --tags v2,ep12,lr3,bs16,ml256,friend` just works.

Usage:
  python src/friend_evidence_export.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_span, token_len  # noqa: E402


def spans_with_conf(tokp, offs, post, thr, max_words):
    out, i, n = [], 0, len(tokp)
    while i < n:
        if tokp[i] < thr:
            i += 1
            continue
        j = i
        while j + 1 < n and tokp[j + 1] >= thr:
            j += 1
        s, e = int(offs[i][0]), int(offs[j][1])
        if e > s:
            t = post[s:e].strip()
            if len(t) >= 3 and token_len(t) <= max_words:
                out.append((t, float(tokp[i:j + 1].max())))
        i = j + 1
    keep = []
    for t, c in sorted(out, key=lambda x: (-token_len(x[0]), -len(x[0]))):
        nt = normalize_span(t)
        if nt and not any(nt in normalize_span(k) for k, _ in keep):
            keep.append((t, c))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof",
                    default="C:/Users/PC5/Desktop/pipeline/outputs/oof.npy")
    ap.add_argument("--outdir", default="artifacts/evidence_oof")
    ap.add_argument("--tag", default="friend")
    # 0.20 keeps the raw pool wide; min_conf below is what the vote filters at.
    # Exporting sub-threshold candidates matters -- the other taggers' files do
    # the same, and evidence_conf_vote reads the raw pool.
    ap.add_argument("--emit-thr", type=float, default=0.20)
    ap.add_argument("--min-conf", type=float, default=0.40,
                    help="the sweep's optimum for this tagger (F1 0.5790)")
    ap.add_argument("--max-words", type=int, default=14)
    args = ap.parse_args()

    if not os.path.exists(args.oof):
        raise SystemExit(f"missing {args.oof}")
    d = np.load(args.oof, allow_pickle=True).item()

    rows = []
    n_post = 0
    for f in sorted(d):
        o = d[f]
        tp = np.asarray(o["tokp"], dtype=float)
        of = np.asarray(o["offsets"])
        for i, r in enumerate(o["row_id"]):
            n_post += 1
            for t, c in spans_with_conf(tp[i], of[i], str(o["post"][i]),
                                        args.emit_thr, args.max_words):
                rows.append({"row_id": str(r), "span": t, "conf": c,
                             "min_conf": args.min_conf})

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.tag}_oof_candidates.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    surv = int((df["conf"] >= args.min_conf).sum())
    print(f"wrote {out}")
    print(f"  {n_post} posts   {len(df)} candidates   "
          f"{surv} survive min_conf={args.min_conf}  "
          f"({surv / max(n_post, 1):.2f}/post)")
    print(f"  conf range {df['conf'].min():.3f}-{df['conf'].max():.3f}")
    print("\nFor comparison the shipped OOF taggers emit 1.20-1.48 spans/post")
    print("after their own min_conf. A much lower rate here means this member will")
    print("rarely be the second vote and cannot help; a much higher one means")
    print("it may push weak spans over the k=2 line.")
    print("\nNext (on the cluster, after scp'ing this file):")
    print("  python src/evidence_vote_oof.py --tags v2,ep12,lr3,bs16,ml256")
    print("  python src/evidence_vote_oof.py "
          "--tags v2,ep12,lr3,bs16,ml256,friend")
    print("\nRead the vote>=2 row in both. The second must beat the first on")
    print("BOTH author halves, or the extra member is noise.")


if __name__ == "__main__":
    main()
