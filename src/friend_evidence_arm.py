"""
friend_evidence_arm.py

THE TEAMMATE'S THIRD HEAD, NEVER USED.

Their model is a TRI-head MentalRoBERTa: risk + factors + evidence tokens. The
deployed pipeline uses their risk head (35% of the risk blend) and their
factor head (`friendfac`, a contributor to the factor pool). **The evidence
head has never been used.**

WHY IT IS WORTH A LOOK. The seven RC7 voters are one model in seven costumes --
same backbone (deberta-v3-large), same data, same BIO objective, differing only
by seed/lr/epochs, so their vote count carries almost no information about
correctness (n_votes AUC 0.610 on RC7's own pool). Architecturally diverse BIO
taggers add little for the same reason: every one of them is still a tagger
trained on SPANS ALONE. The friend's head is trained JOINTLY with risk and
factor supervision on a 125M mental-health-pretrained encoder. That is a
different training signal, not just a different backbone.

WHAT THIS WRITES. Candidate CSVs in exactly the RC7 dump format
(`{run}_{split}_candidates.csv` with row_id, span, conf) so the shipped
`evidence_vote_test.py` can take it as an 8th member with no other change:

  val  <- pipeline/outputs/oof.npy  (per-fold row_id/tokp/offsets/post; every
          row is HELD OUT for the fold that scored it, so it is honest)
  test <- pipeline/outputs/nidhi.csv (their already-decoded submission spans)

`conf` is the mean token probability over the span, so the existing min_conf
machinery still means something.

SECURITY: reads only local files under the user's own directories. Nothing
leaves the machine.

Usage (CPU, seconds):
  python src/friend_evidence_arm.py --pipeline src/friend_pipeline \
      --outdir artifacts/evidence_runs_friend --thr 0.40
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    # The tri-head pipeline is vendored in this repository, so the default
    # points at the in-tree copy rather than an external checkout. Override
    # only to read outputs/ from a separate run of that project.
    ap.add_argument("--pipeline",
                    default=os.path.join(os.path.dirname(
                        os.path.abspath(__file__)), "friend_pipeline"))
    ap.add_argument("--outdir", default="artifacts/evidence_runs_friend")
    ap.add_argument("--run", default="evidence_friend",
                    help="member name; dumps land as {run}_{split}_candidates.csv")
    ap.add_argument("--thr", type=float, default=None,
                    help="token threshold. Default: sweep the friend's own "
                         "EVID_THRESHOLD_GRID and report each, writing the "
                         "middle value 0.40 unless --thr is given.")
    ap.add_argument("--test-only", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.pipeline, "src"))
    import postprocess as PP  # the friend's own decoder, not a reimplementation
    import config as C

    os.makedirs(args.outdir, exist_ok=True)
    out = args.pipeline + os.sep + "outputs"

    # ---- TEST: their spans are already decoded in nidhi.csv -----------------
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scoring import parse_evidence
    nid = pd.read_csv(os.path.join(out, "nidhi.csv"), keep_default_na=False)
    rows = []
    for r in nid.itertuples():
        for s in parse_evidence(r.evidence):
            rows.append({"row_id": str(r.row_id), "span": s, "conf": 1.0})
    te = pd.DataFrame(rows, columns=["row_id", "span", "conf"])
    p = os.path.join(args.outdir, f"{args.run}_test_candidates.csv")
    te.to_csv(p, index=False)
    print(f"TEST  {len(te)} spans over {te.row_id.nunique()} posts "
          f"({len(te)/max(len(nid),1):.2f}/post)  -> {p}")
    print("  NOTE conf=1.0: nidhi.csv carries no per-span probability. The vote")
    print("  gates on min_conf, so this member is effectively ungated on test.")
    if args.test_only:
        return

    # ---- VAL: decode from the held-out OOF token probabilities ---------------
    o = np.load(os.path.join(out, "oof.npy"), allow_pickle=True).item()
    grid = C.EVID_THRESHOLD_GRID if args.thr is None else [args.thr]
    chosen = args.thr if args.thr is not None else 0.40
    for thr in grid:
        rows, nposts = [], 0
        for fold in sorted(o):
            f = o[fold]
            for rid, post, offs, tokp in zip(f["row_id"], f["post"],
                                             f["offsets"], f["tokp"]):
                nposts += 1
                spans = PP.decode_spans(post, offs, tokp, thr)
                for s in spans:
                    # mean token prob inside the span == a usable conf
                    lo = post.find(s)
                    cf = 1.0
                    if lo >= 0:
                        hi = lo + len(s)
                        ps = [p_ for (a, b), p_ in zip(offs, tokp)
                              if b > a and a >= lo and b <= hi]
                        if ps:
                            cf = float(np.mean(ps))
                    rows.append({"row_id": str(rid), "span": s, "conf": cf})
        df = pd.DataFrame(rows, columns=["row_id", "span", "conf"])
        mark = "  <- written" if abs(thr - chosen) < 1e-9 else ""
        print(f"VAL   thr={thr:.2f}  {len(df)} spans over "
              f"{df.row_id.nunique()} of {nposts} posts "
              f"({len(df)/max(nposts,1):.2f}/post){mark}")
        if abs(thr - chosen) < 1e-9:
            q = os.path.join(args.outdir, f"{args.run}_val_candidates.csv")
            df.to_csv(q, index=False)
            print(f"      -> {q}")

    print("\nNEXT -- add as an 8th member and judge by WIN RATE, not the mean:")
    print("  copy the RC7 val dumps into the same dir, then")
    print("  python src/evidence_cap_sweep.py --dir <that dir> \\")
    print("      --runs <RC7>,evidence_friend")
    print("The bar is the same as every other member change: >=90% win over")
    print("author-disjoint splits. Member LOO on the shipped 7 shows dropping")
    print("any one member is essentially costless, so a genuine 8th member")
    print("has to do better than that.")


if __name__ == "__main__":
    main()
