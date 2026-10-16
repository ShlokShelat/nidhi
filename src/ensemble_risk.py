"""
Cross-model risk ensemble.

Averages the DeBERTa 3-seed risk probabilities with an externally supplied
MentalRoBERTa risk distribution. The two members differ in architecture and
in pretraining corpus, so their errors are less correlated than two seeds of
one model: averaging them cancels member-specific error instead of merely
reducing seed variance.

The external file (artifacts/friend_risk_probs.csv):
    row_id,Indicator,Ideation,Behavior,Attempt   (softmax, one row per test post)

The DeBERTa side is computed live from the saved risk runs by the same
predict_risk() call predict.py uses, so the blended member is the same
model the rest of the pipeline serves.

Usage:
  python src/ensemble_risk.py \
      --test data/test.csv \
      --our-risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
      --friend-probs artifacts/friend_risk_probs.csv \
      --weight-ours 0.5 \
      --out artifacts/ensembled_risk.csv

Writes ensembled_risk.csv with row_id + a pred_risk_level column + the 4
averaged probs. predict.py can then read pred_risk_level directly, OR use
--risk-override to consume this file.

--weight-ours controls the blend (0.5 = equal weight on both members). The
external member ships as test-set probabilities only, with no matching
out-of-fold predictions, so the weight cannot be tuned offline; 0.5 is the
weight that assumes nothing about the relative accuracy of the members.

--our-weights controls how the base DeBERTa models named by --our-risk-run
are averaged BEFORE blending with the external member. Comma-separated,
e.g. "0.5,0.3,0.2", one weight per run dir. Omitting it averages them
equally.
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from scoring import RISK_LEVELS
from predict import predict_risk

# the external file's columns use the capitalized official label names
FRIEND_COLS = ["Indicator", "Ideation", "Behavior", "Attempt"]
# the local RISK_LEVELS may be lowercase, so the two label sets are matched by
# POSITION, which is well defined because both list the same clinical ordering
assert len(RISK_LEVELS) == 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--our-risk-run", required=True,
                    help="comma-separated local risk run dirs (seed ensemble)")
    ap.add_argument("--friend-probs", required=True)
    ap.add_argument("--allow-missing-friend", action="store_true",
                    help="blend even when the external member covers <98%% of "
                         "rows. Uncovered rows fall back to an effective "
                         "w_ours=1.0, i.e. they are predicted by a single "
                         "member while the rest of the file is predicted by "
                         "the two-member blend. Use only when a single-member "
                         "prediction on those rows is acceptable.")
    ap.add_argument("--weight-ours", type=float, default=0.5,
                    help="blend weight on the local seed-ensemble probs "
                         "(0..1); the external member gets 1-w")
    ap.add_argument("--our-weights", default=None,
                    help="comma-sep weights for the models in --our-risk-run, "
                         "e.g. '0.5,0.3,0.2'. Default: equal weighting.")
    ap.add_argument("--out", default="artifacts/ensembled_risk.csv")
    # THE HEAD SELECTION IS PART OF THE MODEL, SO IT IS AN ARGUMENT, NOT AN
    # AMBIENT ENVIRONMENT VARIABLE.
    #
    # DualHeadModel.forward returns the categorical logits when RISK_CAT_ONLY=1
    # and a fixed 50/50 mixture of the categorical and ordinal heads otherwise.
    # Those are two different models. Leaving the choice to an exported shell
    # variable means forgetting to export it produces a complete, well-formed,
    # silently different risk column -- no exception, no warning, and nothing in
    # the output file recording which head was read.
    #
    # This flag sets the variable itself, so the deployed behaviour is the
    # default and cannot be lost by forgetting to export anything, and the
    # choice is written to <out>.meta.json so a later step can verify it.
    ap.add_argument("--cat-only", dest="cat_only", action="store_true",
                    default=True,
                    help="read the CATEGORICAL head only (the deployed "
                         "configuration; sets RISK_CAT_ONLY=1). Default.")
    ap.add_argument("--head-mixture", dest="cat_only", action="store_false",
                    help="read the fixed 50/50 categorical+ordinal mixture "
                         "instead. This is a DIFFERENT model from the deployed "
                         "one and its probabilities are not interchangeable.")
    args = ap.parse_args()

    prev = os.environ.get("RISK_CAT_ONLY")
    os.environ["RISK_CAT_ONLY"] = "1" if args.cat_only else "0"
    if prev is not None and prev != os.environ["RISK_CAT_ONLY"]:
        print(f"NOTE: RISK_CAT_ONLY was {prev!r} in the environment; the "
              f"--cat-only/--head-mixture flag wins and it is now "
              f"{os.environ['RISK_CAT_ONLY']!r}.")
    print("risk head: "
          + ("CATEGORICAL only (deployed)" if args.cat_only
             else "50/50 categorical+ordinal MIXTURE (not the deployed head)"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # local probabilities (predict_risk returns (labels, prob_matrix))
    df = load(args.test)

    our_weights = None
    if args.our_weights:
        our_weights = [float(x) for x in args.our_weights.split(",")]
        if any(w < 0 for w in our_weights):
            raise ValueError(f"--our-weights must be non-negative, got {our_weights}")
        if sum(our_weights) == 0:
            raise ValueError("--our-weights cannot all be zero")

    _, our_probs = predict_risk(args.our_risk_run, df, device, weights=our_weights)  # (N,4), softmax
    our_ids = df["row_id"].tolist()

    # external member's probabilities, aligned by row_id
    fp = pd.read_csv(args.friend_probs)
    fp = fp.set_index("row_id")
    friend_probs = np.zeros((len(our_ids), 4), dtype=np.float64)
    missing = 0
    for i, rid in enumerate(our_ids):
        if rid in fp.index:
            friend_probs[i] = fp.loc[rid, FRIEND_COLS].values.astype(float)
        else:
            friend_probs[i] = our_probs[i]  # uncovered row -> single-member pred
            missing += 1
    if missing:
        frac = missing / max(len(our_ids), 1)
        msg = (f"{missing} of {len(our_ids)} row_ids ({100*frac:.0f}%) are "
               f"missing from {args.friend_probs}; those rows fall back to our "
               f"own probabilities, i.e. an effective w_ours of 1.0 on them.")
        # A PER-ROW FALLBACK IS A SILENT MODEL SWAP. The external member is the
        # only architecturally distinct model in the risk blend, so a row that
        # falls back is predicted by a strictly weaker, less diverse ensemble
        # than the one the blend weight was chosen for. The failure this guard
        # prevents is that the swap is invisible: a probability file produced
        # for one test set shares no row_id with a fresh one, so the fallback
        # can fire on every row while the log still reports w_ours=0.5.
        if frac > 0.02 and not args.allow_missing_friend:
            raise SystemExit(
                "REFUSING: " + msg + " Generate the friend's probabilities for "
                "these rows first, or pass --allow-missing-friend to blend "
                "without them and accept the loss knowingly.")
        print("WARNING: " + msg)

    # renormalize both (safety) and blend
    def norm(P):
        s = P.sum(1, keepdims=True)
        s[s == 0] = 1.0
        return P / s
    our_n = norm(our_probs.astype(np.float64))
    fr_n = norm(friend_probs)
    w = args.weight_ours
    blended = w * our_n + (1.0 - w) * fr_n

    pred_idx = blended.argmax(1)
    pred_labels = [RISK_LEVELS[k] for k in pred_idx]

    out = pd.DataFrame({
        "row_id": our_ids,
        "pred_risk_level": pred_labels,
    })
    for j, c in enumerate(FRIEND_COLS):
        out[f"p_{c}"] = blended[:, j]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    # Provenance sidecar. A probability CSV on its own cannot say which head
    # produced it, so downstream steps have no way to tell the deployed column
    # from the mixture column. This records it next to the file.
    import json
    meta = {
        "risk_cat_only": bool(args.cat_only),
        "our_risk_run": args.our_risk_run,
        "our_weights": our_weights,
        "weight_ours": w,
        "friend_probs": args.friend_probs,
        "friend_missing_rows": int(missing),
        "n_rows": int(len(our_ids)),
    }
    with open(args.out + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    from collections import Counter
    print(f"blended risk dist (w_ours={w}, our_weights={our_weights}): "
          f"{dict(Counter(pred_labels))}")
    # how many labels the blend moves relative to the seed ensemble alone
    ours_labels = [RISK_LEVELS[k] for k in our_n.argmax(1)]
    changed = sum(a != b for a, b in zip(pred_labels, ours_labels))
    print(f"changed {changed}/{len(pred_labels)} risk labels vs our-seed-ensemble alone")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
