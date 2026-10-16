"""
evidence_vote_ensemble.py

Builds a span-level VOTE ensemble over several evidence taggers and writes an
override CSV consumable by predict.py --evidence-override or by
patch_submission.py.

Voting rather than probability averaging (ensemble_evidence_seeds.py) attacks
the dominant error mode of a single tagger: spurious spans that match no gold
span at all, rather than boundary error. Requiring two independently-trained
taggers to agree on a span filters exactly those -- an invented span rarely
survives a second seed, a real one usually does.

Agreement is decided with the OFFICIAL containment match rule, not string
equality -- separate seeds almost never produce byte-identical boundaries, so
exact-match voting would throw away nearly every real agreement. When a group
of runs agree, the SHORTEST member is kept, because the metric's 3x length cap
punishes over-long spans while containment favours the tighter one.

The change is GLOBAL -- it touches every post rather than firing on a rare
pattern -- so its effect does not depend on the prevalence of some trigger in
the test set.

Usage (GPU, test set):
  python src/evidence_vote_ensemble.py \
      --test data/test.csv \
      --runs runs/evidence_v2,runs/evidence_v1,runs/evidence_s14 \
      --k 2 --out artifacts/evidence_vote2_test.csv

Verify on a labelled split:
  python src/evidence_vote_ensemble.py \
      --test artifacts/val_split.csv --eval-gold \
      --runs runs/evidence_v2,runs/evidence_v1,runs/evidence_s14 \
      --k 2 --out artifacts/evidence_vote2_val.csv
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1, _span_pair_matches, token_len

# NOTE: torch / transformers / predict are imported lazily inside main(), so
# that other CPU-only tools (evidence_combo_gate.py) can import vote_combine
# from here without needing the GPU stack. Keeping ONE definition of the
# voting logic matters more than import tidiness: a second copy could drift
# out of sync and make two callers disagree about which spans were voted in.


def vote_combine(per_run, k):
    """per_run: list (one per run) of list-per-post of spans. Returns spans-per-post."""
    n_posts = len(per_run[0])
    out = []
    for i in range(n_posts):
        cands = [r[i] for r in per_run]
        pool = [s for spans in cands for s in spans]
        kept, used = [], set()
        for j, s in enumerate(pool):
            if j in used:
                continue
            group = [s]
            used.add(j)
            for j2 in range(j + 1, len(pool)):
                if j2 in used:
                    continue
                if _span_pair_matches(pool[j2], s):
                    group.append(pool[j2])
                    used.add(j2)
            n_runs = sum(1 for spans in cands
                         if any(any(_span_pair_matches(x, g) for g in group)
                                for x in spans))
            if n_runs >= k:
                kept.append(min(group, key=token_len))
        out.append(kept)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True,
                    help="posts to predict on (data/test.csv, or a val split "
                         "with --eval-gold to verify)")
    ap.add_argument("--runs", required=True, help="comma-separated evidence run dirs")
    ap.add_argument("--k", type=int, default=2, help="minimum agreeing runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-gold", action="store_true",
                    help="input has an 'evidence' gold column; report phrase-F1")
    ap.add_argument("--cache-dir", default="artifacts/evidence_per_run",
                    help="per-run predictions are cached here so re-voting at a "
                         "different k needs no GPU")
    args = ap.parse_args()

    import torch
    from predict import predict_evidence  # the same decoder used at inference

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    df = pd.read_csv(args.test)
    run_dirs = [r.strip().rstrip("/") for r in args.runs.split(",")]
    os.makedirs(args.cache_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.test))[0]

    per_run = []
    for rd in run_dirs:
        name = os.path.basename(rd)
        cache = os.path.join(args.cache_dir, f"{name}__{tag}.csv")
        if os.path.exists(cache):
            print(f"  [cached] {name}")
            c = pd.read_csv(cache, keep_default_na=False).set_index("row_id")["evidence"]
            spans = [parse_evidence(c.get(r, "")) for r in df["row_id"]]
        else:
            print(f"  [gpu] scoring {name} ...")
            spans = predict_evidence(rd, df, device)
            pd.DataFrame({"row_id": df["row_id"],
                          "evidence": ["; ".join(s) for s in spans]}).to_csv(
                cache, index=False)
        per_run.append(spans)
        if args.eval_gold:
            g = [parse_evidence(x) for x in df["evidence"]]
            print(f"      {name} alone: F1 {phrase_f1(g, spans).f1:.4f}")

    combined = vote_combine(per_run, args.k)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame({"row_id": df["row_id"],
                  "evidence": ["; ".join(s) for s in combined]}).to_csv(
        args.out, index=False)

    n_spans = sum(len(s) for s in combined)
    n_empty = sum(1 for s in combined if not s)
    print(f"\nvote>={args.k} over {len(run_dirs)} runs: {n_spans} spans, "
          f"{n_empty}/{len(combined)} posts with no evidence")
    if args.eval_gold:
        g = [parse_evidence(x) for x in df["evidence"]]
        r = phrase_f1(g, combined)
        print(f"ENSEMBLE phrase-F1: P {r.precision:.4f}  R {r.recall:.4f}  "
              f"F1 {r.f1:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
