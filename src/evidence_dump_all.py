"""
evidence_dump_all.py

One GPU pass that produces the two artefacts every downstream evidence
experiment needs, so neither of them requires its own GPU job.

For each tagger run and each split it writes:

  {run}_{split}_candidates.csv   every candidate span with its confidence,
                                 collected at min_conf = 0.0
  {run}_{split}_pred.csv         the same run decoded at its OWN shipped
                                 evidence_min_conf

The unfiltered candidate dump serves two purposes.

1. CROSS-RUN AGREEMENT. Keeping only spans that several independently trained
   taggers propose is a precision filter: disagreement concentrates on spans
   one model invented, so agreement discards proportionally more bad spans
   than good ones. Building that filter needs each run's predictions on the
   same split, which is what this pass materialises.

2. THE DISCARDED-CONFIDENCE BAND. Phrase-F1 is micro-averaged with precision
   approximately equal to recall, and in that regime emitting one more span is
   net positive whenever its chance of matching exceeds about F1/2. Any span
   scored above that break-even but below a run's shipped min_conf is being
   discarded even though emitting it pays. Dumping every candidate with its
   confidence lets the hit rate per confidence band be measured on CPU, so the
   cut follows from the metric's own break-even identity instead of from a
   search over val for whichever number happens to score best -- a search that
   fits the split rather than the metric.

Reuses train_evidence.TaggerModel / collect_candidates / spans_at and mirrors
predict.predict_evidence's loading exactly, so this dump cannot drift away
from what the submission path decodes.

Usage (GPU, ~15 min for 4 runs x 750 posts):
  python src/evidence_dump_all.py \
      --runs evidence_v2,evidence_v1,evidence_s14,evidence_s15
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="evidence_v2,evidence_v1,"
                                      "evidence_s14,evidence_s15")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--outdir", default="artifacts/evidence_runs")
    ap.add_argument("--batch-note", default="")
    ap.add_argument("--risk-cond", action="store_true",
                    help="taggers were trained with --risk-cond; condition "
                         "inference on PREDICTED risk levels")
    ap.add_argument("--test-risk", default=None,
                    help="CSV row_id,risk_level for test (the risk predictions "
                         "being deployed). Required with --risk-cond.")
    ap.add_argument("--val-risk", default=None,
                    help="CSV row_id,risk_level (or pred_risk_level) for val, "
                         "OUT OF FOLD. Required with --risk-cond.")
    args = ap.parse_args()

    if args.risk_cond and not (args.test_risk and args.val_risk):
        raise SystemExit("--risk-cond needs --test-risk and --val-risk")

    import torch
    from transformers import AutoTokenizer
    from train_evidence import TaggerModel, collect_candidates, spans_at

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")
    os.makedirs(args.outdir, exist_ok=True)

    # AVOID TRAIN/SERVE TEXT-SPACE SKEW: read every split through
    # data_audit.load, exactly as the training splits were read.
    #
    # load() applies clean_text (HTML unescape, NFKC, folding of typographic
    # characters). The taggers were therefore trained on canonical text, and a
    # curly apostrophe tokenizes differently from a straight one, so serving
    # them raw text changes the token sequence on any post that uses
    # typographic punctuation. An offline check against the training folds
    # cannot detect this, because those folds are already canonical -- the
    # only defence is to canonicalise on both sides by construction.
    #
    # It also settles the output side: spans lifted from canonical text are
    # emitted in canonical space, which is the space the gold spans use.
    #
    # The other components load the same way (src/predict_factors_nli.py,
    # src/blend_risk_saved.py).
    from data_audit import load
    te = load(args.test)
    va = load(args.val)
    print(f"test {len(te)} posts, val {len(va)} posts  (canonicalized via "
          f"data_audit.load, matching how the taggers were trained)")

    risk_of = {}
    if args.risk_cond:
        from scoring import normalize_label
        for split, path, df in (("test", args.test_risk, te),
                                ("val", args.val_risk, va)):
            r = pd.read_csv(path, keep_default_na=False)
            col = ("risk_level" if "risk_level" in r.columns
                   else "pred_risk_level")
            m = {str(a): normalize_label(b) for a, b in zip(r["row_id"], r[col])}
            absent = [str(x) for x in df["row_id"] if str(x) not in m]
            if absent:
                raise SystemExit(
                    f"{path} lacks risk labels for {len(absent)} {split} rows "
                    f"(e.g. {absent[:3]}). Conditioning on a wrong or missing "
                    f"level is worse than not conditioning at all.")
            risk_of[split] = [m[str(x)] for x in df["row_id"]]
            from collections import Counter
            print(f"  {split} risk from {path}: "
                  f"{dict(Counter(risk_of[split]))}")

    for run in [r.strip() for r in args.runs.split(",")]:
        d = os.path.join("runs", run)
        cfgp = os.path.join(d, "run_config.json")
        if not os.path.exists(cfgp):
            print(f"  skip {run}: no run_config.json")
            continue
        cfg = json.load(open(cfgp))
        ckpt = next((os.path.join(d, f) for f in
                     ("model_final.pt", "model.pt")
                     if os.path.exists(os.path.join(d, f))), None)
        if ckpt is None:
            print(f"  skip {run}: no checkpoint")
            continue

        base = cfg.get("model", "microsoft/deberta-v3-large")
        max_len = int(cfg.get("max_len", 512))
        min_conf = float(cfg.get("evidence_min_conf", 0.0))
        print(f"\n=== {run} ===")
        print(f"  base={base}  max_len={max_len}  shipped min_conf={min_conf}")

        tok = AutoTokenizer.from_pretrained(base)
        model = TaggerModel(base)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device).eval()

        for split, df in (("test", te), ("val", va)):
            posts = [str(x) for x in df["post"]]
            # min_conf is applied later on CPU, so collect EVERYTHING here.
            cands = collect_candidates(model, tok, posts, max_len, device,
                                       risks=risk_of.get(split))

            # long form: one row per candidate span, with its confidence
            rows = []
            for i, rid in enumerate(df["row_id"]):
                for s, c in cands[i]:
                    rows.append((rid, s, float(c)))
            pd.DataFrame(rows, columns=["row_id", "span", "conf"]).to_csv(
                os.path.join(args.outdir, f"{run}_{split}_candidates.csv"),
                index=False)

            # and the decoded view, at this run's own min_conf
            kept = spans_at(cands, min_conf)
            pd.DataFrame({
                "row_id": df["row_id"],
                "pred_evidence": ["; ".join(s) for s in kept],
            }).to_csv(os.path.join(args.outdir, f"{run}_{split}_pred.csv"),
                      index=False)

            n = sum(len(s) for s in kept)
            print(f"  {split}: {len(rows)} candidates, "
                  f"{n} kept at min_conf={min_conf} "
                  f"({n / max(len(df), 1):.2f}/post)")

        del model
        torch.cuda.empty_cache()

    print(f"\nwrote to {args.outdir}/")
    print("\nNext, both CPU:")
    print("  1. vote>=2 on test  -> src/evidence_vote_test.py")
    print("  2. confidence-band hit rates -> src/evidence_conf_bands.py")


if __name__ == "__main__":
    main()
