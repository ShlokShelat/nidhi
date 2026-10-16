"""
evidence_redump.py

THE GENERATOR THROWS AWAY SPANS THE SCORER WOULD HAVE ACCEPTED.

`verify_spans(..., max_tokens=12)` in train_evidence.py:114-127 REJECTS any
candidate longer than 12 whitespace tokens, justified by:

    "gold spans in the sample run 2-8 tokens and the 3x rule punishes long
     spans, so anything long is suspect"

That gets the arithmetic backwards. `_span_pair_matches` (scoring.py:141-149)
accepts a prediction that CONTAINS gold as long as
`token_len(pred) <= 3 * token_len(gold)`. Against an 8-token gold that permits
**24 tokens**; against a 6-token gold, 18. The 3x rule is what LICENSES long
spans, not what punishes them -- what it punishes is a long span against a
SHORT gold. Capping at 12 discards matches that would have been paid for.

WHY THIS IS THE RIGHT TARGET. Measured on the 1590-post OOF:

    candidate clusters/post              1.39   (gold spans/post 1.12)
    oracle ceiling of that pool          0.8894 micro
    posts where NO candidate matches     92 of 1020 with non-empty gold

Simply dropping the `min_conf` gate -- also applied at dump time -- lifted the
ceiling to 0.9279 and recovered 14 unreachable posts for zero GPU. Both gates
were tuned under MICRO averaging and GOLD suppression, two defects that have
since been corrected. Generation, not selection, is the binding constraint.

DUMP LIBERALLY, FILTER DOWNSTREAM. `collect_candidates` never forwards
max_tokens, so the 12 is hardcoded into every dump in the pipeline. This re-decodes at a
HIGH cap and writes the token length of every span, which turns max_tokens into
a CPU-tunable knob exactly like min_conf -- one GPU pass, then sweep offline.
Nothing is committed by running it: it writes to `{tag}_long_oof_candidates.csv`
and does not touch the existing files.

Everything else mirrors scripts/evidence_5fold.sh lines 96-138 exactly -- same
run_config, same tokenizer, same model_final.pt, same fold-scores-own-val
discipline -- so the only variable is the cap.

Usage (cluster, one GPU, ~20 min for 5 tags x 5 folds):
  python src/evidence_redump.py
  python src/evidence_redump.py --split test --test data/test.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label  # noqa: E402
from train_evidence import (O, RISK_PREFIX, TaggerModel,  # noqa: E402
                            _decode_candidates, verify_spans)

TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]
DEFAULT_MODEL = "microsoft/deberta-v3-large"


def soft_spans(post, probs, offsets, taus, max_tokens):
    """SPANS THE ARGMAX DECODE CAN NEVER PROPOSE.

    `_decode_candidates` walks `probs.argmax(-1)` and only emits runs tagged
    B/I. A span the model scores at 45% confidence has argmax O everywhere and
    is invisible -- no min_conf and no length cap can recover it, because it
    was never a candidate. That is why 92 of 1020 non-empty-gold posts have NO
    candidate matching gold: on those posts the argmax never fired at all.

    This takes contiguous runs where P(not-O) clears a LOW threshold, at
    several thresholds, off the same forward pass. tau ~ 0.5 reproduces the
    argmax decode; lower tau proposes the spans it silently dropped.
    """
    pnot = 1.0 - probs[:, O]
    out = []
    for tau in taus:
        runs, cur = [], None
        for k, (a, b) in enumerate(offsets):
            if a == b:
                continue
            if pnot[k] >= tau:
                if cur is None:
                    cur = [a, b, float(pnot[k]), 1]
                else:
                    cur[1] = b
                    cur[2] += float(pnot[k])
                    cur[3] += 1
            elif cur:
                runs.append(cur)
                cur = None
        if cur:
            runs.append(cur)
        for a, b, cs, nn in runs:
            text = post[a:b].strip()
            acc, _ = verify_spans([text], post, max_tokens=max_tokens)
            if acc:
                out.append((acc[0], cs / nn, f"tau{tau:g}"))
    return out


def collect_with_cap(model, tok, posts, max_len, device, max_tokens, taus,
                     risks=None):
    """The argmax decode with a parameterised length cap, PLUS soft-threshold
    candidates. Deduped on normalised text, keeping the highest confidence and
    recording which generator first proposed each span."""
    import torch
    from scoring import normalize_span
    out = []
    model.eval()
    with torch.no_grad():
        for i, post in enumerate(posts):
            if risks is None:
                enc = tok(post, truncation=True, max_length=max_len,
                          return_offsets_mapping=True, return_tensors="pt")
                seq = None
            else:
                # A RISK-CONDITIONED tagger was trained on a PAIR: the risk
                # statement as sequence 1, the post as sequence 2. Encoding
                # the post alone feeds it a shape it never saw -- the first
                # attempt did exactly that and candidate counts collapsed from
                # ~800 to ~240 per fold.
                enc = tok(RISK_PREFIX.format(normalize_label(risks[i])), post,
                          truncation="only_second", max_length=max_len,
                          return_offsets_mapping=True, return_tensors="pt")
                seq = enc.sequence_ids(0)
            offsets = enc.pop("offset_mapping").squeeze(0).tolist()
            enc = {k: v.to(device) for k, v in enc.items()
                   if k in ("input_ids", "attention_mask")}
            logits = model(**enc)["logits"].squeeze(0).float()
            probs = torch.softmax(logits, -1).cpu().numpy()
            if seq is not None:
                keep = [t for t in range(len(offsets)) if seq[t] == 1]
                probs = probs[keep]
                offsets = [offsets[t] for t in keep]
            cands = [(s, c, "argmax") for s, c in
                     _decode_candidates(post, probs, offsets,
                                        max_tokens=max_tokens)]
            if taus:
                cands += soft_spans(post, probs, offsets, taus, max_tokens)
            best = {}
            for s, c, src in cands:
                k = normalize_span(s)
                if k and (k not in best or c > best[k][1]):
                    best[k] = (s, c, src)
            out.append(list(best.values()))
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--tags", default=",".join(TAGS))
    a_.add_argument("--runs", default="runs/ev5f_{tag}_f{f}")
    a_.add_argument("--outdir", default="artifacts/evidence_oof")
    a_.add_argument("--suffix", default="_long")
    a_.add_argument("--max-tokens", type=int, default=64,
                    help="dump cap. Keep it HIGH -- the point is to filter "
                         "downstream, not here. 64 is far above anything the "
                         "3x rule could accept, so nothing is pre-judged.")
    a_.add_argument("--taus", default="0.20,0.35,0.50",
                    help="soft-decode thresholds on P(not-O). These propose "
                         "spans the argmax decode never can -- the only way to "
                         "reach the 92 posts where no candidate exists. Empty "
                         "string disables them.")
    a_.add_argument("--risk-cond", action="store_true",
                    help="the tagger was trained with --risk-cond, so it must "
                         "be fed the risk statement as sequence 1. Required "
                         "for runs/evcond5_cond_* -- without it the model sees "
                         "an input shape it never trained on.")
    a_.add_argument("--risks-oof", default="artifacts/champion_oof.csv",
                    help="PREDICTED risk per OOF post (deployment-faithful; "
                         "gold would be leakage)")
    a_.add_argument("--risks-test", default="submissions/NIDHI_factfit.csv",
                    help="the risk labels the CHAMPION shipped -- the tagger "
                         "must be conditioned on exactly what it was "
                         "conditioned on at submission time")
    a_.add_argument("--split", default="oof", choices=["oof", "test"])
    a_.add_argument("--test", default="data/test.csv")
    a_.add_argument("--folds", default="0,1,2,3,4")
    args = a_.parse_args()

    import torch
    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")
    folds = [int(x) for x in args.folds.split(",")]
    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    os.makedirs(args.outdir, exist_ok=True)
    print(f"length cap {args.max_tokens} (was 12 hardcoded), "
          f"soft taus {taus or 'disabled'}\n")

    for tag in [t.strip() for t in args.tags.split(",")]:
        nm = f"{tag}{args.suffix}"
        dst = (f"{args.outdir}/{nm}_oof_candidates.csv" if args.split == "oof"
               else f"{args.outdir}/{nm}_test_candidates.csv")
        if os.path.exists(dst):
            print(f"--- {dst} exists, skipping ---")
            continue
        rows = []
        for f in folds:
            d = args.runs.format(tag=tag, f=f)
            cfgp = f"{d}/run_config.json"
            if not os.path.exists(cfgp):
                raise SystemExit(f"missing {cfgp}")
            cfg = json.load(open(cfgp))
            ml = int(cfg.get("max_len", 512))
            base = cfg.get("model", DEFAULT_MODEL)
            if args.split == "oof":
                df = pd.read_csv(f"artifacts/folds/val_f{f}.csv")
            else:
                df = pd.read_csv(args.test)
            tok = AutoTokenizer.from_pretrained(base)
            m = TaggerModel(base)
            m.load_state_dict(torch.load(f"{d}/model_final.pt",
                                         map_location="cpu"))
            m.to(dev)
            rk = None
            if args.risk_cond:
                rs = pd.read_csv(args.risks_test if args.split == "test"
                                 else args.risks_oof, keep_default_na=False)
                rs["row_id"] = rs["row_id"].astype(str)
                rcol = next(c for c in ("pred_risk_level", "risk_level")
                            if c in rs.columns)
                rmap = dict(zip(rs["row_id"], rs[rcol]))
                ids = [str(x) for x in df["row_id"]]
                miss = [r for r in ids if r not in rmap]
                if miss:
                    raise SystemExit(
                        f"{len(miss)} rows have no risk label to condition on "
                        f"-- a conditioned tagger CANNOT be run without one")
                rk = [rmap[r] for r in ids]
            cands = collect_with_cap(m, tok, [str(x) for x in df["post"]],
                                     ml, dev, args.max_tokens, taus, rk)
            mc = float(cfg.get("evidence_min_conf", 0.0))
            long_n = soft_n = 0
            for i, rid in enumerate(df["row_id"]):
                for s, c, src in cands[i]:
                    tl = len(str(s).split())
                    long_n += tl > 12
                    soft_n += src != "argmax"
                    rows.append((rid, s, float(c), mc, tl, src))
            del m
            torch.cuda.empty_cache()
            print(f"  {tag} fold {f}: {len(df)} posts, "
                  f"{sum(len(c) for c in cands)} candidates  "
                  f"({long_n} over 12 tokens, {soft_n} only the soft decode "
                  f"found)", flush=True)
        out = pd.DataFrame(rows, columns=["row_id", "span", "conf",
                                          "min_conf", "toks", "src"])
        out.to_csv(dst, index=False)
        over = int((out["toks"] > 12).sum())
        soft = int((out["src"] != "argmax").sum())
        print(f"wrote {dst}  {len(out)} candidates: {over} over 12 tokens "
              f"({over/max(len(out),1):.1%}), {soft} from the soft decode "
              f"({soft/max(len(out),1):.1%}) -- both previously impossible\n",
              flush=True)

    print("Next, on CPU -- sweep the cap that used to be hardcoded:")
    print("  python src/evidence_matchprob.py --suffix _long --pool-only")
    print("  python src/evidence_matchprob.py --suffix _long --max-tokens 24")
    print("\nRead the ORACLE CEILING and the unreachable-post count. If they")
    print("move the way removing min_conf did (0.8894 -> 0.9279, 92 -> 78),")
    print("the pool was starved by two gates rather than by the tagger.")


if __name__ == "__main__":
    main()
