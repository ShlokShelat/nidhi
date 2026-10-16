"""
evidence_conditioning_oracle.py

Measure what conditioning the evidence tagger on the WRONG risk level costs.

The pipeline is one-directional: risk -> evidence. The tagger is TOLD the
predicted risk and asked to justify it. On the posts where risk is wrong it is
therefore asked for evidence of a label that is not present -- conditioned on
a lower severity level it extracts the phrase matching that level while the
gold span marks the higher one, and phrase-F1 on that post is 0. This is a
conditioning failure rather than a tagger failure, and it is charged at the
0.3 evidence weight.

The same four conditionings are a weak instrument for predicting RISK, because
the tagger is the weaker judge of risk than the risk model itself. They are a
different proposition for EVIDENCE, where the tagger is the only judge and the
risk label is just an input it is handed -- sometimes wrongly.

FOUR DECODES, SAME SPANS, MEASURED WITH THE OFFICIAL SCORER:

  predicted   the deployed decode: condition on the predicted risk label
  gold        ORACLE: condition on the true label. The gap between this and
              `predicted` bounds everything a conditioning fix can recover.
  best-conf   condition on whichever label the tagger is most confident about
  weighted    UNION over conditionings, each span kept if its confidence times
              the risk model's probability for that conditioning clears a
              threshold. Marginalises over the risk label instead of
              committing to it, and needs no extra model.

Only `weighted` is deployable: `gold` requires the answer, and `best-conf`
relies on the tagger's own risk ranking, which is weaker than the risk model's.

WHY THE MACRO AVERAGE MATTERS HERE. Evidence is scored MACRO per post, so a
post scoring 0 costs as much as a long post scoring 0.9 -- and the risk-wrong
posts are concentrated 0s. Recovering part of them moves a term carrying 0.3
of the composite.

The gold-empty rule is preserved exactly: indicator posts get empty evidence,
which is the scorer's own convention.

Usage (GPU, ~10 min for 372 val posts x 4 conditionings):
  python src/evidence_conditioning_oracle.py
  python src/evidence_conditioning_oracle.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span,  # noqa: E402
                     parse_evidence, phrase_f1, token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def pf1(gold, pred):
    """phrase_f1 takes GOLD FIRST and returns a PhraseF1Result, not a float.
    Wrapping it once removes the chance of scoring the arguments the wrong way
    round, which produces a plausible-looking number rather than an error."""
    return phrase_f1(gold, pred).f1


def dedupe(spans):
    out = []
    for s in sorted(spans, key=lambda x: (-token_len(x), -len(x))):
        n = normalize_span(s)
        if n and not any(n in normalize_span(k) for k in out):
            out.append(s)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--runs", default="runs/evcond5_cond_f{f}")
    a_.add_argument("--fold", type=int, default=-1,
                    help="-1 = ensemble all 5 fold taggers (val is held out "
                         "of none of them individually, so all 5 are usable)")
    a_.add_argument("--cache", default="artifacts/evcond_val_spans.json")
    a_.add_argument("--limit", type=int, default=0)
    a_.add_argument("--min-conf", type=float, default=0.35)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    if args.limit:
        va = va.head(args.limit)
    rows = va["row_id"].tolist()
    gold_risk = np.array([normalize_label(v) for v in va["risk_level"]])
    gold_ev = [parse_evidence(v) for v in va["evidence"]]
    posts = list(va["post"].astype(str))

    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pv = pv.set_index("row_id")
    P = pv.loc[rows, [f"prob_{c}" for c in LEVELS]].to_numpy(float)
    pred_risk = np.array([normalize_label(v)
                          for v in pv.loc[rows, "pred_risk_level"]])

    # ---- spans per (post, conditioning) --------------------------------
    if os.path.exists(args.cache):
        S = json.load(open(args.cache))
        print(f"reusing {args.cache}")
    else:
        import torch
        from transformers import AutoTokenizer

        from train_evidence import (RISK_PREFIX, TaggerModel,  # noqa: E402
                                    _decode_candidates)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            raise SystemExit("no GPU -- submit via sbatch")
        folds = range(5) if args.fold < 0 else [args.fold]
        S = {r: {c: [] for c in LEVELS} for r in rows}
        for f in folds:
            d = args.runs.format(f=f)
            cfg = json.load(open(f"{d}/run_config.json"))
            ml, base = int(cfg.get("max_len", 512)), cfg.get(
                "model", "microsoft/deberta-v3-large")
            tok = AutoTokenizer.from_pretrained(base)
            m = TaggerModel(base)
            m.load_state_dict(torch.load(f"{d}/model_final.pt",
                                         map_location="cpu"))
            m.to(dev).eval()
            with torch.no_grad():
                for r, post in zip(rows, posts):
                    for lv in LEVELS:
                        enc = tok(RISK_PREFIX.format(lv), post,
                                  truncation="only_second", max_length=ml,
                                  return_offsets_mapping=True,
                                  return_tensors="pt")
                        seq = enc.sequence_ids(0)
                        off = enc.pop("offset_mapping").squeeze(0).tolist()
                        e = {k: v.to(dev) for k, v in enc.items()
                             if k in ("input_ids", "attention_mask")}
                        pr = torch.softmax(
                            m(**e)["logits"].squeeze(0).float(), -1).cpu().numpy()
                        keep = [t for t in range(len(off)) if seq[t] == 1]
                        for sp, cf in _decode_candidates(
                                post, pr[keep], [off[t] for t in keep]):
                            S[r][lv].append([sp, float(cf)])
            del m
            torch.cuda.empty_cache()
            print(f"  fold {f} done", flush=True)
        json.dump(S, open(args.cache, "w"))
        print(f"wrote {args.cache}")

    # ---- four decodes ---------------------------------------------------
    def pick(r, lv):
        return dedupe([s for s, c in S[r][lv] if c >= args.min_conf])

    def weighted(r, i):
        """Marginalise over the risk label: keep a span when its confidence
        times P(that label) clears the same bar, so a span the tagger is sure
        of under a label the risk model half-believes still survives."""
        acc = {}
        for j, lv in enumerate(LEVELS):
            for s, c in S[r][lv]:
                k = normalize_span(s)
                acc[k] = max(acc.get(k, 0.0), c * P[i, j])

        keep = [s for lv in LEVELS for s, c in S[r][lv]
                if acc.get(normalize_span(s), 0) >= args.min_conf * 0.5]
        return dedupe(keep)

    decodes = {}
    for nm in ("predicted", "gold", "best-conf", "weighted"):
        out = []
        for i, r in enumerate(rows):
            # indicator posts emit EMPTY evidence -- the scorer's convention.
            # Preserved in every decode so the comparison isolates
            # conditioning, not suppression.
            if nm == "gold":
                lv, supp = gold_risk[i], gold_risk[i] == "indicator"
            else:
                supp = pred_risk[i] == "indicator"
                if nm == "predicted":
                    lv = pred_risk[i]
                elif nm == "best-conf":
                    lv = LEVELS[int(np.argmax(
                        [max([c for _, c in S[r][x]], default=0.0)
                         for x in LEVELS]))]
                else:
                    lv = None
            out.append([] if supp else
                       (weighted(r, i) if lv is None else pick(r, lv)))
        decodes[nm] = out

    ok = pred_risk == gold_risk
    print(f"\n{len(rows)} val posts   risk correct on {ok.mean():.1%}\n")
    print(f"  {'decode':<12}{'ALL':>9}{'risk RIGHT':>12}{'risk WRONG':>12}"
          f"{'n wrong':>9}")
    iok, ibad = np.flatnonzero(ok), np.flatnonzero(~ok)
    for nm, sp in decodes.items():
        a = pf1(gold_ev, sp)
        b = pf1([gold_ev[i] for i in iok], [sp[i] for i in iok])
        c = pf1([gold_ev[i] for i in ibad], [sp[i] for i in ibad])
        star = "  <- deployed decode" if nm == "predicted" else (
            "  <- ORACLE, not shippable" if nm == "gold" else "")
        print(f"  {nm:<12}{a:>9.4f}{b:>12.4f}{c:>12.4f}"
              f"{len(ibad):>9}{star}")

    g, p = decodes["gold"], decodes["predicted"]
    gap = pf1(gold_ev, g) - pf1(gold_ev, p)
    print(f"\n  ORACLE GAP: {gap:+.4f} evidence  =  {0.3*gap:+.4f} composite")
    print("  That bounds everything a conditioning fix can recover. Under")
    print("  ~0.01 composite the lane does not repay the plumbing; above it,")
    print("  `weighted` is the deployable share.")
    for nm in ("weighted", "best-conf"):
        d = pf1(gold_ev, decodes[nm]) - pf1(gold_ev, p)
        share = f" ({d/gap:.0%} of the oracle)" if abs(gap) > 1e-9 else ""
        print(f"  {nm:<10} {0.3*d:+.4f} composite vs the deployed decode"
              f"{share}")
    print("\n  Read the `risk WRONG` column: `predicted` collapsing there")
    print("  while `gold` holds up attributes the loss to the conditioning")
    print("  rather than to those posts being intrinsically hard.")


if __name__ == "__main__":
    main()
