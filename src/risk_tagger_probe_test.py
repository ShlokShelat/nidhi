"""
risk_tagger_probe_test.py

Build the test-side severity dump the risk-severity nudge consumes.

THE KNOB. `champion_risk_inject.py --severity <csv> --w <x>` tilts the risk
logits toward higher severity in proportion to the evidence taggers' opinion,
applied in LOG space before the argmax so the rules and route_attempt still run
on top:

    Mn = log(M) + w * outer(s, [0,1,2,3]) / 3

The taggers hold a 4-level severity meter -- forced to `indicator` they emit
0.54 spans/post, forced to `ideation`, 1.29, on identical text. The signal is
too weak to decide a post on its own, but it is genuinely independent of the
risk model, which makes it useful for breaking ties the risk model is already
unsure about. `--w` defaults to 0.0, so the nudge is off unless requested.

WHY THIS SCRIPT EXISTS. `risk_tagger_probe.py` runs each fold's tagger over
that fold's HELD-OUT slice, which is the right construction for OOF and
produces `artifacts/risk_probe.csv` (1635 labelled rows, no test rows). For test
there is no held-out fold, so every fold model scores every post and the
confidences are averaged -- the standard bag, and the same construction every
other test artifact in this repo uses.

REGIME CAVEAT. The OOF dump is SINGLE-MODEL per row while this dump is a
5-FOLD BAG, so an offline number measured on the OOF dump is an upper bound for
this one. Judge the nudge by how many labels it moves rather than by the
offline delta.

SECURITY: reads only local files, runs only local checkpoints, no network.

Usage (GPU, ~30 min for 5 folds x 378 posts x 4 conditionings):
  python src/risk_tagger_probe_test.py --out artifacts/risk_probe_test.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_audit import load as audit_load  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
METHOD = re.compile(
    r"\b(pills?|overdose|od|hang|hung|noose|rope|jump|bridge|gun|shoot|"
    r"knife|cut|blade|wrist|bleach|poison|carbon monoxide|train|drown)\b",
    re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/evcond5_cond_f{f}")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default="artifacts/risk_probe_test.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from train_evidence import RISK_PREFIX, TaggerModel, _decode_candidates

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")

    missing = [args.runs.format(f=f) for f in range(args.folds)
               if not os.path.exists(os.path.join(args.runs.format(f=f),
                                                  "model_final.pt"))]
    if missing:
        raise SystemExit(
            f"REFUSING: {len(missing)} of {args.folds} fold checkpoints "
            f"missing: {missing[:3]}. A partial bag is a DIFFERENT model, so "
            f"its severity scores are not on the calibrated scale.")

    # canonical text, exactly as the taggers were trained
    df = audit_load(args.test)
    df["row_id"] = df["row_id"].astype(str)
    tcol = next(c for c in ("clean_text", "post", "text", "body")
                if c in df.columns)
    if args.limit:
        df = df.head(args.limit)
    rows = df["row_id"].tolist()
    posts = [str(x) for x in df[tcol]]
    print(f"{len(rows)} test posts x 4 conditionings x {args.folds} folds")

    # accumulate per (row, level) across folds, then average
    acc = {r: {lv: {"conf": [], "n": [], "meth": []} for lv in LEVELS}
           for r in rows}

    for f in range(args.folds):
        d = args.runs.format(f=f)
        cfg = json.load(open(os.path.join(d, "run_config.json")))
        ml = int(cfg.get("max_len", 512))
        base = cfg.get("model", "microsoft/deberta-v3-large")
        tok = AutoTokenizer.from_pretrained(base)
        m = TaggerModel(base)
        m.load_state_dict(torch.load(os.path.join(d, "model_final.pt"),
                                     map_location="cpu"))
        m.to(dev).eval()
        with torch.no_grad():
            for rid, post in zip(rows, posts):
                for lv in LEVELS:
                    enc = tok(RISK_PREFIX.format(lv), post,
                              truncation="only_second", max_length=ml,
                              return_offsets_mapping=True,
                              return_tensors="pt")
                    seq = enc.sequence_ids(0)
                    off = enc.pop("offset_mapping").squeeze(0).tolist()
                    e = {k: v.to(dev) for k, v in enc.items()
                         if k in ("input_ids", "attention_mask")}
                    pr = torch.softmax(m(**e)["logits"].squeeze(0).float(),
                                       -1).cpu().numpy()
                    keep = [t for t in range(len(off)) if seq[t] == 1]
                    cands = _decode_candidates(post, pr[keep],
                                               [off[t] for t in keep])
                    confs = [c for _, c in cands]
                    txt = " ... ".join(s for s, _ in cands)
                    a = acc[rid][lv]
                    a["conf"].append(max(confs) if confs else 0.0)
                    a["n"].append(len(cands))
                    a["meth"].append(int(bool(METHOD.search(txt))))
        del m
        torch.cuda.empty_cache()
        print(f"  fold {f} done", flush=True)

    out = []
    for r in rows:
        rec = {"row_id": r}
        for lv in LEVELS:
            a = acc[r][lv]
            rec[f"conf_{lv}"] = float(np.mean(a["conf"]))
            rec[f"n_{lv}"] = float(np.mean(a["n"]))
            rec[f"meth_{lv}"] = int(np.mean(a["meth"]) >= 0.5)
        out.append(rec)
    o = pd.DataFrame(out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    o.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  {o.shape}")
    print("  mean conf per level: "
          + "  ".join(f"{lv} {o[f'conf_{lv}'].mean():.3f}" for lv in LEVELS))
    print("  mean spans per level: "
          + "  ".join(f"{lv} {o[f'n_{lv}'].mean():.2f}" for lv in LEVELS))
    print("\n  SANITY: spans/post should RISE from indicator to ideation")
    print("  (0.54 -> 1.29 on OOF). If it is flat, the conditioning prefix")
    print("  never reached the model and the dump is worthless.")


if __name__ == "__main__":
    main()
