"""
risk_head_diagnosis.py

Dumps the CATEGORICAL and ORDINAL head probabilities of the risk model
separately, and attributes the risk errors to one head or the other.

WHY THE HEADS ARE INVISIBLE OTHERWISE. DualHeadModel carries two heads over one
encoder (train_risk.py):
    cat_head  Linear(h, 4)   plain categorical softmax
    ord_head  Linear(h, 3)   CORN, logit k = P(y > k | y > k-1)
`_combine()` merges them as log(0.5*cat_p + 0.5*ord_p) -- a HARDCODED 50/50 mix
regardless of --alpha-ordinal, which only shapes the training loss. And
`forward()` returns ONLY the merged result, so no artefact on disk records the
two heads separately. The ordinal head's k=1 logit is a trained "is this worse
than ideation?" gate -- a plan/self-harm discriminator learned from every post
with y >= ideation -- and the merge hides it.

WHAT THIS DECIDES. For the errors that matter, the two heads either DISAGREE
(one of them is right, so a class-conditional mix could recover those posts --
one parameter, to be judged over many author-disjoint splits) or AGREE (both
confidently wrong, so the boundary needs new supervision rather than new
decoding, and decode-side ideas can be dropped).

The distinction is not reachable from the merged probabilities, because a
confident merged margin can come either from agreement or from one head
dominating.

Usage:
  python src/risk_head_diagnosis.py \
      --runs runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
      --data artifacts/val_split.csv --tag val
  python src/risk_head_diagnosis.py \
      --runs runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
      --data artifacts/full_train.csv --tag full
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data_audit import load
from scoring import normalize_label

LEV = ["indicator", "ideation", "behavior", "attempt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="comma-separated run dirs (the 3 shipped seeds)")
    ap.add_argument("--data", default="artifacts/val_split.csv")
    ap.add_argument("--tag", default="val")
    ap.add_argument("--outdir", default="artifacts/risk_heads")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from train_risk import DualHeadModel, PostDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")
    use_bf16 = torch.cuda.is_bf16_supported()
    os.makedirs(args.outdir, exist_ok=True)

    df = load(args.data)
    have_gold = "risk_level" in df.columns
    gold = (np.array([LEV.index(normalize_label(x)) for x in df["risk_level"]])
            if have_gold else None)
    print(f"{len(df)} posts from {args.data}   gold: {have_gold}")
    if not have_gold:
        # PostDataset builds a label tensor unconditionally, and data/test.csv
        # has no risk_level. The value is never read: the batch's "labels" key
        # is popped before the forward pass, and this script computes the two
        # heads directly from the encoder rather than calling model(**batch).
        df = df.copy()
        df["risk_level"] = LEV[0]

    runs = [r.strip() for r in args.runs.split(",")]
    CAT, ORD = [], []
    for rd in runs:
        cfgp = os.path.join(rd, "run_config.json")
        cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
        base = cfg.get("model", "microsoft/deberta-v3-large")
        max_len = int(cfg.get("max_len", 512))
        uc = bool(cfg.get("user_context", True))
        ckpt = next((os.path.join(rd, f) for f in ("model.pt", "model_final.pt")
                     if os.path.exists(os.path.join(rd, f))), None)
        if ckpt is None:
            raise SystemExit(f"no checkpoint in {rd}")
        print(f"\n=== {rd}   base={base} max_len={max_len} user_context={uc} ===")

        tok = AutoTokenizer.from_pretrained(base)
        model = DualHeadModel(base, alpha_ordinal=float(cfg.get("alpha_ordinal", 0.5)))
        state = torch.load(ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  WARNING missing={list(missing)[:4]} "
                  f"unexpected={list(unexpected)[:4]}")
        model.to(device).eval()

        loader = DataLoader(PostDataset(df.copy(), tok, max_len, uc),
                            batch_size=args.batch, shuffle=False)
        cps, ops = [], []
        with torch.no_grad():
            for batch in loader:
                batch.pop("idx", None)
                batch.pop("labels", None)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    out = model.encoder(**batch)
                    pooled = out.last_hidden_state[:, 0]
                pooled = pooled.to(model.cat_head.weight.dtype)
                cat_logits = model.cat_head(pooled).float()
                ord_logits = model.ord_head(pooled).float()
                cat_p = torch.softmax(cat_logits, -1)
                # reproduce _combine()'s CORN chain exactly
                cond = torch.sigmoid(ord_logits)
                cum = torch.cumprod(cond, -1)
                ones = torch.ones_like(cum[:, :1])
                upper = torch.cat([ones, cum], -1)
                lower = torch.cat([cum, torch.zeros_like(cum[:, :1])], -1)
                ord_p = (upper - lower).clamp_min(1e-8)
                cps.append(cat_p.cpu().numpy())
                ops.append(ord_p.cpu().numpy())
        CAT.append(np.concatenate(cps))
        ORD.append(np.concatenate(ops))
        del model
        torch.cuda.empty_cache()

    cat_p = np.mean(CAT, axis=0)
    ord_p = np.mean(ORD, axis=0)
    mix = 0.5 * cat_p + 0.5 * ord_p          # what the pipeline actually uses
    np.savez(os.path.join(args.outdir, f"heads_{args.tag}.npz"),
             cat_p=cat_p, ord_p=ord_p, mix=mix,
             row_id=np.array(df["row_id"].tolist(), dtype=object))
    print(f"\nwrote {args.outdir}/heads_{args.tag}.npz")

    # How many posts change if the ordinal head is dropped from the average?
    # This sets whether the change is resolvable at all: a change touching only
    # a handful of the test posts cannot be distinguished from scoring noise.
    pc0, pm0 = cat_p.argmax(1), mix.argmax(1)
    movers = int((pc0 != pm0).sum())
    print("\n" + "=" * 74)
    print("IF THE ORDINAL HEAD WERE DROPPED AT INFERENCE (categorical only)")
    print("=" * 74)
    print(f"  posts whose predicted class changes: {movers} of {len(df)} "
          f"({100.0*movers/len(df):.1f}%)")
    print(f"  {'from':<12}{'to':<12}{'n':>5}")
    from collections import Counter
    cc = Counter((LEV[a], LEV[b]) for a, b in zip(pm0, pc0) if a != b)
    for (a, b), n in cc.most_common():
        print(f"  {a:<12}{b:<12}{n:>5}")
    print("\n  NOTE this is the DeBERTa member only. It enters the shipped")
    print("  blend at 0.35 effective weight (0.5*deberta + 0.5*friend, then")
    print("  0.70 of that + 0.30 NLI), so the number of posts that change in")
    print("  the FINAL predictions will be smaller still.")

    if not have_gold:
        return

    from sklearn.metrics import f1_score
    pc, po, pm = cat_p.argmax(1), ord_p.argmax(1), mix.argmax(1)
    print("\n" + "=" * 74)
    print("EACH HEAD ALONE vs THE SHIPPED 50/50 MIX")
    print("=" * 74)
    for name, p in (("categorical", pc), ("ordinal (CORN)", po),
                    ("mix 0.5/0.5 (shipped)", pm)):
        print(f"  {name:<24} wF1 {f1_score(gold, p, average='weighted'):.4f}"
              f"   acc {(p == gold).mean():.4f}")

    print("\nper-class F1")
    print(f"  {'class':<12}{'cat':>8}{'ord':>8}{'mix':>8}")
    for k, l in enumerate(LEV):
        f = lambda p: f1_score((gold == k).astype(int), (p == k).astype(int),
                               zero_division=0)
        print(f"  {l:<12}{f(pc):>8.3f}{f(po):>8.3f}{f(pm):>8.3f}")

    print("\n" + "=" * 74)
    print("THE CONFIDENT ERRORS -- does either head know better?")
    print("=" * 74)
    i_id, i_be, i_at = LEV.index("ideation"), LEV.index("behavior"), LEV.index("attempt")
    for aname, a, b in (("ideation<->behavior", i_id, i_be),
                        ("behavior<->attempt", i_be, i_at)):
        m = (((gold == a) & (pm == b)) | ((gold == b) & (pm == a)))
        n = int(m.sum())
        if n == 0:
            print(f"\n  {aname}: no errors")
            continue
        marg = np.abs(mix[m, a] - mix[m, b])
        catright = int((pc[m] == gold[m]).sum())
        ordright = int((po[m] == gold[m]).sum())
        both = int(((pc[m] == gold[m]) & (po[m] == gold[m])).sum())
        print(f"\n  {aname}: {n} errors under the shipped mix")
        print(f"    median |margin| between the two classes : {np.median(marg):.3f}")
        print(f"    categorical head alone gets right       : {catright}/{n}")
        print(f"    ordinal head alone gets right           : {ordright}/{n}")
        print(f"    both heads right (mix still wrong)      : {both}/{n}")
        print(f"    NEITHER head right                      : "
              f"{n - catright - ordright + both}/{n}")

    print("\n" + "=" * 74)
    print("HOW TO READ IT")
    print("=" * 74)
    print("  One head clearly better on these errors -> a class-conditional mix")
    print("  can recover them. That is ONE parameter, so judge it over many")
    print("  author-disjoint splits before it goes anywhere near a submission.")
    print()
    print("  Neither head right on most of them -> the boundary is not a")
    print("  decoding problem. Drop every remaining decode-side idea and spend")
    print("  the effort on new supervision (the gold evidence spans are the")
    print("  only untapped source that is not derived from risk_level itself).")


if __name__ == "__main__":
    main()
