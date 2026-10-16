"""
train_risk_boundary.py

A dedicated binary model for the single costliest decision in the risk chain:
does this post describe a PLAN or a SELF-HARM ACT?

WHY THIS BOUNDARY. Measured on the deployed chain's 1590 out-of-fold posts
(`src/risk_error_stats.py`):

    class       gold  pred   prec    rec     F1    weight   loss
    indicator    593   562  0.897  0.850  0.873    0.373  0.0475
    ideation     499   614  0.684  0.842  0.755    0.314  0.0770
    behavior     388   330  0.782  0.665  0.719    0.244  0.0687
    attempt      110    84  0.774  0.591  0.670    0.069  0.0228

`behavior -> ideation` alone is **102 errors, 30% of all 343**. Behavior recall
is 0.665 and ideation precision 0.684 -- ideation is absorbing a third of the
behavior posts. Those two classes carry 0.146 of the 0.216 total loss (68%),
while `attempt` carries only 0.023.

The error is charged TWICE: once against behavior's recall, once against
ideation's precision, so even a partial fix carries disproportionate weight
in the composite score.

WHY A NEW MODEL AND NOT A NEW DECISION RULE. Shifting the operating point
toward behavior was measured as flat-to-negative over 50 author splits. The
class marginals are already correct (the probe intervals confirm this on
test). The
model RANKS those 102 posts wrong, and no threshold, prior or reweighting
reaches a ranking error. New information is required.

WHY THIS IS NOT THE ORDINAL HEAD THAT WAS HARMFUL. The CORN head shared an
encoder with the categorical head and was then mixed 50/50 in `_combine()`
regardless of `--alpha-ordinal`. This is a SEPARATE model trained on ONE
question, entering the chain as a member rather than as half of a hardcoded
mixture.

THE SUPERVISION ARGUMENT. The official definitions make the boundary explicit:
ideation is intent WITHOUT a plan; behavior is a plan OR self-harm. The 4-way
model must infer that distinction as a byproduct of a softmax over four
classes. A binary model gets all 1635 posts as direct supervision for exactly
that question -- 498 positive (behavior + attempt), 1137 negative -- which is
far denser than what the 4-way objective provides for it.

OUTPUT. Per-fold out-of-fold P(plan-or-self-harm), written in the same shape as
the other risk OOF dumps so the honest 50-split instrument can judge it, plus
a test-set column. It ships, if at all, as a MEMBER -- never as a re-decider on
its own, because a member's value measured on `champion_oof.csv` alone has
previously been overstated relative to the deployed evaluation.

Usage:
  python src/train_risk_boundary.py --fold 0 --outdir runs/risk_bnd_f0
  python src/train_risk_boundary.py --predict --runs runs/risk_bnd_f0,... \
      --test data/test.csv --out artifacts/risk_boundary_test.csv
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label  # noqa: E402

POSITIVE = {"behavior", "attempt"}   # a plan, a method, preparation, or an act


def label_of(risk):
    return int(normalize_label(risk) in POSITIVE)


def build(df, tok, max_len):
    import torch
    x = tok([str(p) for p in df["post"]], truncation=True,
            max_length=max_len, padding="max_length", return_tensors="pt")
    y = torch.tensor([label_of(r) for r in df["risk_level"]],
                     dtype=torch.float)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--train", default=None)
    ap.add_argument("--val", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default="artifacts/risk_boundary_test.csv")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, get_linear_schedule_with_warmup)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")

    # ------------------------------------------------------- inference
    if args.predict:
        if not args.runs:
            raise SystemExit("--predict needs --runs")
        te = pd.read_csv(args.test)
        acc = []
        for rd in [r.strip() for r in args.runs.split(",")]:
            cfg = json.load(open(os.path.join(rd, "run_config.json")))
            tok = AutoTokenizer.from_pretrained(cfg["model"])
            m = AutoModelForSequenceClassification.from_pretrained(
                cfg["model"], num_labels=1).to(dev)
            m.load_state_dict(torch.load(os.path.join(rd, "model.pt"),
                                         map_location=dev))
            m.eval()
            x, _ = build(te.assign(risk_level="indicator"), tok,
                         int(cfg["max_len"]))
            out = []
            with torch.no_grad():
                for i in range(0, len(te), 32):
                    b = {k: v[i:i + 32].to(dev) for k, v in x.items()}
                    out.append(torch.sigmoid(
                        m(**b).logits.squeeze(-1)).cpu().numpy())
            acc.append(np.concatenate(out))
            print(f"  scored {rd}")
            del m
            torch.cuda.empty_cache()
        p = np.mean(acc, axis=0)
        pd.DataFrame({"row_id": te["row_id"].astype(str),
                      "p_plan_or_act": p}).to_csv(args.out, index=False)
        print(f"wrote {args.out}   mean {p.mean():.3f}")
        return

    # ------------------------------------------------------- training
    if args.fold is not None:
        tr_p = f"artifacts/folds/train_f{args.fold}.csv"
        va_p = f"artifacts/folds/val_f{args.fold}.csv"
        out = args.outdir or f"runs/risk_bnd_f{args.fold}"
    else:
        tr_p, va_p, out = args.train, args.val, args.outdir
    if not (tr_p and va_p and out):
        raise SystemExit("need --fold, or --train/--val/--outdir")
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tr, va = pd.read_csv(tr_p), pd.read_csv(va_p)
    tok = AutoTokenizer.from_pretrained(args.model)
    xtr, ytr = build(tr, tok, args.max_len)
    xva, yva = build(va, tok, args.max_len)
    print(f"train {len(tr)} ({int(ytr.sum())} plan-or-act, "
          f"{100*float(ytr.mean()):.1f}%)   val {len(va)} "
          f"({int(yva.sum())} positive)")

    m = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1).to(dev)
    dl = DataLoader(TensorDataset(xtr["input_ids"], xtr["attention_mask"], ytr),
                    batch_size=args.batch, shuffle=True)
    steps = (len(dl) // args.grad_accum + 1) * args.epochs
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    use_bf16 = torch.cuda.is_bf16_supported()

    def val_probs():
        m.eval()
        o = []
        with torch.no_grad():
            for i in range(0, len(va), 32):
                b = {"input_ids": xva["input_ids"][i:i + 32].to(dev),
                     "attention_mask": xva["attention_mask"][i:i + 32].to(dev)}
                o.append(torch.sigmoid(
                    m(**b).logits.squeeze(-1).float()).cpu().numpy())
        m.train()
        return np.concatenate(o)

    from sklearn.metrics import average_precision_score as APS
    # FIXED epoch count, no best-on-val selection. Every other trainer here
    # selects the best epoch by val score, which is what makes the OOF
    # instrument optimistic (a factor bag's OOF was measured over-predicting
    # held-out evaluation by roughly 4.6x). The dumps below are meant to be
    # honest.
    for ep in range(args.epochs):
        m.train()
        opt.zero_grad()
        for s, (ii, am, yy) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                lg = m(input_ids=ii.to(dev),
                       attention_mask=am.to(dev)).logits.squeeze(-1).float()
            loss = lossf(lg, yy.to(dev))
            (loss / args.grad_accum).backward()
            if (s + 1) % args.grad_accum == 0:
                opt.step()
                sch.step()
                opt.zero_grad()
        p = val_probs()
        print(f"== epoch {ep+1}: val AP = {APS(yva.numpy(), p):.4f}  "
              f"(base rate {float(yva.mean()):.3f})")

    p = val_probs()
    torch.save(m.state_dict(), os.path.join(out, "model.pt"))
    pd.DataFrame({"row_id": va["row_id"].astype(str),
                  "p_plan_or_act": p,
                  "gold": [label_of(r) for r in va["risk_level"]]}).to_csv(
        os.path.join(out, "oof_boundary.csv"), index=False)
    json.dump({**vars(args), "final_val_ap": float(APS(yva.numpy(), p))},
              open(os.path.join(out, "run_config.json"), "w"), indent=2)
    print(f"\nwrote {out}/model.pt and oof_boundary.csv")
    print("LAST epoch saved deliberately -- no val selection.")


if __name__ == "__main__":
    main()
