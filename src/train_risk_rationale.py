"""
train_risk_rationale.py

Risk classifier with RATIONALE SUPERVISION: teach the encoder WHERE to look.

WHY. src/risk_head_diagnosis.py showed the risk errors are CONFIDENTLY wrong --
on the ideation<->behavior boundary NEITHER the categorical nor the ordinal head
is right on 38 of 39 val errors, median margin 0.640. That rules out every
decoding fix (blend weights, thresholds, calibration, head mixes): a confidently
wrong model cannot be corrected by re-decoding its existing output. The boundary
needs new SUPERVISION.

WHERE THE SUPERVISION IS. Verified independently on the 39 confident errors:

    classify gold-ideation vs gold-behavior using ...
      the annotator's GOLD EVIDENCE SPAN text     0.692
      majority baseline                           0.538
      the FULL POST text                          0.308   <- worse than chance
      the 24 factor labels (tested separately)    0.56    <- also below majority

The annotator's highlighted words carry the signal. The surrounding post text
ACTIVELY MISLEADS -- its despair vocabulary drowns out the one clause that
decides the label. Behaviour spans name concrete actions and objects (tonight,
jump, note, bridge, pills); ideation spans express abstract desire (want to die,
wish, wanna). On all 188 val ideation/behavior posts the span text reaches 0.840
against a 0.590 majority.

WHY A TRAINING-TIME HEAD AND NOT A TEST-TIME FEATURE. With PREDICTED spans the
same classifier scores only 0.590 on those errors -- the evidence model fails on
the same posts the risk model does. So spans cannot be fed in at inference. But
supervising a span head during TRAINING needs no spans at test time: the encoder
learns to attend to rationale-bearing tokens and carries that forward.

    loss = (1 - alpha) * CE  +  alpha * CORN  +  lambda * spanBCE

lambda is FIXED at 0.5 a priori and never swept -- a single fitted parameter
has historically transferred poorly on this task, and a larger fitted
parameter count has cost real regressions before.

WHAT ELSE THIS FIXES. Trained as a 5-fold AUTHOR-DISJOINT bag over
artifacts/folds/, it also retires two long-standing defects: the shipped seeds
train on only ~1263 posts and burn 372 on epoch selection, and all five shipped
checkpoints pick their epoch on that same 372-post split (risk_v1_s15 ships at
equal weight with a degraded best-val of 0.7129).

HONEST SIZING. Even a full resolution of the 39 confident errors bounds a
modest member-level gain, which is a small composite contribution once the
blend weight is applied. The val evidence is n=39 at p~0.03 -- real but
marginal. Judge the result by replicating the deployed recipe on the pooled
OOF, never by this script's own per-fold numbers.

This does NOT touch src/train_risk.py or any shipped run directory.

Usage (one fold):
  python src/train_risk_rationale.py --fold 0 --outdir runs/risk_rat_f0
Usage (all five): scripts/risk_rationale_5fold.sh
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

from scoring import normalize_label, parse_evidence
from train_risk import LABEL2ID, build_text

LEV = ["indicator", "ideation", "behavior", "attempt"]


def span_char_ranges(post: str, evidence) -> list[tuple[int, int]]:
    """Character ranges of the gold evidence spans inside the post.

    Both sides come through data_audit.load (clean_text), so a direct find
    succeeds for the large majority; a lowercase retry catches the rest.
    Roughly 1.4% of gold spans are unlocatable because the annotator lightly
    paraphrased -- those are simply skipped, which is the honest behaviour:
    an unlocatable rationale supervises nothing rather than supervising noise.
    """
    out = []
    lo = post.lower()
    for sp in parse_evidence(evidence):
        s = str(sp).strip()
        if not s:
            continue
        i = post.find(s)
        if i < 0:
            i = lo.find(s.lower())
        if i >= 0:
            out.append((i, i + len(s)))
    return out


class RationaleDataset:
    """PostDataset plus a token-level rationale mask.

    build_text() may prepend "[USER_POST k/N] ", which shifts every character
    offset; the prefix length is measured per row rather than assumed.
    """

    def __init__(self, df, tokenizer, max_len, use_user_context):
        import torch
        self.torch = torch
        df = df.copy()
        if "anon_user_id" in df:
            df["user_post_count"] = df.groupby("anon_user_id")["post"].transform("size")
            df["post_index_in_user"] = df.groupby("anon_user_id").cumcount() + 1
        self.rows = []
        for _, r in df.iterrows():
            post = str(r["post"])
            text = build_text(r, use_user_context)
            shift = len(text) - len(post)
            if text[shift:] != post:          # defensive: prefix assumption broken
                shift = text.find(post)
                shift = shift if shift >= 0 else 0
            ranges = [(a + shift, b + shift)
                      for a, b in span_char_ranges(post, r.get("evidence", ""))]
            self.rows.append((text, LABEL2ID[normalize_label(r["risk_level"])],
                              ranges))
        self.tok = tokenizer
        self.max_len = max_len
        n_with = sum(1 for _, _, rr in self.rows if rr)
        print(f"  rationale masks: {n_with}/{len(self.rows)} posts have at "
              f"least one locatable gold span")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        text, label, ranges = self.rows[i]
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       padding="max_length", return_offsets_mapping=True,
                       return_tensors="pt")
        offs = enc["offset_mapping"].squeeze(0)
        mask = self.torch.zeros(offs.shape[0], dtype=self.torch.float)
        for a, b in ranges:
            # a token is rationale if its character range overlaps a gold span
            ov = (offs[:, 0] < b) & (offs[:, 1] > a) & (offs[:, 1] > offs[:, 0])
            mask[ov] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": self.torch.tensor(label),
            "span_mask": mask,
        }


def build_model(model_name, alpha_ordinal, lam):
    import torch
    import torch.nn as nn
    from train_risk import DualHeadModel

    class DualHeadRationale(DualHeadModel):
        """DualHeadModel + a token-level rationale head.

        The extra head is discarded at inference; its only job is to force the
        shared encoder to represent which tokens justify the label.
        """

        def __init__(self, name, alpha_ordinal=0.5, lam=0.5):
            super().__init__(name, alpha_ordinal=alpha_ordinal)
            h = self.encoder.config.hidden_size
            self.span_head = nn.Linear(h, 1)
            self.lam = lam

        def forward(self, input_ids=None, attention_mask=None, labels=None,
                    span_mask=None):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            seq = out.last_hidden_state
            pooled = self.dropout(seq[:, 0]).to(self.cat_head.weight.dtype)
            cat_logits = self.cat_head(pooled)
            ord_logits = self.ord_head(pooled)

            loss = None
            if labels is not None:
                ce = nn.CrossEntropyLoss(weight=self.class_weights)(cat_logits, labels)
                loss = (1 - self.alpha) * ce + self.alpha * self._corn_loss(
                    ord_logits, labels)
                if span_mask is not None and self.lam > 0:
                    sl = self.span_head(seq.to(self.span_head.weight.dtype)
                                        ).squeeze(-1)
                    keep = attention_mask.bool()
                    sb = nn.functional.binary_cross_entropy_with_logits(
                        sl[keep], span_mask.to(sl.dtype)[keep])
                    loss = loss + self.lam * sb
            return {"loss": loss,
                    "logits": self._combine(cat_logits, ord_logits)}

    return DualHeadRationale(model_name, alpha_ordinal, lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None,
                    help="use artifacts/folds/{train,val}_f<fold>.csv")
    ap.add_argument("--train", default=None)
    ap.add_argument("--val", default=None)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--alpha-ordinal", type=float, default=0.5)
    ap.add_argument("--lambda-span", type=float, default=0.5,
                    help="FIXED a priori; do not sweep")
    ap.add_argument("--user-context", type=int, default=1)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from sklearn.metrics import f1_score
    from data_audit import load

    if args.fold is not None:
        args.train = args.train or f"artifacts/folds/train_f{args.fold}.csv"
        args.val = args.val or f"artifacts/folds/val_f{args.fold}.csv"
    if not args.train or not args.val:
        raise SystemExit("give --fold or both --train and --val")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")
    use_bf16 = torch.cuda.is_bf16_supported()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    tr, va = load(args.train), load(args.val)
    for d, n in ((tr, "train"), (va, "val")):
        if "evidence" not in d.columns:
            raise SystemExit(f"{n} split has no evidence column -- rationale "
                             "supervision needs the gold spans")
    print(f"train {len(tr)}  val {len(va)}  model {args.model}")
    print(f"alpha_ordinal={args.alpha_ordinal}  lambda_span={args.lambda_span}")

    tok = AutoTokenizer.from_pretrained(args.model)
    uc = bool(args.user_context)
    print(" train:")
    tr_ds = RationaleDataset(tr, tok, args.max_len, uc)
    print(" val:")
    va_ds = RationaleDataset(va, tok, args.max_len, uc)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=args.batch * 2, num_workers=0)

    model = build_model(args.model, args.alpha_ordinal, args.lambda_span).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = (len(tr_dl) // args.grad_accum + 1) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    ygold = np.array([LEV.index(normalize_label(x)) for x in va["risk_level"]])
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        run = []
        for step, b in enumerate(tr_dl):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=use_bf16):
                loss = model(**b)["loss"]
            (loss / args.grad_accum).backward()
            run.append(float(loss))
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
            if (step + 1) % 200 == 0:
                print(f"  epoch {ep+1} step {step+1} loss "
                      f"{np.mean(run[-200:]):.4f}")

        model.eval()
        L = []
        with torch.no_grad():
            for b in va_dl:
                b.pop("labels", None)
                b.pop("span_mask", None)
                b = {k: v.to(device) for k, v in b.items()}
                with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    L.append(model(**b)["logits"].float().cpu().numpy())
        P = np.concatenate(L)
        wf1 = f1_score(ygold, P.argmax(1), average="weighted")
        print(f"== epoch {ep+1}: val weighted-F1 = {wf1:.4f}")
        if wf1 > best:
            best = wf1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            np.savez(os.path.join(args.outdir, "val_probs.npz"),
                     probs=np.exp(P), row_id=np.array(va["row_id"].tolist(),
                                                     dtype=object))

    if best_state:
        torch.save(best_state, os.path.join(args.outdir, "model.pt"))
    json.dump({**vars(args), "best_val_weighted_f1": best},
              open(os.path.join(args.outdir, "run_config.json"), "w"), indent=2)
    print(f"\nbest val weighted-F1 = {best:.4f}")
    print("Compare against the 3-seed mix's 0.7498 on the 372-post val split --")
    print("but this trains on author-disjoint folds, so the honest comparison is")
    print("the pooled OOF via src/champion_oof.py with the deployed recipe.")


if __name__ == "__main__":
    main()
