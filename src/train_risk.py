"""
Risk-level classifier: encoder + dual head (categorical softmax + ordinal CORN).

Rationale (from 2024/2025 editions of this cup):
  - Fine-tuned encoders beat prompted LLMs by ~15 wF1 points on this task.
  - The labels are an ordinal severity ladder (indicator < ideation <
    behavior < attempt); the 2025 edition's strong entry paired an ordinal
    head with a categorical head and combined them.

Model: --model mental/mental-roberta-base (domain-pretrained, try first)
       or microsoft/deberta-v3-large (bigger general encoder).

User context: optionally prepends compact user-history metadata to the text
("[USER_POST k/N]"). The 2025 ablations found user-sequence signal was the
single most valuable non-text feature; this is the cheapest way to inject it.

Usage (single H100, bf16):
  python src/train_risk.py --train artifacts/train_split.csv \
      --val artifacts/val_split.csv --model microsoft/deberta-v3-large \
      --outdir runs/risk_deberta_v1
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")          # no hub calls; cache only (override: HF_HUB_OFFLINE=0)
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1

LABEL2ID = {l: i for i, l in enumerate(RISK_LEVELS)}


def build_text(row: pd.Series, use_user_context: bool) -> str:
    if not use_user_context:
        return str(row["post"])
    k = row.get("post_index_in_user", "?")
    n = row.get("user_post_count", "?")
    return f"[USER_POST {k}/{n}] {row['post']}"


class PostDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int,
                 use_user_context: bool, weights: dict | None = None):
        df = df.copy()
        if "anon_user_id" in df:
            df["user_post_count"] = df.groupby("anon_user_id")["post"].transform("size")
            df["post_index_in_user"] = df.groupby("anon_user_id").cumcount() + 1
        self.texts = [build_text(r, use_user_context) for _, r in df.iterrows()]
        self.labels = [LABEL2ID[normalize_label(x)] for x in df["risk_level"]]
        # Per-example weights for hard-example mining. None -> every batch omits
        # `sample_weight` entirely and the plain unweighted loss path is used.
        self.weights = ([float(weights.get(str(r), 1.0)) for r in df["row_id"]]
                        if weights is not None else None)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i]),
            "idx": torch.tensor(i),
        }
        if self.weights is not None:
            out["sample_weight"] = torch.tensor(self.weights[i],
                                                dtype=torch.float)
        return out


class DualHeadModel(nn.Module):
    """Categorical head + CORN ordinal head over a shared encoder.

    CORN (Conditional Ordinal Regression): K-1 binary logits, logit k
    predicts P(y > k | y > k-1). Rank-consistent, simple to train.
    """

    def __init__(self, model_name: str, n_classes: int = 4,
                 alpha_ordinal: float = 0.5, class_weights=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.cat_head = nn.Linear(h, n_classes)
        self.ord_head = nn.Linear(h, n_classes - 1)
        self.alpha = alpha_ordinal
        self.n_classes = n_classes
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None,
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                sample_weight=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        # mixed-precision safety: encoder may emit fp16/bf16 while the heads
        # are fp32 and not wrapped by autocast on all stacks; cast explicitly
        pooled = pooled.to(self.cat_head.weight.dtype)
        cat_logits = self.cat_head(pooled)
        ord_logits = self.ord_head(pooled)

        loss = None
        if labels is not None:
            if sample_weight is None:
                ce = nn.CrossEntropyLoss(weight=self.class_weights)(cat_logits,
                                                                    labels)
                corn = self._corn_loss(ord_logits, labels)
            else:
                # PER-EXAMPLE weights for hard-example mining. Both terms are
                # weighted -- weighting only the CE would dilute the effect,
                # since alpha=0.5 makes CORN half the objective.
                w = sample_weight.to(cat_logits.dtype)
                ce_i = nn.CrossEntropyLoss(weight=self.class_weights,
                                           reduction="none")(cat_logits, labels)
                ce = (ce_i * w).sum() / w.sum().clamp_min(1e-8)
                corn = self._corn_loss(ord_logits, labels, w)
            loss = (1 - self.alpha) * ce + self.alpha * corn
        # RISK_CAT_ONLY=1 returns the CATEGORICAL logits instead of the 50/50
        # mixture. _combine() hardcodes that equal average regardless of
        # --alpha-ordinal, which weights only the LOSS, so the inference-time
        # mixture is not a tuned quantity. The two heads are not symmetric
        # either: under CORN, P(y = attempt) is the product of all three
        # conditionals, so the ordinal head is structurally shy of the top class.
        #
        # No retraining is needed: cat_head is trained in every checkpoint. Set
        # this at INFERENCE only -- during training it would also change which
        # epoch validation selects, which is a different configuration.
        #
        # predict.py applies softmax to whatever comes back. softmax(cat_logits)
        # = cat_p, and softmax(_combine(...)) = the mixture, because _combine
        # returns the log of a distribution that already sums to 1. So both
        # paths stay correct probabilities.
        if os.environ.get("RISK_CAT_ONLY") == "1":
            return {"loss": loss, "logits": cat_logits}
        return {"loss": loss, "logits": self._combine(cat_logits, ord_logits)}

    def _corn_loss(self, ord_logits, labels, w=None):
        """Conditional training-set losses per CORN formulation.

        `w` optionally supplies per-example weights. Without it the sum is
        unweighted and the denominator is the plain count of conditional
        terms, so the loss is the ordinary CORN mean."""
        total, n_terms = 0.0, 0.0
        for k in range(self.n_classes - 1):
            mask = labels >= k  # conditional subset: y > k-1
            if mask.sum() == 0:
                continue
            target = (labels[mask] > k).float()
            bce = nn.functional.binary_cross_entropy_with_logits(
                ord_logits[mask, k], target, reduction="none")
            if w is None:
                total = total + bce.sum()
                n_terms += float(mask.sum().item())
            else:
                total = total + (bce * w[mask]).sum()
                n_terms += float(w[mask].sum().item())
        return total / max(n_terms, 1.0)

    def _combine(self, cat_logits, ord_logits):
        """Average categorical probs with ordinal-derived class probs."""
        cat_p = torch.softmax(cat_logits, dim=-1)
        cond = torch.sigmoid(ord_logits)                       # P(y>k | y>k-1)
        cum = torch.cumprod(cond, dim=-1)                      # P(y>k)
        ones = torch.ones_like(cum[:, :1])
        upper = torch.cat([ones, cum], dim=-1)                 # P(y>=k)
        lower = torch.cat([cum, torch.zeros_like(cum[:, :1])], dim=-1)
        ord_p = (upper - lower).clamp_min(1e-8)
        return torch.log(0.5 * cat_p + 0.5 * ord_p)


@torch.no_grad()
def evaluate(model, loader, device, use_bf16):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        idx = batch.pop("idx")
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=use_bf16):
            logits = model(**batch)["logits"]
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    gold = [RISK_LEVELS[i] for i in labels.tolist()]
    pred = [RISK_LEVELS[i] for i in logits.argmax(-1).tolist()]
    return risk_weighted_f1(gold, pred), logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--alpha-ordinal", type=float, default=0.5)
    ap.add_argument("--class-weighting", action="store_true")
    ap.add_argument("--sample-weights", default=None,
                    help="CSV of row_id,weight from src/make_hard_weights.py. "
                         "Applied to TRAINING only; validation stays unweighted "
                         "so epoch selection is not gamed.")
    ap.add_argument("--user-context", action="store_true", default=True)
    ap.add_argument("--no-val", action="store_true",
                    help="full-data mode: no eval, save last epoch")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="keep 0: transformers 5.9 tokenizer misbehaves in "
                         "forked dataloader workers (suspected)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = (args.precision == "bf16" and device == "cuda"
                and torch.cuda.is_bf16_supported())
    print(f"[precision] {'bf16 autocast' if use_bf16 else 'fp32'} on {device}")
    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tr_df, va_df = pd.read_csv(args.train), pd.read_csv(args.val)

    weights = None
    if args.class_weighting:
        counts = tr_df["risk_level"].map(normalize_label).value_counts()
        weights = [len(tr_df) / (len(RISK_LEVELS) * counts.get(l, 1))
                   for l in RISK_LEVELS]

    # Hard-example weights, if supplied. TRAINING ONLY -- the val loader never
    # gets them, so epoch selection stays on the plain objective and cannot be
    # gamed by the weighting.
    sw = None
    if getattr(args, "sample_weights", None):
        wdf = pd.read_csv(args.sample_weights)
        sw = {str(r): float(w) for r, w in zip(wdf["row_id"], wdf["weight"])}
        have = [str(r) for r in tr_df["row_id"] if str(r) in sw]
        vals = np.array([sw[r] for r in have])
        print(f"sample weights: {args.sample_weights}   "
              f"{len(have)}/{len(tr_df)} matched   "
              f"mean {vals.mean():.3f}   "
              f"{int((vals > 1.0).sum())} upweighted   max {vals.max():.1f}")
        if len(have) < 0.9 * len(tr_df):
            raise SystemExit("under 90% of training posts matched a weight -- "
                             "wrong file or wrong split?")

    tr_loader = DataLoader(PostDataset(tr_df, tok, args.max_len,
                                       args.user_context, weights=sw),
                           batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    va_loader = DataLoader(PostDataset(va_df, tok, args.max_len, args.user_context),
                           batch_size=args.batch * 2, shuffle=False, num_workers=args.num_workers)

    model = DualHeadModel(args.model, alpha_ordinal=args.alpha_ordinal,
                          class_weights=weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(tr_loader) // args.grad_accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    best_f1, best_state, step = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        opt.zero_grad()
        for b, batch in enumerate(tr_loader):
            idx = batch.pop("idx")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=use_bf16):
                loss = model(**batch)["loss"]

            # NaN tripwire: catch the FIRST bad loss with the exact rows
            if not torch.isfinite(loss):
                rows = tr_df.iloc[idx.tolist()]
                print(f"\nNON-FINITE LOSS at epoch {epoch} batch {b}")
                print("row_ids in batch:", rows.get("row_id", rows.index).tolist())
                ii = batch["input_ids"]
                emb_rows = model.encoder.get_input_embeddings().weight.shape[0]
                print(f"input_ids range in failing batch: [{ii.min().item()}, "
                      f"{ii.max().item()}] vs embedding rows {emb_rows} "
                      f"{'<-- GARBAGE IDS FROM WORKERS' if (ii.max().item() >= emb_rows or ii.min().item() < 0) else '(ids look valid)'}")
                rows.to_csv(os.path.join(args.outdir, "poison_batch.csv"), index=False)
                sys.exit("aborted; poison_batch.csv written — send it for analysis")

            (loss / args.grad_accum).backward()
            running.append(loss.item())
            if (b + 1) % args.grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gn):
                    print(f"\nNON-FINITE GRAD NORM at epoch {epoch} batch {b}")
                    sys.exit("aborted at gradient step")
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step} "
                          f"loss {np.mean(running[-40:]):.4f} grad_norm {gn:.2f} "
                          f"lr {sched.get_last_lr()[0]:.2e}")

        if args.no_val:
            print(f"== epoch {epoch}: (full-data mode, no eval)")
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_f1 = -1.0
        else:
            f1, _ = evaluate(model, va_loader, device, use_bf16)
            print(f"== epoch {epoch}: val weighted_f1 = {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    f1, logits = evaluate(model, va_loader, device, use_bf16)
    print(f"\nbest val weighted_f1 = {f1:.4f}")

    va_out = va_df.copy()
    va_out["pred_risk_level"] = [RISK_LEVELS[i] for i in logits.argmax(-1).tolist()]
    probs = torch.softmax(logits, -1)
    for i, l in enumerate(RISK_LEVELS):
        va_out[f"prob_{l}"] = probs[:, i].numpy()
    va_out.to_csv(os.path.join(args.outdir, "val_predictions.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model_final.pt"))
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "best_val_weighted_f1": float(best_f1)}, f, indent=2)


if __name__ == "__main__":
    main()
