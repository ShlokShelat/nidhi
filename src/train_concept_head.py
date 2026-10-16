"""
train_concept_head.py

STAGE 1 of two-stage training. Trains encoder + concept_head ONLY,
against the concept BCE loss (with per-concept pos_weight). There is
no risk loss, no fusion layer, no risk heads here at all -- the point
is to get concept predictions to converge on their own merits, with
no competing gradient signal that could pull them toward collapse.

Saves a state_dict containing only `encoder.*` and `concept_head.*`
keys. Load it into train_risk_concepts_bottleneck.py via
--init-concept-checkpoint (loaded with strict=False, since that
script's model also has fusion/cat_head/ord_head which this
checkpoint doesn't cover).

Usage:
  python src/train_concept_head.py \
      --train artifacts/train_split.csv --val artifacts/val_split.csv \
      --model microsoft/deberta-v3-large \
      --outdir runs/concept_head_stage1 \
      --concept-labels artifacts/train_concept_labels.csv \
      --val-concept-labels artifacts/val_concept_labels.csv \
      --seed 13
"""
from __future__ import annotations
import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import json
import os
import random
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
sys.path.insert(0, os.path.dirname(__file__))

CONCEPTS = [
    "method_mechanism", "plan_present", "completed_act", "urgency_imminence",
    "speaker_is_author", "farewell_framing", "protective_factor", "negation",
    "past_attempt_history",
]


def build_text(row, use_user_context):
    if not use_user_context:
        return str(row["post"])
    k = row.get("post_index_in_user", "?")
    n = row.get("user_post_count", "?")
    return f"[USER_POST {k}/{n}] {row['post']}"


class ConceptDataset(Dataset):
    """Same text-building convention as PostDataset, but only ever
    needs concept labels -- risk_level is irrelevant here."""
    def __init__(self, df, tokenizer, max_len, use_user_context, concept_df):
        df = df.copy()
        if "anon_user_id" in df:
            df["user_post_count"] = df.groupby("anon_user_id")["post"].transform("size")
            df["post_index_in_user"] = df.groupby("anon_user_id").cumcount() + 1
        self.texts = [build_text(r, use_user_context) for _, r in df.iterrows()]
        merged = df[["row_id"]].merge(concept_df, on="row_id", how="left")
        for c in CONCEPTS:
            merged[c] = merged[c].fillna(False).astype(float)
        self.concepts = merged[CONCEPTS].values
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "concept_labels": torch.tensor(self.concepts[i], dtype=torch.float),
            "idx": torch.tensor(i),
        }


class ConceptOnlyModel(nn.Module):
    def __init__(self, model_name, n_concepts=9, pos_weight=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.concept_head = nn.Linear(h, n_concepts)
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.float) if pos_weight is not None else None,
        )

    def forward(self, input_ids=None, attention_mask=None, concept_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        pooled = pooled.to(self.concept_head.weight.dtype)
        concept_logits = self.concept_head(pooled)
        loss = None
        if concept_labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                concept_logits.to(torch.float32), concept_labels, pos_weight=self.pos_weight)
        return {"loss": loss, "concept_logits": concept_logits}


def compute_pos_weight(concept_df, concepts):
    weights = []
    for c in concepts:
        pos = concept_df[c].astype(bool).sum()
        neg = len(concept_df) - pos
        w = neg / max(pos, 1)
        weights.append(float(np.clip(w, 1.0, 50.0)))
    return weights


@torch.no_grad()
def evaluate_concepts(model, loader, device, use_bf16):
    model.eval()
    all_probs, all_gold = [], []
    for batch in loader:
        batch.pop("idx")
        gold = batch.pop("concept_labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(**batch)["concept_logits"]
        all_probs.append(torch.sigmoid(logits).float().cpu())
        all_gold.append(gold)
    probs = torch.cat(all_probs).numpy()
    gold = torch.cat(all_gold).numpy()
    preds = (probs >= 0.5).astype(int)
    print(f"{'concept':<25}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}")
    mean_f1 = []
    for j, c in enumerate(CONCEPTS):
        g, p = gold[:, j].astype(int), preds[:, j]
        tp = int(((p == 1) & (g == 1)).sum())
        fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        tn = int(((p == 0) & (g == 0)).sum())
        acc = (tp + tn) / max(len(g), 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        mean_f1.append(f1)
        print(f"{c:<25}{acc:>8.1%}{prec:>8.1%}{rec:>8.1%}{f1:>8.1%}")
    macro_f1 = float(np.mean(mean_f1))
    print(f"\n  macro-F1 across concepts: {macro_f1:.4f}")
    return macro_f1


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
    ap.add_argument("--concept-labels", required=True)
    ap.add_argument("--val-concept-labels", required=True,
                    help="Needed here (unlike Stage 2) since this script's only "
                         "metric IS concept quality.")
    ap.add_argument("--user-context", action="store_true", default=True)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = (args.precision == "bf16" and device == "cuda"
                and torch.cuda.is_bf16_supported())
    print(f"[precision] {'bf16 autocast' if use_bf16 else 'fp32'} on {device}")
    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tr_df, va_df = pd.read_csv(args.train), pd.read_csv(args.val)
    tr_concept_df = pd.read_csv(args.concept_labels)
    va_concept_df = pd.read_csv(args.val_concept_labels)
    for c in CONCEPTS:
        assert c in tr_concept_df.columns, f"missing concept column: {c}"
        assert c in va_concept_df.columns, f"missing concept column in val: {c}"

    pos_weight = compute_pos_weight(tr_concept_df, CONCEPTS)
    print("[pos_weight] " + ", ".join(f"{c}={w:.1f}" for c, w in zip(CONCEPTS, pos_weight)))

    g = torch.Generator()
    g.manual_seed(args.seed)
    tr_loader = DataLoader(
        ConceptDataset(tr_df, tok, args.max_len, args.user_context, tr_concept_df),
        batch_size=args.batch, shuffle=True, generator=g, num_workers=args.num_workers)
    va_loader = DataLoader(
        ConceptDataset(va_df, tok, args.max_len, args.user_context, va_concept_df),
        batch_size=args.batch * 2, shuffle=False, num_workers=args.num_workers)

    model = ConceptOnlyModel(args.model, pos_weight=pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(tr_loader) // args.grad_accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    best_f1, best_state, step = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        opt.zero_grad()
        for b, batch in enumerate(tr_loader):
            batch.pop("idx")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch)["loss"]
            (loss / args.grad_accum).backward()
            running.append(loss.item())
            if (b + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step} loss {np.mean(running[-40:]):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e}")
        print(f"\n== epoch {epoch} val concept metrics ==")
        macro_f1 = evaluate_concepts(model, va_loader, device, use_bf16)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model_final.pt"))
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "best_val_macro_f1": float(best_f1)}, f, indent=2)
    print(f"\nbest val macro-F1 across concepts: {best_f1:.4f}")
    print(f"saved to {os.path.join(args.outdir, 'model_final.pt')}")


if __name__ == "__main__":
    main()
