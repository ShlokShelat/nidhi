"""
Definition-aware factor classification (NLI / entailment style).

Motivation: our per-label analysis shows factor F1 is r=0.806 correlated
with training-example count — rare factors (n=8-35) simply lack examples.
The organizers' paper (arXiv 2507.10008) provides exact clinical
DEFINITIONS for all 24 factors. Instead of 24 anonymous sigmoid outputs
that must each be learned from scratch, we reformulate as entailment:

    premise    = the post
    hypothesis = "This post shows <factor>: <definition>"
    label      = 1 if the factor is present, else 0

The encoder sees the factor's MEANING every time, so signal is shared
across labels: learning "traumatic experience" partly teaches the model
what "stressful life event" is NOT (their definitions contrast), and a
rare factor benefits from the model's general language understanding of
its definition rather than needing many examples. This is the standard
route to beating label scarcity (cf. entailment-based zero/few-shot
classification).

Cost: one forward pass per (post, factor) pair. Test set is 378 posts x
24 factors ~= 9072 pairs — trivial. Training builds pairs from the labeled
posts (balanced: all positives + sampled negatives).

This is a SEPARATE script; it doesn't touch the working train_factors.py.
Validate via val_split macro-F1 and (if promising) the CV harness before
any submission — same discipline as always.

Usage:
  python src/train_factors_nli.py \
      --train artifacts/train_split.csv --val artifacts/val_split.csv \
      --model microsoft/deberta-v3-large --outdir runs/factors_nli \
      --neg-ratio 3 --epochs 3
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from scoring import FACTOR_TAXONOMY, parse_factors, factors_macro_f1

# exact definitions from arXiv 2507.10008 Table III (same as label_external.py)
FACTOR_DEFINITIONS = {
    "mental health issues": "existing mental disorders such as depression or personality disorders",
    "physical health/characteristic": "physical health issues such as illness, obesity, or being underweight",
    "substance use": "the uncontrollable use of drugs, alcohol, or tobacco",
    "hopelessness": "feelings of hopelessness, or feeling trapped or stuck",
    "emotion dysregulation": "emotion regulation difficulties such as uncontrolled anxiety or anger",
    "low self-esteem": "self-negative feelings such as worthlessness, being a burden, or self-hate",
    "poor school performance": "low school performance such as failing tests or bad grades",
    "low socio-economic status": "unemployment, poverty, or homelessness",
    "interpersonal violence": "violence or assault occurring outside home settings",
    "prior self-harm or suicidal thought/attempt": "past, resolved, previous self-harm or suicidal thought, plan, or attempt",
    "poor social support": "lacking friends, loneliness, being isolated, rejected, neglected, or abandoned",
    "interpersonal difficulty": "difficulty making new friends or socializing",
    "dysfunctional family": "family-related issues that negatively impact the individual",
    "exposure to others' suicide": "mentioning or describing others' suicidal thoughts, attempt, or death",
    "stressful life event": "events that pose challenges but do not lead to traumatic responses",
    "traumatic experience": "an event overwhelming an individual's ability to cope",
    "cognitive deficits": "difficulty in cognitive abilities",
    "suicide means (with access)": "description of potential suicide means or access to those means",
    "sexual orientation related issues": "gender or sexual identity issues, or same-sex relationship concerns",
    "social support": "support from family, partners, friends, or healthcare professionals",
    "coping strategy": "activities the individual engages in to deal with stressful situations",
    "psychological capital": "the individual's positive psychological state such as resilience or optimism",
    "sense of responsibility": "awareness of responsibility to one's own health or survival and to others",
    "meaning in life": "a sense of purpose, goals, or reasons for living",
}
N = len(FACTOR_TAXONOMY)


# Confusable-factor groups: definitions that are semantically close and easy
# to mistake. Sampling negatives from a post's confusable neighbors (given its
# positives) creates HARD negatives that teach the model the boundary, e.g.
# "stressful life event" vs "traumatic experience", the exact distinction the
# annotators' definitions hinge on.
CONFUSABLE = {
    "stressful life event": ["traumatic experience"],
    "traumatic experience": ["stressful life event", "interpersonal violence"],
    "interpersonal violence": ["traumatic experience", "dysfunctional family", "interpersonal difficulty"],
    "interpersonal difficulty": ["poor social support", "interpersonal violence"],
    "poor social support": ["interpersonal difficulty", "social support"],
    "social support": ["poor social support", "psychological capital"],
    "dysfunctional family": ["interpersonal violence", "stressful life event"],
    "hopelessness": ["low self-esteem", "emotion dysregulation", "meaning in life"],
    "low self-esteem": ["hopelessness", "emotion dysregulation"],
    "emotion dysregulation": ["hopelessness", "low self-esteem"],
    "meaning in life": ["hopelessness", "psychological capital", "sense of responsibility"],
    "psychological capital": ["coping strategy", "meaning in life", "social support"],
    "coping strategy": ["psychological capital", "sense of responsibility"],
    "sense of responsibility": ["meaning in life", "coping strategy"],
    "prior self-harm or suicidal thought/attempt": ["suicide means (with access)", "exposure to others' suicide"],
    "suicide means (with access)": ["prior self-harm or suicidal thought/attempt"],
    "exposure to others' suicide": ["prior self-harm or suicidal thought/attempt"],
    "substance use": ["mental health issues"],
    "mental health issues": ["substance use", "emotion dysregulation", "cognitive deficits"],
    "cognitive deficits": ["mental health issues"],
}
_CONF_IDX = {FACTOR_TAXONOMY.index(k): [FACTOR_TAXONOMY.index(v) for v in vs if v in FACTOR_TAXONOMY]
             for k, vs in CONFUSABLE.items() if k in FACTOR_TAXONOMY}


def make_pairs(df, neg_ratio, rng, hard_frac=0.5):
    """Build (text, factor_idx, label) rows. All positives + sampled negatives,
    where a fraction `hard_frac` of negatives are drawn from factors CONFUSABLE
    with the post's positives (hard negatives), the rest random (easy).

    Called once PER EPOCH with a fresh rng draw so negatives vary across epochs
    (regularization + better coverage of the easy-negative space)."""
    rows = []
    for _, r in df.iterrows():
        post = str(r["post"])
        pos = {FACTOR_TAXONOMY.index(f) for f in parse_factors(r["factors"])
               if f in FACTOR_TAXONOMY}
        for j in pos:
            rows.append((post, j, 1))
        n_neg = max(1, len(pos) * neg_ratio) if pos else neg_ratio
        # hard-negative candidates: confusable with any positive, not itself positive
        hard_pool = set()
        for j in pos:
            for c in _CONF_IDX.get(j, []):
                if c not in pos:
                    hard_pool.add(c)
        hard_pool = list(hard_pool)
        rng.shuffle(hard_pool)
        n_hard = min(len(hard_pool), int(round(n_neg * hard_frac)))
        chosen = hard_pool[:n_hard]
        # fill the rest with random easy negatives
        easy_pool = [j for j in range(N) if j not in pos and j not in set(chosen)]
        rng.shuffle(easy_pool)
        chosen += easy_pool[: n_neg - n_hard]
        for j in chosen:
            rows.append((post, j, 0))
    return rows


class PairDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        post, j, y = self.rows[i]
        hyp = f"This post shows {FACTOR_TAXONOMY[j]}: {FACTOR_DEFINITIONS[FACTOR_TAXONOMY[j]]}"
        enc = self.tok(post, hyp, truncation="only_first", max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(float(y)),
                "factor": torch.tensor(j)}


class NLIScorer(nn.Module):
    def __init__(self, model_name, pos_weight=None):
        nn.Module.__init__(self)
        self.enc = AutoModel.from_pretrained(model_name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, 1)
        self.register_buffer("pos_weight",
            torch.tensor([pos_weight], dtype=torch.float) if pos_weight else None)

    def forward(self, input_ids=None, attention_mask=None, label=None, **kw):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        x = self.drop(out.last_hidden_state[:, 0])
        x = x.to(self.head.weight.dtype)
        logit = self.head(x).squeeze(-1)
        loss = None
        if label is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit, label.to(logit.dtype), pos_weight=self.pos_weight)
        return {"loss": loss, "logit": logit}


@torch.no_grad()
def score_all(model, df, tok, max_len, device, use_bf16, batch=64):
    """Return (N_posts, 24) probability matrix by scoring every pair."""
    model.eval()
    posts = df["post"].astype(str).tolist()
    probs = np.zeros((len(posts), N), dtype=np.float32)
    pairs = [(pi, j) for pi in range(len(posts)) for j in range(N)]
    for s in range(0, len(pairs), batch):
        chunk = pairs[s : s + batch]
        texts = [posts[pi] for pi, _ in chunk]
        hyps = [f"This post shows {FACTOR_TAXONOMY[j]}: {FACTOR_DEFINITIONS[FACTOR_TAXONOMY[j]]}"
                for _, j in chunk]
        enc = tok(texts, hyps, truncation="only_first", max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            logit = model(**enc)["logit"]
        p = torch.sigmoid(logit.float()).cpu().numpy()
        for (pi, j), pv in zip(chunk, p):
            probs[pi, j] = pv
    return probs


def tune_thresholds(probs, gold):
    ths = np.full(N, 0.5)
    for j in range(N):
        best_t, best_f = 0.5, -1
        for t in np.arange(0.1, 0.9, 0.05):
            pred = (probs[:, j] >= t).astype(int)
            tp = int(((pred == 1) & (gold[:, j] == 1)).sum())
            fp = int(((pred == 1) & (gold[:, j] == 0)).sum())
            fn = int(((pred == 0) & (gold[:, j] == 1)).sum())
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best_f:
                best_f, best_t = f1, t
        ths[j] = best_t
    return ths


def gold_matrix(df):
    Y = np.zeros((len(df), N), dtype=int)
    for r, cell in enumerate(df["factors"]):
        for f in parse_factors(cell):
            if f in FACTOR_TAXONOMY:
                Y[r, FACTOR_TAXONOMY.index(f)] = 1
    return Y


def to_lists(binary):
    return [[FACTOR_TAXONOMY[j] for j in range(N) if row[j]] for row in binary]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--neg-ratio", type=int, default=3)
    ap.add_argument("--external", default=None,
                    help="external labeled CSV (new users) — added to training pairs, never to val")
    ap.add_argument("--hard-frac", type=float, default=0.5,
                    help="fraction of negatives drawn from confusable factors (hard negs)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")
    if device == "cpu":
        print("WARNING: on CPU — this trains a full model, use a GPU node.")

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    tr_df = pd.read_csv(args.train)
    va_df = pd.read_csv(args.val)
    if args.external:
        ext = pd.read_csv(args.external, keep_default_na=False)
        keep = [c for c in ["row_id", "post", "factors"] if c in ext.columns]
        tr_df = pd.concat([tr_df, ext[keep]], ignore_index=True)
        print(f"external: +{len(ext)} posts merged into training (now {len(tr_df)})")
    tok = AutoTokenizer.from_pretrained(args.model)

    # pos_weight approximates neg:pos ratio so the head doesn't collapse to
    # predicting "absent" (clamped to avoid over-correction)
    pw = float(np.clip(args.neg_ratio, 1.0, 5.0))
    model = NLIScorer(args.model, pos_weight=pw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # estimate steps from a first pair build (size is stable across epochs)
    probe = make_pairs(tr_df, args.neg_ratio, rng, args.hard_frac)
    steps_per_epoch = len(probe) // args.batch // args.grad_accum + 1
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    print(f"~{len(probe)} pairs/epoch ({sum(r[2] for r in probe)} pos), "
          f"pos_weight={pw}, hard_frac={args.hard_frac}")

    va_gold = gold_matrix(va_df)
    best_macro, best_state = -1.0, None
    for epoch in range(args.epochs):
        # FRESH negatives each epoch (regularization + coverage)
        train_rows = make_pairs(tr_df, args.neg_ratio, rng, args.hard_frac)
        loader = DataLoader(PairDataset(train_rows, tok, args.max_len),
                            batch_size=args.batch, shuffle=True, num_workers=2)
        model.train(); opt.zero_grad()
        run = []

        for b, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(**batch)
                loss = out["loss"]

            loss_val = loss.item()

            # --- DEBUG: catch the FIRST NaN the instant it happens ---
            if np.isnan(loss_val) or np.isinf(loss_val):
                print(f"\n{'!'*80}\nNaN/Inf loss detected at epoch {epoch+1} step {b+1}\n{'!'*80}")
                # dump every example in this batch so we can see exactly
                # which (post, factor) pair caused it
                batch_start = b * loader.batch_size
                for i in range(min(len(train_rows) - batch_start, loader.batch_size)):
                    idx = batch_start + i
                    if idx >= len(train_rows):
                        break
                    post_text, factor_idx, label = train_rows[idx]
                    from scoring import FACTOR_TAXONOMY as _FT
                    print(f"  [{i}] factor={_FT[factor_idx]!r} label={label} "
                          f"len={len(post_text)} post={post_text[:200]!r}")
                # also check the logits themselves for nan/inf
                logit = out["logit"]
                print(f"  logit stats: min={logit.min().item()} max={logit.max().item()} "
                      f"any_nan={torch.isnan(logit).any().item()} "
                      f"any_inf={torch.isinf(logit).any().item()}")
                print("Stopping here — do not continue training on a poisoned model.")
                import sys; sys.exit(1)
            # --- end debug ---

            (loss / args.grad_accum).backward()
            run.append(loss_val)
            if (b + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if (b + 1) % 200 == 0:
                print(f"epoch {epoch+1} step {b+1} loss {np.mean(run[-200:]):.4f}")
        # eval
        vp = score_all(model, va_df, tok, args.max_len, device, use_bf16)
        ths = tune_thresholds(vp, va_gold)
        on = (vp >= ths).astype(int)
        macro = factors_macro_f1(to_lists(va_gold), to_lists(on))
        print(f"== epoch {epoch+1}: val macro-F1 (tuned) = {macro:.4f}")
        if macro > best_macro:
            best_macro = macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            np.save(os.path.join(args.outdir, "thresholds.npy"), ths)

    if best_state:
        torch.save(best_state, os.path.join(args.outdir, "model.pt"))
    with open(os.path.join(args.outdir, "tok"), "w") as f:
        f.write(args.model)
    print(f"\nbest val macro-F1 = {best_macro:.4f}")
    print("compare against plain-factor baseline ~0.4165 (unaug single split)")


if __name__ == "__main__":
    main()
