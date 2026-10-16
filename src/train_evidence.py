"""
Evidence span extraction: BIO token tagger + hard grounding verifier.

Design choice: extractive tagging, not generation. A tagger's output is a
substring of the post BY CONSTRUCTION, which eliminates the entire class of
"hallucinated span scores zero" failures that generative extraction has.
An LLM proposer can be ensembled later, but every span from ANY source must
pass verify_spans() before entering a submission.

Training targets are built by locating each gold span in the post
(normalized, case-insensitive) and labeling its tokens B/I. Gold spans that
cannot be located are logged, not silently dropped: a large count means the
span-locating chain, not the annotation, is at fault and must be fixed first.

Usage:
  python src/train_evidence.py --train artifacts/train_split.csv \
      --val artifacts/val_split.csv --model microsoft/deberta-v3-large \
      --outdir runs/evidence_v1
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")          # no hub calls; cache only (override: HF_HUB_OFFLINE=0)
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from scoring import (RISK_LEVELS, normalize_label, normalize_span,
                     parse_evidence, phrase_f1)

O, B, I = 0, 1, 2

# --risk-cond: condition the tagger on the risk level (see EvidenceDataset).
RISK_PREFIX = "The suicide risk level of this post is {}."


# ----------------------------------------------------------------------
# Grounding verifier — the non-negotiable gate for ALL evidence sources
# ----------------------------------------------------------------------

def find_span_in_post(span: str, post: str) -> tuple[int, int] | None:
    """Locate span in post, returning char offsets in the post as given.
    Tries exact match first, then case-insensitive, then
    whitespace-flexible regex. Returns None if not found."""
    s, p = str(span), str(post)
    idx = p.find(s)
    if idx >= 0:
        return idx, idx + len(s)
    idx = p.lower().find(s.lower())
    if idx >= 0:
        return idx, idx + len(s)
    pattern = r"\s+".join(re.escape(w) for w in s.split())
    m = re.search(pattern, p, flags=re.IGNORECASE)
    if m:
        return m.start(), m.end()
    return None


def _find_fuzzy(span: str, post: str, min_ratio: float = 0.88) -> tuple[int, int] | None:
    """Training-target fallback for annotator typos and spacing drift, where
    the recorded span differs from the post by a character or by a word
    boundary and so cannot be located exactly.
    Finds the best same-size word window by difflib ratio, then locates
    that window (which is post text, so it always locates) via the
    flexible matcher above. NOT used at prediction time."""
    import difflib
    sw = str(span).lower().split()
    pw = str(post).split()
    if not sw or not pw:
        return None
    best, best_win = 0.0, None
    for size in range(max(1, len(sw) - 1), len(sw) + 2):
        for i in range(0, max(1, len(pw) - size + 1)):
            cand_words = pw[i:i + size]
            r = difflib.SequenceMatcher(
                None, " ".join(sw), " ".join(w.lower() for w in cand_words)).ratio()
            if r > best:
                best, best_win = r, cand_words
    if best >= min_ratio and best_win:
        return find_span_in_post(" ".join(best_win), post)
    return None


def locate_for_training(span: str, post: str) -> list[tuple[int, int]]:
    """Full chain for building B/I training targets: exact/ci/ws-flex,
    punctuation-stripped regex, then fuzzy. Discontinuous spans are
    already split by parse_evidence. Returns [] if unrecoverable."""
    loc = find_span_in_post(span, post)
    if loc:
        return [loc]
    # punctuation-tolerant: words joined by any non-word gap
    words = re.findall(r"\w+", str(span).lower())
    if words:
        pattern = r"\W{0,6}".join(re.escape(w) for w in words)
        m = re.search(pattern, post, flags=re.IGNORECASE)
        if m:
            return [(m.start(), m.end())]
    loc = _find_fuzzy(span, post)
    return [loc] if loc else []


def verify_spans(spans: list[str], post: str,
                 max_tokens: int = 12) -> tuple[list[str], list[str]]:
    """Return (accepted, rejected). Accepted spans are re-extracted from the
    ORIGINAL post text at the located offsets, so the submitted string is
    verbatim even if the proposal had normalization drift.
    max_tokens: hard sanity cap; gold spans in the sample run 2-8 tokens and
    the 3x rule punishes long spans, so anything long is suspect."""
    accepted, rejected = [], []
    seen = set()
    for sp in spans:
        loc = find_span_in_post(sp, post)
        n_alpha = sum(ch.isalpha() for ch in str(sp))
        if loc is None or len(str(sp).split()) > max_tokens or n_alpha < 3:
            rejected.append(sp)   # ungrounded, too long, or punctuation shrapnel
            continue
        verbatim = post[loc[0]:loc[1]]
        key = normalize_span(verbatim)
        if key in seen:      # one-to-one matching makes duplicates worthless
            continue
        seen.add(key)
        accepted.append(verbatim)
    return accepted, rejected


# ----------------------------------------------------------------------
# BIO dataset construction
# ----------------------------------------------------------------------

class EvidenceDataset(Dataset):
    """BIO targets over the post.

    With risk_cond=True the post is encoded as the SECOND sequence of a pair,
    the first being a one-line statement of the risk level. The gold spans are
    the annotator's justification FOR THAT LEVEL -- a countdown for `behavior`,
    a death wish for `ideation`, a completed act for `attempt` -- so telling the
    tagger which level it is arguing for changes which sentence it should pick.
    Without it the tagger can only look for generically risky-sounding text.

    Only sequence-1 tokens carry labels; the prefix is masked to -100 and
    dropped at decode time, so char offsets still index the post.
    """

    def __init__(self, df, tokenizer, max_len, log_misses: list | None = None,
                 risk_cond: bool = False, cond_dropout: float = 0.0,
                 seed: int = 0):
        self.tok, self.max_len = tokenizer, max_len
        self.risk_cond = risk_cond
        self.prefixes = None
        # CONDITIONING DROPOUT. During training the prefix is ALWAYS the gold
        # level, so the tagger never sees a wrong one and learns to obey it
        # unconditionally. At deployment it is handed the PREDICTED level,
        # which is wrong on a minority of posts, and on exactly those posts an
        # obedient tagger argues for a level the text does not support and
        # picks the wrong sentence. That is a training/serving mismatch, not a
        # weak tagger.
        #
        # With probability `cond_dropout` the prefix is replaced -- half the
        # time by `unknown`, half by a WRONG level -- while the target spans
        # stay the gold ones. The tagger then learns to use the hint when it
        # agrees with the text and ignore it when it does not, which is the
        # same idea as classifier-free guidance. It also creates a principled
        # `unknown` conditioning to fall back on for uncertain posts, which an
        # always-gold-conditioned model has no way to express.
        self.cond_dropout = float(cond_dropout)
        self.rng = np.random.default_rng(seed)
        self.levels = None
        if risk_cond:
            if "risk_level" not in df.columns:
                raise SystemExit("--risk-cond needs a risk_level column")
            self.levels = [normalize_label(x) for x in df["risk_level"]]
            self.prefixes = [RISK_PREFIX.format(x) for x in self.levels]
        self.posts = df["post"].astype(str).tolist()
        self.char_labels = []
        for r, (post, cell) in enumerate(zip(self.posts, df["evidence"])):
            spans = parse_evidence(cell)
            marks = np.zeros(len(post), dtype=np.int8)
            for sp in spans:
                locs = locate_for_training(sp, post)
                if not locs:
                    if log_misses is not None:
                        log_misses.append({"row": r, "span": sp})
                    continue
                for a, b in locs:
                    marks[a:b] = 1
            self.char_labels.append(marks)

    def __len__(self):
        return len(self.posts)

    def _prefix(self, i):
        """The conditioning for example i, corrupted with probability
        cond_dropout. Resampled EVERY epoch (this runs in __getitem__, not
        __init__), so the model sees the same post under several conditionings
        over training rather than memorising one pairing."""
        if self.cond_dropout <= 0 or self.rng.random() >= self.cond_dropout:
            return self.prefixes[i]
        if self.rng.random() < 0.5:
            return RISK_PREFIX.format("unknown")
        wrong = [c for c in RISK_LEVELS if c != self.levels[i]]
        return RISK_PREFIX.format(wrong[self.rng.integers(len(wrong))])

    def __getitem__(self, i):
        if self.risk_cond:
            enc = self.tok(self._prefix(i), self.posts[i],
                           truncation="only_second", max_length=self.max_len,
                           padding="max_length", return_offsets_mapping=True,
                           return_tensors="pt")
            seq = enc.sequence_ids(0)
        else:
            enc = self.tok(self.posts[i], truncation=True,
                           max_length=self.max_len, padding="max_length",
                           return_offsets_mapping=True, return_tensors="pt")
            seq = None
        offsets = enc.pop("offset_mapping").squeeze(0).tolist()
        marks = self.char_labels[i]
        labels, prev_inside = [], False
        for t, (a, b) in enumerate(offsets):
            # special/pad token, or a prefix token whose offsets index the
            # prefix rather than the post
            if a == b or (seq is not None and seq[t] != 1):
                labels.append(-100)
                prev_inside = False
                continue
            inside = bool(marks[a:b].any())
            labels.append((I if prev_inside else B) if inside else O)
            prev_inside = inside
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels),
            "idx": torch.tensor(i),
        }


class TaggerModel(nn.Module):
    """BIO tagger.

    `per_post` matches the loss reduction to the metric's reduction. The scorer
    MACRO-averages phrase-F1 over POSTS -- every post counts once, whether it is
    40 tokens or 400. The default CrossEntropyLoss averages over every token in
    the batch, so a 400-token post pulls ~5x the gradient of an 80-token one:
    the tagger is trained disproportionately on long posts and scored equally
    across all of them. Per-post normalisation averages within each post first,
    then across posts, which is exactly what the scorer does.

    OFF by default, so the reduction changes only when it is asked for and a
    run reproduces bit-for-bit otherwise.
    """

    def __init__(self, model_name, per_post=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.encoder.config.hidden_size, 3)
        self.per_post = per_post

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(out.last_hidden_state)
        hidden = hidden.to(self.head.weight.dtype)  # mixed-precision safety
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            if not self.per_post:
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    logits.view(-1, 3), labels.view(-1))
            else:
                tok = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
                    logits.view(-1, 3), labels.view(-1)).view(labels.shape)
                keep = (labels != -100).to(tok.dtype)
                per = (tok * keep).sum(1) / keep.sum(1).clamp(min=1.0)
                # a row with no supervised token must not drag the mean to 0
                alive = keep.sum(1) > 0
                loss = per[alive].mean() if alive.any() else per.sum() * 0.0
        return {"loss": loss, "logits": logits}


# ----------------------------------------------------------------------
# Decoding: token tags -> verbatim char spans
# ----------------------------------------------------------------------

def _decode_candidates(post: str, probs, offsets,
                       max_tokens: int = 12) -> list[tuple[str, float]]:
    """Decode B/I/O argmax tags into candidate spans, each with a
    confidence = mean over its tokens of P(not-O). Grounding-verified."""
    tags = probs.argmax(-1)
    raw = []                      # [start, end, conf_sum, n_tokens]
    cur = None
    for t, (a, b), p in zip(tags.tolist(), offsets, probs.tolist()):
        if a == b:
            continue
        conf = 1.0 - p[O]
        if t == B or (t == I and cur is None):
            if cur:
                raw.append(cur)
            cur = [a, b, conf, 1]
        elif t == I and cur is not None:
            cur[1] = b
            cur[2] += conf
            cur[3] += 1
        else:
            if cur:
                raw.append(cur)
            cur = None
    if cur:
        raw.append(cur)

    out = []
    for a, b, cs, n in raw:
        text = post[a:b].strip()
        accepted, _ = verify_spans([text], post, max_tokens=max_tokens)
        if accepted:
            out.append((accepted[0], cs / n))
    return out


@torch.no_grad()
def collect_candidates(model, tokenizer, posts: list[str], max_len: int,
                       device: str = "cuda",
                       risks: list[str] | None = None
                       ) -> list[list[tuple[str, float]]]:
    """risks, when given, must be one risk level per post -- GOLD levels give
    the oracle ceiling, PREDICTED levels give the shippable number."""
    model.eval().to(device)
    out = []
    for i, post in enumerate(posts):
        if risks is not None:
            enc = tokenizer(RISK_PREFIX.format(normalize_label(risks[i])), post,
                            truncation="only_second", max_length=max_len,
                            return_offsets_mapping=True, return_tensors="pt")
            seq = enc.sequence_ids(0)
        else:
            enc = tokenizer(post, truncation=True, max_length=max_len,
                            return_offsets_mapping=True, return_tensors="pt")
            seq = None
        offsets = enc.pop("offset_mapping").squeeze(0).tolist()
        enc = {k: v.to(device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        logits = model(**enc)["logits"].squeeze(0).float()
        probs = torch.softmax(logits, -1).cpu().numpy()
        if seq is not None:
            keep = [t for t in range(len(offsets)) if seq[t] == 1]
            probs = probs[keep]
            offsets = [offsets[t] for t in keep]
        out.append(_decode_candidates(post, probs, offsets))
    return out


def spans_at(cands: list[list[tuple[str, float]]],
             min_conf: float) -> list[list[str]]:
    return [[s for s, c in post_c if c >= min_conf] for post_c in cands]


def predict_spans(model, tokenizer, posts: list[str], max_len: int,
                  device: str = "cuda", min_conf: float = 0.0) -> list[list[str]]:
    """Backwards-compatible entry point used by predict.py."""
    return spans_at(collect_candidates(model, tokenizer, posts, max_len, device),
                    min_conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-val", action="store_true",
                    help="full-data mode: no eval; use --fixed-min-conf")
    ap.add_argument("--fixed-min-conf", type=float, default=0.65,
                    help="min_conf to store when --no-val (from dev run)")
    ap.add_argument("--per-post", action="store_true",
                    help="average the token loss WITHIN each post before "
                         "averaging across posts. The scorer macro-averages "
                         "phrase-F1 over POSTS, but the default token-mean loss "
                         "gives a 400-token post ~5x the gradient of an 80-token "
                         "one. OFF by default so a run reproduces exactly "
                         "unless the reduction is changed deliberately.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--risk-cond", action="store_true",
                    help="condition the tagger on the risk level by encoding "
                         "'The suicide risk level of this post is X.' as the "
                         "first sequence of a pair")
    ap.add_argument("--cond-dropout", type=float, default=0.0,
                    help="probability of corrupting the training prefix (half "
                         "`unknown`, half a wrong level) while keeping the gold "
                         "spans. Fixes the train/serve mismatch: training "
                         "always shows the GOLD level, deployment shows the "
                         "PREDICTED one, so a tagger that obeys the prefix "
                         "unconditionally picks the wrong sentence whenever "
                         "the risk prediction is wrong. TRAINING ONLY -- "
                         "validation conditioning stays gold, so the reported "
                         "val number holds the conditioning fixed and measures "
                         "the tagger. Try 0.3.")
    ap.add_argument("--val-risk-from", default=None,
                    help="CSV with row_id,risk_level of PREDICTED levels for "
                         "validation. Without it, --risk-cond validates on "
                         "GOLD levels, which is an oracle ceiling and NOT a "
                         "shippable number.")
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

    misses: list = []
    # dropout on TRAIN only: the val set keeps gold conditioning, so the val
    # score varies with the tagger and not with the conditioning
    tr_ds = EvidenceDataset(tr_df, tok, args.max_len, log_misses=misses,
                            risk_cond=args.risk_cond,
                            cond_dropout=args.cond_dropout,
                            seed=getattr(args, "seed", 0))
    va_ds = EvidenceDataset(va_df, tok, args.max_len, risk_cond=args.risk_cond)
    if misses:
        pd.DataFrame(misses).to_csv(
            os.path.join(args.outdir, "unlocatable_gold_spans.csv"), index=False)
        print(f"note: {len(misses)} gold spans unlocatable after full recovery "
              f"chain (logged, excluded from training targets)")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           num_workers=args.num_workers)

    model = TaggerModel(args.model, per_post=args.per_post).to(device)
    if args.per_post:
        print("[loss] PER-POST normalisation: each post contributes equally, "
              "matching the macro-per-post scorer")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(tr_loader) // args.grad_accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    va_posts = va_df["post"].astype(str).tolist() if not args.no_val else []
    gold = [parse_evidence(c) for c in va_df["evidence"]] if not args.no_val else []

    va_risks = None
    if args.risk_cond and not args.no_val:
        if args.val_risk_from:
            rk = pd.read_csv(args.val_risk_from, keep_default_na=False)
            col = ("risk_level" if "risk_level" in rk.columns
                   else "pred_risk_level")
            m = {str(a): b for a, b in zip(rk["row_id"], rk[col])}
            absent = [str(r) for r in va_df["row_id"] if str(r) not in m]
            if absent:
                raise SystemExit(f"--val-risk-from lacks {len(absent)} val rows")
            va_risks = [m[str(r)] for r in va_df["row_id"]]
            print(f"[risk-cond] val conditioned on PREDICTED levels from "
                  f"{args.val_risk_from} -- this number is shippable")
        else:
            va_risks = [str(x) for x in va_df["risk_level"]]
            print("[risk-cond] val conditioned on GOLD levels -- ORACLE "
                  "CEILING, not a shippable number")

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
            if not torch.isfinite(loss):
                rows = tr_df.iloc[idx.tolist()]
                print(f"\nNON-FINITE LOSS at epoch {epoch} batch {b}")
                print("row_ids:", rows.get("row_id", rows.index).tolist())
                rows.to_csv(os.path.join(args.outdir, "poison_batch.csv"), index=False)
                sys.exit("aborted; poison_batch.csv written")
            (loss / args.grad_accum).backward()
            running.append(loss.item())
            if (b + 1) % args.grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gn):
                    sys.exit(f"NON-FINITE GRAD NORM at epoch {epoch} batch {b}")
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step} "
                          f"loss {np.mean(running[-40:]):.4f} grad_norm {gn:.2f}")

        if args.no_val:
            print(f"== epoch {epoch}: (full-data mode, no eval)")
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            cands = collect_candidates(model, tok, va_posts, args.max_len,
                                       device, risks=va_risks)
            res = phrase_f1(gold, spans_at(cands, 0.0))
            print(f"== epoch {epoch}: val phrase F1 = {res.f1:.4f} "
                  f"(P={res.precision:.4f} R={res.recall:.4f})")
            if res.f1 > best_f1:
                best_f1 = res.f1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

    model.load_state_dict(best_state); model.to(device)
    if args.no_val:
        best_conf = args.fixed_min_conf
        print(f"\nfull-data mode: stored min_conf={best_conf:.2f} (no sweep)")
        torch.save(model.state_dict(), os.path.join(args.outdir, "model_final.pt"))
        with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
            json.dump({**vars(args), "evidence_min_conf": float(best_conf),
                       "trained_on": "full_data"}, f, indent=2)
    else:
        cands = collect_candidates(model, tok, va_posts, args.max_len, device,
                                   risks=va_risks)
        best_conf, best_res = 0.0, phrase_f1(gold, spans_at(cands, 0.0))
        for th in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            r = phrase_f1(gold, spans_at(cands, th))
            if r.f1 > best_res.f1:
                best_conf, best_res = th, r
        res, pred = best_res, spans_at(cands, best_conf)
        print(f"\nbest val phrase F1 = {res.f1:.4f} at min_conf={best_conf:.2f} "
              f"(P={res.precision:.4f} R={res.recall:.4f}, "
              f"matched {res.n_matched}/{res.n_gold} gold spans)")
        va_out = va_df.copy()
        va_out["pred_evidence"] = ["; ".join(p) for p in pred]
        va_out.to_csv(os.path.join(args.outdir, "val_predictions.csv"), index=False)
        torch.save(model.state_dict(), os.path.join(args.outdir, "model_final.pt"))
        with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
            json.dump({**vars(args), "best_val_phrase_f1": float(res.f1),
                       "evidence_min_conf": float(best_conf)}, f, indent=2)


if __name__ == "__main__":
    main()
