"""
train_span_reranker.py

A CROSS-ENCODER THAT READS THE SPAN, INSTEAD OF RE-WEIGHTING VOTE COUNTS.

The selection oracle on the deployed pool is macro 0.8824 -> 0.9493: the
right spans are usually in the pool and the decode picks wrong. Five prior
attempts to close that gap have not succeeded, and every one re-weighted the
SAME signals the vote already uses -- vote count, confidence, length,
position. Their combined AUC for "this candidate matches a gold span" is
0.752, barely above `n_votes` alone at 0.658. There is nothing left in those
features.

**A cross-encoder asks a question none of them can express: does this clause
read like an annotator's justification?** `'kill myself'` and the clause the
annotator actually chose are indistinguishable by vote count and confidence,
and obviously different to anything that reads the text.

TRAINING DATA, produced by the `evoofc` re-dump: 3762 conditioned candidates
over 1590 OOF posts, 62.8% of which match a gold span under the scorer's own
containment+3x rule. Each post was scored by the fold model that held it out,
so nothing leaks.

    premise    = "The suicide risk level of this post is {level}." + post
    hypothesis = the candidate span
    label      = 1 if it matches a gold span

The risk level is included because evidence is risk-conditioned: the same
clause justifies `behavior` and not `ideation`, so a judge that cannot see the
level is being asked an underspecified question.

HONEST FRAMING. This is the last untried instrument, not a likely win. Five
selection methods failed, and a sixth may fail too -- but they failed for a
reason this one does not share, which is the only argument for spending the
GPU. It is validated author-disjoint before anything ships.

Usage (one GPU, ~40 min):
  python src/train_span_reranker.py --folds 5
  python src/train_span_reranker.py --eval-only --ckpt runs/span_rerank_f0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_span_pair_matches, normalize_label,  # noqa: E402
                     normalize_span, parse_evidence)

RISK_PREFIX = "The suicide risk level of this post is {}."


def auc(score, pos):
    pos = np.asarray(pos, bool)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def build_examples(cand_csv, labeled, risk_csv):
    lab = pd.read_csv(labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rk = pd.read_csv(risk_csv, keep_default_na=False)
    rk["row_id"] = rk["row_id"].astype(str)
    col = ("pred_risk_level" if "pred_risk_level" in rk.columns
           else "risk_level")
    risk = {r: normalize_label(v) for r, v in zip(rk["row_id"], rk[col])}

    d = pd.read_csv(cand_csv, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    rows = []
    for r, grp in d.groupby("row_id"):
        if r not in lab.index:
            continue
        gold = parse_evidence(lab.loc[r, "evidence"])
        post = str(lab.loc[r, "post"])
        lv = risk.get(r, "ideation")
        seen = set()
        for s, c in zip(grp["span"].astype(str),
                        grp["conf"].astype(float) if "conf" in grp
                        else [1.0] * len(grp)):
            n = normalize_span(s)
            if not n or n in seen:
                continue
            seen.add(n)
            rows.append({
                "row_id": r, "post": post, "span": s, "risk": lv,
                "conf": float(c), "author": str(lab.loc[r, "anon_user_id"]),
                "y": int(any(_span_pair_matches(s, g) for g in gold)),
            })
    return pd.DataFrame(rows)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--cands",
                    default="artifacts/evidence_oof/evcond5_c5_oof_candidates.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--risk", default="artifacts/nligaur/oof_base.csv")
    a_.add_argument("--model", default="microsoft/deberta-v3-large")
    a_.add_argument("--outdir", default="runs/span_rerank")
    a_.add_argument("--folds", type=int, default=5)
    a_.add_argument("--epochs", type=int, default=2)
    a_.add_argument("--lr", type=float, default=1e-5)
    a_.add_argument("--batch", type=int, default=16)
    a_.add_argument("--max-len", type=int, default=384)
    a_.add_argument("--seed", type=int, default=17)
    a_.add_argument("--out-scores", default="artifacts/span_rerank_oof.csv")
    args = a_.parse_args()

    ex = build_examples(args.cands, args.labeled, args.risk)
    print(f"{len(ex)} candidates over {ex.row_id.nunique()} posts / "
          f"{ex.author.nunique()} authors   {ex.y.mean():.1%} positive")
    print(f"  baseline AUC from confidence alone: "
          f"{auc(ex['conf'].to_numpy(), ex['y'].to_numpy()):.3f}")

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, get_linear_schedule_with_warmup)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("no GPU -- submit via sbatch")
    tok = AutoTokenizer.from_pretrained(args.model)

    class DS(Dataset):
        def __init__(self, df):
            self.d = df.reset_index(drop=True)

        def __len__(self):
            return len(self.d)

        def __getitem__(self, i):
            r = self.d.iloc[i]
            enc = tok(RISK_PREFIX.format(r["risk"]) + " " + r["post"],
                      r["span"], truncation="only_first",
                      max_length=args.max_len, padding="max_length",
                      return_tensors="pt")
            return ({k: v.squeeze(0) for k, v in enc.items()},
                    torch.tensor(float(r["y"])))

    ua = pd.unique(ex["author"])
    rng = np.random.default_rng(args.seed)
    order = list(ua)
    rng.shuffle(order)
    fold_of = {a: i % args.folds for i, a in enumerate(order)}
    ex["fold"] = [fold_of[a] for a in ex["author"]]

    scores = np.full(len(ex), np.nan)
    for f in range(args.folds):
        tr = ex[ex.fold != f]
        va = ex[ex.fold == f]
        print(f"\n=== fold {f}: train {len(tr)} / val {len(va)} "
              f"({va.y.mean():.1%} positive)")
        m = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=1).to(dev)
        dl = DataLoader(DS(tr), batch_size=args.batch, shuffle=True,
                        drop_last=True)
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
        sch = get_linear_schedule_with_warmup(
            opt, int(0.1 * len(dl) * args.epochs), len(dl) * args.epochs)
        lossf = torch.nn.BCEWithLogitsLoss()
        m.train()
        for ep in range(args.epochs):
            tot = nb = 0
            for enc, y in dl:
                enc = {k: v.to(dev) for k, v in enc.items()}
                out = m(**enc).logits.squeeze(-1)
                loss = lossf(out, y.to(dev))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
                tot += float(loss)
                nb += 1
            print(f"  epoch {ep+1} loss {tot/max(nb,1):.4f}", flush=True)
        m.eval()
        preds = []
        with torch.no_grad():
            for enc, _ in DataLoader(DS(va), batch_size=32):
                enc = {k: v.to(dev) for k, v in enc.items()}
                preds.extend(torch.sigmoid(
                    m(**enc).logits.squeeze(-1)).float().cpu().numpy())
        scores[va.index] = preds
        print(f"  fold AUC {auc(np.array(preds), va.y.to_numpy()):.3f}"
              f"   (confidence alone "
              f"{auc(va['conf'].to_numpy(), va.y.to_numpy()):.3f})")
        os.makedirs(f"{args.outdir}_f{f}", exist_ok=True)
        m.save_pretrained(f"{args.outdir}_f{f}")
        tok.save_pretrained(f"{args.outdir}_f{f}")
        del m
        torch.cuda.empty_cache()

    ex["rerank"] = scores
    # SAVE THE FOLD. Val authors appear in this re-ranker's own training folds,
    # so anything scoring the deployed 7-tagger candidates must use the fold
    # model that EXCLUDED that author. Without this column the mapping has to
    # be re-derived from the seed, which is fragile and silently wrong if the
    # author set ever changes.
    ex[["row_id", "span", "risk", "conf", "y", "author", "fold",
        "rerank"]].to_csv(args.out_scores, index=False)
    pd.DataFrame({"author": list(fold_of), "fold": list(fold_of.values())}
                 ).to_csv(args.out_scores.replace(".csv", "_folds.csv"),
                          index=False)
    ok = ~np.isnan(scores)
    print(f"\nwrote {args.out_scores}")
    print("=" * 68)
    print(f"  OOF AUC  reranker {auc(scores[ok], ex.y.to_numpy()[ok]):.3f}"
          f"   confidence {auc(ex['conf'].to_numpy()[ok], ex.y.to_numpy()[ok]):.3f}")
    print("\n  The hand-feature ceiling was 0.752 and n_votes alone 0.658.")
    print("  If this clears ~0.85 the instrument is genuinely different and")
    print("  worth wiring into the decode; if it lands near 0.75 it has")
    print("  learned the same thing the features already encode and the")
    print("  selection lane is closed for good.")


if __name__ == "__main__":
    main()
