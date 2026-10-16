"""
factor_sentence_screen.py

Is the annotation unit the SENTENCE rather than the POST?

THE EVIDENCE FOR ASKING. The organizers' 2022 ancestor paper (Li et al., HKIE
Transactions 29(4)) annotated triggers at the SENTENCE level -- 5,353 sentences,
799 carrying triggers -- and aggregated to posts. If the PFA dataset kept that
protocol, then a factor is present because ONE sentence asserts it, while the
cross-encoder here reads the whole post against each hypothesis and has to
notice a single clause inside up to 320 tokens of other material.

There is a published precedent for the fix on a structurally identical task:
Peng's BioNLP lab built a retrieve-verify-decide pipeline for 18-factor
suicide-related SDoH extraction and reported **+17.7% F1 on infrequent
factors** over fine-tuned encoders (arXiv 2508.05003). Infrequent factors are
exactly where this system is weak -- roughly 10 of the 24 labels sit below
0.67 and they set the macro score.

WHAT THIS SCREENS, WITHOUT RETRAINING ANYTHING. Take one EXISTING fold
checkpoint. Score every (sentence, hypothesis) pair and MAX-POOL to the post,
then compare against that same checkpoint's post-level score on the same posts.
If the annotation unit really is the sentence, max-pooling should help most on
the rare labels -- a single asserting sentence stops being diluted.

This is a screen, not a build. A positive result justifies retraining an arm on
sentence-level pairs; a negative one closes the hypothesis for the cost of one
short GPU job.

FALSIFIER, FIXED BEFORE LOOKING. Sentence max-pooling must raise mean average
precision by >= +0.02 overall, OR by >= +0.05 on the labels currently below
0.67 AP. Anything less does not justify a retrain, because the same checkpoint
is being asked a question it was not trained for -- the honest test of the
hypothesis is a retrained arm, and this screen only decides whether that is
worth the GPU.

Usage (GPU, ~20 min for one fold):
  python src/factor_sentence_screen.py --run runs/factors_mnli_cv_f0 --fold 0
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)


def sentences(post, lo=3):
    """Split into sentence-like units. Reddit posts are punctuation-poor, so
    newlines count as boundaries too. Fragments under `lo` tokens are merged
    forward rather than scored alone -- a 2-word fragment cannot assert a
    factor and would only add noise to a max-pool."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", str(post))
             if p.strip()]
    out, buf = [], ""
    for p in parts:
        buf = (buf + " " + p).strip() if buf else p
        if len(buf.split()) >= lo:
            out.append(buf)
            buf = ""
    if buf:
        if out:
            out[-1] = (out[-1] + " " + buf).strip()
        else:
            out.append(buf)
    return out or [str(post)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from train_factors_nli import NLIScorer, hypothesis_for, score_all

    va = pd.read_csv(f"artifacts/folds/val_f{args.fold}.csv")
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(va), N), dtype=int)
    for i, r in enumerate(va["row_id"].astype(str)):
        for x in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(x).strip().lower())
            if j is not None:
                Y[i, j] = 1

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")
    name = open(os.path.join(args.run, "tok")).read().strip()
    tok = AutoTokenizer.from_pretrained(name)
    model = NLIScorer(name).to(dev)
    model.load_state_dict(torch.load(os.path.join(args.run, "model.pt"),
                                     map_location=dev), strict=False)
    model.eval()
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()

    print(f"{len(va)} posts from fold {args.fold}, model {name}")
    P_post = score_all(model, va, tok, args.max_len, dev, use_bf16)

    sents = [sentences(p) for p in va["post"]]
    ns = [len(s) for s in sents]
    print(f"sentences/post: mean {np.mean(ns):.1f}  median {int(np.median(ns))}"
          f"  max {max(ns)}   total pairs {sum(ns) * N}")

    hyp = [hypothesis_for(FACTOR_TAXONOMY[j]) for j in range(N)]
    flat = [(i, s, j) for i, ss in enumerate(sents) for s in ss
            for j in range(N)]
    P_sent = np.zeros((len(va), N))
    with torch.no_grad():
        for b in range(0, len(flat), args.batch):
            ch = flat[b:b + args.batch]
            enc = tok([s for _, s, _ in ch], [hyp[j] for _, _, j in ch],
                      truncation="only_first", max_length=args.max_len,
                      padding=True, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                # NLIScorer.forward returns {"loss", "logit"}, not a tensor
                # (train_factors_nli.py:540). This line had never executed --
                # the script was written and never run.
                lg = model(**enc)["logit"].squeeze(-1).float()
            pr = torch.sigmoid(lg).cpu().numpy()
            for (i, _, j), v in zip(ch, pr):
                P_sent[i, j] = max(P_sent[i, j], float(v))
            if b % (args.batch * 200) == 0:
                print(f"  {b}/{len(flat)} pairs", flush=True)

    from sklearn.metrics import average_precision_score as APS

    def ap(P):
        return np.array([APS(Y[:, j], P[:, j]) if Y[:, j].sum() else np.nan
                         for j in range(N)])

    a_post, a_sent = ap(P_post), ap(P_sent)
    ok = ~np.isnan(a_post)
    weak = ok & (a_post < 0.67)

    print(f"\n{'factor':<44}{'n':>5}{'post':>8}{'sent':>8}{'delta':>9}")
    print("-" * 74)
    for j in range(N):
        if not ok[j]:
            continue
        print(f"{FACTOR_TAXONOMY[j]:<44}{int(Y[:, j].sum()):>5}"
              f"{a_post[j]:>8.3f}{a_sent[j]:>8.3f}{a_sent[j] - a_post[j]:>+9.3f}")

    d_all = np.nanmean(a_sent[ok]) - np.nanmean(a_post[ok])
    d_weak = (np.nanmean(a_sent[weak]) - np.nanmean(a_post[weak])
              if weak.any() else float("nan"))
    print(f"\nMEAN AP   post {np.nanmean(a_post[ok]):.4f}   "
          f"sent {np.nanmean(a_sent[ok]):.4f}   delta {d_all:+.4f}")
    print(f"WEAK LABELS (post AP < 0.67, n={int(weak.sum())})   "
          f"delta {d_weak:+.4f}")

    print("\n" + "=" * 70)
    if d_all >= 0.02 or d_weak >= 0.05:
        print("PASSES. The annotation unit looks like the SENTENCE. Retraining")
        print("an arm on sentence-level pairs is worth the GPU -- and this is")
        print("an UNDERSTATEMENT, since the checkpoint was never trained to")
        print("read isolated sentences.")
    else:
        print("FAILS the pre-stated bar (>= +0.02 overall or +0.05 on weak")
        print("labels). Whole-post scoring is not diluting the signal, so the")
        print("sentence-level lane does not justify a retrain.")
    print("=" * 70)


if __name__ == "__main__":
    main()
