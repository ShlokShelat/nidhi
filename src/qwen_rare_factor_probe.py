"""
qwen_rare_factor_probe.py

Can a local LLM, given only the published definition, recognise the factors the
supervised model cannot learn?

THE ASYMMETRY THIS TESTS. Five factors have under 40 positive examples in all
1635 training posts -- `sexual orientation related issues` has EIGHT. No
supervised model learns a concept from eight examples; that is a property of the
data, not a failure of the architecture. But macro-F1 weights all 24 labels
equally, so those five carry 21% of S2, and a threshold fitted by
F1-maximisation on roughly two validation positives is nearly unconstrained.

An LLM does not need the eight examples. It needs the definition, which is
published. Moving those five labels from ~0.15 to ~0.50 would be worth a
meaningful composite gain, given each carries an equal 1/24 share of macro-F1
regardless of how rare it is.

WHERE AN LLM CAN AND CANNOT HELP. Asking an LLM for a calibrated 4-way ordinal
risk judgement competes with a supervised model trained on 1263 examples, which
is the supervised model's strongest case. This asks instead for definitional
recognition on labels where the supervised model has eight examples, which is
its weakest. The two tasks are not interchangeable.

NOTHING LEAVES THE MACHINE. The model runs locally. No dataset post is sent to
any external API, which is why this route is available at all.

FOUR THINGS THAT MAKE THE RESULT TRUSTWORTHY.

1. CONTROL LABELS. Common labels are scored too. A null result on the rare
   labels means nothing unless the harness demonstrably works, so if
   `hopelessness` (45.6% prevalence) does not score well, the harness is broken
   and the rare-label numbers must be discarded.

2. PROBABILITIES, NOT LABELS. It scores logP(" Yes") against logP(" No") rather
   than parsing generated text. That yields a continuous score, so it can be
   thresholded, calibrated, and ENSEMBLED with the NLI model instead of only
   replacing it. It also makes AP -- a threshold-free ranking metric -- the
   headline number, which cannot be gamed by a lucky cutoff.

3. THRESHOLDS ARE CROSS-FITTED OVER AUTHOR HALVES. Fitting a cutoff and scoring
   it on the same posts makes any reading optimistic, and posts by one author
   are correlated, so the split must be by author rather than by post.

4. THE SAME DEFINITION STRING THE ENCODER USES. It imports `_definitions()` from
   train_factors_nli, so FACTOR_DEFS_JSON overrides apply identically and the
   comparison is genuinely one variable.

Usage:
  python src/qwen_rare_factor_probe.py --model <local-qwen> --limit 400   # smoke
  python src/qwen_rare_factor_probe.py --model <local-qwen>               # full
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402
from train_factors_nli import _definitions  # noqa: E402

# The targets are the labels the encoder is WEAKEST on, which is not the same
# set as the rarest labels. Rarity alone does not predict weakness here: the NLI
# formulation reads the definition rather than learning the concept from
# examples, so `poor school performance` scores 0.737 off sixteen positives.
#
# The weakness is ABSTRACT constructs, regardless of how much data they have.
# Held-out F1 (arm `mnli`, 5 author-disjoint folds) with gold counts:
#     meaning in life            45   0.231   <- at its 0.235 oracle ceiling
#     exposure to others' suicide 14  0.303
#     sense of responsibility     58  0.411
#     physical health            78   0.522
#     social support            112   0.533
#     stressful life event      182   0.544
#     psychological capital     238   0.577   <- 238 examples and still weak
#     interpersonal difficulty  195   0.591
# `psychological capital` has 238 positives, so it is not data-starved; the
# concept is simply hard to pin down from a cross-encoder hypothesis. That is
# where a definition-reading LLM has a plausible edge.
#
# `meaning in life` is excluded: at 0.231 against a measured oracle ceiling of
# 0.235 there is no headroom to recover, so including it would only add noise.
WEAK = [
    "exposure to others' suicide",
    "sense of responsibility",
    "physical health/characteristic",
    "social support",
    "stressful life event",
    "psychological capital",
]
# the rarest labels, kept so --factors can still target them
RARE = [
    "sexual orientation related issues",      # 8   F1 0.387
    "exposure to others' suicide",            # 14  F1 0.303
    "poor school performance",                # 16  F1 0.737
    "substance use",                          # 33  F1 0.568
    "cognitive deficits",                     # 33  F1 0.507
]

# The encoder's OWN held-out F1 from src/factor_oof_instrument.py on 1635 posts
# (arm `mnli`, pooled over 5 author-disjoint folds). Printed alongside the LLM's
# so the comparison needs no second window. The LLM has to BEAT these to matter.
ENCODER_OOF_F1 = {
    "meaning in life": 0.231,
    "exposure to others' suicide": 0.303,
    "sense of responsibility": 0.411,
    "physical health/characteristic": 0.522,
    "social support": 0.533,
    "stressful life event": 0.544,
    "psychological capital": 0.577,
    "interpersonal difficulty": 0.591,
    "cognitive deficits": 0.507,
    "substance use": 0.568,
    "sexual orientation related issues": 0.387,
    "poor school performance": 0.737,
    "hopelessness": 0.816,
    "mental health issues": 0.734,
}
# harness controls: if these do not score well, the rare numbers mean nothing
CONTROL = ["hopelessness", "mental health issues"]


def _one_turn(post: str, factor: str, definition: str, max_chars: int,
              with_codebook: bool) -> str:
    post = post.strip()
    if len(post) > max_chars:
        post = post[:max_chars] + " ..."
    head = ("You are annotating posts from an online mental-health support "
            "forum for a clinical research study, using a fixed codebook.\n\n"
            f'Codebook entry -- "{factor}": {definition}\n\n'
            if with_codebook else "")
    return (f"{head}Post:\n\"\"\"\n{post}\n\"\"\"\n\n"
            f'Question: according to that codebook entry, does the post show '
            f'"{factor}"?\n'
            "Answer with exactly one word: Yes or No.")


def build_prompt(tok, post: str, factor: str, definition: str,
                 max_post_chars: int, demos=None) -> str:
    """demos: list of (post, is_positive) shown as prior turns.

    WHY FEW-SHOT MATTERS HERE MORE THAN MODEL SIZE. The published definition is
    NOT the rule the annotators applied. `mental health issues` is defined as
    "existing mental disorders such as depression..." and in r/SuicideWatch
    nearly every post describes depression -- so a literal reading says Yes to
    everything and precision collapses. The encoder learned the implicit
    boundary from 228 labelled examples. A LARGER model applies the literal
    definition better, so scaling makes this failure worse, not better.
    Demonstrations are the only channel that carries the annotators' actual rule.
    """
    msgs = []
    for dpost, pos in (demos or []):
        msgs.append({"role": "user",
                     "content": _one_turn(dpost, factor, definition,
                                          max_post_chars // 3,
                                          with_codebook=not msgs)})
        msgs.append({"role": "assistant", "content": "Yes" if pos else "No"})
    msgs.append({"role": "user",
                 "content": _one_turn(post, factor, definition, max_post_chars,
                                      with_codebook=not msgs)})
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    flat = "\n\n".join(
        m["content"] if m["role"] == "user" else f"Answer: {m['content']}"
        for m in msgs)
    return flat + "\n\nAnswer:"


def yes_no_ids(tok) -> tuple[int, int]:
    """First token of each answer under this tokenizer. Both spaced and bare
    forms are tried because the chat template may already end mid-turn."""
    def first(s):
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if ids else None
    for y, n in ((" Yes", " No"), ("Yes", "No"), (" yes", " no"),
                 ("yes", "no")):
        a, b = first(y), first(n)
        if a is not None and b is not None and a != b:
            return a, b
    raise SystemExit("could not resolve distinct Yes/No token ids")


def author_halves(auth) -> np.ndarray:
    s = pd.Series(auth)
    c = s.value_counts()
    half, size = {}, {"A": 0, "B": 0}
    for a in sorted(c.index, key=lambda x: (-c[x], str(x))):
        p = "A" if size["A"] <= size["B"] else "B"
        half[a] = p
        size[p] += c[a]
    return s.map(half).to_numpy()


def best_f1(score, y):
    """Best achievable F1 and its cutoff, over the observed score values."""
    o = np.argsort(-score, kind="stable")
    ys = y[o]
    cum = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    npos = max(int(y.sum()), 1)
    f1 = 2 * cum / (k + npos)
    i = int(np.argmax(f1))
    return float(f1[i]), float(score[o][i])


def f1_at(score, y, thr):
    p = (score >= thr).astype(int)
    tp = int((p & y).sum())
    den = 2 * tp + int((p & (1 - y)).sum()) + int(((1 - p) & y).sum())
    return (2 * tp / den) if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="local path or cached HF id of an instruct model")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--out", default="artifacts/qwen_rare_probe.npz")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-post-chars", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N posts (smoke test)")
    ap.add_argument("--factors", default=None,
                    help="comma-separated override of which factors to score")
    ap.add_argument("--shots", type=int, default=0,
                    help="labelled demos per class. Demos are drawn from the "
                         "OPPOSITE author half, so no post ever sees an "
                         "example from its own author.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pd.read_csv(args.labeled, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    defs = _definitions()

    factors = ([f.strip() for f in args.factors.split(",")]
               if args.factors else WEAK + CONTROL)
    for f in factors:
        if f not in FACTOR_TAXONOMY:
            raise SystemExit(f"{f!r} is not in the taxonomy")
        if f not in defs:
            raise SystemExit(f"{f!r} has no definition")

    Y = np.zeros((len(df), len(factors)), dtype=int)
    for i, cell in enumerate(df["factors"]):
        got = {str(x).strip().lower() for x in parse_factors(cell)}
        for j, f in enumerate(factors):
            Y[i, j] = int(f in got)

    print(f"{len(df)} posts x {len(factors)} factors = "
          f"{len(df) * len(factors)} scored pairs")
    print(f"model: {args.model}")
    for j, f in enumerate(factors):
        tag = ("control" if f in CONTROL else
               "TARGET" if f in WEAK else "rare")
        print(f"  {f:<44}{int(Y[:, j].sum()):>5} positives   [{tag}]")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # last position must be the real one
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    yid, nid = yes_no_ids(tok)
    print(f"Yes/No token ids: {yid}/{nid}")

    # ---- demo selection, author-disjoint ------------------------------------
    # Posts in half A get demos drawn only from half B and vice versa, so a post
    # can never see a labelled example written by its own author. Two fixed demo
    # sets per factor (one per half) rather than per-post sampling: it removes a
    # noise source and keeps the prompts cacheable.
    demo_for = {}
    if args.shots > 0:
        dhalf = author_halves(df["anon_user_id"])
        rng = np.random.default_rng(0)
        for j, f in enumerate(factors):
            for h, src in (("A", dhalf == "B"), ("B", dhalf == "A")):
                pos = np.flatnonzero(src & (Y[:, j] == 1))
                neg = np.flatnonzero(src & (Y[:, j] == 0))
                if len(pos) < args.shots:
                    print(f"  NOTE {f} half {h}: only {len(pos)} positives "
                          f"available for {args.shots} shots")
                take_p = rng.choice(pos, min(args.shots, len(pos)),
                                    replace=False) if len(pos) else []
                take_n = rng.choice(neg, min(args.shots, len(neg)),
                                    replace=False) if len(neg) else []
                d = [(df["post"].iloc[k], True) for k in take_p] + \
                    [(df["post"].iloc[k], False) for k in take_n]
                rng.shuffle(d)          # don't let all Yes precede all No
                demo_for[(j, h)] = d
        print(f"few-shot: {args.shots} positive + {args.shots} negative demos "
              f"per factor, drawn from the opposite author half")
    else:
        dhalf = np.array(["A"] * len(df))

    pairs = [(i, j) for i in range(len(df)) for j in range(len(factors))]
    prompts = [build_prompt(tok, df["post"].iloc[i], factors[j],
                            defs[factors[j]], args.max_post_chars,
                            demo_for.get((j, dhalf[i])))
               for i, j in pairs]
    print(f"prompt chars: median "
          f"{int(np.median([len(p) for p in prompts]))}  "
          f"max {max(len(p) for p in prompts)}")
    # sort by length so batches are uniformly padded -- pure speed, no effect
    order = sorted(range(len(prompts)), key=lambda k: len(prompts[k]))
    P = np.zeros((len(df), len(factors)), dtype=float)

    done = 0
    with torch.no_grad():
        for s in range(0, len(order), args.batch):
            chunk = order[s:s + args.batch]
            enc = tok([prompts[k] for k in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=3072)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            logits = model(**enc).logits[:, -1, :].float()
            two = torch.stack([logits[:, yid], logits[:, nid]], dim=-1)
            pyes = torch.softmax(two, dim=-1)[:, 0].cpu().numpy()
            for k, p in zip(chunk, pyes):
                i, j = pairs[k]
                P[i, j] = float(p)
            done += len(chunk)
            if done % (args.batch * 50) < args.batch:
                print(f"  {done}/{len(order)} pairs", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, probs=P, factors=np.array(factors, dtype=object),
             row_id=np.array(df["row_id"].tolist(), dtype=object))
    print(f"\ndumped -> {args.out}")

    # ---- scoring -------------------------------------------------------------
    from sklearn.metrics import average_precision_score
    half = author_halves(df["anon_user_id"])
    A, B = half == "A", half == "B"

    print("\n" + "=" * 78)
    print("RESULTS   AP is threshold-free; xfit-F1 fits the cutoff on the other")
    print("          author half, so it cannot flatter itself")
    print("=" * 78)
    print(f"{'factor':<44}{'pos':>5}{'base AP':>9}{'AP':>8}{'xfitF1':>8}"
          f"{'encoder':>8}")
    print("-" * 86)
    rare_ap, ctrl_ap = [], []
    for j, f in enumerate(factors):
        y = Y[:, j]
        if y.sum() == 0:
            print(f"{f:<44}{0:>5}   no positives, skipped")
            continue
        base = float(y.mean())
        apv = float(average_precision_score(y, P[:, j]))
        _, tA = best_f1(P[:, j][A], y[A])
        _, tB = best_f1(P[:, j][B], y[B])
        xf = np.zeros(len(y), dtype=int)
        xf[A] = (P[:, j][A] >= tB).astype(int)
        xf[B] = (P[:, j][B] >= tA).astype(int)
        tp = int((xf & y).sum())
        den = 2 * tp + int((xf & (1 - y)).sum()) + int(((1 - xf) & y).sum())
        xf1 = (2 * tp / den) if den else 0.0
        (ctrl_ap if f in CONTROL else rare_ap).append(apv / max(base, 1e-9))
        # the encoder's own held-out F1, so the comparison is on one line
        enc = ENCODER_OOF_F1.get(f)
        vs = f"{enc:>8.3f}" if enc is not None else f"{'-':>8}"
        print(f"{f:<44}{int(y.sum()):>5}{base:>9.3f}{apv:>8.3f}{xf1:>8.3f}"
              + vs + ("" if f in CONTROL else "   <- TARGET"))

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if ctrl_ap:
        c = float(np.mean(ctrl_ap))
        print(f"control labels: AP is {c:.1f}x their base rate")
        if c < 1.5:
            print("  *** THE HARNESS IS BROKEN. The model cannot even pick out")
            print("  common labels, so the rare-label numbers below say nothing")
            print("  about the idea. Check the prompt, the chat template, and")
            print("  the Yes/No token ids before drawing any conclusion. ***")
            return
        print("  harness works -- the rare numbers are interpretable")
    if rare_ap:
        r = float(np.mean(rare_ap))
        print(f"rare labels:    AP is {r:.1f}x their base rate")
        print("\nThe bar: xfit-F1 around 0.40+ on the rare labels would beat a")
        print("supervised model that has 8-33 examples to learn from, and 21% of")
        print("S2 rides on those five. Around 0.10 means the approach carries no")
        print("usable signal on them.")
    print("\nNEXT, if it clears the bar: ensemble rather than replace. These are")
    print("calibrated probabilities, so they can be averaged with the NLI")
    print("member's, and the rare-label thresholds re-fitted on the OOF")
    print("instrument -- never on the posts being scored.")


if __name__ == "__main__":
    main()
