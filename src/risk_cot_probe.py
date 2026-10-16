"""
risk_cot_probe.py

Can a local instruct model REASON its way to the risk label without being told
the answer?

WHY THIS COMES BEFORE ANY CoT TRAINING. The proposal is to write reasoning
traces for the training posts and distil them. But a trace generated FROM the
gold label is a rationalisation: the model will justify whatever label it is
handed, and for the substantial share of labels that rest on annotator context
rather than post text (the "11/12/13/14 days and counting" countdown series, the
near-duplicate posts, an author's history) it will invent a reason that is not
there. Training on invented reasons teaches invention.

So the prior question is whether reasoning helps AT ALL, measured with the gold
label hidden. If a strong model with the definitions in front of it, thinking
step by step, cannot separate these classes, distillation has nothing to distil.

WHY THERE IS REASON TO HOPE, from measurements already in this repo:
  - the 2024 competition winner was few-shot CoT Qwen2-72B + LoRA generative
    models; the Qwen member trained so far used no reasoning in the prompt at all
  - on the confident ideation/behavior errors, classifying from the ANNOTATOR'S
    SPAN scores 0.692 while classifying from the FULL POST scores 0.308 --
    worse than chance. The signal is in one clause and the surrounding despair
    vocabulary drowns it. Isolating that clause is exactly what reasoning does.

WHAT DECIDES IT. Not raw accuracy -- the encoder chain is at 0.7985 wF1 and a
zero-shot LLM will not beat that. What matters is COMPLEMENTARITY: does it get
right the posts the encoder gets wrong? A member only pays if it breaks the
chain's failure pattern, and the risk audit found the chain's members all fail
on the SAME posts.

PRE-STATED BAR, fixed before looking: on the posts the encoder gets WRONG, the
CoT model must be right on >= 35% (chance among the 3 remaining classes is
~33%, and the encoder's own second choice is right about 45% of the time -- so
below 35% it adds nothing a cheaper reranking of the encoder's own
probabilities would not). Miss it and the CoT lane closes before any GPU goes
into training.

EVERYTHING IS LOCAL. No dataset post leaves the cluster.

Usage:
  python src/risk_cot_probe.py --model /path/to/local/model
  python src/risk_cot_probe.py --list                 # what is on disk
  python src/risk_cot_probe.py --model ... --n 400 --no-cot   # ablation
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402

DEFS = {
    "indicator": "distress with no explicit mention of suicide, self-harm or "
                 "wanting to die",
    "ideation": "an explicit suicidal thought or wish, and nothing further: no "
                "method, no time, no self-harm, nothing prepared, no attempt",
    "behavior": "action or preparation short of an attempt: self-harm, a named "
                "or available method, a stated time, or preparation for death",
    "attempt": "an act the author ALREADY CARRIED OUT to end their life",
}

# Hand-written exemplars. Deliberately chosen as the CONTRASTIVE pair the whole
# argument turns on, plus one indicator and one behavior, so the model sees the
# distinction demonstrated rather than described.
SHOTS = [
    ("I have been failing every class and I cannot get out of bed. My parents "
     "say I am lazy.",
     "The post describes distress and school failure. It never mentions "
     "suicide, self-harm or wanting to die, and names no method or time.",
     "indicator"),
    ("I just want to die. I think about it every day but I would never "
     "actually do anything.",
     "The author explicitly says they want to die, so this is past indicator. "
     "But it is only a thought: no method, no time, no self-harm and nothing "
     "prepared. The author explicitly rules out acting.",
     "ideation"),
    ("I've been trying to find ways to hang myself. I keep looking things up "
     "at night.",
     "The author is seeking a method of dying, which goes beyond a wish. But "
     "nothing has been carried out yet -- they are still searching.",
     "behavior"),
    ("I tried to hang myself last month. I woke up in hospital and nothing has "
     "changed since.",
     "The author states an act already carried out to end their life, and the "
     "hospital confirms it happened. This is a completed attempt in the past, "
     "not a plan.",
     "attempt"),
]


def build_prompt(post: str, cot: bool) -> str:
    lines = ["You are labelling Reddit posts for suicide risk level.",
             "The four levels, from lowest to highest:"]
    for k in RISK_LEVELS:
        lines.append(f"  {k}: {DEFS[k]}")
    lines += [
        "",
        "The levels are a ladder. If a post reaches more than one level, the "
        "HIGHEST one applies.",
        "",
    ]
    for p, r, y in SHOTS:
        lines.append(f"Post: {p}")
        if cot:
            lines.append(f"Reasoning: {r}")
        lines.append(f"Level: {y}")
        lines.append("")
    lines.append(f"Post: {post}")
    lines.append("Reasoning:" if cot else "Level:")
    return "\n".join(lines)


def parse_level(text: str) -> str | None:
    m = re.findall(r"\b(indicator|ideation|behaviou?r|attempt)\b", text, re.I)
    if not m:
        return None
    last = m[-1].lower()
    return "behavior" if last.startswith("behavio") else last


def list_models():
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
        "hub")
    # NIDHI_MODEL_DIR: an extra directory of local model copies, if any.
    roots = [hub, "models"] + ([os.environ["NIDHI_MODEL_DIR"]]
                               if os.environ.get("NIDHI_MODEL_DIR") else [])
    seen = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for p in glob.glob(os.path.join(r, "*")):
            if os.path.isdir(p):
                seen.append(p)
    print("candidate local model directories:")
    for p in sorted(seen):
        try:
            sz = sum(os.path.getsize(f) for f in glob.glob(
                os.path.join(p, "**", "*"), recursive=True)
                if os.path.isfile(f) and not os.path.islink(f))
            print(f"  {p}   {sz / 1e9:.1f} GB")
        except OSError:
            print(f"  {p}")
    print("\nNOTE: `du -sh` under-reports HF snapshots because they are "
          "symlinks. Use `du -shL`.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--oof", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--n", type=int, default=400,
                    help="sample size; 0 for all")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--no-cot", action="store_true",
                    help="ablation: same prompt, label only, no reasoning")
    ap.add_argument("--max-post-words", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/cot_probe.csv")
    ap.add_argument("--extra-csv", default=None,
                    help="score these posts too (row_id,post; no gold needed) "
                         "in the SAME process. Loading a 4-bit 72B took 1h34m, "
                         "so a second job to score test would cost more than "
                         "the scoring itself. Written to <out>.extra.csv.")
    ap.add_argument("--extra-risk-from", default=None,
                    help="risk levels for --extra-csv, only needed if the "
                         "prompt were conditioned on them; unused today but "
                         "kept so the flag pair stays symmetric")
    ap.add_argument("--load-4bit", action="store_true",
                    help="NF4 -- a 72B fits on one 80GB card at ~40GB")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--batch", type=int, default=8,
                    help="batched greedy generation with LEFT padding")
    ap.add_argument("--samples", type=int, default=1,
                    help="SELF-CONSISTENCY: sample this many reasoning chains "
                         "per post and take the majority label, with the VOTE "
                         "FRACTION as the probability.\n"
                         "This fixes TWO things. (1) Majority over sampled "
                         "chains is the standard accuracy boost for reasoning "
                         "models. (2) It replaces a USELESS confidence signal: "
                         "the 72B's log-prob confidence saturates near 1.0 with "
                         "almost no gap between right and wrong answers, so no "
                         "gate can be built on it. A vote fraction over k chains has "
                         "real spread and is a genuine uncertainty estimate. "
                         "Implies do_sample with temperature.")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="only used when --samples > 1")
    ap.add_argument("--soft", action="store_true",
                    help="after the reasoning, score each of the 4 level "
                         "strings by total log-probability and softmax them. "
                         "Gives a blendable distribution instead of a hard "
                         "label, and removes text parsing entirely -- 10.7% of "
                         "generations were unparseable and defaulted to "
                         "`indicator`, which understated the CoT arm.")
    ap.add_argument("--focus-errors", action="store_true",
                    help="sample half from posts the encoder gets WRONG. The "
                         "rescue rate is the quantity that decides this lane, "
                         "so spend generations there rather than on posts the "
                         "encoder already handles. Overall wF1 on such a "
                         "sample is NOT comparable to the full set.")
    args = ap.parse_args()

    if args.list or not args.model:
        list_models()
        if not args.model:
            raise SystemExit("\npass --model <dir>")
        return

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    d = pd.read_csv(args.oof, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    d = d[d["row_id"].isin(lab.index)].copy()
    d["g"] = [normalize_label(x) for x in d["gold"]]
    d["enc"] = [normalize_label(x) for x in d["pred_risk_level"]]

    rng = np.random.default_rng(args.seed)
    if args.n and args.n < len(d):
        if args.focus_errors:
            # half from encoder-wrong (rescue rate), half from encoder-right
            # (breakage rate). Both are needed; neither needs the natural mix.
            wrong = d.index[d.g != d.enc].to_numpy()
            right = d.index[d.g == d.enc].to_numpy()
            kw = min(args.n // 2, len(wrong))
            kr = min(args.n - kw, len(right))
            take = list(rng.choice(wrong, kw, replace=False)) + \
                list(rng.choice(right, kr, replace=False))
        else:
            take = []
            for c in RISK_LEVELS:
                sub = d.index[d.g == c].to_numpy()
                k = max(1, int(round(args.n * len(sub) / len(d))))
                take += list(rng.choice(sub, min(k, len(sub)), replace=False))
        d = d.loc[sorted(take)].reset_index(drop=True)
    n_wrong = int((d.g != d.enc).sum())
    print(f"{len(d)} posts ({n_wrong} the encoder gets wrong)   "
          f"encoder wF1 on this sample "
          f"{risk_weighted_f1(list(d.g), list(d.enc)):.4f}")
    if args.focus_errors:
        print("  NOTE: errors are deliberately over-sampled -- the wF1 above "
              "is NOT\n  comparable to the full set. Read rescue and breakage, "
              "not accuracy.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"device_map": "auto"}
    if args.load_4bit or args.load_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise SystemExit("bitsandbytes/transformers too old for 4-bit")
        if args.load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
            print("[quant] NF4 4-bit")
        else:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print("[quant] 8-bit")
    else:
        kw["torch_dtype"] = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    mdl.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # LEFT padding: with right padding the generated continuation starts after
    # the pad run and the decode slice below is wrong for every short prompt.
    tok.padding_side = "left"

    prompts = []
    for r in d.itertuples():
        post = " ".join(str(lab.loc[r.row_id, "post"]).split())
        post = " ".join(post.split()[:args.max_post_words])
        prompts.append(build_prompt(post, cot=not args.no_cot))

    def label_logprobs(ctx: str) -> np.ndarray:
        """Total log P(level string | ctx) for each of the 4 levels.

        Teacher-forces each candidate continuation and sums the token
        log-probs. Scoring the FULL string avoids depending on whether the
        tokenizer gives `indicator` and `ideation` distinct first tokens.
        """
        outs = []
        for lv in RISK_LEVELS:
            full = tok(ctx + " " + lv, return_tensors="pt").to(mdl.device)
            n_ctx = tok(ctx, return_tensors="pt")["input_ids"].shape[1]
            with torch.no_grad():
                lg = mdl(**full)["logits"][0].float()
            lp = torch.log_softmax(lg[:-1], -1)
            tgt = full["input_ids"][0, 1:]
            tok_lp = lp[torch.arange(len(tgt)), tgt]
            outs.append(float(tok_lp[n_ctx - 1:].sum()))
        a = np.array(outs)
        a = a - a.max()
        e = np.exp(a)
        return e / e.sum()

    preds, raws, softs = [], [], []
    B = max(1, args.batch)
    K = max(1, args.samples)
    if K > 1:
        print(f"[self-consistency] {K} sampled chains at T={args.temperature}; "
              f"label = majority, probability = vote fraction")
    for s in range(0, len(prompts), B):
        chunk = prompts[s:s + B]
        enc = tok(chunk, return_tensors="pt", padding=True).to(mdl.device)
        cut = enc["input_ids"].shape[1]

        if K > 1:
            votes = [[] for _ in chunk]
            first = [""] * len(chunk)
            for k in range(K):
                with torch.no_grad():
                    out = mdl.generate(**enc, max_new_tokens=args.max_new,
                                       do_sample=True,
                                       temperature=args.temperature,
                                       top_p=0.95,
                                       pad_token_id=tok.pad_token_id)
                for bi, row in enumerate(out):
                    g = tok.decode(row[cut:], skip_special_tokens=True)
                    g = g.split("Post:")[0].strip()
                    if k == 0:
                        first[bi] = g.replace("\n", " ")[:300]
                    lv = parse_level(g)
                    if lv:
                        votes[bi].append(lv)
            for bi in range(len(chunk)):
                v = votes[bi]
                raws.append(first[bi])
                if not v:
                    preds.append("indicator")
                    softs.append(np.full(len(RISK_LEVELS),
                                         1.0 / len(RISK_LEVELS)))
                    continue
                frac = np.array([v.count(c) / len(v) for c in RISK_LEVELS])
                softs.append(frac)
                preds.append(RISK_LEVELS[int(frac.argmax())])
        else:
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=args.max_new,
                                   do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for bi, row in enumerate(out):
                gen = tok.decode(row[cut:], skip_special_tokens=True)
                gen = gen.split("Post:")[0].strip()
                raws.append(gen.replace("\n", " ")[:300])
                if args.soft:
                    # cut at its own "Level:" so the model is not asked to
                    # re-decide after it already committed in text
                    reason = gen.split("Level:")[0].rstrip()
                    P = label_logprobs(f"{chunk[bi]}{reason}\nLevel:")
                    softs.append(P)
                    preds.append(RISK_LEVELS[int(P.argmax())])
                else:
                    p = parse_level(gen)
                    preds.append(p if p else "indicator")
        if (s // B) % 5 == 0:
            print(f"  {min(s + B, len(prompts))}/{len(prompts)}")

    # self-consistency produces a probability vector too (the vote fraction),
    # so the prob columns must be written in that mode as well or the whole
    # point of the run is discarded
    soft_out = bool(args.soft) or args.samples > 1

    d["cot"] = preds
    d["trace"] = raws
    cols = ["row_id", "g", "enc", "cot", "trace"]
    if soft_out:
        S = np.stack(softs)
        for j, c in enumerate(RISK_LEVELS):
            d[f"prob_{c}"] = S[:, j]
        cols += [f"prob_{c}" for c in RISK_LEVELS]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    d[cols].to_csv(args.out, index=False)

    y = d.g.to_numpy()
    enc_p = d.enc.to_numpy()
    cot_p = d.cot.to_numpy()
    if soft_out:
        if args.samples > 1:
            print(f"\nself-consistency over {args.samples} chains: "
                  f"probability = vote fraction")
        else:
            print("\nsoft mode: labels come from log-prob scoring, so nothing "
                  "is unparseable")
        conf = S.max(1)
        print(f"  mean confidence {conf.mean():.3f}   "
              f"below 0.5: {int((conf < 0.5).sum())} of {len(conf)}")
        right = cot_p == y
        print(f"  mean confidence when RIGHT {conf[right].mean():.3f}   "
              f"when WRONG {conf[~right].mean():.3f}")
        print("  A gap here means confidence is usable as a blend gate; no gap")
        print("  means it is confidently wrong like our encoder is.")
    else:
        n_unparsed = sum(1 for t in raws if parse_level(t) is None)
        print(f"\nunparseable generations: {n_unparsed} "
              f"(defaulted to indicator)")
    print(f"\n{'model':<10}{'wF1':>9}{'acc':>8}")
    print("-" * 27)
    print(f"{'encoder':<10}{risk_weighted_f1(list(y), list(enc_p)):>9.4f}"
          f"{(y == enc_p).mean():>8.3f}")
    print(f"{'CoT LLM':<10}{risk_weighted_f1(list(y), list(cot_p)):>9.4f}"
          f"{(y == cot_p).mean():>8.3f}")

    print(f"\n{'class':<12}{'gold':>6}{'enc rec':>9}{'cot rec':>9}")
    for c in RISK_LEVELS:
        m = y == c
        if m.sum():
            print(f"{c:<12}{int(m.sum()):>6}{(enc_p[m] == c).mean():>9.3f}"
                  f"{(cot_p[m] == c).mean():>9.3f}")

    print("\n" + "=" * 60)
    print("COMPLEMENTARITY -- the number that actually decides this")
    print("=" * 60)
    wrong = enc_p != y
    if wrong.sum():
        rescue = float((cot_p[wrong] == y[wrong]).mean())
        print(f"encoder wrong on {int(wrong.sum())} posts; CoT right on "
              f"{rescue:.3f} of them")
        agree_wrong = float((cot_p[wrong] == enc_p[wrong]).mean())
        print(f"  CoT repeats the encoder's exact error {agree_wrong:.3f} "
              f"of the time")
        right = ~wrong
        break_rate = float((cot_p[right] != y[right]).mean())
        print(f"encoder right on {int(right.sum())} posts; CoT breaks "
              f"{break_rate:.3f} of them")
        print(f"\nBAR: rescue >= 0.35. {'PASSES' if rescue >= 0.35 else 'FAILS'}"
              f" -- {'build the CoT training set' if rescue >= 0.35 else 'the CoT lane closes before any training GPU'}")
    print("=" * 60)
    print(f"\nwrote {args.out} -- read the traces on rescued posts; if the")
    print("reasoning is sound they are worth distilling, if it is lucky")
    print("guessing dressed as reasoning they are not.")

    if args.extra_csv:
        from data_audit import load as _load
        ex = _load(args.extra_csv)
        ex["row_id"] = ex["row_id"].astype(str)
        print(f"\n{'=' * 60}")
        print(f"EXTRA PASS -- {len(ex)} posts from {args.extra_csv}, no gold")
        print("=" * 60)
        eposts = [" ".join(str(p).split()[:args.max_post_words])
                  for p in ex["post"]]
        eprompts = [build_prompt(p, cot=not args.no_cot) for p in eposts]
        epred, eraw, esoft = [], [], []
        # MUST mirror the main loop exactly. Running greedy here while the OOF
        # pass sampled would recreate the regime mismatch this mode exists to
        # remove -- the deployed event must be the measured event.
        for s in range(0, len(eprompts), B):
            chunk = eprompts[s:s + B]
            enc = tok(chunk, return_tensors="pt", padding=True).to(mdl.device)
            cutx = enc["input_ids"].shape[1]

            if K > 1:
                votes = [[] for _ in chunk]
                first = [""] * len(chunk)
                for k in range(K):
                    with torch.no_grad():
                        out = mdl.generate(**enc, max_new_tokens=args.max_new,
                                           do_sample=True,
                                           temperature=args.temperature,
                                           top_p=0.95,
                                           pad_token_id=tok.pad_token_id)
                    for bi, row in enumerate(out):
                        g = tok.decode(row[cutx:], skip_special_tokens=True)
                        g = g.split("Post:")[0].strip()
                        if k == 0:
                            first[bi] = g.replace("\n", " ")[:300]
                        lv = parse_level(g)
                        if lv:
                            votes[bi].append(lv)
                for bi in range(len(chunk)):
                    v = votes[bi]
                    eraw.append(first[bi])
                    if not v:
                        epred.append("indicator")
                        esoft.append(np.full(len(RISK_LEVELS),
                                             1.0 / len(RISK_LEVELS)))
                        continue
                    frac = np.array([v.count(c) / len(v) for c in RISK_LEVELS])
                    esoft.append(frac)
                    epred.append(RISK_LEVELS[int(frac.argmax())])
            else:
                with torch.no_grad():
                    out = mdl.generate(**enc, max_new_tokens=args.max_new,
                                       do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                for bi, row in enumerate(out):
                    gen = tok.decode(row[cutx:], skip_special_tokens=True)
                    gen = gen.split("Post:")[0].strip()
                    eraw.append(gen.replace("\n", " ")[:300])
                    if args.soft:
                        reason = gen.split("Level:")[0].rstrip()
                        P = label_logprobs(f"{chunk[bi]}{reason}\nLevel:")
                        esoft.append(P)
                        epred.append(RISK_LEVELS[int(P.argmax())])
                    else:
                        p = parse_level(gen)
                        epred.append(p if p else "indicator")
            if (s // B) % 5 == 0:
                print(f"  {min(s + B, len(eprompts))}/{len(eprompts)}")
        eo = pd.DataFrame({"row_id": list(ex["row_id"]), "cot": epred,
                           "trace": eraw})
        if soft_out:
            ES = np.stack(esoft)
            for j, c in enumerate(RISK_LEVELS):
                eo[f"prob_{c}"] = ES[:, j]
        ep = args.out + ".extra.csv"
        eo.to_csv(ep, index=False)
        from collections import Counter
        print(f"  predicted mix: {dict(Counter(epred))}")
        print(f"  wrote {ep}")


if __name__ == "__main__":
    main()
