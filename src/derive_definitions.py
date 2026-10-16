"""
derive_definitions.py

Derives each factor's definition FROM THE ANNOTATIONS rather than taking the
published wording as given.

THE PROBLEM. For several factors the published definition and the annotators'
applied boundary diverge: a post can satisfy the published wording and still
be labelled negative, and vice versa. An NLI hypothesis built from the
published definition is then scoring the wrong criterion, and the model is
penalised for applying that definition correctly.

The rule the annotators actually used exists only in their labels, so it has
to be recovered from them. This script prompts a model with labelled positives
and negatives for one factor and asks for a replacement definition that sorts
those posts the same way.

TWO THINGS THIS SCRIPT GETS RIGHT, both of which are easy to get wrong.

  GRAMMAR. The output is spliced into "This post shows <factor>: <text>", so
  the text must complete that sentence as a property OF THE POST. A prompt
  asking for "the rule the annotators followed" yields a rule-shaped answer
  ("applied when the person names ..."), which splices into a malformed
  sentence describing annotators rather than the post. The prompt below asks
  for the completion directly, and clean() validates the output against that
  shape and rejects rule-shaped openings.

  LEAKAGE. Definitions are derived PER FOLD from that fold's TRAINING portion
  only. Fold f's definitions never see fold f's posts. A definition written
  from the evaluation posts would produce a large gain that is pure
  circularity, and it would look exactly like success.

EXAMPLE BUDGET. deepseek-chat has a 64K context, so there is no reason to be
stingy: up to 60 positives and 60 negatives at 600 characters each is roughly
18K tokens, comfortably inside it. Rare labels contribute every positive they
have (8 for `sexual orientation related issues`); common labels are sampled.

This sends real dataset posts to an external API. There is no local-model path
in this script -- if that is unacceptable for this data, do not run it.

Usage:
  export DEEPSEEK_API_KEY=...
  python src/derive_definitions.py --out artifacts/derived_defs
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from scoring import FACTOR_TAXONOMY, parse_factors

API = "https://api.deepseek.com/chat/completions"
BAD_STARTS = (
    "applied when", "annotators", "the annotators", "this label",
    "label is", "labelled when", "labeled when", "used when", "marked when",
)


def call(key, prompt, temperature, max_tokens, retries=5):
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            if a == retries - 1:
                print(f"      API failed after {retries} tries: "
                      f"{type(e).__name__}")
                return None
            time.sleep(2 ** a)
    return None


def clean(txt, factor=None):
    """Normalise the completion and reject anything that will not slot into
    'This post shows <factor>: <text>'.

    The factor name is needed because models frequently echo the whole stem
    back. Stripping only "this post shows" leaves "<factor>: ..." behind, which
    splices into "This post shows meaning in life: meaning in life: ..." --
    the label name duplicated in every affected hypothesis."""
    if not txt:
        return None
    t = " ".join(txt.strip().split())
    t = t.strip().strip('"').strip("'").strip()
    # models sometimes echo the sentence they were shown, in whole or part
    for pre in ("this post shows", "the post shows"):
        if t.lower().startswith(pre):
            t = t[len(pre):].lstrip(" :-")
    if factor and t.lower().startswith(factor.lower()):
        rest = t[len(factor):].lstrip()
        if rest.startswith(":"):
            t = rest.lstrip(": ").strip()
    t = t.rstrip(".").strip()
    if not t or len(t.split()) > 60 or len(t) < 12:
        return None
    if t.lower().startswith(BAD_STARTS):
        return None
    return t[0].lower() + t[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-dir", default="artifacts/folds")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-pos", type=int, default=60)
    ap.add_argument("--n-neg", type=int, default=60)
    ap.add_argument("--chars", type=int, default=600)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--out", default="artifacts/derived_defs")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("set DEEPSEEK_API_KEY")

    # Never derive definitions THROUGH an existing override -- that would
    # compound one derivation on top of another.
    os.environ.pop("FACTOR_DEFS_JSON", None)
    from train_factors_nli import FACTOR_DEFINITIONS

    missing = [t for t in FACTOR_TAXONOMY if t not in FACTOR_DEFINITIONS]
    if missing:
        raise SystemExit(f"no published definition for: {missing}")

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    for f in range(args.k):
        dst = os.path.join(args.out, f"defs_f{f}.json")
        if os.path.exists(dst):
            print(f"--- {dst} exists, skipping ---")
            continue
        p = os.path.join(args.folds_dir, f"train_f{f}.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} (scripts/nli_risk_5fold.sh writes it)")
        tr = pd.read_csv(p)
        sets = [set(parse_factors(c)) & set(FACTOR_TAXONOMY)
                for c in tr["factors"]]
        posts = [str(x) for x in tr["post"]]
        print(f"\n=== fold {f}: {len(tr)} TRAINING posts "
              f"(fold {f}'s own posts are NOT visible here) ===")

        def build(factor):
            pos = [i for i, s in enumerate(sets) if factor in s]
            # hard negatives: posts that carry SOME factor, so the boundary is
            # drawn against annotated posts rather than against blanks
            hard = [i for i, s in enumerate(sets) if factor not in s and s]
            if not pos:
                return None, 0, 0
            rng.shuffle(pos)
            rng.shuffle(hard)
            P, N = pos[:args.n_pos], hard[:args.n_neg]
            pt = "\n\n".join(f"[{n+1}] {posts[i][:args.chars]}"
                             for n, i in enumerate(P))
            nt = "\n\n".join(f"[{n+1}] {posts[i][:args.chars]}"
                             for n, i in enumerate(N))
            prompt = (
                f"Annotators labelled posts from a suicide-support forum with "
                f"risk and protective factors. One factor is \"{factor}\".\n\n"
                f"Its published definition is:\n"
                f"  \"{FACTOR_DEFINITIONS[factor]}\"\n\n"
                f"That definition does not always match what the annotators "
                f"actually did. Study these examples.\n\n"
                f"=== {len(P)} POSTS THEY LABELLED \"{factor}\" ===\n{pt}\n\n"
                f"=== {len(N)} POSTS THEY DID NOT ===\n{nt}\n\n"
                f"Your job: produce a replacement definition that a reader "
                f"could apply to sort these posts the same way the annotators "
                f"did.\n\n"
                f"It must complete this sentence naturally:\n"
                f"  \"This post shows {factor}: ______\"\n\n"
                f"Requirements:\n"
                f"- Describe what is present IN THE POST. Do not mention "
                f"annotators, labels, or labelling.\n"
                f"- Start with a noun phrase, exactly like the published "
                f"definition does.\n"
                f"- If their boundary is narrower or broader than the "
                f"published definition, encode that difference.\n"
                f"- Where a similar-looking case does NOT count, say so "
                f"briefly (e.g. \"...; not X\").\n"
                f"- Lowercase, under 45 words, no trailing full stop.\n"
                f"- Output ONLY the completion text. No preamble, no quotes."
            )
            return prompt, len(P), len(N)

        built = {t: build(t) for t in FACTOR_TAXONOMY}
        defs, kept, fell_back = {}, 0, []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut = {ex.submit(call, key, pr, args.temperature, args.max_tokens): t
                   for t, (pr, _, _) in built.items() if pr}
            for fu in as_completed(fut):
                t = fut[fu]
                c = clean(fu.result(), factor=t)
                if c:
                    defs[t] = c
                    kept += 1
                else:
                    defs[t] = FACTOR_DEFINITIONS[t]
                    fell_back.append(t)
        for t in FACTOR_TAXONOMY:
            defs.setdefault(t, FACTOR_DEFINITIONS[t])
            if t not in fell_back and defs[t] == FACTOR_DEFINITIONS[t]:
                fell_back.append(t)

        json.dump(defs, open(dst, "w"), indent=2)
        print(f"  derived {kept}/24, fell back to published for "
              f"{len(fell_back)}: {fell_back if fell_back else 'none'}")
        print(f"  wrote {dst}")
        for t in ("meaning in life", "sense of responsibility",
                  "exposure to others' suicide", "hopelessness"):
            _, np_, nn_ = built[t]
            print(f"\n  {t}   ({np_} pos / {nn_} neg shown)")
            print(f"     published: {FACTOR_DEFINITIONS[t]}")
            print(f"     derived  : {defs[t]}")
            print(f"     sentence : \"This post shows {t}: {defs[t]}\"")

    print("\n" + "=" * 78)
    print("BEFORE TRAINING, read the 'sentence' lines above. Each must be")
    print("grammatical and describe the POST. If any reads as a rule about")
    print("annotators, the completion slipped through and the NLI model would")
    print("be scoring a malformed hypothesis.")
    print("\nThen: sbatch --export=ALL scripts/factors_derived_defs.sh")
    print("Each fold uses only its own training-derived definitions.")
    print("=" * 78)


if __name__ == "__main__":
    main()
