"""
gen_rare_factors_api.py

Generate synthetic training posts for the rare factor labels via the DeepSeek
API, with length and diversity designed into the generation.

WHY LENGTH AND DIVERSITY ARE GENERATION-TIME CONCERNS. Surface style (capital
"I", apostrophes, trailing periods, newlines) can be normalised after the fact
by src/style_match_synthetic.py. Length and lexical diversity cannot: truncating
a post can remove the very factor it was written to express, and no rewrite can
turn a set of near-paraphrases into distinct posts. A generator left to its own
defaults clusters around one length and reuses a handful of openings, which
gives the classifier a synthetic dialect to learn instead of the real one.

HOW THAT IS ENFORCED
  - each post's target length is sampled from the REAL corpus length
    distribution, so the output inherits its spread (many short posts, a long
    tail) rather than clustering at one modal length
  - a large prompt grid -- 30 situations x 8 voices x 6 opening styles
  - the generic openings LLMs default to are banned outright (see BANNED)
  - the real corpus's surface conventions are applied during generation rather
    than patched on afterwards

RUNS ON THE LOGIN NODE. Compute nodes have no network. This needs none of the
GPU queue, so it can run alongside the factor jobs.

API KEY comes from the DEEPSEEK_API_KEY environment variable and is never
written to disk or logs:

    export DEEPSEEK_API_KEY=...
    python src/gen_rare_factors_api.py --out artifacts/syn_rare_api.csv

A key that has been pasted into a chat, a terminal history, or a shell script
should be rotated. Nothing real leaves the cluster here -- the prompts contain
only factor definitions from the published task paper, never a dataset post --
but the key itself is worth treating as compromised once it has been shared.

Usage:
  export DEEPSEEK_API_KEY=...
  python src/gen_rare_factors_api.py --target-count 150 --workers 8
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from scoring import FACTOR_TAXONOMY, parse_factors

API = "https://api.deepseek.com/chat/completions"

PROTECTIVE = {
    "social support", "coping strategy", "psychological capital",
    "sense of responsibility", "meaning in life",
}

SITUATIONS = [
    "a first-year university student living in halls",
    "someone in their thirties unemployed for eight months",
    "a sixteen-year-old still at school",
    "a single parent of a toddler",
    "someone six months after a divorce",
    "a night-shift warehouse worker",
    "someone who moved abroad for work and knows nobody",
    "a person in their forties caring for a parent with dementia",
    "a recent graduate on their fortieth job application",
    "someone eight years into therapy",
    "a person sharing a flat with strangers",
    "someone whose partner left three weeks ago",
    "a nurse coming off a run of double shifts",
    "a truck driver who is away from home most weeks",
    "someone in their fifties made redundant after twenty years",
    "a postgraduate student whose funding just ended",
    "a retail worker on a zero-hours contract",
    "someone living back with their parents at twenty-eight",
    "a person recently released from hospital",
    "an only child whose parents recently separated",
    "someone in a small town with no transport",
    "a person on a long NHS waiting list",
    "a freelancer who has had no work for two months",
    "someone whose best friend moved away last year",
    "a person in their twenties with a new baby",
    "a teacher in their third year",
    "someone doing an apprenticeship they hate",
    "a person who has just failed their driving test again",
    "someone whose visa application was refused",
    "a person working from home who has not spoken to anyone in days",
]
VOICES = [
    "numb and matter-of-fact", "exhausted, rambling, no paragraphs",
    "angry and bitter", "quiet and resigned",
    "panicked, jumping between thoughts", "flat, listing things",
    "apologetic, over-explaining", "detached, almost clinical",
]
OPENINGS = [
    "start mid-thought, as if continuing something",
    "start with a concrete detail about today",
    "start with a question",
    "start by apologising for posting",
    "start with a time reference (last night, this morning, for months)",
    "start with a flat statement of fact",
]
BANNED = ["i feel so lost", "i can't take it", "i feel so alone",
          "i feel like i'm", "i cant do this", "i don't know what to do"]


def call(key, prompt, temperature, retries=4):
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 400,
    }).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, urllib.error.HTTPError,
                KeyError, TimeoutError) as e:
            if a == retries - 1:
                return None
            time.sleep(2 ** a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--target-count", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--out", default="artifacts/syn_rare_api.csv")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("set DEEPSEEK_API_KEY in the environment")

    from train_factors_nli import FACTOR_DEFINITIONS

    rng = random.Random(args.seed)
    real = pd.read_csv(args.labeled)
    cnt = collections.Counter(
        f for c in real["factors"]
        for f in set(parse_factors(c)) & set(FACTOR_TAXONOMY))

    # Real length distribution, so generated lengths inherit its spread instead
    # of clustering. Clipped to 15-160 to avoid asking for a 3-word post.
    lens = real["post"].astype(str).str.split().str.len()
    pool = [int(x) for x in lens if 15 <= x <= 160]
    print(f"real length pool: n={len(pool)} median={int(np.median(pool))} "
          f"p90={int(np.percentile(pool, 90))}")

    need = {f: min(args.target_count - cnt.get(f, 0), args.target_count)
            for f in FACTOR_TAXONOMY if cnt.get(f, 0) < args.target_count}
    print(f"{'factor':<44}{'real':>6}{'to gen':>8}")
    for f, n in sorted(need.items(), key=lambda x: cnt.get(x[0], 0)):
        print(f"{f:<44}{cnt.get(f,0):>6}{n:>8}")
    total = sum(need.values())
    print(f"\ntotal {total} posts via DeepSeek, {args.workers} workers")

    def build(factor, i):
        d = FACTOR_DEFINITIONS[factor]
        who = SITUATIONS[rng.randrange(len(SITUATIONS))]
        voice = VOICES[rng.randrange(len(VOICES))]
        opener = OPENINGS[rng.randrange(len(OPENINGS))]
        words = pool[rng.randrange(len(pool))]
        if factor in PROTECTIVE:
            core = (f"The post must clearly show this PROTECTIVE element:\n"
                    f"  {factor}: {d}\n"
                    f"It is genuinely present and pulling against the "
                    f"despair, doing real work in the post. The person is "
                    f"still in a bad place -- do NOT resolve their distress.")
        else:
            core = (f"The post must show this RISK FACTOR:\n"
                    f"  {factor}: {d}\n"
                    f"It must be a real contributing cause of how they feel, "
                    f"not a passing mention of the topic.")
        return (
            f"Write one anonymous post for a suicide-support forum.\n\n"
            f"Author: {who}\nVoice: {voice}\nOpening: {opener}\n"
            f"Length: about {words} words (this matters - match it closely)\n\n"
            f"{core}\n\n"
            f"Rules:\n"
            f"- Never name the factor or use clinical language.\n"
            f"- First person, informal, specific concrete details.\n"
            f"- Do NOT begin with any of: {', '.join(BANNED)}.\n"
            f"- Write the way people actually type online: some posts have "
            f"typos, missing apostrophes, run-on sentences, or stop abruptly.\n"
            f"- Output ONLY the post text. No title, no quotes, no commentary."
        )

    jobs = [(f, i) for f, n in need.items() for i in range(n)]
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut = {ex.submit(call, key, build(f, i), args.temperature): f
               for f, i in jobs}
        done = 0
        for fu in as_completed(fut):
            done += 1
            f = fut[fu]
            t = fu.result()
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}  kept {len(rows)}")
            if not t:
                continue
            t = re.sub(r'^["\'`]+|["\'`]+$', "", t).strip()
            t = t.replace("’", "'").replace("‘", "'")
            t = t.replace("“", '"').replace("”", '"')
            if not (12 <= len(t.split()) <= 220):
                continue
            if any(t.lower().startswith(b) for b in BANNED):
                continue
            rows.append({"row_id": f"API{len(rows):05d}", "post": t,
                         "factors": str([f]), "target_factor": f})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    w = out["post"].str.split().str.len()
    op = out["post"].str.lower().str.split().str[:4].str.join(" ")
    print(f"\nwrote {len(out)} posts to {args.out}")
    print(f"  median words   {w.median():.0f}   (real corpus 38)")
    print(f"  p90 words      {w.quantile(.9):.0f}   (real corpus 150)")
    print(f"  unique openings {100 * op.nunique() / len(op):.0f}%"
          f"   (real corpus 90%)")
    print(f"  per factor: {dict(collections.Counter(out.target_factor))}")
    print("\nAudit before use, then run the 5-fold arm with")
    print("  SYN=<this file> TAG=synapi")


if __name__ == "__main__":
    main()
