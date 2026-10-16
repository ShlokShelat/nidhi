"""
evidence_llm_reranker.py

Uses an LLM as a SELECTOR over span candidates produced by the BIO taggers,
not as an extractor.

Why this is worth trying when LLM extraction already failed:

    headroom_analysis.py, val, official phrase metric
        best single tagger              0.6969
        union of 4 taggers (as-is)      0.6491
        PERFECT selector over the union 0.8617   (the achievable ceiling)

    The candidate pool already holds 75.7% of all gold spans. The deficit is
    selection, not generation. src/evidence_llm.py previously asked a model to
    GENERATE spans and reached 0.6374 -- worse than the taggers. That result
    says nothing about selection, which is a far easier task: judging whether
    a given quote supports a given risk level, with the candidate handed over.

    A learned feature-based reranker (src/evidence_reranker.py) reached only
    0.7010 cross-fitted, barely above the best single tagger. Its features --
    model agreement, length, position, nesting -- never read the text, and
    its top weights are pure geometry. Capturing the ceiling requires
    semantics.

The property that matters most: this reranker fits NOTHING on val. It is a
zero-shot judgement, so there are no parameters to overfit. Val-tuned changes
have a documented history of failing to transfer, while zero-val-parameter
changes have transferred cleanly. This is in the second class.

One call per post (all candidates judged together), so val is 372 calls and
test 378 -- negligible cost. Responses are cached to disk, so re-running or
changing the decode is free.

Usage:
  # inspect prompts without calling anything
  python src/evidence_llm_reranker.py --dry-run --limit 3

  # DeepSeek API (needs DEEPSEEK_API_KEY)
  python src/evidence_llm_reranker.py --backend deepseek \
      --out artifacts/evidence_llm_rerank_val.csv

  # local Qwen already on disk
  python src/evidence_llm_reranker.py --backend qwen --qwen /path/to/Qwen2.5-14B-Instruct \
      --out artifacts/evidence_llm_rerank_val.csv
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1, token_len, _span_pair_matches
from rule_ablation_v2 import author_split

RUN_ORDER = ["evidence_v2", "evidence_v1", "evidence_s14", "evidence_s15"]

SYSTEM = (
    "You are a clinical annotator for a suicide-risk research corpus. You "
    "judge whether short quotations taken verbatim from a social-media post "
    "are valid EVIDENCE for that post's assessed suicide-risk level.\n\n"
    "A quotation is valid evidence when it, by itself, expresses the "
    "suicidal thought, intent, plan, preparation or act that justifies the "
    "risk level. It is NOT valid evidence when it is generic distress, "
    "context, background, someone else's words, or a fragment too vague to "
    "stand alone.\n\n"
    "Be selective. Most posts have between zero and three valid evidence "
    "quotations. Choosing extra weak quotations is penalised exactly as "
    "harshly as missing good ones."
)

# ---------------------------------------------------------------------------
# SCORE mode. The binary prompt above measured P 0.7374 / R 0.6117 / F1 0.6687
# on val -- best-in-class precision but the worst recall of any method, i.e.
# it over-rejected. "Be selective" plus the "zero to three" anchor pushed the
# model to drop borderline candidates that were in fact gold.
#
# Rather than re-guess the wording, this asks for a graded 0-100 score per
# candidate and leaves the cut-off to be fitted -- ONE parameter, chosen on
# one author-disjoint half and applied to the other. That also separates two
# very different failure modes: if the ranking is good (high average
# precision) but the binary cut was wrong, the method is sound and only the
# operating point was off. If the ranking itself is poor, the idea is dead.
# ---------------------------------------------------------------------------
SYSTEM_SCORE = (
    "You are a clinical annotator for a suicide-risk research corpus. For each "
    "candidate quotation taken verbatim from a post, you rate how likely it is "
    "to have been selected by expert annotators as EVIDENCE for that post's "
    "suicide-risk level.\n\n"
    "Evidence quotations are the specific words expressing the suicidal "
    "thought, intent, plan, preparation or act that justifies the risk level. "
    "Generic distress, background context, or another person's words score "
    "low. Direct statements of the author's own suicidality score high.\n\n"
    "Use the full 0-100 range. Do not cluster your scores."
)

USER_SCORE = """Post (risk level assessed as: {risk}):
\"\"\"{post}\"\"\"

Candidate quotations, each taken verbatim from the post:
{cands}

Score every candidate from 0 to 100 for how likely an expert annotator marked
it as evidence. Return ONLY JSON of the form {{"scores": {{"0": 87, "1": 12}}}}
with one entry per candidate index."""

USER = """Post (risk level assessed as: {risk}):
\"\"\"{post}\"\"\"

Candidate quotations, each taken verbatim from the post:
{cands}

Return ONLY a JSON object of the form {{"keep": [indices]}} listing the
indices of the candidates that are valid evidence for this risk level.
Return {{"keep": []}} if none qualify."""


def load_runs(df, cache_dir, tag):
    runs = {}
    if cache_dir and os.path.isdir(cache_dir):
        for f in sorted(glob.glob(os.path.join(cache_dir, f"*__{tag}.csv"))):
            name = os.path.basename(f).split("__")[0]
            d = pd.read_csv(f, keep_default_na=False).set_index("row_id")["evidence"]
            runs[name] = [parse_evidence(d.get(r, "")) for r in df["row_id"]]
    if not runs:
        for r in RUN_ORDER:
            p = f"runs/{r}/val_predictions.csv"
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            col = next(c for c in d.columns if "pred" in c.lower())
            m = df[["row_id"]].merge(d[["row_id", col]], on="row_id", how="left")
            runs[r] = [parse_evidence(x) for x in m[col]]
    return runs


def candidates_for(runs, i):
    """Deduplicated union for one post, keeping the shortest surface form."""
    cands = []
    for n in runs:
        for s in runs[n][i]:
            hit = next((c for c in cands if _span_pair_matches(s, c["span"])), None)
            if hit is None:
                cands.append({"span": s, "by": {n}})
            else:
                hit["by"].add(n)
                if token_len(s) < token_len(hit["span"]):
                    hit["span"] = s
    return cands


def parse_scores(text, n):
    """Return a length-n list of 0-100 scores, or None if unparseable."""
    if not text:
        return None
    m = re.search(r'\{[^{}]*"scores"\s*:\s*\{[^{}]*\}[^{}]*\}', text, re.S)
    out = [None] * n
    if m:
        try:
            d = json.loads(m.group(0))["scores"]
            for k, v in d.items():
                i = int(k)
                if 0 <= i < n:
                    out[i] = float(v)
        except Exception:
            pass
    if all(v is None for v in out):
        pairs = re.findall(r'"?(\d+)"?\s*[:=]\s*(\d+(?:\.\d+)?)', text)
        for k, v in pairs:
            i = int(k)
            if 0 <= i < n and out[i] is None:
                out[i] = float(v)
    if all(v is None for v in out):
        return None
    return [0.0 if v is None else v for v in out]


def parse_keep(text, n):
    if not text:
        return None
    m = re.search(r'\{[^{}]*"keep"\s*:\s*\[[^\]]*\][^{}]*\}', text, re.S)
    blob = m.group(0) if m else text
    try:
        got = json.loads(blob).get("keep", [])
    except Exception:
        nums = re.findall(r"\d+", blob)
        if not nums:
            return None
        got = [int(x) for x in nums]
    return sorted({int(i) for i in got if isinstance(i, (int, float)) and 0 <= int(i) < n})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="artifacts/val_split.csv")
    ap.add_argument("--risk-pred", default=None,
                    help="CSV row_id,pred_risk_level. Omit on val to use gold "
                         "risk ONLY for a ceiling reading; a real run must "
                         "pass predicted risk so val and test match.")
    ap.add_argument("--cache-dir", default="artifacts/evidence_per_run")
    ap.add_argument("--tag", default="val_split")
    ap.add_argument("--backend", choices=["deepseek", "qwen"], default="deepseek")
    ap.add_argument("--qwen", default=None)
    ap.add_argument("--cache", default="artifacts/llm_rerank_cache.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.split)
    has_gold = "evidence" in df.columns
    gold = [parse_evidence(x) for x in df["evidence"]] if has_gold else None
    runs = load_runs(df, args.cache_dir, args.tag)
    if not runs:
        raise SystemExit("no tagger predictions found")
    print(f"taggers: {list(runs)}")

    if args.risk_pred:
        rp = pd.read_csv(args.risk_pred).set_index("row_id")
        col = next((c for c in ("pred_risk_level", "champion_final", "risk_level")
                    if c in rp.columns), None)
        if col is None:
            raise SystemExit(f"--risk-pred needs one of pred_risk_level / "
                             f"champion_final / risk_level; got {list(rp.columns)}")
        risks = [str(rp.loc[r, col]) for r in df["row_id"]]
    elif "risk_level" in df.columns:
        risks = [str(x) for x in df["risk_level"]]
        print("WARNING: using GOLD risk level. Fine for a ceiling reading, "
              "but pass --risk-pred for a deployable run.")
    else:
        risks = ["unknown"] * len(df)

    cache = {}
    if os.path.exists(args.cache):
        for line in open(args.cache, encoding="utf-8"):
            try:
                r = json.loads(line)
                cache[r["key"]] = r["response"]
            except Exception:
                pass
    print(f"cache entries: {len(cache)}")

    client = None
    if not args.dry_run:
        if args.backend == "deepseek":
            from deepseek_client import DeepSeekClient
            client = DeepSeekClient()
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.qwen)
            mdl = AutoModelForCausalLM.from_pretrained(
                args.qwen, torch_dtype=torch.bfloat16).to("cuda").eval()
            client = (tok, mdl)

    n_posts = len(df) if not args.limit else min(args.limit, len(df))
    selected, n_calls, n_fail = [], 0, 0
    fh = None if args.dry_run else open(args.cache, "a", encoding="utf-8")

    for i in range(n_posts):
        cands = candidates_for(runs, i)
        if not cands:
            selected.append([])
            continue
        listing = "\n".join(f"[{j}] {c['span']}" for j, c in enumerate(cands))
        prompt = USER.format(risk=risks[i], post=str(df['post'].iloc[i])[:4000],
                             cands=listing)
        key = hashlib.md5((SYSTEM + prompt).encode("utf-8")).hexdigest()

        if args.dry_run:
            print(f"\n----- post {df['row_id'].iloc[i]} -----\n{prompt}\n")
            selected.append([c["span"] for c in cands])
            continue

        if key in cache:
            raw = cache[key]
        else:
            if args.backend == "deepseek":
                # DeepSeekClient.generate(system_prompt, user_prompt) -- positional
                raw = client.generate(SYSTEM, prompt)
            else:
                tok, mdl = client
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt}]
                text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
                import torch
                inp = tok(text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    o = mdl.generate(**inp, max_new_tokens=120, do_sample=False)
                raw = tok.decode(o[0][inp["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
            n_calls += 1
            fh.write(json.dumps({"key": key, "response": raw}) + "\n")
            fh.flush()

        keep = parse_keep(raw, len(cands))
        if keep is None:
            n_fail += 1
            # fall back to the validated vote>=2 rule rather than guessing
            keep = [j for j, c in enumerate(cands) if len(c["by"]) >= 2]
        selected.append([cands[j]["span"] for j in keep])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_posts} posts ({n_calls} new calls)")

    if fh:
        fh.close()
    if args.dry_run:
        print("dry run complete -- no API calls made")
        return

    print(f"\ncalls made {n_calls}   parse failures {n_fail} "
          f"(fell back to vote>=2)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame({"row_id": df["row_id"][:n_posts],
                      "evidence": ["; ".join(s) for s in selected]}).to_csv(
            args.out, index=False)
        print(f"wrote {args.out}")

    if has_gold:
        g = gold[:n_posts]
        s = phrase_f1(g, selected)
        print(f"\nLLM RERANKER   P {s.precision:.4f}  R {s.recall:.4f}  "
              f"F1 {s.f1:.4f}")
        for n, sp in runs.items():
            print(f"  {n:<16} F1 {phrase_f1(g, sp[:n_posts]).f1:.4f}")
        vote2 = [[c["span"] for c in candidates_for(runs, i) if len(c["by"]) >= 2]
                 for i in range(n_posts)]
        print(f"  {'vote>=2':<16} F1 {phrase_f1(g, vote2).f1:.4f}")
        if n_posts == len(df):
            half = author_split(df)[0].values
            for h in ("A", "B"):
                idx = [i for i in range(n_posts) if half[i] == h]
                print(f"  half {h}: LLM F1 "
                      f"{phrase_f1([g[i] for i in idx], [selected[i] for i in idx]).f1:.4f}"
                      f"   vote>=2 "
                      f"{phrase_f1([g[i] for i in idx], [vote2[i] for i in idx]).f1:.4f}")


if __name__ == "__main__":
    main()
