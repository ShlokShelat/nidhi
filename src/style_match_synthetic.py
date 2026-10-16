"""
style_match_synthetic.py

Strip the surface tells that let a model separate synthetic posts from real ones.

THE PROBLEM, measured on artifacts/syn_rare_factors.csv against the real corpus:

                            SYNTHETIC    REAL
    lowercase "i"              99.8%    24.2%
    capital "I"                 0.0%    68.6%
    contains a newline          0.0%     7.6%
    ends with a period         81.4%    39.1%
    curly apostrophes          15.1%     0.0%
    median length           81 words  38 words

Two of these separate the sets PERFECTLY. Real posts use capital "I" 69% of the
time and the synthetic ones never do; real posts contain no curly apostrophes
and 15% of synthetic ones do. A DeBERTa encoder will pick that up immediately.

WHY THAT IS DANGEROUS HERE. Every synthetic post carries exactly ONE factor
label, while real posts carry 2-4. If synthetic posts are trivially
identifiable from surface form, the model can learn

    "lowercase i, no newlines, closing period  =>  this one factor, and NOT
     the other twenty-three"

which is worthless at test time (no real post looks like that) and actively
teaches false negatives on the common labels. The generation prompt asked for
"lowercase-ish" text, which is what produced this.

WHAT THIS FIXES AND WHAT IT DOES NOT.

  Fixed: capitalisation of "i", apostrophe style, trailing punctuation,
  occasional newlines -- all pure surface, none of it touches meaning.

  NOT fixed: length (median 81 vs 38) and opening diversity (40% unique vs
  90%). Truncating posts would cut sentences mid-thought and change which
  factor is expressed, which is worse than the length mismatch. Those remain
  as known residual differences and are reported at the end.

Deterministic given --seed, so the output is reproducible.

Usage:
  python src/style_match_synthetic.py \
      --in artifacts/syn_rare_factors.csv \
      --real artifacts/full_train.csv \
      --out artifacts/syn_rare_factors_styled.csv
"""

from __future__ import annotations

import argparse
import os
import random
import re

import pandas as pd


# Openings that appear in 0% of the real corpus but were produced in bulk by
# the DeepSeek generator, because the prompt offered "apologise for posting"
# and "start mid-thought" as options and the model over-used both:
#     "sorry for posting..."   16.8% of generations, 0.0% of real posts
#     leading ellipsis         12.0% of generations, 0.0% of real posts
# Nearly 29% of the corpus carrying an opening that never occurs in real data
# is a separator the encoder can learn instead of the factor.
APOLOGY = re.compile(
    r"^\s*(?:sorry|apologies|sorry for posting|sorry to post)"
    r"[^.!?\n]{0,60}[.!?]\s*", re.I)
LEAD_ELLIPSIS = re.compile(r"^\s*(?:…|\.\.\.)\s*")


def profile(x: pd.Series) -> dict:
    return {
        "cap_i": x.str.contains(r"\bI\b").mean(),
        "newline": x.str.contains("\n").mean(),
        "end_dot": x.str.rstrip().str.endswith(".").mean(),
        "curly": x.str.contains("’").mean(),
        "start_cap": x.apply(lambda t: t[:1].isupper() if t else False).mean(),
        "apology": x.str.lower().str.startswith("sorry").mean(),
        "ellipsis": x.apply(lambda t: bool(LEAD_ELLIPSIS.match(t))).mean(),
        "words": x.str.split().str.len().median(),
    }


def show(name, p):
    print(f"{name:<12} capI {100 * p['cap_i']:5.1f}%  "
          f"startcap {100 * p['start_cap']:5.1f}%  "
          f"newline {100 * p['newline']:5.1f}%  "
          f"end-dot {100 * p['end_dot']:5.1f}%  "
          f"curly {100 * p['curly']:5.1f}%  "
          f"sorry {100 * p['apology']:5.1f}%  "
          f"lead... {100 * p['ellipsis']:5.1f}%  "
          f"med {p['words']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="artifacts/syn_rare_factors.csv")
    ap.add_argument("--real", default="artifacts/full_train.csv")
    ap.add_argument("--out", default="artifacts/syn_rare_factors_styled.csv")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    syn = pd.read_csv(args.inp)
    real = pd.read_csv(args.real)
    rp = profile(real["post"].astype(str))
    sp = profile(syn["post"].astype(str))
    print("BEFORE")
    show("real", rp)
    show("synthetic", sp)

    def topup(cur, tgt):
        """Probability to apply a transform to the posts that LACK a feature,
        so the final rate lands on the target. Applying the transform at the
        target rate regardless of the current rate over-corrects: the first
        version pushed capital-I from 60.4% to 82.3% against a target of
        68.6%, because 68.6% of posts got it on top of the 60.4% that already
        had it."""
        if tgt <= cur:
            return 0.0
        return min(1.0, (tgt - cur) / max(1e-9, 1.0 - cur))

    p_capi = topup(sp["cap_i"], rp["cap_i"])
    p_nl = topup(sp["newline"], rp["newline"])
    print(f"\ntop-up rates: capital-I {p_capi:.2f}  newline {p_nl:.2f}"
          f"   (applied only to posts lacking the feature)")

    out = []
    for t in syn["post"].astype(str):
        # curly -> straight: real corpus has ZERO curly apostrophes
        t = t.replace("’", "'").replace("‘", "'")
        t = t.replace("“", '"').replace("”", '"')

        # Drop generator-artifact openings down to the real rate (both are 0%
        # in the real corpus). Keep a token fraction so the edit itself is not
        # a signature -- a corpus with exactly zero apologies is as unnatural
        # as one with 17%.
        if APOLOGY.match(t) and rng.random() > rp["apology"] + 0.01:
            stripped = APOLOGY.sub("", t).strip()
            if len(stripped.split()) >= 10:      # do not gut a short post
                t = stripped
        if LEAD_ELLIPSIS.match(t) and rng.random() > rp["ellipsis"] + 0.01:
            t = LEAD_ELLIPSIS.sub("", t).strip()

        # capitalise standalone "i" (and i'm / i've / i'll / i'd), per POST
        # rather than per token -- a post that mixes both would itself be an
        # unnatural signature. Only posts that do NOT already contain a
        # capital "I" are candidates, at the top-up rate.
        if not re.search(r"\bI\b", t) and rng.random() < p_capi:
            t = re.sub(r"\bi\b", "I", t)
            t = re.sub(r"\bi('(?:m|ve|ll|d|s))", lambda m: "I" + m.group(1), t)
            # sentence-initial capitals too, otherwise capital-I mid-sentence
            # with lowercase sentence starts is its own giveaway
            t = re.sub(r"(^|[.!?]\s+)([a-z])",
                       lambda m: m.group(1) + m.group(2).upper(), t)

        # trailing period at the real rate
        ends = t.rstrip().endswith(".")
        if ends and rng.random() > rp["end_dot"]:
            t = t.rstrip().rstrip(".")
        elif not ends and rng.random() < rp["end_dot"] * 0.5:
            t = t.rstrip() + "."

        # first character capitalised at the real rate
        if t:
            if t[0].islower() and rng.random() < rp["start_cap"]:
                t = t[0].upper() + t[1:]
            elif t[0].isupper() and rng.random() > rp["start_cap"]:
                t = t[0].lower() + t[1:]

        # occasional paragraph break, as in real posts
        if "\n" not in t and rng.random() < p_nl:
            parts = re.split(r"(?<=[.!?])\s+", t)
            if len(parts) > 2:
                k = len(parts) // 2
                t = " ".join(parts[:k]) + "\n\n" + " ".join(parts[k:])

        out.append(t)

    syn["post"] = out
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    syn.to_csv(args.out, index=False)

    print("\nAFTER")
    show("real", rp)
    show("styled", profile(syn["post"].astype(str)))
    print(f"\nwrote {args.out}")

    fp = profile(syn["post"].astype(str))
    ro = (real["post"].astype(str).str.lower().str.split().str[:4]
          .str.join(" "))
    so = (syn["post"].astype(str).str.lower().str.split().str[:4]
          .str.join(" "))
    print("\nRESIDUAL DIFFERENCES, not fixable by rewriting:")
    print(f"  length   median {fp['words']:.0f} vs {rp['words']:.0f} words -- "
          f"truncating would cut sentences mid-thought")
    print(f"           and could remove the factor being taught")
    print(f"  openings {100 * so.nunique() / len(so):.0f}% unique vs "
          f"{100 * ro.nunique() / len(ro):.0f}% -- needs regeneration with "
          f"more varied prompts")
    print("\nSo this removes the PERFECT separators but not every difference.")
    print("If the styled run still hurts the common labels, the next fix is")
    print("pseudo-labelling: run the incumbent factor model over these posts,")
    print("keep its predictions for the other 23 labels, force the target to 1,")
    print("and train on realistic multi-label targets instead of single ones.")


if __name__ == "__main__":
    main()
