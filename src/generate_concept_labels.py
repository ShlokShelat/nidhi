"""
generate_concept_labels.py

Generates weak binary labels for the auxiliary concepts that the
4-level risk taxonomy actually turns on — tense, concreteness,
imminence, and who the suicidal content is about:

  1. method_mechanism      - explicit suicide method/mechanism mentioned
                              (merged: generic method + specific named mechanism)
  2. plan_present           - a concrete plan is described (not just intent)
  3. completed_act           - self-harm/attempt already occurred (past tense)
  4. urgency_imminence        - imminent timing language ("about to", "tonight")
  5. speaker_is_author        - is the suicidal content about the POSTER
                                 themselves, not someone else (persecutory
                                 belief / third-person discussion)
  6. farewell_framing         - goodbye/finality/suicide-note framing
  7. protective_or_negated    - protective factor present, OR the suicidal
                                 statement is explicitly negated
  8. past_attempt_history      - mentions a PRIOR (not current) attempt/history

Uses DeepSeek (same client as the factor-generation pipeline). Binary
yes/no per concept, one JSON call per post.

Usage:
  python src/generate_concept_labels.py \
      --input artifacts/train_split.csv \
      --out artifacts/train_concept_labels.csv \
      --sleep 0.3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))


def stratified_sample(df, n, seed=42):
    """Sample ~n rows, proportionally by risk_level, so rare classes
    (Behavior, Attempt) aren't underrepresented in a small pilot —
    exactly the classes where plan/method/completed_act matter most.

    NOTE: train_split.csv has inconsistent risk_level casing/whitespace
    ("Indicator" vs "indicator" vs "Ideation " etc — same issue
    scoring.py's normalize_label() already handles for real training).
    Normalize here too so stratification groups by the true 4 classes,
    not 10 string variants."""
    from scoring import normalize_label
    df = df.copy()
    strat_col = df["risk_level"].map(normalize_label) if "risk_level" in df.columns else None
    if strat_col is None or n >= len(df):
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    frac = n / len(df)
    df["_strat_key"] = strat_col
    sampled = df.groupby("_strat_key", group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=seed) if len(g) > 0 else g)
    return sampled.drop(columns=["_strat_key"]).reset_index(drop=True)

CONCEPTS = [
    "method_mechanism",
    "plan_present",
    "completed_act",
    "urgency_imminence",
    "speaker_is_author",
    "farewell_framing",
    "protective_factor",   # split from protective_or_negated: reason to live / support
    "negation",            # split from protective_or_negated: explicit "I don't want to..."
    "past_attempt_history",
]

# Recorded in the metadata sidecar so a labels file can always be traced back
# to the exact prompt that produced it.
PROMPT_VERSION = "v6_protective_factor_tightened"

SYSTEM_PROMPT = (
    "You are an expert clinical annotator labeling Reddit posts for a "
    "suicide-risk research dataset. You read carefully and answer only "
    "based on what the text actually states, not what it might imply "
    "in the worst case. You are precise and conservative — if something "
    "is genuinely ambiguous, you say so rather than guessing."
)

LABEL_TEMPLATE = """Read this post and answer 9 yes/no questions about it.

POST:
\"\"\"
{post}
\"\"\"

For each question, answer true or false, based ONLY on what the text
states. Do not infer beyond what's written.

1. method_mechanism: Does the post mention or seriously consider a
   specific suicide method or mechanism (e.g. pills, a weapon,
   cutting, hanging, overdose, jumping)? Mark TRUE even if the person
   states they do NOT currently have access to it, cannot obtain it,
   are unsure whether it would work, or ultimately decide against
   using it — the question is whether the method is DISCUSSED, not
   whether it is available or will be used.
2. plan_present: Does the post describe a CONCRETE, ACTIONABLE suicide
   plan — e.g. a specific method, a specific timing, or a clearly
   stated intended course of action ("I've decided I'm going to hang
   myself tomorrow") — not merely a wish or vague intent? (Preparation
   steps are NOT required for this to be true.)
3. completed_act: Does the post describe a self-harm or suicide
   ATTEMPT THAT HAS ALREADY HAPPENED (past tense — "I took", "I've
   cut", not future intent)?
4. urgency_imminence: Does the post express IMMINENT timing or
   near-term readiness for a suicidal act, WITHOUT necessarily
   describing a concrete plan? This includes explicit time phrases
   ("about to", "tonight", "right now", "now", "soon") AND clear
   implied near-term intent, such as a conditional trigger for
   imminent action ("if X doesn't happen, I will...") or a stated
   intention to act very soon. Do NOT mark this true based on: (a)
   general emotional intensity or hopelessness alone (e.g. "I'm
   done", "I can't do this anymore") without any timing signal, (b)
   wanting an effective or fast METHOD (e.g. "quickest way") — that
   is about method choice, not timing, or (c) mentioning a PAST or
   RECURRING act (e.g. "again", "I've done this before") — that is
   about history, not imminent future timing.
5. speaker_is_author: Determine ONLY who is speaking, regardless of
   the post's severity or how credible the risk seems. If the post is
   primarily about the AUTHOR's own thoughts, feelings, experiences,
   or intentions, mark TRUE — even if no explicit suicidal statement
   is present. Answer false only if the post is about someone else, a
   general discussion, a belief that OTHERS want the poster dead, or
   the poster discussing ANOTHER PERSON's suicidal thoughts. Do NOT
   use suicide risk level or emotional intensity to infer authorship.
6. farewell_framing: Does the post read as a farewell/goodbye message,
   a "suicide note," or use finality language addressed to specific
   people ("I'm sorry", "goodbye", "you'll be better without me")?
7. protective_factor: Does the post show something that is ACTUALLY
   REDUCING OR DELAYING suicide risk RIGHT NOW — not just mentioning
   a person, place, or resource? Examples that count: explicit
   reasons to live ("my daughter is why I'm alive"), someone actively
   asking to talk so they don't act ("anyone to talk, so I stay in
   this bed instead of trying again"), a third party intervening
   (someone called 911/an ambulance), or a stated past instance of
   resisting/overcoming an urge. Mark FALSE, even if these words
   appear, in these two common traps:
   (a) Merely MENTIONING therapy, a nurse, a doctor, medication, or
       being in a hospital/psych ward does NOT by itself count. Only
       mark TRUE if that contact is described as actually helping or
       being engaged with genuinely — NOT if the post describes lying
       to a therapist, the ward/nurse not helping, remaining actively
       suicidal despite being there, or seeking/obtaining means WHILE
       in that setting.
   (b) The words "help" or "please help" do NOT automatically count.
       If the post is asking for HELP FINDING A METHOD OR WAY TO DIE
       (e.g. "I need help in ways I can kill myself"), that is
       method-seeking, not a protective factor — mark FALSE.
8. negation: Does the post EXPLICITLY DENY or negate current suicidal
   intent — e.g. "I don't want to die", "I won't do it", "I'm not
   going to"? This is about the ABSENCE of stated intent, not a
   reason for that absence.
9. past_attempt_history: Does the post mention a PRIOR suicide
   attempt or self-harm history (not the current post's content) —
   e.g. "I've tried before", "last year I attempted"?

Also rate your OVERALL confidence in ALL 9 of these answers TOGETHER
as one number, 0-100 (100 = completely certain about every answer, 50
= genuinely unsure on at least one of them). This is confidence in the
whole annotation, not a separate score per question.

Return ONLY JSON in exactly this shape, no other text:
{{
  "method_mechanism": true/false,
  "plan_present": true/false,
  "completed_act": true/false,
  "urgency_imminence": true/false,
  "speaker_is_author": true/false,
  "farewell_framing": true/false,
  "protective_factor": true/false,
  "negation": true/false,
  "past_attempt_history": true/false,
  "confidence": 0-100
}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--start-row", type=int, default=0,
                    help="resume from this row index if interrupted")
    ap.add_argument("--limit", type=int, default=None,
                    help="only label this many posts (for a pilot run before "
                         "committing to the full dataset) — STRATIFIED by "
                         "risk_level so rare classes aren't underrepresented. "
                         "Omit for full run.")
    args = ap.parse_args()

    from deepseek_client import DeepSeekClient, GenerationConfig
    gen_config = GenerationConfig(temperature=0.0)
    client = DeepSeekClient(gen_config)

    import datetime
    run_metadata = {
        "model_name": getattr(client, "config", gen_config).model if hasattr(client, "config") else "deepseek-chat",
        "temperature": gen_config.temperature,
        "prompt_version": PROMPT_VERSION,
        "timestamp": datetime.datetime.now().isoformat(),
        "n_concepts": len(CONCEPTS),
    }

    df = pd.read_csv(args.input)
    if args.limit is not None:
        df = stratified_sample(df, args.limit)
        print(f"PILOT MODE: labeling {len(df)} posts, stratified by risk_level")
        print(f"  class distribution: {dict(df['risk_level'].value_counts())}")

    out_path = args.out
    if args.start_row == 0 and os.path.exists(out_path):
        os.remove(out_path)

    header_written = args.start_row > 0 and os.path.exists(out_path)

    for i in range(args.start_row, len(df)):
        row = df.iloc[i]
        safe_post = str(row["post"])[:3000].replace("{", "{{").replace("}", "}}")
        prompt = LABEL_TEMPLATE.format(post=safe_post)

        try:
            result = client.generate_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
            labels = {c: bool(result.get(c, False)) for c in CONCEPTS}
            confidence = int(result.get("confidence", -1))
        except Exception as e:
            print(f"  ! row {i} failed: {e} — labeling all concepts False as fallback")
            labels = {c: False for c in CONCEPTS}
            confidence = -1  # sentinel: generation failed, not a real low-confidence answer

        record = {"row_id": row.get("row_id", i), **labels, "confidence": confidence}
        pd.DataFrame([record]).to_csv(
            out_path, mode="a", header=not header_written, index=False)
        header_written = True

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(df)}] labeled")
        time.sleep(args.sleep)

    print(f"\nwrote {out_path}: {len(df) - args.start_row} rows labeled")

    # metadata sidecar — model/temp/prompt version/timestamp, so you always
    # know exactly what produced a given labels file later
    meta_path = out_path.replace(".csv", "_metadata.json")
    import json as _json
    with open(meta_path, "w") as f:
        _json.dump(run_metadata, f, indent=2)
    print(f"wrote {meta_path}")

    result_df = pd.read_csv(out_path)
    print("\nPositive rate per concept:")
    for c in CONCEPTS:
        if c in result_df.columns:
            rate = result_df[c].mean()
            print(f"  {c:<25} {rate:.1%}")

    if "confidence" in result_df.columns:
        conf = result_df[result_df["confidence"] >= 0]["confidence"]
        if len(conf) > 0:
            print(f"\nConfidence: mean={conf.mean():.1f}  std={conf.std():.1f}  "
                 f"min={conf.min()}  max={conf.max()}")
            bins = [0, 50, 70, 85, 95, 101]
            labels_bins = ["<50", "50-69", "70-84", "85-94", "95-100"]
            hist = pd.cut(conf, bins=bins, labels=labels_bins, right=False).value_counts().sort_index()
            print("Confidence histogram:")
            for label, count in hist.items():
                print(f"  {label:<8} {'#' * int(count / max(1, len(conf)) * 50)} {count}")


if __name__ == "__main__":
    main()
