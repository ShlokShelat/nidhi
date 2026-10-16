"""
evidence_sweep_rc.py

SOFT-DECODE SWEEP FOR THE SEVEN TAGGERS THAT ACTUALLY SHIP.

THE MEASURED CONSTRAINT. The candidate pool holds 1.02-1.30 spans/post against
1.12 gold spans -- fewer candidates than correct answers -- and 42% of missed
golds are ABSENT from the pool entirely. Five selection rules, four
representatives, cluster on/off, gates and length caps have all failed on that
pool, and they were always going to: selection cannot choose what generation
never proposed.

WHY THE POOL IS STARVED. `_decode_candidates` walks `probs.argmax(-1)` and only
emits runs tagged B/I. A span the model scores at 45% has argmax O on every
token and is invisible -- no min_conf and no length cap can recover it. The NER
literature says the gold is usually sitting just off the best path (Shibuya &
Hovy, TACL 2020, built a whole nested-NER method on second-best CRF paths;
AdaK-NER likewise), and the two-stage span literature runs stage-1 recall at
>99% before it lets a ranker near the problem. Ours is nowhere close.

WHAT THIS IS AND IS NOT. `evidence_redump.py` already runs this sweep, but on
the `ev5f_*` FOLD runs and against `artifacts/folds/val_f*.csv`. Those are not
what ships. This runs the identical decode against the SEVEN `evidence_*_rc`
taggers that build the champion's evidence column, on the val twin AND on test,
in one pass -- which matters because the evidence pipeline is the one component
whose val and test artifacts are matched by construction, which is precisely
why an evidence change is the kind of thing that transfers cleanly to the
official score.

  redump          ev5f folds  / fold vals        -> does the pool starve?
  this            RC7 shipped / val twin + test  -> is the fix shippable?

THREE THINGS IT DOES NOT DO. It does not touch a single existing file; every
output carries a new `_cand_soft` name. It does not change the champion's
decode -- argmax at max_tokens=12 is still dumped, as the first row of the
sweep, so the incumbent is inside the pool rather than replaced by it. And it
commits to nothing: choosing tau and the cap is CPU work afterwards.

THE ENCODING GATE. A risk-conditioned tagger was trained on a PAIR -- the risk
statement as sequence 1, the post as sequence 2. Feeding it the post alone is
a shape it never saw, and that exact mistake once made three GPU-trained
taggers look useless (trained conditioned, dumped unconditioned; candidate
counts collapsed ~800 -> ~240). So before dumping anything this re-derives the
argmax candidates for the first --gate-posts posts through the CANONICAL
`collect_candidates` and refuses to run unless they match span for span.

--test-risk MUST BE NIDHI_cluster.csv, NOT THE CURRENT CHAMPION. The shipped
`artifacts/evidence_runs_rc/*_candidates.csv` were dumped by
`scripts/evidence_cond_deploy.sh` with `CHAMP=submissions/NIDHI_cluster.csv`,
and NIDHI_ev7's risk column differs from it on 5 posts. Conditioning on ev7
would move those 5 for a reason that has nothing to do with the soft decode and
make the comparison unattributable -- the exact failure mode that cost three
submissions this month. The reproduction check below enforces this.

Usage (GPU, ~25 min for 7 runs x 750 posts):
  python src/evidence_sweep_rc.py \
      --test-risk submissions/NIDHI_cluster.csv \
      --val-risk artifacts/nligaur/oof_base.csv
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The champion's seven. evidence_v2_rc is deliberately EXCLUDED -- dropping it
# is the change that made the current champion, and re-admitting it here
# would quietly undo the only change that has worked.
RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--runs", default=",".join(RC7))
    a_.add_argument("--test", default="data/test.csv")
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--outdir", default="artifacts/evidence_runs_rc")
    a_.add_argument("--test-risk", required=True,
                    help="the CHAMPION submission. A conditioned tagger must "
                         "be fed exactly the risk column that shipped.")
    a_.add_argument("--val-risk", required=True,
                    help="OUT-OF-FOLD predicted risk for the val twin. Gold "
                         "would be leakage and would inflate every number.")
    a_.add_argument("--taus", default="0.20,0.35,0.50",
                    help="soft thresholds on P(not-O). Matches "
                         "evidence_redump.py so the two measurements are "
                         "comparable.")
    a_.add_argument("--max-tokens", type=int, default=64,
                    help="dump cap. HIGH on purpose: the 3x containment rule "
                         "licenses a 24-token span against an 8-token gold, "
                         "so the hardcoded 12 was discarding paid matches. "
                         "Filter downstream, not here.")
    a_.add_argument("--gate-posts", type=int, default=25)
    args = a_.parse_args()

    import torch
    from transformers import AutoTokenizer
    from train_evidence import (TaggerModel, collect_candidates,
                                _decode_candidates, RISK_PREFIX)
    from evidence_redump import soft_spans

    def forward_and_decode(model, tok, posts, max_len, device, max_tokens,
                           taus, risks):
        """ONE forward pass, three products: the argmax candidates, the
        soft-threshold candidates, and the RAW TOKEN POSTERIORS.

        `collect_with_cap` in evidence_redump.py does the first two and throws
        the posteriors away. Keeping them costs nothing on this pass and is
        what makes the enumerate-then-rank lane possible on CPU afterwards:
        `evidence_generator_ceiling.py` measured that exhaustive 3-8 grams
        cover 98.1% of gold spans (96.8% of posts fully) against ~1.1
        candidates/post from the argmax decode, so the pool CAN be saturated
        -- but only if something can score an arbitrary span, and mean
        P(not-O) over its tokens is exactly that scorer, already computed here.

        The decoding itself still comes from the shared functions, so this
        cannot drift from the champion's decode; the encoding gate below
        proves it against `collect_candidates` before anything is written.
        """
        from scoring import normalize_span as _ns
        out, post_tok = [], []
        model.eval()
        with torch.no_grad():
            for i, post in enumerate(posts):
                if risks is None:
                    enc = tok(post, truncation=True, max_length=max_len,
                              return_offsets_mapping=True, return_tensors="pt")
                    seq = None
                else:
                    # A conditioned tagger was trained on a PAIR: risk
                    # statement as sequence 1, post as sequence 2. Encoding the
                    # post alone is a shape it never saw and collapses the
                    # candidate count ~800 -> ~240.
                    enc = tok(RISK_PREFIX.format(normalize_label(risks[i])),
                              post, truncation="only_second",
                              max_length=max_len,
                              return_offsets_mapping=True, return_tensors="pt")
                    seq = enc.sequence_ids(0)
                offsets = enc.pop("offset_mapping").squeeze(0).tolist()
                enc = {k: v.to(device) for k, v in enc.items()
                       if k in ("input_ids", "attention_mask")}
                logits = model(**enc)["logits"].squeeze(0).float()
                probs = torch.softmax(logits, -1).cpu().numpy()
                if seq is not None:
                    keep = [t for t in range(len(offsets)) if seq[t] == 1]
                    probs = probs[keep]
                    offsets = [offsets[t] for t in keep]
                # drop special/empty-offset positions so the saved arrays are
                # pure content tokens and line up with post character offsets
                real = [t for t, (a, b) in enumerate(offsets) if a != b]
                post_tok.append((probs[real].astype(np.float16),
                                 np.asarray([offsets[t] for t in real],
                                            dtype=np.int32)))
                cands = [(s, c, "argmax") for s, c in
                         _decode_candidates(post, probs, offsets,
                                            max_tokens=max_tokens)]
                if taus:
                    cands += soft_spans(post, probs, offsets, taus, max_tokens)
                best = {}
                for s, c, src in cands:
                    k = _ns(s)
                    if not k:
                        continue
                    if k not in best:
                        best[k] = (s, c, src)
                    elif c > best[k][1]:
                        # KEEP THE argmax LABEL STICKY. A soft-decode span can
                        # normalise to the same key as an argmax span and, if
                        # its confidence is higher, overwrite the `src` tag.
                        # The reproduction check below filters on src=="argmax"
                        # to rebuild the shipped pool, so a relabelled entry
                        # would look like a MISSING shipped span and could trip
                        # the >1% abort -- killing Part 1 with a message
                        # blaming the risk column for a bookkeeping detail.
                        keep_src = ("argmax" if best[k][2] == "argmax" else src)
                        best[k] = (s, c, keep_src)
                out.append(list(best.values()))
        return out, post_tok

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")
    os.makedirs(args.outdir, exist_ok=True)
    taus = [float(x) for x in args.taus.split(",") if x.strip()]

    # Canonical text, exactly as evidence_dump_all.py does it. 29.9% of
    # data/test.csv changes under clean_text against 0.3% of the val split, so
    # reading test raw deploys the taggers into a text space they never saw for
    # nearly a third of the set -- invisible to every offline instrument.
    from data_audit import load
    from scoring import normalize_label, normalize_span
    te = load(args.test)
    va = load(args.val)
    print(f"test {len(te)} posts, val {len(va)} posts (canonicalized)")

    risk_of = {}
    for split, path, df in (("test", args.test_risk, te),
                            ("val", args.val_risk, va)):
        r = pd.read_csv(path, keep_default_na=False)
        # Column preference is copied verbatim from evidence_dump_all.py:121
        # so the argmax half still reproduces the stored pool. But if a val
        # file carries BOTH columns, `risk_level` is the GOLD one and this
        # conditions the tagger on the answer -- an oracle reading, and the
        # exact defect that inflated evidence_vote_oof.py. Do not silently
        # change the preference (that would break reproduction); shout instead.
        if ("risk_level" in r.columns and "pred_risk_level" in r.columns):
            print(f"  WARNING: {path} has BOTH risk_level and "
                  f"pred_risk_level. Using risk_level to match the stored "
                  f"dump, but if that column is GOLD then every {split} "
                  f"number from this run is an oracle reading, not a "
                  f"shippable one. Check before believing the val score.")
        col = ("risk_level" if "risk_level" in r.columns
               else "pred_risk_level")
        m = {str(a): normalize_label(b) for a, b in zip(r["row_id"], r[col])}
        absent = [str(x) for x in df["row_id"] if str(x) not in m]
        if absent:
            raise SystemExit(
                f"{path} lacks risk labels for {len(absent)} {split} rows "
                f"(e.g. {absent[:3]}). Conditioning on a missing level is "
                f"worse than not conditioning at all.")
        risk_of[split] = [m[str(x)] for x in df["row_id"]]
    from collections import Counter
    for s in ("test", "val"):
        print(f"  {s} risk: {dict(Counter(risk_of[s]))}")
    print(f"taus {taus}, max_tokens {args.max_tokens} (argmax@12 also kept)\n")

    runs = [r.strip() for r in args.runs.split(",")]
    for run in runs:
        d = os.path.join("runs", run)
        cfgp = os.path.join(d, "run_config.json")
        if not os.path.exists(cfgp):
            print(f"  skip {run}: no run_config.json")
            continue
        ckpt = next((os.path.join(d, f) for f in ("model_final.pt", "model.pt")
                     if os.path.exists(os.path.join(d, f))), None)
        if ckpt is None:
            print(f"  skip {run}: no checkpoint")
            continue
        cfg = json.load(open(cfgp))
        base = cfg.get("model", "microsoft/deberta-v3-large")
        max_len = int(cfg.get("max_len", 512))
        print(f"\n=== {run} ===  base={base} max_len={max_len}")

        tok = AutoTokenizer.from_pretrained(base)
        model = TaggerModel(base)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device).eval()

        # ---- encoding gate -------------------------------------------------
        # Reproduce the CANONICAL argmax candidates before trusting anything
        # this file computes. If the pair encoding, the sequence-id masking or
        # the risk prefix is wrong, the sweep is measuring a different model
        # and every downstream number is fiction.
        g = min(args.gate_posts, len(va))
        ref = collect_candidates(model, tok, [str(x) for x in va["post"][:g]],
                                 max_len, device, risks=risk_of["val"][:g])
        mine, _ = forward_and_decode(model, tok,
                                     [str(x) for x in va["post"][:g]],
                                     max_len, device, 12, [],
                                     risk_of["val"][:g])
        # compare on NORMALISED span keys: forward_and_decode dedupes and
        # collect_candidates does not, so raw lists can differ on a repeated
        # span without the model differing at all.
        a = [sorted({normalize_span(s) for s, _ in p} - {""}) for p in ref]
        b = [sorted({normalize_span(s) for s, _, _ in p} - {""}) for p in mine]
        if a != b:
            k = next(i for i in range(g) if a[i] != b[i])
            raise SystemExit(
                f"ENCODING GATE FAILED for {run} at post {k}:\n"
                f"  canonical {a[k]}\n  here      {b[k]}\n"
                f"The soft decode would be running a different model than the "
                f"champion does. Refusing to dump.")
        print(f"  encoding gate OK ({g} posts reproduce collect_candidates)")

        for split, df in (("val", va), ("test", te)):
            posts = [str(x) for x in df["post"]]
            cands, ptok = forward_and_decode(model, tok, posts, max_len,
                                             device, args.max_tokens, taus,
                                             risk_of[split])
            rows = []
            for i, rid in enumerate(df["row_id"]):
                for s, c, src in cands[i]:
                    rows.append((rid, s, float(c), len(str(s).split()), src))
            out = pd.DataFrame(rows, columns=["row_id", "span", "conf",
                                              "toks", "src"])
            dst = os.path.join(args.outdir, f"{run}_{split}_cand_soft.csv")
            out.to_csv(dst, index=False)
            n_arg = int((out["src"] == "argmax").sum())
            n_long = int((out["toks"] > 12).sum())
            print(f"  {split}: {len(out)} candidates "
                  f"({len(out)/max(len(df),1):.2f}/post) -- {n_arg} argmax, "
                  f"{len(out)-n_arg} soft-only, {n_long} over 12 tokens",
                  flush=True)

            # ---- raw token posteriors ---------------------------------
            # Written AFTER the CSV and inside try/except: this is the new,
            # speculative half of the pass and it must never put at risk the
            # part that is already proven. Ragged per post, so flat arrays plus an
            # index; float16 because these feed a ranker, not a likelihood.
            try:
                idx = np.zeros(len(ptok) + 1, dtype=np.int64)
                for i, (pr, _) in enumerate(ptok):
                    idx[i + 1] = idx[i] + len(pr)
                np.savez_compressed(
                    os.path.join(args.outdir, f"{run}_{split}_tokprobs.npz"),
                    probs=np.concatenate([p for p, _ in ptok])
                    if len(ptok) else np.zeros((0, 3), np.float16),
                    offsets=np.concatenate([o for _, o in ptok])
                    if len(ptok) else np.zeros((0, 2), np.int32),
                    index=idx,
                    row_id=np.array([str(x) for x in df["row_id"]],
                                    dtype=object))
                print(f"    token posteriors: {idx[-1]} content tokens "
                      f"({idx[-1]/max(len(ptok),1):.0f}/post)")
            except Exception as e:
                print(f"    token-posterior dump FAILED ({e}) -- candidates "
                      f"are still written, only the enumerate-then-rank lane "
                      f"is affected")

            # ---- reproduction check against the SHIPPED pool ---------------
            # The argmax spans at <=12 tokens are, by construction, the
            # incumbent pool: verify_spans only REJECTS on the cap, it never
            # truncates. So they must match the stored dump. This is the
            # strongest available statement that the sweep is a superset of
            # what ships rather than a different model's output.
            old = os.path.join(args.outdir, f"{run}_{split}_candidates.csv")
            if os.path.exists(old):
                o = pd.read_csv(old, keep_default_na=False)
                o["row_id"] = o["row_id"].astype(str)
                sub = out[(out["src"] == "argmax") & (out["toks"] <= 12)]
                def keyed(frame):
                    g = {}
                    for r, s in zip(frame["row_id"].astype(str), frame["span"]):
                        k = normalize_span(str(s))
                        if k:
                            g.setdefault(r, set()).add(k)
                    return g
                ga, gb = keyed(o), keyed(sub)
                ids = {str(x) for x in df["row_id"]}
                diff = sum(ga.get(r, set()) != gb.get(r, set()) for r in ids)
                rate = diff / max(len(ids), 1)
                print(f"    vs shipped pool: {diff}/{len(ids)} posts differ "
                      f"({rate:.1%})")
                if rate > 0.01:
                    raise SystemExit(
                        f"REPRODUCTION CHECK FAILED for {run}/{split}: "
                        f"{rate:.1%} of posts have a different argmax pool "
                        f"than the shipped dump. Something other than the "
                        f"soft decode changed -- most likely the risk column "
                        f"passed to --{split}-risk is not the one these "
                        f"candidates were originally conditioned on. Fix that "
                        f"before measuring anything.")
            else:
                print(f"    (no {os.path.basename(old)} to check against)")

        del model
        torch.cuda.empty_cache()

    print(f"\nwrote *_cand_soft.csv to {args.outdir}/  "
          f"(no existing file was touched)")
    print("\nNext, all CPU, and IN THIS ORDER:")
    print("  1. does the pool now contain the answers?")
    print("     oracle recall on val at each (tau, cap) -- if the 42% absent")
    print("     golds are STILL absent at tau 0.20, no decoding trick will")
    print("     ever reach them and the whole lane closes on one measurement.")
    print("  2. only if it does: MBR subset selection against the pool, with")
    print("     the seven taggers as samples and phrase-F1 as the utility.")
    print("     Vote>=2 is a crude approximation of exactly that, so the")
    print("     matched-objective version is the first thing to try and it")
    print("     needs no training and cannot overfit 372 posts.")


if __name__ == "__main__":
    main()
