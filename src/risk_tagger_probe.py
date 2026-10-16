"""
risk_tagger_probe.py

ASK THE EVIDENCE TAGGER TO VOTE ON RISK, BY CONDITIONING IT ON ALL FOUR LEVELS.

THE PIPELINE RUNS ONE WAY:

    post -> RISK model -> predicted level -> RISK-CONDITIONED tagger -> spans

The tagger is TOLD the predicted level ("The suicide risk level of this post
is {level}") and asked for the clause justifying THAT level. It therefore
consumes risk and never corrects it: when the risk model is wrong, the tagger
is asked the wrong question and answers it. Any symbolic rule built on the
evidence column inherits that circularity.

THE PROBE closes the loop. Condition the SAME post on all four levels and
score each. The tagger was trained to find evidence for a level that is
actually present, so it should struggle when asked for evidence of a level
that is not -- and `argmax over conditionings` becomes a risk signal computed
by a model that is NOT the risk classifier. That independence is the point:
risk errors are CONFIDENTLY wrong, so nothing derived from the risk
probabilities themselves can fix them.

WHY A MODEL SIGNAL RATHER THAN MORE VOCABULARY. On the behavior->ideation
errors, a method word appears in the POST 21.3% of the time and in the gold
span 15.7%, the same rate as in the extracted span -- extraction loses nothing
that was there. 55.7% of behavior posts contain NEITHER a method NOR a
self-harm word, so the lexical ceiling is already reached and any further gain
must come from a model. Those spans were the UNCONDITIONED tagger vote, so the
conditioning effect specifically is what section 4 measures.

TWO CALIBRATION POINTS THAT MATTER
  * Raw confidences are NOT comparable across conditionings. The tagger has a
    prior over levels from training frequency, so a column can sit low for
    every post. Everything below therefore compares RANK WITHIN A COLUMN.
  * The override needs a MARGIN, not an argmax: an unweighted vote dilutes a
    strong member. Section 5 sweeps the margin and prints the whole curve --
    judge a PLATEAU as real and a SPIKE as a fit.

THE FAILURE MODE TO WATCH. The tagger only ever saw GOLD conditioning during
training; it never saw a mismatched pair. It may have learned to TRUST the
prefix rather than verify it, in which case every conditioning is confident,
section 1's rows are flat, and the probe carries no signal. Section 1 comes
first for exactly that reason.

Each post is scored ONLY by the fold model that held it out, so nothing leaks.

Usage (one GPU, ~30 min for 1635 posts x 4 conditionings):
  python src/risk_tagger_probe.py
  python src/risk_tagger_probe.py --limit 300          # smoke test
  python src/risk_tagger_probe.py --cache artifacts/risk_probe.csv --skip-gpu
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
METHOD = re.compile(
    r"\b(jump(?:ing|ed)?|bridge|pills?|knife|knives|gun|shoot|hang(?:ing)?|"
    r"rope|noose|overdose|od|blade|razor|bleach|poison|wrists?|veins?|"
    r"cliff|roof|balcony)\b", re.I)


def wf1(gold, pred):
    """Weighted F1, the scorer's risk metric."""
    tot, n = 0.0, len(gold)
    for c in LEVELS:
        g, p = (gold == c), (pred == c)
        tp = float((g & p).sum())
        pr = tp / max(p.sum(), 1e-9)
        rc = tp / max(g.sum(), 1e-9)
        f = 2 * pr * rc / max(pr + rc, 1e-9)
        tot += f * g.sum()
    return tot / max(n, 1)


def rank_norm(x):
    """Rank within a column, mapped to [0,1]. Removes the tagger's prior over
    conditionings so the four columns are comparable per post."""
    o = np.argsort(np.argsort(x))
    return o / max(len(x) - 1, 1)


def dump_probe(args):
    """Run every held-out post through its fold's tagger under all four
    conditionings. Returns a frame of conf_/n_/meth_ per level."""
    import torch
    from transformers import AutoTokenizer

    from train_evidence import (RISK_PREFIX, TaggerModel,  # noqa: E402
                                _decode_candidates)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")

    rows = []
    for f in range(5):
        d = args.runs.format(f=f)
        cfg = json.load(open(f"{d}/run_config.json"))
        ml = int(cfg.get("max_len", 512))
        base = cfg.get("model", "microsoft/deberta-v3-large")
        va = pd.read_csv(args.folds.format(f=f), keep_default_na=False)
        va["row_id"] = va["row_id"].astype(str)
        if args.limit:
            va = va.head(max(args.limit // 5, 1))
        tok = AutoTokenizer.from_pretrained(base)
        m = TaggerModel(base)
        m.load_state_dict(torch.load(f"{d}/model_final.pt", map_location="cpu"))
        m.to(dev).eval()
        with torch.no_grad():
            for rid, post in zip(va["row_id"], va["post"].astype(str)):
                rec = {"row_id": rid}
                for lv in LEVELS:
                    enc = tok(RISK_PREFIX.format(lv), post,
                              truncation="only_second", max_length=ml,
                              return_offsets_mapping=True,
                              return_tensors="pt")
                    seq = enc.sequence_ids(0)
                    off = enc.pop("offset_mapping").squeeze(0).tolist()
                    e = {k: v.to(dev) for k, v in enc.items()
                         if k in ("input_ids", "attention_mask")}
                    pr = torch.softmax(m(**e)["logits"].squeeze(0).float(),
                                       -1).cpu().numpy()
                    keep = [t for t in range(len(off)) if seq[t] == 1]
                    pr, off2 = pr[keep], [off[t] for t in keep]
                    cands = _decode_candidates(post, pr, off2)
                    confs = [c for _, c in cands]
                    txt = " ... ".join(s for s, _ in cands)
                    rec[f"conf_{lv}"] = max(confs) if confs else 0.0
                    rec[f"n_{lv}"] = len(cands)
                    rec[f"meth_{lv}"] = int(bool(METHOD.search(txt)))
                rows.append(rec)
        del m
        torch.cuda.empty_cache()
        print(f"  fold {f}: {len(va)} posts x 4 conditionings", flush=True)
    return pd.DataFrame(rows)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--runs", default="runs/evcond5_cond_f{f}")
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--folds", default="artifacts/folds/val_f{f}.csv")
    a_.add_argument("--cache", default="artifacts/risk_probe.csv")
    a_.add_argument("--skip-gpu", action="store_true",
                    help="reuse --cache instead of re-running the taggers")
    a_.add_argument("--limit", type=int, default=0)
    a_.add_argument("--boot", type=int, default=400)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    if args.skip_gpu and os.path.exists(args.cache):
        df = pd.read_csv(args.cache)
        df["row_id"] = df["row_id"].astype(str)
        print(f"reusing {args.cache}  ({len(df)} posts)\n")
    else:
        df = dump_probe(args)
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        df.to_csv(args.cache, index=False)
        print(f"\nwrote {args.cache}  ({len(df)} posts)\n")

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    oof = pd.read_csv(args.oof, keep_default_na=False)
    oof["row_id"] = oof["row_id"].astype(str)
    ixo = oof.set_index("row_id")
    df = df[df["row_id"].isin(ixo.index) & df["row_id"].isin(lab.index)]
    df = df.reset_index(drop=True)
    rid = df["row_id"].tolist()

    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rid])
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rid])
    P = np.stack([ixo.loc[rid, f"prob_{c}"].astype(float).to_numpy()
                  for c in LEVELS], axis=1)
    pred = np.array(LEVELS)[P.argmax(1)]
    C = df[[f"conf_{c}" for c in LEVELS]].to_numpy()
    # rank within each conditioning column -- see header
    R = np.stack([rank_norm(C[:, j]) for j in range(4)], axis=1)
    probe = np.array(LEVELS)[R.argmax(1)]

    ok = (gold == "behavior") & (pred == "behavior")
    cell = (gold == "behavior") & (pred == "ideation")
    idn = (gold == "ideation") & (pred == "ideation")
    named = (("gold beh, pred beh (ok)", ok),
             ("gold beh, pred ide (ERR)", cell),
             ("gold ide, pred ide (ok)", idn))

    print("=" * 88)
    print("1. BEST-SPAN CONFIDENCE UNDER EACH CONDITIONING, BY GOLD CLASS")
    print("=" * 88)
    print(f"{'gold':<12}{'n':>6}" + "".join(f"{'cond '+c[:6]:>13}"
                                            for c in LEVELS)
          + "   peaks on own?")
    for j, c in enumerate(LEVELS):
        m = gold == c
        if not m.sum():
            continue
        mu = [C[m, k].mean() for k in range(4)]
        print(f"{c:<12}{int(m.sum()):>6}"
              + "".join(f"{v:>13.3f}" for v in mu)
              + f"   {'YES' if int(np.argmax(mu)) == j else 'no'}")
    print("\n  A row peaking on its OWN conditioning means the tagger can tell")
    print("  which question it was asked. FLAT ROWS mean it learned to trust")
    print("  the prefix instead of verifying it, and the probe is dead here.")

    print("\n" + "=" * 88)
    print("2. THE PROBE AS A STANDALONE RISK CLASSIFIER")
    print("=" * 88)
    print(f"  risk model accuracy   {np.mean(pred == gold):.3f}")
    print(f"  probe argmax (rank)   {np.mean(probe == gold):.3f}")
    print(f"  agree with risk model {np.mean(probe == pred):.1%}")
    dis = probe != pred
    if dis.sum():
        print(f"  where they DISAGREE ({int(dis.sum())} posts): "
              f"risk right {np.mean(pred[dis] == gold[dis]):.1%}, "
              f"probe right {np.mean(probe[dis] == gold[dis]):.1%}")
        print("\n  The probe does not need to be BETTER overall -- only better")
        print("  than chance where it disagrees. That is what an override buys.")

    print("\n" + "=" * 88)
    print("3. THE CELL -- 89 posts, gold behavior predicted ideation")
    print("=" * 88)
    for nm, m in named:
        if not m.sum():
            continue
        cb, ci = R[m, 2], R[m, 1]
        print(f"  {nm:<26}{int(m.sum()):>5}   rank cond-beh {cb.mean():.3f}   "
              f"cond-ide {ci.mean():.3f}   diff {cb.mean()-ci.mean():+.3f}"
              f"   beh>ide {np.mean(cb > ci):.0%}")
    print("\n  If the ERR row leans toward cond-behavior the way the (ok) row")
    print("  does, the tagger sees the plan the risk model missed.")

    print("\n" + "=" * 88)
    print("4. DOES cond=behavior SURFACE A METHOD THAT cond=ideation HIDES?")
    print("=" * 88)
    mb = df[[f"meth_{c}" for c in LEVELS]].to_numpy()
    for nm, m in named:
        if not m.sum():
            continue
        print(f"  {nm:<26}{int(m.sum()):>5}   method in span: cond-ideation "
              f"{mb[m, 1].mean():>5.1%}   cond-behavior {mb[m, 2].mean():>5.1%}")
    print("\n  risk_method_coverage.py showed the UNCONDITIONED taggers lose")
    print("  nothing (14 in gold span, 14 in ours). This is the conditioned")
    print("  test it could not run.")

    # ---------------------------------------------------------------- 5
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([by[a] for a in rng.choice(ua, len(ua), True)])
             for _ in range(args.boot)]
    base_w = wf1(gold, pred)

    def judge(newpred):
        ch = newpred != pred
        if not ch.sum():
            return None
        right = int(((newpred == gold) & ~(pred == gold) & ch).sum())
        wrong = int(((pred == gold) & ~(newpred == gold) & ch).sum())
        d = [wf1(gold[b], newpred[b]) - wf1(gold[b], pred[b]) for b in boots]
        d = np.array(d)
        return (int(ch.sum()), right, wrong, wf1(gold, newpred),
                d.mean(), float(np.mean(d > 0)),
                float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))

    print("\n" + "=" * 88)
    print("5. YOUR RULE, SIMULATED -- switch when another conditioning beats")
    print("   the assigned one by a margin.  base risk wF1 "
          f"{base_w:.4f}, {args.boot} author bootstraps")
    print("=" * 88)
    print(f"  {'rule':<22}{'margin':>7}{'fires':>7}{'right':>6}{'wrong':>6}"
          f"{'wF1':>9}{'d comp':>9}{'win':>6}   one-draw 95%")
    for nm, tgt in (("ideation->behavior", "behavior"),
                    ("any (argmax)", None)):
        for mg in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
            new = pred.copy()
            for i in range(len(pred)):
                j = LEVELS.index(pred[i])
                if tgt is not None:
                    if pred[i] != "ideation":
                        continue
                    k = LEVELS.index(tgt)
                else:
                    k = int(R[i].argmax())
                    if k == j:
                        continue
                if R[i, k] - R[i, j] > mg:
                    new[i] = LEVELS[k]
            r = judge(new)
            if r is None:
                continue
            n, ri, wr, w, dm, win, lo, hi = r
            print(f"  {nm:<22}{mg:>7.2f}{n:>7}{ri:>6}{wr:>6}{w:>9.4f}"
                  f"{0.4*dm:>+9.4f}{win:>5.0%}   "
                  f"[{0.4*lo:+.4f},{0.4*hi:+.4f}]")
    print("\n  d comp = 0.4 x d(risk wF1), evidence and factors held fixed.")
    print("  READ THE CURVE, NOT THE BEST CELL. A broad plateau of positive")
    print("  win rates is a real effect; a single spike surrounded by losses")
    print("  is a margin fitted to this sample. The bracket is the ONE-DRAW")
    print("  spread, and win rate is the verdict.")

    print("\n" + "=" * 88)
    print("6. AS A FEATURE INSTEAD OF AN OVERRIDE -- blend into the risk probs")
    print("=" * 88)
    Pn = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    for w_ in (0.05, 0.10, 0.20, 0.30, 0.50):
        new = np.array(LEVELS)[(np.log(np.clip(Pn, 1e-9, None))
                                + w_ * R).argmax(1)]
        r = judge(new)
        if r is None:
            print(f"  weight {w_:<5.2f}  no change")
            continue
        n, ri, wr, w, dm, win, lo, hi = r
        print(f"  weight {w_:<5.2f}{n:>7}{ri:>6}{wr:>6}{w:>9.4f}"
              f"{0.4*dm:>+9.4f}{win:>5.0%}   "
              f"[{0.4*lo:+.4f},{0.4*hi:+.4f}]")
    print("\n  A weak independent signal typically helps as a FEATURE and hurts")
    print("  as an override, because an override discards the risk model's")
    print("  confidence entirely. If this section beats section 5, the blend is")
    print("  the right form, not the switch.")


if __name__ == "__main__":
    main()
