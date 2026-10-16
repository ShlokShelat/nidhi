"""
friend_evidence_eval.py

Score the friend's EVIDENCE tagger, which has never been used.

WHY IT MATTERS. Evidence is 0.30 of the composite and the least-modernised
component: all eight of the shipped taggers are `microsoft/deberta-v3-large`,
and four of them (v2, v1, s14, s15) are the SAME configuration differing only
by seed. The bag's whole diversity is seed noise. The friend's tagger is a
different family entirely -- mental-roberta-base, trained multi-task alongside
risk and factor heads -- so its errors have real reason to be decorrelated
from the shipped taggers'.

WHY IT IS ONLY NOW POSSIBLE. `friend_export_oof.py` exports risk and factor
probabilities but NOT evidence. The raw arrays are in
`pipeline/outputs/oof.npy`: per fold it holds `tokp` (n x 320 token
probabilities), `offsets` (n x 320 x 2 character spans) and `post`. That is
everything needed to decode spans and score them against gold.

WHAT THIS DECIDES. The `evidence_v2` member alone scores 0.6840 on the OOF
instrument and the shipped vote>=2 over five OOF taggers reaches 0.7152. If the
friend's tagger lands anywhere near those, it is worth adding as a member of a
DIFFERENT family -- which is exactly the parity-plus-diversity condition the
LoRA experiment showed an ensemble needs, and which the eight near-identical
seeds do not provide. If it lands far below, the lane closes for zero GPU time.

CPU only, seconds. Needs the tri-head's stage-2 outputs (src/friend_pipeline).

Usage:
  python src/friend_evidence_eval.py \
      --oof src/friend_pipeline/outputs/oof.npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span,  # noqa: E402
                     parse_evidence, phrase_f1, token_len)


def decode(tokp, offs, post, thr, max_words):
    """Contiguous runs of tokens above `thr` -> verbatim substrings.

    Mirrors the friend's own postprocessing: keep the span verbatim (the scorer
    compares literally) and hard-cap the length, because a prediction more than
    3x its gold span's token count scores 0 even when it contains the gold text.
    """
    spans, i, n = [], 0, len(tokp)
    while i < n:
        if tokp[i] < thr:
            i += 1
            continue
        j = i
        while j + 1 < n and tokp[j + 1] >= thr:
            j += 1
        s, e = int(offs[i][0]), int(offs[j][1])
        if e > s:
            t = post[s:e].strip()
            if len(t) >= 3 and token_len(t) <= max_words:
                spans.append(t)
        i = j + 1
    # drop any span contained in a longer sibling: one-to-one matching makes the
    # shorter one a guaranteed false positive
    out = []
    for s in sorted(spans, key=lambda x: (-token_len(x), -len(x))):
        ns = normalize_span(s)
        if ns and not any(ns in normalize_span(k) for k in out):
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof",
                    default=os.path.join(
                        os.environ.get("FRIEND_OUT_DIR", os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "friend_pipeline", "outputs")), "oof.npy"),
                    help="the tri-head's stage-2 OOF dump (default: "
                         "src/friend_pipeline/outputs/oof.npy, or "
                         "$FRIEND_OUT_DIR/oof.npy)")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--max-words", type=int, default=14,
                    help="the friend's own MAX_SPAN_WORDS")
    args = ap.parse_args()

    if not os.path.exists(args.oof):
        raise SystemExit(f"missing {args.oof}")
    d = np.load(args.oof, allow_pickle=True).item()

    rid, TP, OF, POST = [], [], [], []
    for f in sorted(d):
        o = d[f]
        rid += [str(x) for x in o["row_id"]]
        TP.append(np.asarray(o["tokp"], dtype=float))
        OF.append(np.asarray(o["offsets"]))
        POST += [str(x) for x in o["post"]]
    TP = np.concatenate(TP, 0)
    OF = np.concatenate(OF, 0)
    print(f"{len(rid)} OOF posts   token probs {TP.shape}   "
          f"prob range {TP.min():.3f}-{TP.max():.3f}")

    lab = pd.read_csv(args.labeled, keep_default_na=False).set_index("row_id")
    keep = [i for i, r in enumerate(rid) if r in lab.index]
    if len(keep) != len(rid):
        print(f"NOTE {len(rid) - len(keep)} rows have no gold; dropped")
    G = [parse_evidence(lab.loc[rid[i], "evidence"]) for i in keep]
    risk = [normalize_label(lab.loc[rid[i], "risk_level"]) for i in keep]

    print(f"\n{'thr':>6}{'F1':>9}{'P':>8}{'R':>8}{'spans':>8}{'/post':>7}"
          f"{'+rule':>9}")
    print("-" * 60)
    best = (None, -1.0)
    for thr in np.round(np.arange(0.20, 0.85, 0.05), 2):
        P = [decode(TP[i], OF[i], POST[i], thr, args.max_words) for i in keep]
        r = phrase_f1(G, P)
        # the shipped consistency rule, applied identically for a fair read
        Pc = [[] if risk[k] == "indicator" else p for k, p in enumerate(P)]
        rc = phrase_f1(G, Pc)
        if rc.f1 > best[1]:
            best = (thr, rc.f1)
        print(f"{thr:>6.2f}{r.f1:>9.4f}{r.precision:>8.4f}{r.recall:>8.4f}"
              f"{r.n_pred:>8}{r.n_pred/len(keep):>7.2f}{rc.f1:>9.4f}")

    print(f"\nbest with the consistency rule: {best[1]:.4f} at thr={best[0]}")
    print("\n" + "=" * 60)
    print("COMPARE AGAINST THE SHIPPED OOF EVIDENCE NUMBERS")
    print("=" * 60)
    print("  evidence_v2 member alone       0.6840")
    print("  shipped vote>=2                0.7152")
    print("  union of all shipped taggers   0.7027")
    if best[1] >= 0.65:
        print("\n  The friend's tagger is competitive AND from a different")
        print("  family. The eight shipped members are near-identical seeds, so")
        print("  this is the first real diversity available for evidence. Worth")
        print("  exporting its TEST spans and adding it as a 9th voter.")
    elif best[1] >= 0.55:
        print("\n  Behind the shipped members but not hopeless. A vote member")
        print("  does not need to be the best -- vote>=2 uses agreement, not")
        print("  quality -- so test it as a 9th voter before dismissing it.")
    else:
        print("\n  Too far behind to add to the vote. Closed, for zero GPU.")
    print("\nCAVEAT: the friend truncates at 320 tokens against the shipped")
    print("512, so long posts are penalised here in a way the shipped taggers")
    print("are not.")


if __name__ == "__main__":
    main()
