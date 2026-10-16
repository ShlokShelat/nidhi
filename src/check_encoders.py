"""
check_encoders.py

Which encoder checkpoints can this environment actually load, offline?

WHY THIS EXISTS. Three factor/evidence jobs were queued and all three died at
model-load time, for three different reasons:
  - answerdotai/ModernBERT-large : transformers 4.46.3 has no `modernbert`
  - AIMH/mental-roberta-large    : gated repo, 403; the on-disk dir was empty
  - (and the same ModernBERT failure again on the evidence arm)
Every one of those was a ten-second check away. Run this BEFORE sbatch.

It reports, per candidate: does the config load, does the tokenizer load, does
AutoModel instantiate, hidden size, and whether the weights are really on disk
(HF snapshots are symlink farms, so `du -sh` under-reports -- this counts
resolved sizes).

Usage:
  python src/check_encoders.py
  python src/check_encoders.py --models microsoft/deberta-v3-large,facebook/bart-large-mnli
"""

from __future__ import annotations

import argparse
import glob
import os

DEFAULTS = [
    # already the basis of shipped arms -- controls, should all pass
    "microsoft/deberta-v3-large",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "FacebookAI/roberta-large-mnli",
    # genuinely unused families, the actual candidates
    "facebook/bart-large-mnli",
    "microsoft/deberta-v2-xlarge",
    "microsoft/deberta-large-mnli",
    "google/electra-large-discriminator",
    "mental/mental-bert-base-uncased",
    # known-bad, kept so the report shows WHY
    "answerdotai/ModernBERT-large",
    "AIMH/mental-roberta-large",
]


def snapshot_bytes(repo: str) -> float:
    """Resolved size of the local snapshot in GB, following symlinks."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
        "hub")
    d = os.path.join(hub, "models--" + repo.replace("/", "--"))
    if not os.path.isdir(d):
        return 0.0
    tot = 0
    for f in glob.glob(os.path.join(d, "**", "*"), recursive=True):
        try:
            if os.path.isfile(f):          # follows symlinks, unlike islink
                tot += os.path.getsize(os.path.realpath(f))
        except OSError:
            pass
    return tot / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--online", action="store_true",
                    help="allow hub lookups; default is offline, matching how "
                         "the training jobs run")
    args = ap.parse_args()

    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    print(f"transformers {transformers.__version__}   "
          f"offline={'no' if args.online else 'yes'}")

    models = ([m.strip() for m in args.models.split(",")] if args.models
              else DEFAULTS)
    print(f"\n{'checkpoint':<58}{'GB':>7}{'cfg':>5}{'tok':>5}{'model':>7}"
          f"{'hidden':>8}")
    print("-" * 90)
    ok, bad = [], []
    for m in models:
        gb = snapshot_bytes(m)
        c = t = mo = "-"
        hid = ""
        why = ""
        try:
            cfg = AutoConfig.from_pretrained(m)
            c = "ok"
            hid = str(getattr(cfg, "hidden_size", "?"))
        except Exception as e:
            why = type(e).__name__ + ": " + str(e).split("\n")[0][:70]
        if c == "ok":
            try:
                AutoTokenizer.from_pretrained(m)
                t = "ok"
            except Exception as e:
                why = type(e).__name__ + ": " + str(e).split("\n")[0][:70]
            try:
                AutoModel.from_pretrained(m)
                mo = "ok"
            except Exception as e:
                why = type(e).__name__ + ": " + str(e).split("\n")[0][:70]
        print(f"{m[:57]:<58}{gb:>7.1f}{c:>5}{t:>5}{mo:>7}{hid:>8}")
        if why:
            print(f"    {why}")
        (ok if mo == "ok" else bad).append(m)

    print("\n" + "=" * 66)
    print(f"USABLE ({len(ok)}):")
    for m in ok:
        print(f"  {m}")
    print(f"\nNOT USABLE ({len(bad)}):")
    for m in bad:
        print(f"  {m}")
    print("=" * 66)
    print("\nOnly submit an arm whose checkpoint appears under USABLE.")
    print("Note deberta-v2-xlarge wants a lower LR -- factors_encoder.sh's own")
    print("docstring uses LR=5e-6 for it; 1e-5 may diverge at that size.")


if __name__ == "__main__":
    main()
