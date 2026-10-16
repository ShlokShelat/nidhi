"""Does --risk-cond keep the BIO targets aligned to the POST?

THE FAILURE MODE THIS GUARDS AGAINST. With a text PAIR, offset_mapping restarts
at 0 for the second sequence. If the prefix tokens are not masked out, their
offsets index into the POST and mislabel its opening characters -- silently,
and in a way that looks like a mediocre model rather than a bug.

This decodes the B/I tokens back to text and compares against the target span,
with conditioning off and on. What is being checked is the offset arithmetic,
not the model, so the fixtures below are synthetic sentences written for this
test -- no corpus row is reproduced here.

Requires a fast tokenizer, so it skips cleanly where transformers is absent.

Usage:
  python -m pytest tests/test_risk_cond_alignment.py -q
  python tests/test_risk_cond_alignment.py
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# Import-time guards. Without these a missing torch/transformers raises during
# COLLECTION, which aborts the whole tests/ directory -- the scoring tests in
# test_scoring.py would then never run, and the suite would report no failures
# because it never got as far as running anything.
pytest.importorskip("torch", reason="risk-cond alignment needs torch")
pytest.importorskip("transformers",
                    reason="risk-cond alignment needs a fast tokenizer")

from transformers import AutoTokenizer  # noqa: E402

from train_evidence import RISK_PREFIX, EvidenceDataset  # noqa: E402

MODEL = os.environ.get("TOK_MODEL", "microsoft/deberta-v3-large")

# Synthetic fixtures. Each `evidence` string must appear verbatim in its `post`
# so the BIO builder has something to locate; the third row is the empty-span
# case, which must produce no B/I tokens at all.
DF = pd.DataFrame({
    "row_id": ["A", "B", "C"],
    "post": [
        "third week of the same routine and still counting the days",
        "looking for someone to talk to because I keep scratching my arms",
        "the weather turned cold and the bus was late again this morning",
    ],
    "risk_level": ["Behavior", "Behavior", "Indicator"],
    "evidence": ["still counting the days", "keep scratching my arms", ""],
})


def _misaligned(verbose=True):
    """Return the number of rows whose decoded span != the target span."""
    # The tokenizer is fetched from the Hub. This test sets HF_HUB_OFFLINE=1
    # above, so on a machine where it is not already cached the load raises
    # instead of downloading. That is a missing fixture, not a failure of the
    # offset arithmetic under test, so skip rather than fail -- otherwise a
    # reviewer running the suite offline sees a red test that says nothing
    # about this code.
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:
        pytest.skip(f"tokenizer {MODEL} is not available offline: "
                    f"{type(e).__name__}. Pre-cache it, or set TOK_MODEL to a "
                    f"local fast tokenizer, to run this test.")
    if verbose:
        print(f"tokenizer: {MODEL}   fast={tok.is_fast}")
    if not tok.is_fast:
        pytest.skip("need a fast tokenizer for sequence_ids/offsets")

    bad = 0
    for cond in (False, True):
        ds = EvidenceDataset(DF, tok, max_len=64, risk_cond=cond)
        if verbose:
            print(f"\n{'=' * 62}\nrisk_cond={cond}\n{'=' * 62}")
        for i in range(len(DF)):
            item = ds[i]
            post = DF.loc[i, "post"]
            if cond:
                enc = tok(RISK_PREFIX.format(DF.loc[i, "risk_level"].lower()),
                          post, truncation="only_second", max_length=64,
                          padding="max_length", return_offsets_mapping=True)
            else:
                enc = tok(post, truncation=True, max_length=64,
                          padding="max_length", return_offsets_mapping=True)
            offs = enc["offset_mapping"]
            lab = item["labels"].tolist()
            picked = [post[a:b] for l, (a, b) in zip(lab, offs) if l in (1, 2)]
            n_lab = sum(1 for l in lab if l != -100)
            got = " ".join("".join(picked).split())
            want = DF.loc[i, "evidence"]
            ok = (got.lower().replace(" ", "") == want.lower().replace(" ", ""))
            bad += (not ok)
            if verbose:
                print(f"  row {DF.loc[i, 'row_id']}  "
                      f"labelled_tokens={n_lab:>3}  "
                      f"{'OK ' if ok else 'BAD'}  decoded={got!r}  "
                      f"want={want!r}")
    return bad


def test_risk_cond_alignment():
    bad = _misaligned(verbose=False)
    assert bad == 0, (
        f"{bad} rows decode to the wrong span -- the conditioning prefix is "
        f"leaking into the post's offsets")


def main():
    bad = _misaligned(verbose=True)
    print("\n" + "=" * 62)
    if bad:
        raise SystemExit(f"{bad} misaligned rows -- do NOT train this arm")
    print("targets align to the post in both modes; the arm is safe to train")


if __name__ == "__main__":
    main()
