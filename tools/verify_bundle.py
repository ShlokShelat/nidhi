#!/usr/bin/env python3
"""Check that the gated, text-bearing reproduction files are present and are
byte-identical to the ones the submission was built from.

    python tools/verify_bundle.py            # run from the repository root

Exits non-zero if any file is missing or differs. Expected hashes live in
tools/gated_files.txt; the human-readable list is artifacts/MANIFEST.md.
"""
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad = 0
with open(os.path.join(ROOT, "tools", "gated_files.txt")) as fh:
    for line in fh:
        want, rel = line.split()
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"MISSING   {rel}")
            bad += 1
            continue
        with open(path, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        print(f"{'ok' if got == want else 'DIFFERS':<9} {rel}")
        bad += got != want
print("\nREADY: every gated file matches." if not bad else f"\n{bad} problem(s), listed above.")
sys.exit(1 if bad else 0)
