#!/usr/bin/env python3
"""For the authors: pack the gated, text-bearing files from the original
competition submission into one archive, for distribution ONLY through an
access-controlled channel (for example the gated Hugging Face repository that
already hosts the weights). Never attach this archive to a public release.

    python tools/pack_gated_bundle.py --submission-dir /path/to/NIDHI \
        --out nidhi_gated_bundle.tar.gz

Refuses to pack if any file is missing or does not match tools/gated_files.txt.
"""
import argparse
import hashlib
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--submission-dir", required=True)
ap.add_argument("--out", default="nidhi_gated_bundle.tar.gz")
a = ap.parse_args()
with open(os.path.join(ROOT, "tools", "gated_files.txt")) as fh:
    entries = [line.split() for line in fh if line.strip()]
for want, rel in entries:
    p = os.path.join(a.submission_dir, rel)
    if not os.path.exists(p):
        sys.exit(f"refusing: {rel} is missing from {a.submission_dir}")
    with open(p, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != want:
            sys.exit(f"refusing: {rel} does not match its recorded hash")
with tarfile.open(a.out, "w:gz") as t:
    for _, rel in entries:
        t.add(os.path.join(a.submission_dir, rel), arcname=rel)
print(f"wrote {a.out} with {len(entries)} files; distribute it through a gated channel only")
