#!/usr/bin/env python3
"""Generate a synthetic dataset for smoke-testing the pipeline."""

import argparse

from a2clm.data.synthetic import generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--per-family", type=int, default=20)
    ap.add_argument("--benign", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    meta = generate(args.out, args.per_family, args.benign, seed=args.seed)
    print(f"metadata written to {meta}")
