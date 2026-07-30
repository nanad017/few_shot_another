"""Generate a tiny random dataset (train/val/test) to smoke-test the
pipeline before running on the real Filtered LargePE data.

    python scripts/make_fake_dataset.py --root data-fake
    python train.py --data-root data-fake --smoke-test --image-size 64
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data-fake")
    p.add_argument("--classes-per-split", type=int, default=6)
    p.add_argument("--samples-per-class", type=int, default=20)
    p.add_argument("--size", type=int, default=64)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        for c in range(args.classes_per_split):
            d = Path(args.root) / split / f"{split}_class_{c:02d}"
            d.mkdir(parents=True, exist_ok=True)
            base = rng.integers(0, 256, (args.size, args.size), dtype=np.uint8)
            for i in range(args.samples_per_class):
                noise = rng.integers(0, 64, (args.size, args.size), dtype=np.uint8)
                img = ((base.astype(np.int16) + noise) % 256).astype(np.uint8)
                Image.fromarray(img, mode="L").save(d / f"{i:03d}.png")
    print(f"fake dataset written to {args.root}/")


if __name__ == "__main__":
    main()
