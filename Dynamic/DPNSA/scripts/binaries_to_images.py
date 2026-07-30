"""Convert malware binaries into grayscale images (Nataraj et al. [21]).

Each byte of the file becomes one pixel (0-255). The byte stream is
reshaped to a fixed width (chosen from the file size, per the original
paper's table), then the image is resized to a fixed square, matching
the fixed-width fixed-length images used by DPNSA (256x256).

Input layout:  in_root/<class_name>/<binary files>
Output layout: out_root/<class_name>/<name>.png

    python scripts/binaries_to_images.py --in-root raw/train --out-root data/train
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def width_for_size(n_bytes: int) -> int:
    kb = n_bytes / 1024
    for limit, width in [
        (10, 32), (30, 64), (60, 128), (100, 256), (200, 384),
        (500, 512), (1000, 768),
    ]:
        if kb < limit:
            return width
    return 1024


def binary_to_image(path: Path, size: int) -> Image.Image:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"empty file: {path}")
    width = width_for_size(data.size)
    height = max(data.size // width, 1)
    img = Image.fromarray(data[: width * height].reshape(height, width), mode="L")
    return img.resize((size, size), Image.BILINEAR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()

    in_root, out_root = Path(args.in_root), Path(args.out_root)
    n = 0
    for cls_dir in sorted(d for d in in_root.iterdir() if d.is_dir()):
        out_dir = out_root / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(p for p in cls_dir.iterdir() if p.is_file()):
            try:
                binary_to_image(f, args.size).save(out_dir / (f.stem + ".png"))
                n += 1
            except ValueError as e:
                print(f"skip: {e}")
    print(f"converted {n} binaries -> {out_root}")


if __name__ == "__main__":
    main()
