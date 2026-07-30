"""Malware-image dataset and episodic (N-way K-shot) sampling.

Expected directory layout (one folder per malware class, any image
format PIL can read; images are grayscale, resized to `image_size`):

    root/
      train/ classA/ *.png ...
      val/   classB/ *.png ...
      test/  classC/ *.png ...

Use scripts/binaries_to_images.py to convert raw malware binaries into
grayscale images (Nataraj et al. visualisation, as used by the paper).
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}


class MalwareImageFolder(Dataset):
    def __init__(self, root: str, image_size: int = 256, augment: bool = False):
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        if not self.classes:
            raise FileNotFoundError(f"No class folders found under {self.root}")
        self.samples_by_class = {
            c: sorted(
                p for p in (self.root / c).iterdir() if p.suffix.lower() in IMG_EXTS
            )
            for c in self.classes
        }
        empty = [c for c, s in self.samples_by_class.items() if not s]
        if empty:
            raise FileNotFoundError(f"Classes with no images: {empty}")

    def load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.augment and random.random() < 0.5:  # horizontal flip (paper's aug)
            arr = arr[:, ::-1].copy()
        return torch.from_numpy(arr).unsqueeze(0)

    def sample_episode(self, n_way: int, k_shot: int, n_query: int):
        """Returns support (N*K,1,H,W), query (Q,1,H,W), query labels (Q,)."""
        classes = random.sample(self.classes, n_way)
        support, query, labels = [], [], []
        for idx, cls in enumerate(classes):
            paths = self.samples_by_class[cls]
            need = k_shot + n_query
            chosen = (
                random.sample(paths, need)
                if len(paths) >= need
                else [random.choice(paths) for _ in range(need)]
            )
            support += [self.load_image(p) for p in chosen[:k_shot]]
            query += [self.load_image(p) for p in chosen[k_shot:]]
            labels += [idx] * n_query
        return (
            torch.stack(support),
            torch.stack(query),
            torch.tensor(labels, dtype=torch.long),
        )
