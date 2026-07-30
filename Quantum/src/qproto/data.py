"""Dataset loading and a synthetic generator for end-to-end smoke tests.

The paper's datasets must be obtained separately:
  - CCCS-CIC-AndMal-2020: https://www.unb.ca/cic/datasets/andmal2020.html
  - KronoDroid (real-device subset): https://github.com/aleguma/kronodroid

`load_csv` expects a CSV with feature columns, a label column, and optionally a
timestamp column (KronoDroid) for drift evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path, label_col, timestamp_col=None, drop_cols=()):
    df = pd.read_csv(path)
    y = df[label_col].to_numpy()
    ts = df[timestamp_col].to_numpy() if timestamp_col else None
    drop = [label_col, *([timestamp_col] if timestamp_col else []), *drop_cols]
    X = df.drop(columns=drop).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return X, y, ts


def stratified_split(X, y, seed=42, extra=None):
    """60/20/20 stratified train/val/test split (paper, "Data partitioning strategy")."""
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    out = [(X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])]
    if extra is not None:
        out.append((extra[tr], extra[va], extra[te]))
    return out


def make_synthetic(n_classes=15, n_per_class=400, d=200, d_informative=40,
                   drift_strength=1.5, seed=42):
    """Synthetic APK-like data: sparse count features, class-dependent Gaussian
    means on an informative subspace, pseudo-timestamps with gradual drift."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.0, size=(n_classes, d_informative))
    X, y, ts = [], [], []
    for c in range(n_classes):
        t = rng.uniform(0, 1, size=n_per_class)  # normalized collection time
        base = rng.normal(0, 1, size=(n_per_class, d))
        drift_dir = rng.normal(0, 1, size=d_informative)
        base[:, :d_informative] += centers[c] + np.outer(t, drift_dir) * drift_strength
        base[:, d_informative:] *= 0.5
        mask = rng.uniform(size=(n_per_class, d)) < 0.3  # sparsity like count features
        base[mask] = 0.0
        X.append(base)
        y.append(np.full(n_per_class, c))
        ts.append(t)
    X, y, ts = np.concatenate(X), np.concatenate(y), np.concatenate(ts)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], ts[perm]
