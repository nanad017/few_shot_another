"""Episodic sampling for N-way K-shot tasks (paper, "Episodic training paradigm").

Each episode samples N classes uniformly, then K support and Q query samples
per class (disjoint); labels are remapped to {0, ..., N-1}.
"""

import numpy as np


class EpisodeSampler:
    def __init__(self, X: np.ndarray, y: np.ndarray, n_way: int, k_shot: int,
                 q_query: int, seed: int = 42):
        self.X = X
        self.n_way, self.k_shot, self.q_query = n_way, k_shot, q_query
        self.rng = np.random.default_rng(seed)
        self.class_indices = {}
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if len(idx) >= k_shot + q_query:
                self.class_indices[c] = idx
        if len(self.class_indices) < n_way:
            raise ValueError(
                f"only {len(self.class_indices)} classes have >= {k_shot + q_query} "
                f"samples; need at least {n_way}")
        self.classes = list(self.class_indices)

    def sample(self):
        """Return (support_X, support_y, query_X, query_y) with remapped labels."""
        chosen = self.rng.choice(len(self.classes), size=self.n_way, replace=False)
        sx, sy, qx, qy = [], [], [], []
        for new_label, ci in enumerate(chosen):
            idx = self.rng.choice(self.class_indices[self.classes[ci]],
                                  size=self.k_shot + self.q_query, replace=False)
            sx.append(self.X[idx[: self.k_shot]])
            qx.append(self.X[idx[self.k_shot:]])
            sy.append(np.full(self.k_shot, new_label))
            qy.append(np.full(self.q_query, new_label))
        return (np.concatenate(sx), np.concatenate(sy),
                np.concatenate(qx), np.concatenate(qy))
