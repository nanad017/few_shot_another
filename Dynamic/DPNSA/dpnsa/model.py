"""DPNSA: Dynamic Prototype Network based on Sample Adaptation.

Pipeline per episode (Fig. 1):
  1. Dynamic feature embedding of support + query images.
  2. Prototype = mean dynamic embedding of each class's support samples.
  3. Dual-sample dynamic activation of every (query, prototype) pair.
  4. Cosine distance between the activated pair; softmax over -distance
     gives class probabilities (Eq. 3); NLL loss (Eq. 4); prediction is
     the nearest prototype (Eq. 5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import DynamicEmbedding
from .activation import DualSampleActivation


class DPNSA(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_kernels: int = 4,
        num_funcs: int = 2,
        variant: str = "ds1",
        distance: str = "cosine",
    ):
        super().__init__()
        assert distance in ("cosine", "euclidean")
        self.encoder = DynamicEmbedding(in_channels, num_kernels=num_kernels)
        self.activation = DualSampleActivation(
            self.encoder.out_channels, num_funcs=num_funcs, variant=variant
        )
        self.distance = distance

    def set_temperature(self, value: float) -> None:
        self.encoder.set_temperature(value)

    def _distance(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        q = q.flatten(1)
        p = p.flatten(1)
        if self.distance == "cosine":
            return 1.0 - F.cosine_similarity(q, p, dim=1)
        return ((q - p) ** 2).sum(dim=1)

    def forward(self, support: torch.Tensor, query: torch.Tensor, n_way: int, k_shot: int):
        """Compute per-query logits over the N episode classes.

        support: (N*K, C, H, W), grouped by class (class 0 first, ...).
        query:   (Q, C, H, W).
        Returns logits (Q, N): higher = closer prototype.
        """
        z = self.encoder(torch.cat([support, query], dim=0))
        z_support, z_query = z[: support.shape[0]], z[support.shape[0]:]

        c, h, w = z_support.shape[1:]
        prototypes = z_support.view(n_way, k_shot, c, h, w).mean(dim=1)  # (N, C, H, W)

        n_query = z_query.shape[0]
        # Every (query, prototype) pair goes through the dual-sample
        # activation, since the activation parameters depend on the pair.
        q_rep = z_query.unsqueeze(1).expand(n_query, n_way, c, h, w).reshape(-1, c, h, w)
        p_rep = prototypes.unsqueeze(0).expand(n_query, n_way, c, h, w).reshape(-1, c, h, w)
        q_act, p_act = self.activation(q_rep, p_rep)

        dist = self._distance(q_act, p_act).view(n_query, n_way)
        return -dist

    def episode_loss(self, support, query, query_labels, n_way, k_shot):
        logits = self.forward(support, query, n_way, k_shot)
        loss = F.cross_entropy(logits, query_labels)
        acc = (logits.argmax(dim=1) == query_labels).float().mean()
        return loss, acc, logits
