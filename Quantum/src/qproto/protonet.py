"""Few-shot prototypical network (paper, "Few-shot prototypical network architecture").

Embedding network f_phi: R^d -> R^128:
    h1 = Dropout(ReLU(BN(W1 x + b1)), p=0.3)     W1: 512 x d
    h2 = ReLU(BN(W2 h1 + b2))                    W2: 256 x 512
    e  = W3 h2 + b3                              W3: 128 x 256  (no activation)

Prototypes are class centroids of support embeddings (Eq. 3); classification is
a softmax over negative squared Euclidean distances (Eq. 4); training minimizes
the episodic negative log-likelihood (Eq. 5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, in_dim: int, hidden1: int = 512, hidden2: int = 256,
                 out_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.BatchNorm1d(hidden1),
            nn.ReLU(), nn.Dropout(dropout))
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU())
        self.layer3 = nn.Linear(hidden2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.layer2(self.layer1(x)))


def prototypes(support_emb: torch.Tensor, support_y: torch.Tensor,
               n_way: int) -> torch.Tensor:
    """Eq. 3: class centroids c_k of the embedded support set. -> (n_way, d_e)"""
    return torch.stack([support_emb[support_y == k].mean(dim=0)
                        for k in range(n_way)])


def proto_logits(query_emb: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    """Eq. 4 logits: negative squared Euclidean distance to each prototype."""
    return -torch.cdist(query_emb, protos).pow(2)


def prototypical_loss(query_emb: torch.Tensor, query_y: torch.Tensor,
                      protos: torch.Tensor) -> torch.Tensor:
    """Eq. 5: episodic negative log-likelihood over query samples."""
    return F.cross_entropy(proto_logits(query_emb, protos), query_y)
