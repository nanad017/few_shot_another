"""The A2-CLM framework (Algorithm 1): three encoders GAT_o / GAT_p / GAT_q
with projection heads MLP_o / MLP_p / MLP_q. Only the online branch (o) is
trained by backpropagation; p and q are momentum-updated copies
(I_p = l1*I_p + (1-l1)*I_o, I_q = l2*I_q + (1-l2)*I_o, lines 19-20)."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph import SHGFM
from .encoder import MetaGraphGATEncoder, ProjectionHead


class A2CLM(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4,
                 proj_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.enc_o = MetaGraphGATEncoder(hidden_dim, num_layers, dropout)
        self.head_o = ProjectionHead(self.enc_o.out_dim, proj_dim)
        self.enc_p = copy.deepcopy(self.enc_o)
        self.head_p = copy.deepcopy(self.head_o)
        self.enc_q = copy.deepcopy(self.enc_o)
        self.head_q = copy.deepcopy(self.head_o)
        for m in (self.enc_p, self.head_p, self.enc_q, self.head_q):
            for p in m.parameters():
                p.requires_grad_(False)

    def _branch(self, which: str):
        return {"o": (self.enc_o, self.head_o),
                "p": (self.enc_p, self.head_p),
                "q": (self.enc_q, self.head_q)}[which]

    def embed(self, g: SHGFM, branch: str = "o",
              delta: torch.Tensor | None = None) -> torch.Tensor:
        enc, head = self._branch(branch)
        z = head(enc(g, delta=delta))
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def momentum_update(self, lambda1: float = 0.99, lambda2: float = 0.99):
        for tgt, lam in ((self.enc_p, lambda1), (self.enc_q, lambda2)):
            for p_t, p_o in zip(tgt.parameters(), self.enc_o.parameters()):
                p_t.mul_(lam).add_(p_o, alpha=1.0 - lam)
        for tgt, lam in ((self.head_p, lambda1), (self.head_q, lambda2)):
            for p_t, p_o in zip(tgt.parameters(), self.head_o.parameters()):
                p_t.mul_(lam).add_(p_o, alpha=1.0 - lam)
