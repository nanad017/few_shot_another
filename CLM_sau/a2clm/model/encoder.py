"""Graph encoder of A2-CLM (Sec. III-D1/D2).

Per meta-graph M_i, K layers of GAT-style attention (Eq. 8) with GIN-style
aggregation (Eq. 9) run over the meta-graph-restricted edge set; the target
process representations of all K layers are concatenated (Eq. 10). A
semantic attention over meta-graphs (Eq. 11) produces the graph-level
representation h_G (Eq. 12). A non-linear projection head implements
Eq. 13."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph import SHGFM
from ..schema import ENTITY_TYPES, F_IN, NUM_META_GRAPHS


class GATGINLayer(nn.Module):
    """h_i^(k) = MLP((1+eps_k) h_i^(k-1) + sum_j alpha_ij h_j^(k-1))."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.att = nn.Linear(2 * dim, 1)              # W^T [x_i || x_j] + b (Eq. 8)
        self.eps = nn.Parameter(torch.zeros(1))       # trainable balance (Eq. 9)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(dim, dim))
        self.act = nn.ReLU()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        agg = torch.zeros_like(h)
        if edge_index.numel():
            src, dst = edge_index[0], edge_index[1]
            e = F.leaky_relu(self.att(torch.cat([h[dst], h[src]], dim=-1))).squeeze(-1)
            # softmax over incoming edges of each dst node
            e_max = torch.full((n,), float("-inf"), device=h.device)
            e_max = e_max.index_reduce(0, dst, e, "amax", include_self=True)
            e_exp = torch.exp(e - e_max[dst])
            denom = torch.zeros(n, device=h.device).index_add_(0, dst, e_exp)
            alpha = e_exp / denom[dst].clamp_min(1e-12)
            agg.index_add_(0, dst, alpha.unsqueeze(-1) * h[src])
        return self.act(self.mlp((1.0 + self.eps) * h + agg))


class MetaGraphGATEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Eq. 2: per-type projection W_T into a common space.
        self.type_proj = nn.ModuleList(
            [nn.Linear(F_IN, hidden_dim) for _ in ENTITY_TYPES])
        self.layers = nn.ModuleList(
            [GATGINLayer(hidden_dim, dropout) for _ in range(num_layers)])
        # Eq. 11: semantic attention over meta-graphs.
        self.att_w = nn.Linear(num_layers * hidden_dim, hidden_dim)
        self.att_b = nn.Linear(hidden_dim, 1, bias=False)

    @property
    def out_dim(self) -> int:
        return self.num_layers * self.hidden_dim

    def input_proj(self, g: SHGFM) -> torch.Tensor:
        h = torch.zeros(g.num_nodes, self.hidden_dim, device=g.x.device)
        for t, proj in enumerate(self.type_proj):
            mask = g.node_types == t
            if mask.any():
                h[mask] = proj(g.x[mask])
        return h

    def forward_projected(self, g: SHGFM, h0: torch.Tensor) -> torch.Tensor:
        """Encode from already-projected node features (PGD perturbs h0)."""
        reps = []
        for mg_idx in range(NUM_META_GRAPHS):
            ei, _ = g.metagraph_edges(mg_idx)
            h, per_layer = h0, []
            for layer in self.layers:
                h = layer(h, ei)
                per_layer.append(h[g.target])
            reps.append(torch.cat(per_layer, dim=-1))    # Eq. 10
        H = torch.stack(reps)                            # [M, K*d]
        scores = self.att_b(torch.tanh(self.att_w(H)))   # Eq. 11
        theta = torch.softmax(scores, dim=0)
        return (theta * H).sum(dim=0)                    # Eq. 12

    def forward(self, g: SHGFM, delta: torch.Tensor | None = None) -> torch.Tensor:
        h0 = self.input_proj(g)
        if delta is not None:
            h0 = h0 + delta
        return self.forward_projected(g, h0)


class ProjectionHead(nn.Module):
    """z = MLP(h_G) (Eq. 13)."""

    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                                 nn.Linear(out_dim, out_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)
