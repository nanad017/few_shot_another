"""InfoNCE loss (Eq. 14-15)."""

import torch


def info_nce(z_o: torch.Tensor, z_p: torch.Tensor, z_negs: torch.Tensor,
             tau: float = 0.07) -> torch.Tensor:
    """l_{o,p} for one anchor/positive pair against n negatives.
    All inputs are L2-normalized vectors ([D] and [n, D])."""
    pos = (z_o * z_p).sum() / tau
    if z_negs.numel():
        negs = z_negs @ z_o / tau
        logits = torch.cat([pos.unsqueeze(0), negs])
    else:
        logits = pos.unsqueeze(0)
    return -(pos - torch.logsumexp(logits, dim=0))
