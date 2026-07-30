"""Attribute masking attack (Sec. III-C2, Eq. 5):
S' = S * (1 - L_m) + V * L_m with V ~ N(mu, sigma^2)."""

import torch


class AttributeMaskingAttack:
    name = "mask"

    def __init__(self, ratio: float = 0.3, mu: float = 0.0, sigma: float = 0.1,
                 generator: torch.Generator | None = None):
        self.ratio = ratio
        self.mu = mu
        self.sigma = sigma
        self.generator = generator

    def __call__(self, g):
        out = g.clone()
        mask = (torch.rand(out.x.shape, generator=self.generator,
                           device=out.x.device) < self.ratio).float()
        noise = self.mu + self.sigma * torch.randn(
            out.x.shape, generator=self.generator, device=out.x.device)
        out.x = out.x * (1.0 - mask) + noise * mask
        return out
