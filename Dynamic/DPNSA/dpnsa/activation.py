"""Dual-sample dynamic activation module (Fig. 3 / Fig. 4, Eq. 1-2).

Given a query feature Q' and a class prototype P_n, a hyper function
alpha(Q', P_n) produces per-channel activation parameters (slopes a_c^f
and intercepts b_c^f for F piecewise-linear components). Both inputs are
then activated with

    D(x)_c = max_f ( a_c^f * x_c + b_c^f )

(the dual-sample extension of Dynamic ReLU, ref [49]), after which a
shared conv + max-pool extracts the important information.

Two hyper-function designs are provided:
  DS1 (Fig. 4a): concat -> Conv -> GAP -> FC -> ReLU -> FC -> normalize
  DS2 (Fig. 4b): concat -> spatial attention -> Conv -> GAP -> FC -> ReLU
                 -> FC -> normalize
The paper's ablation (Table 4) finds DS1 the stronger variant, so it is
the default.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention used by the DS2 hyper function."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class DualSampleActivation(nn.Module):
    def __init__(
        self,
        channels: int,
        num_funcs: int = 2,
        reduction: int = 8,
        variant: str = "ds1",
    ):
        super().__init__()
        assert variant in ("ds1", "ds2")
        self.channels = channels
        self.num_funcs = num_funcs
        self.variant = variant

        self.spatial_attn = SpatialAttention() if variant == "ds2" else None
        self.corr_conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 2 * num_funcs * channels)

        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def _hyper(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """alpha(Q', P_n): per-channel (a, b) parameters. Returns (B, 2, F, C)."""
        x = torch.cat([q, p], dim=1)
        if self.spatial_attn is not None:
            x = self.spatial_attn(x)
        x = self.corr_conv(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        # Fig. 4 applies a normalization layer to the generated activation
        # parameters before using them as the dual-sample Dynamic-ReLU
        # coefficients. Normalize the complete (a, b) vector per pair.
        x = F.normalize(x, p=2, dim=1, eps=1e-6)
        return x.view(-1, 2, self.num_funcs, self.channels)

    def _activate(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W); a, b: (B, F, C)
        x = x.unsqueeze(1)                      # (B, 1, C, H, W)
        a = a.unsqueeze(-1).unsqueeze(-1)       # (B, F, C, 1, 1)
        b = b.unsqueeze(-1).unsqueeze(-1)
        return (a * x + b).amax(dim=1)          # max over F -> (B, C, H, W)

    def forward(self, q: torch.Tensor, p: torch.Tensor):
        """Activate a batch of (query, prototype) pairs.

        q, p: (B, C, H, W) -> activated, pooled (B, C, H/2, W/2) each.
        """
        theta = self._hyper(q, p)  # (B, 2, F, C)
        a = theta[:, 0]  # (B, F, C)
        b = theta[:, 1]
        q_out = self.post(self._activate(q, a, b))
        p_out = self.post(self._activate(p, a, b))
        return q_out, p_out
