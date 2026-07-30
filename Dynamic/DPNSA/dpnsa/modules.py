"""Dynamic convolution (attention over convolution kernels).

Implements the dynamic convolution layer from Chen et al., "Dynamic
Convolution: Attention over Convolution Kernels" (CVPR 2020) — reference
[47] of the DPNSA paper. K parallel kernels are aggregated with
input-dependent attention weights computed squeeze-and-excitation style
(GAP -> FC -> ReLU -> FC -> softmax with temperature).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelAttention(nn.Module):
    """Squeeze-and-excitation attention over the K candidate kernels."""

    def __init__(self, in_channels: int, num_kernels: int, reduction: int = 4):
        super().__init__()
        hidden = max(in_channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, num_kernels, kernel_size=1)
        # Softmax temperature; annealed from a high value towards 1 during
        # training to ease joint optimisation of all kernels (see the
        # dynamic convolution paper). Kept as a buffer so it is saved in
        # checkpoints and can be updated from the training loop.
        self.register_buffer("temperature", torch.tensor(30.0))

    def set_temperature(self, value: float) -> None:
        self.temperature.fill_(max(float(value), 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.pool(x)
        a = F.relu(self.fc1(a), inplace=True)
        a = self.fc2(a).flatten(1)  # (B, K)
        return F.softmax(a / self.temperature, dim=1)


class DynamicConv2d(nn.Module):
    """Conv2d whose kernel is an attention-weighted sum of K kernels.

    The aggregated kernel depends on the input sample, so the layer
    realises sample-adaptive feature extraction while keeping the network
    shallow (the few-shot setting of DPNSA).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_kernels: int = 4,
        reduction: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_kernels = num_kernels

        self.attention = KernelAttention(in_channels, num_kernels, reduction)
        self.weight = nn.Parameter(
            torch.empty(num_kernels, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(num_kernels, out_channels)) if bias else None
        for k in range(num_kernels):
            nn.init.kaiming_normal_(self.weight[k], mode="fan_out", nonlinearity="relu")

    def set_temperature(self, value: float) -> None:
        self.attention.set_temperature(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        attn = self.attention(x)  # (B, K)

        # Aggregate a per-sample kernel, then run as a grouped convolution
        # over the batch (standard trick: fold batch into groups).
        weight = self.weight.view(self.num_kernels, -1)
        agg_weight = (attn @ weight).view(
            b * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        agg_bias = (attn @ self.bias).view(-1) if self.bias is not None else None

        out = F.conv2d(
            x.reshape(1, b * c, h, w),
            agg_weight,
            agg_bias,
            stride=self.stride,
            padding=self.padding,
            groups=b,
        )
        return out.view(b, self.out_channels, out.shape[-2], out.shape[-1])
