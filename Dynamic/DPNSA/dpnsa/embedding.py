"""Dynamic feature embedding module (Fig. 2 of the DPNSA paper).

Three blocks, each: conv 3x3 -> batch norm -> ReLU -> 2x2 max-pool.
The middle block uses dynamic convolution; the first and last use static
convolution. Filters: 32 -> 64 -> 128.
"""

import torch.nn as nn

from .modules import DynamicConv2d


class DynamicEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, num_kernels: int = 4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.dynconv = DynamicConv2d(32, 64, kernel_size=3, padding=1, num_kernels=num_kernels)
        self.block2_rest = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.out_channels = 128

    def set_temperature(self, value: float) -> None:
        self.dynconv.set_temperature(value)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2_rest(self.dynconv(x))
        x = self.block3(x)
        return x  # (B, 128, H/8, W/8)
