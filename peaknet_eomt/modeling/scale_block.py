# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    """2D Layer Normalization (replacement for timm.layers.LayerNorm2d)"""
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: [B, C, H, W]
        u = x.mean([2, 3], keepdim=True)  # Mean over spatial dimensions
        s = ((x - u) ** 2).mean([2, 3], keepdim=True)  # Variance over spatial dimensions
        x = (x - u) / torch.sqrt(s + self.eps)
        
        if self.weight is not None:
            x = x * self.weight[:, None, None]
        if self.bias is not None:
            x = x + self.bias[:, None, None]
        return x


class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, conv1_layer=nn.ConvTranspose2d):
        super().__init__()

        self.conv1 = conv1_layer(
            embed_dim,
            embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x