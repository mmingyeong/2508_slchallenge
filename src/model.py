"""
model.py

ConvNeXt V2 for binary classification of strong-lensing images.

References:
- GitHub: https://github.com/facebookresearch/ConvNeXt-V2
- Paper: ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
         arXiv:2301.00808

Modifications for project:
- in_chans=1 for grayscale FITS images (shape: (1, 41, 41))
- num_classes=1 for binary classification (use BCEWithLogitsLoss)
"""

import torch
import torch.nn as nn
from timm.layers import trunc_normal_, DropPath
from utils import LayerNorm, GRN


class Block(nn.Module):
    """
    ConvNeXtV2 Block.
    Depthwise conv -> LayerNorm -> Pointwise conv (Linear) -> GELU -> GRN -> Pointwise conv -> DropPath residual.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return identity + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    """
    ConvNeXt V2 backbone + classification head.

    Args:
        in_chans (int): Number of input channels (1 for grayscale FITS).
        num_classes (int): Output classes (1 for binary logits).
        depths (list[int]): Number of blocks per stage.
        dims (list[int]): Feature dimensions per stage.
        drop_path_rate (float): Stochastic depth rate.
        head_init_scale (float): Init scaling for classifier weights.
    """

    def __init__(
        self,
        in_chans=1,
        num_classes=1,
        depths=[2, 2, 6, 2],
        dims=[40, 80, 160, 320],
        drop_path_rate=0.0,
        head_init_scale=1.0,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        # Stem
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        # Stages with downsampling
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])  # Global average pooling
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


# -------------------------------------------------------------------------
# Model constructors for different scales
# -------------------------------------------------------------------------
def convnextv2_atto(**kwargs) -> ConvNeXtV2:
    return ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)


def convnextv2_nano(**kwargs) -> ConvNeXtV2:
    return ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)


def convnextv2_tiny(**kwargs) -> ConvNeXtV2:
    return ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
