"""
utils.py

Minimal utilities for ConvNeXt V2 blocks used in this project.

References
----------
- GitHub: https://github.com/facebookresearch/ConvNeXt-V2
- Paper:  ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
          arXiv:2301.00808

Notes
-----
- Only the components required by our model are included:
  * LayerNorm: supports both channels_last (NHWC) and channels_first (NCHW).
  * GRN: Global Response Normalization operating on NHWC tensors.

- DropPath is imported from timm in model.py, so it is not duplicated here.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    LayerNorm with support for two data formats:
      - channels_last  : input shape (N, H, W, C)  → uses F.layer_norm
      - channels_first : input shape (N, C, H, W)  → manual normalization over channel dim

    Parameters
    ----------
    normalized_shape : int
        Channel dimension (C).
    eps : float, default=1e-6
        Numerical stability term.
    data_format : {"channels_last", "channels_first"}, default="channels_last"
        Expected memory format of the input tensor.

    Usage
    -----
    * In ConvNeXt V2 Block (NHWC path):    LayerNorm(C)
    * In downsample layers (NCHW path):    LayerNorm(C, data_format="channels_first")
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError(f"Unsupported data_format: {data_format}")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        # for F.layer_norm (channels_last)
        self._shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            # x: (N, H, W, C) → normalize over last dim C
            return F.layer_norm(x, self._shape, self.weight, self.bias, self.eps)
        else:
            # x: (N, C, H, W) → normalize over channel dim C (manual LN)
            mean = x.mean(dim=1, keepdim=True)
            var  = (x - mean).pow(2).mean(dim=1, keepdim=True)
            xhat = (x - mean) / torch.sqrt(var + self.eps)
            # broadcast (C,) to (1,C,1,1)
            return self.weight[:, None, None] * xhat + self.bias[:, None, None]


class GRN(nn.Module):
    """
    Global Response Normalization (GRN).

    Operates on NHWC tensors (N, H, W, C) as in ConvNeXt V2:
      Gx = ||x||_2 over spatial dims (H,W), shape (N,1,1,C)
      Nx = Gx / mean_C(Gx)
      y  = gamma * (x * Nx) + beta + x

    Parameters
    ----------
    dim : int
        Channel dimension C.
    eps : float, default=1e-6
        Numerical stability term.

    Notes
    -----
    * In ConvNeXt V2 Block, tensors are permuted to NHWC before GRN.
    * Ensure input to GRN is NHWC; the model handles permute().
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta  = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, H, W, C)
        # L2 norm over spatial dims
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)          # (N,1,1,C)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)       # (N,1,1,C)
        return self.gamma * (x * Nx) + self.beta + x
