"""
Patch extraction and embedding module for WaveSeekerNet.

Converts a 2-D input matrix into a sequence of patch embeddings using
an unfold-based approach followed by a gated linear unit projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from WaveSeekerNet.sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear
from WaveSeekerNet.sub_modules.lib.pos_encoding import PositionalEncoding


class MakePatches(nn.Module):
    """Extract and embed non-overlapping patches from a 2-D input.

    Uses :class:`torch.nn.Unfold` to extract patches, then projects each
    patch to ``emb_dim`` dimensions via a gated linear unit (GLU).
    Positional encodings are added after projection.

    Parameters
    ----------
    patch_width : int
        Width of each patch.
    patch_height : int
        Height of each patch.
    emb_dim : int
        Output embedding dimension.
    n_channel : int
        Number of input channels. Default ``1``.
    patch_mode : str
        One of ``"patch"``, ``"compress"`` (overlapping stride = patch size)
        or ``"full"`` (stride = 1). Default ``"patch"``.
    """

    def __init__(
        self,
        patch_width: int,
        patch_height: int,
        emb_dim: int,
        n_channel: int = 1,
        patch_mode: str = "patch",
    ) -> None:
        super().__init__()

        stride = (patch_width, patch_height) if patch_mode in ("patch", "compress") else 1

        self.conv = nn.Unfold((patch_width, patch_height), stride=stride)
        self.n_channel = n_channel
        self.pos_encoding = PositionalEncoding(emb_dim)

        patch_dim = patch_width * patch_height
        self.W1 = NoisyFactorizedLinear(patch_dim, 128, sigma_zero=0.1)
        self.W2 = NoisyFactorizedLinear(patch_dim, 128, sigma_zero=0.1)
        self.W3 = nn.Linear(128, emb_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract patches, apply GLU projection, and add positional encoding.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape ``(B, n_patches, emb_dim)``.
        """
        x = self.conv(inputs)                  # (B, patch_dim, n_patches)
        x = x.transpose(1, -1)                 # (B, n_patches, patch_dim)

        # Gated activation
        x_gate = F.mish(self.W1(x))           # (B, n_patches, 128)
        x_inner = self.W2(x)                   # (B, n_patches, 128)
        x = x_gate * x_inner                   # GLU element-wise product
        x = self.W3(x)                         # (B, n_patches, emb_dim)

        return self.pos_encoding(x)
