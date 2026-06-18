"""
gMLP (gated MLP) token-mixing block for WaveSeekerNet.

References
----------
gMLP: https://arxiv.org/abs/2105.08050
Efficient Attention: https://arxiv.org/abs/1812.01243
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import RMSNorm

from WaveSeekerNet.sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear


class SpatialGatingUnit(nn.Module):
    """Spatial gating unit from gMLP.

    Splits the input into two halves, applies a spatial (token-wise)
    projection to one half, and multiplies them element-wise.
    Optionally adds an efficient self-attention branch.

    Parameters
    ----------
    d_ffn : int
        Feature dimension of *each half* (total input is ``2 * d_ffn``).
    n_patches : int
        Sequence length (number of tokens/patches).
    use_attn : bool
        If ``True``, add an efficient linear attention term to the
        spatial gate.
    """

    def __init__(self, d_ffn: int, n_patches: int, use_attn: bool) -> None:
        super().__init__()

        self.norm = RMSNorm(d_ffn, eps=1e-8)
        self.spatial_proj = NoisyFactorizedLinear(n_patches, n_patches)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        self.use_attn = use_attn

        if use_attn:
            self.get_qkv = NoisyFactorizedLinear(d_ffn * 2, 64 * 3)
            self.get_qkv.apply(self._weight_init)
            self.proj_out = nn.Linear(64, d_ffn)
            self.attn_dropout = nn.Dropout1d(0.1)

    @staticmethod
    def _weight_init(module: nn.Module) -> None:
        """Kaiming initialisation for NoisyFactorizedLinear layers."""
        if isinstance(module, NoisyFactorizedLinear):
            nn.init.kaiming_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Efficient linear self-attention over the full input.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, L, 2 * d_ffn)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, L, d_ffn)``.
        """
        x_q, x_k, x_v = self.get_qkv(x).chunk(3, dim=-1)  # each (B, L, 64)

        # Reshape to (B, H=2, L, 32)
        x_qh = x_q.view(*x_q.shape[:2], 2, x_q.shape[-1] // 2).transpose(1, 2)
        x_kh = x_k.view(*x_k.shape[:2], 2, x_k.shape[-1] // 2).transpose(1, 2)
        x_vh = x_v.view(*x_v.shape[:2], 2, x_v.shape[-1] // 2).transpose(1, 2)

        x_kh = nn.functional.softmax(x_kh, dim=-2)
        x_qh = nn.functional.softmax(x_qh, dim=-1)

        ch = x_kh.transpose(-1, -2) @ x_vh          # (B, H, 32, 32)
        x_attn = x_qh @ ch.transpose(-1, -2)        # (B, H, L, 32)

        x_attn = x_attn.transpose(1, 2).flatten(-2, -1)  # (B, L, 64)
        return self.attn_dropout(self.proj_out(x_attn))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial gating.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, L, 2 * d_ffn)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, L, d_ffn)``.
        """
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = torch.transpose(self.spatial_proj(torch.transpose(v, -1, -2)), -1, -2)

        if self.use_attn:
            v = v + self._attention(x)

        return u * v


class gMLPBlock(nn.Module):
    """Gated MLP (gMLP) token-mixing block.

    Applies channel expansion → spatial gating → channel projection,
    following the gMLP architecture.

    Parameters
    ----------
    embedding_dim : int
        Input and output embedding dimension.
    ffn_dropout : float
        Dropout probability in channel projection layers.
    n_patches : int
        Sequence length.
    activation : type
        Activation function class. Default :class:`torch.nn.Mish`.
    dropout_type : str
        Unused; kept for API compatibility.
    use_attn : bool
        Pass attention flag to :class:`SpatialGatingUnit`. Default ``True``.
    """

    def __init__(
        self,
        embedding_dim: int,
        ffn_dropout: float,
        n_patches: int,
        activation: type = nn.Mish,
        dropout_type: str = "patch",
        use_attn: bool = True,
    ) -> None:
        super().__init__()

        self.channel_proj_1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            activation(),
            nn.Dropout1d(ffn_dropout),
        )
        self.spatial_gate = SpatialGatingUnit(embedding_dim, n_patches, use_attn)
        self.channel_proj_2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            activation(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout1d(ffn_dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the gMLP block.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(B, n_patches, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Same shape as *inputs*.
        """
        x = self.channel_proj_1(inputs)
        x = self.spatial_gate(x)
        return self.channel_proj_2(x)