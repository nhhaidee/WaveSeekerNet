"""
Fourier-based token-mixing head for WaveSeekerNet.

References
----------
FNet: https://arxiv.org/abs/2105.03824
FFT Processing: https://openreview.net/pdf?id=EXHG-A3jlM
Efficient Attention: https://arxiv.org/abs/1812.01243
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import RMSNorm

from WaveSeekerNet.sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear


class FNETHead(nn.Module):
    """Fourier-based token-mixing attention head.

    Applies a 2-D real FFT to the input, processes the frequency-domain
    representation with an efficient multi-head self-attention mechanism,
    applies soft-thresholding (shrinkage) to prune uninformative frequencies,
    then reconstructs the signal with an inverse FFT.

    Parameters
    ----------
    emb_dim : int
        Token embedding dimension. Must be divisible by 32.
    activation : type
        Activation function class (instantiated internally).
    """

    def __init__(self, emb_dim: int, activation: type) -> None:
        super().__init__()

        self.heads = emb_dim // 32
        self.dim = emb_dim
        self.d_emb = (emb_dim // 2) + 1

        self.multi_head_proj = nn.Linear(self.d_emb, emb_dim)
        self.get_qkv = nn.Linear(32, 32 * 3)
        self.proj_attn = nn.Linear(32, 32)
        self.attn_dropout = nn.Dropout1d(0.125)
        self.norm = RMSNorm(32, eps=1e-8)
        self.out_proj = NoisyFactorizedLinear(32, 32)
        self.merge_proj = nn.Linear(emb_dim, self.d_emb)

        self.process_scale = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            activation(),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Dropout1d(0.125),
        )

        nn.init.xavier_uniform_(self.multi_head_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.merge_proj.weight)
        nn.init.constant_(self.multi_head_proj.bias, 0)
        nn.init.constant_(self.merge_proj.bias, 0)

    def _create_heads(
        self, x_in: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """Reshape flat embeddings into multi-head format.

        Parameters
        ----------
        x_in : torch.Tensor
            Shape ``(B, L, d_emb)``.

        Returns
        -------
        x : torch.Tensor
            Shape ``(B * heads, L, 32)``.
        batch_size : int
        length : int
        """
        batch_size, length, _ = x_in.size()

        x = self.multi_head_proj(x_in)                  # (B, L, emb_dim)
        x = x.view(batch_size, length, self.heads, 32)  # (B, L, H, 32)
        x = x.permute(0, 2, 1, 3).contiguous()         # (B, H, L, 32)
        x = x.view(batch_size * self.heads, length, 32) # (B*H, L, 32)

        return x, batch_size, length

    def _restore_dim(
        self,
        x: torch.Tensor,
        batch_size: int,
        length: int,
        permute: bool = True,
    ) -> torch.Tensor:
        """Inverse of :meth:`_create_heads`.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B * heads, L, 32)``.
        batch_size, length : int
            Saved from :meth:`_create_heads`.
        permute : bool
            If ``True``, return shape ``(B, L, emb_dim)``; otherwise
            return shape ``(B, heads, L, 32)`` without merging.
        """
        x = x.view(batch_size, self.heads, length, 32)  # (B, H, L, 32)
        if permute:
            x = x.permute(0, 2, 1, 3).contiguous()     # (B, L, H, 32)
            x = x.view(batch_size, length, 32 * self.heads)
        return x

    def _attention(
        self,
        x: torch.Tensor,
        batch_size: int,
        length: int,
    ) -> torch.Tensor:
        """Efficient linear self-attention in the frequency domain.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B * heads, L, 32)``.
        batch_size, length : int
            Saved from :meth:`_create_heads`.

        Returns
        -------
        torch.Tensor
            Shape ``(B * heads, L, 32)``.
        """
        x_q, x_k, x_v = self.get_qkv(x).chunk(3, dim=-1)

        # Efficient attention — reshape to (B, H, L, 32) for key/query softmax
        x_qh = self._restore_dim(x_q, batch_size, length, permute=False)
        x_kh = self._restore_dim(x_k, batch_size, length, permute=False)
        x_vh = self._restore_dim(x_v, batch_size, length, permute=False)

        x_kh = nn.functional.softmax(x_kh, dim=-2)
        x_qh = nn.functional.softmax(x_qh, dim=-1)

        ch = x_kh.transpose(-1, -2) @ x_vh          # (B, H, 32, 32)
        x_attn = x_qh @ ch.transpose(-1, -2)        # (B, H, L, 32)

        x_attn = x_attn.view(batch_size * self.heads, length, 32)
        return self.attn_dropout(self.proj_attn(x_attn))

    def _shrinkage(self, x: torch.Tensor) -> torch.Tensor:
        """Soft-threshold shrinkage to prune low-magnitude frequencies.

        Applies ``x - arctan(x)`` then zeros values with absolute
        magnitude ≤ 0.01.

        Parameters
        ----------
        x : torch.Tensor
            Frequency-domain tensor.

        Returns
        -------
        torch.Tensor
            Sparse frequency tensor of the same shape.
        """
        x_out = x - torch.arctan(x)
        x_gate = torch.where(torch.abs(x_out) > 0.01, 1, 0)
        return x_out * x_gate

    def forward(self, x_p: torch.Tensor) -> torch.Tensor:
        """Apply FFT, efficient attention, shrinkage, and inverse FFT.

        Parameters
        ----------
        x_p : torch.Tensor
            Shape ``(B, L, emb_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, L, emb_dim)``.
        """
        # 2-D real FFT — conjugate symmetry halves the spectrum
        x_fft = torch.real(
            torch.fft.rfftn(x_p.float(), dim=(-2, -1), norm="forward")
        )

        x_fft, batch_size, length = self._create_heads(x_fft)
        x_fft = x_fft + self._attention(x_fft, batch_size, length)
        x_fft = self.norm(x_fft)
        x_fft = self.out_proj(x_fft) + x_fft
        x_fft = self._restore_dim(x_fft, batch_size, length)
        x_fft = self.merge_proj(x_fft)
        x_fft = self._shrinkage(x_fft)

        # Inverse FFT
        x_ifft = torch.real(
            torch.fft.irfftn(x_fft.float(), dim=(-2, -1), norm="forward")
        )
        # Scale tensor to be in range between 5 and -5
        v_min, v_max = x_ifft.min(), x_ifft.max()
        denom = (v_max - v_min).clamp(min=1e-8)
        x_ifft_scale = (x_ifft - v_min) / denom * 10 - 5

        return self.process_scale(x_ifft_scale)
