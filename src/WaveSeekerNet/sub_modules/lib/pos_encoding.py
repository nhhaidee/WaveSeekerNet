"""
Sinusoidal positional encoding with learnable position-sensitive
projection (LSPE) for WaveSeekerNet.

References
----------
Positional Encoding:
https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
LSPE: https://aclanthology.org/2022.findings-aacl.42.pdf
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from WaveSeekerNet.sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with learnable LSPE refinement.

    Adds fixed sinusoidal positional encodings to the input embeddings,
    optionally refined via a learnable two-layer projection (LSPE).

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_length : int
        Maximum supported sequence length. Default ``5000``.
    """

    def __init__(self, d_model: int, max_length: int = 5000) -> None:
        super().__init__()

        pe = torch.zeros(max_length, d_model)
        k = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)
        pe = pe.unsqueeze(0)                   # (1, max_length, d_model)

        self.register_buffer("pe", pe)

        self.lspe = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Mish(),
            NoisyFactorizedLinear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, d_model)``.

        Returns
        -------
        torch.Tensor
            Positionally-encoded embeddings, same shape as *x*.
        """
        pos = self.pe[:, : x.size(1)].requires_grad_(False)
        return x + self.lspe(pos)
