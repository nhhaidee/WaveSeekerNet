"""
Global Expectation Pooling for WaveSeekerNet.

References
----------
Expectation Pooling:
https://academic.oup.com/bioinformatics/article/36/5/1405/5584233
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GlobalExpectationPooling(nn.Module):
    """Global pooling via a learnable soft-attention expectation.

    Computes a weighted expectation over the sequence dimension using
    a learnable temperature parameter ``m``.

    The pooling is defined as:

    .. math::

        p_t = \\frac{\\exp(m \\cdot (x_t - \\bar{x}))}{
                      \\sum_{t'} \\exp(m \\cdot (x_{t'} - \\bar{x}))}

        \\text{output} = \\sum_t p_t \\cdot x_t

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        super().__init__()
        self.m = nn.Parameter(torch.tensor([[1.0]]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pool the sequence dimension to a single vector.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, D)`` — the expectation-pooled embedding.
        """
        now = torch.transpose(inputs, -1, -2)                      # (B, D, T)
        now_diff = now - torch.mean(now, dim=-1, keepdim=True)    # centre
        now_diff_m = now_diff * self.m

        sgn_now = torch.sign(now_diff_m)
        diff_2 = sgn_now * torch.exp(now_diff_m) + torch.exp(now_diff_m)
        prob = diff_2 / 2
        prob = prob / torch.sum(prob, dim=-1, keepdim=True)       # normalise

        return torch.sum(now * prob, dim=-1)                       # (B, D)