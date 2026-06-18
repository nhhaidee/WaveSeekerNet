"""
Noisy factorized linear layer for WaveSeekerNet.

Implements NoisyNet with factorized Gaussian noise, used throughout the
model to encourage exploration and act as a regulariser.

References
----------
Noisy Networks for Exploration: https://arxiv.org/abs/1706.10295
Code adapted from:
https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter07/lib/dqn_model.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Apply the signed square-root: ``sign(x) * sqrt(|x|)``."""
    return torch.sign(x) * torch.sqrt(torch.abs(x))


class NoisyFactorizedLinear(nn.Linear):
    """Linear layer with factorized Gaussian noise (NoisyNet).

    Adds learnable noise to weights and biases during training to provide
    implicit exploration / regularisation.  The noise is factorized across
    input and output dimensions, reducing parameters from
    ``O(in * out)`` to ``O(in + out)``.

    During *evaluation* mode the noise terms are zeroed, so the layer
    behaves identically to a standard :class:`~torch.nn.Linear`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    sigma_zero : float
        Initial standard deviation of the noise parameters, scaled
        internally by ``1 / sqrt(in_features)``. Default ``0.5``.
    bias : bool
        If ``True``, adds a learnable bias. Default ``True``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_zero: float = 0.5,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)

        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init)
        )
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))

        if bias:
            self.sigma_bias = nn.Parameter(
                torch.full((out_features,), sigma_init)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional factorized noise injection.

        Parameters
        ----------
        input : torch.Tensor
            Shape ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., out_features)``.
        """
        if self.training:
            self.epsilon_input.normal_()
            self.epsilon_output.normal_()

        eps_in = _signed_sqrt(self.epsilon_input.data)
        eps_out = _signed_sqrt(self.epsilon_output.data)
        noise_v = torch.mul(eps_in, eps_out)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()

        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)