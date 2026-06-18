"""
Custom activation functions for WaveSeekerNet.

References
----------
Mish: https://arxiv.org/abs/1908.08681
ErMish: combines the error function with Mish-style gating.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ErMish(nn.Module):
    """Error-function Mish activation (ErMish).

    A learnable activation that applies a scaled error-function gate:

    .. math::

        \\text{ErMish}(x) = 1.5 \\cdot x \\cdot
        \\text{erf}(\\alpha + 0.7071 \\cdot \\tanh(\\text{softplus}(x)))

    where ``alpha`` is a trainable scalar initialised to ``0``.

    If ``use_alpha=False``, the learnable shift is omitted.

    Parameters
    ----------
    use_alpha : bool
        Include a learnable shift parameter ``alpha``. Default ``True``.
    """

    def __init__(self, use_alpha: bool = True) -> None:
        super().__init__()

        self.use_alpha = use_alpha
        if self.use_alpha:
            self.alpha = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the ErMish activation.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            Activated tensor, same shape as *inputs*.
        """
        gate = 0.7071 * nn.functional.softplus(inputs).tanh()
        if self.use_alpha:
            gate = self.alpha + gate
        return 1.5 * inputs * torch.erf(gate)