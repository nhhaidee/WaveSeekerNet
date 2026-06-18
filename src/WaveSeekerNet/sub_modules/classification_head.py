"""
Classification head for WaveSeekerNet.

Supports both standard linear layers and KAN (Kolmogorov–Arnold Network)
layers as the logit projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from WaveSeekerNet.sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear
from WaveSeekerNet.sub_modules.lib.kan_layer import KANLinear


class ClassificationHead(nn.Module):
    """Multi-layer classification head.

    Produces class logits (and optionally softmax probabilities) from a
    pooled sequence embedding.

    Parameters
    ----------
    emb_dim : int
        Input embedding dimension.
    n_out : int
        Number of output classes.
    activation : type
        Activation function class (instantiated internally).
    return_probs : bool
        If ``True``, also return softmax class probabilities.
    use_kan : bool
        Use KAN layers instead of standard linear layers.
    ffn_dropout : float
        Dropout probability before the final projection.
    final_hidden_size : int
        Size of the penultimate hidden layer.
    grid_size : int
        Grid size for KAN spline approximation. Default ``8``.
    input_dropout : bool
        If ``True``, apply dropout to the input embedding. Default ``False``.
    """

    def __init__(
        self,
        emb_dim: int,
        n_out: int,
        activation: type,
        return_probs: bool,
        use_kan: bool,
        ffn_dropout: float,
        final_hidden_size: int,
        grid_size: int = 8,
        input_dropout: bool = False,
    ) -> None:
        super().__init__()

        self.return_probs = return_probs
        self.input_dropout = input_dropout

        if self.input_dropout:
            self.dropout = nn.Dropout(0.25)

        if not use_kan:
            self.logits = nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2),
                activation(),
                nn.Linear(emb_dim * 2, emb_dim),
                activation(),
                nn.Linear(emb_dim, final_hidden_size),
                activation(),
                nn.Dropout(0.125),
                NoisyFactorizedLinear(final_hidden_size, n_out),
            )
        else:
            self.logits = nn.Sequential(
                KANLinear(
                    emb_dim,
                    emb_dim * 2,
                    grid_size=grid_size,
                    spline_order=3,
                    base_activation=activation,
                ),
                KANLinear(
                    emb_dim * 2,
                    emb_dim,
                    grid_size=grid_size,
                    spline_order=3,
                    base_activation=activation,
                ),
                KANLinear(
                    emb_dim,
                    final_hidden_size,
                    spline_order=3,
                    base_activation=activation,
                ),
                nn.Dropout(0.125),
                NoisyFactorizedLinear(final_hidden_size, n_out),
            )

        if self.return_probs:
            self.sm_out = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Compute logits (and optionally softmax probabilities).

        Parameters
        ----------
        x : torch.Tensor
            Pooled embedding of shape ``(B, emb_dim)``.

        Returns
        -------
        When ``return_probs=True``:
            ``(logits, probs)`` — both shape ``(B, n_out)``.
        When ``return_probs=False``:
            ``logits`` — shape ``(B, n_out)``.
        """
        if self.input_dropout:
            x = self.dropout(x)

        x_logit = self.logits(x)

        if self.return_probs:
            return x_logit, self.sm_out(x_logit)

        return x_logit