"""
StarNet-inspired element-wise multiplication layer.

Combines an MLP-Mixer patch/channel mixing step with a StarNet-style
element-wise product gate.

References
----------
StarNet: https://arxiv.org/pdf/2403.19967
gMLP:    https://arxiv.org/pdf/2105.01601
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import RMSNorm

from WaveSeekerNet.sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear


class StarLayer(nn.Module):
    """Patch-mixing + channel-mixing + StarNet element-wise product gate.

    First applies an MLP-Mixer-style mixing step (spatial then channel),
    then projects down via a StarNet gating mechanism.

    Parameters
    ----------
    emb_in : int
        Input embedding dimension.
    emb_out : int
        Output embedding dimension.
    n_patches : int
        Sequence length (number of tokens/patches).
    activation : type
        Activation function class (instantiated internally).
    ffn_dropout : float
        Dropout probability inside the mixing layers. Default ``0.1``.
    dropout : str
        Either ``"1d"`` (nn.Dropout1d) or ``"2d"`` (nn.Dropout2d).
        Default ``"1d"``.
    """

    def __init__(
        self,
        emb_in: int,
        emb_out: int,
        n_patches: int,
        activation: type,
        ffn_dropout: float = 0.1,
        dropout: str = "1d",
    ) -> None:
        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_out

        dropout_cls = nn.Dropout2d if dropout == "2d" else nn.Dropout1d

        # MLP Mixer — spatial (patch) mixing
        self.norm_mix_in = RMSNorm(emb_in, eps=1e-8)
        self.patches_mixer = nn.Sequential(
            nn.Linear(n_patches, n_patches),
            activation(),
            nn.Linear(n_patches, n_patches),
            dropout_cls(ffn_dropout),
        )

        # MLP Mixer — channel mixing
        self.norm_mix_out = RMSNorm(emb_in, eps=1e-8)
        self.hidden_mixer = nn.Sequential(
            nn.Linear(emb_in, emb_in),
            activation(),
            nn.Linear(emb_in, emb_in),
            dropout_cls(ffn_dropout),
        )

        # StarNet gating
        self.norm_star = RMSNorm(emb_in, eps=1e-8)
        self.project_down = nn.Linear(emb_in, emb_out)
        self.W_in = NoisyFactorizedLinear(emb_out, emb_out * 2)
        self.act = activation()
        self.W_out = nn.Linear(emb_out, emb_out)

    def _mlp_mixer(
        self, x: torch.Tensor, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Apply patch-then-channel MLP mixing with skip connection.

        Parameters
        ----------
        x : torch.Tensor
            Normalised inputs, shape ``(B, n_patches, emb_in)``.
        inputs : torch.Tensor
            Original (un-normalised) inputs for the skip connection.
        """
        # Patch mixing (operate on transposed sequence dimension)
        x = torch.transpose(self.patches_mixer(torch.transpose(x, -1, -2)), -1, -2)
        x = x + inputs
        x = self.norm_mix_out(x)

        # Channel mixing
        return self.hidden_mixer(x) + x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through StarLayer.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(B, n_patches, emb_in)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, n_patches, emb_out)``.
        """
        x = self.norm_mix_in(inputs)
        x = self._mlp_mixer(x, inputs)

        # StarNet element-wise product gate
        x = self.norm_star(x)
        x = self.project_down(x)
        w_1, w_2 = self.W_in(x).chunk(2, dim=-1)
        w = self.act(w_1) * w_2

        return self.W_out(w)