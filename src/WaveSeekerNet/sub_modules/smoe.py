"""
Sparse and dense feed-forward expert modules for WaveSeekerNet.

Implements the Sparsely Gated Mixture-of-Experts (SMoE) routing
mechanism used inside :class:`~WaveSeekerNet.model.WaveSeekerBlock`.

References
----------
Sparsely Gated MoE: https://arxiv.org/pdf/1701.06538
Stable MoE / Z-loss: https://arxiv.org/pdf/2202.08906
Multihead MoE: https://arxiv.org/pdf/2404.15045

Code adapted from:
https://huggingface.co/blog/AviSoori1x/seemoe
https://github.com/kyegomez/MHMoE/blob/main/mh_moe/main.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from WaveSeekerNet.sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear


class WaveExpert(nn.Module):
    """Single gated feed-forward expert (used when SMoE is disabled).

    Parameters
    ----------
    in_embed : int
        Input and output embedding dimension.
    ffn_dropout : float
        Dropout probability applied after the output projection.
    activation : type
        Activation function class (instantiated internally).
    """

    def __init__(self, in_embed: int, ffn_dropout: float, activation: type) -> None:
        super().__init__()

        self.linear_in = nn.Linear(in_embed, in_embed * 2, bias=False)
        self.activation_1 = activation()
        self.dropout = nn.Dropout1d(ffn_dropout)
        self.bias = nn.Parameter(torch.ones(in_embed))
        self.linear_out = NoisyFactorizedLinear(in_embed, in_embed, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply gated linear unit followed by a noisy linear projection.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(B, L, in_embed)``.

        Returns
        -------
        torch.Tensor
            Same shape as *inputs*.
        """
        x_1, x_2 = self.linear_in(inputs).chunk(2, dim=-1)
        x_1 = self.activation_1(x_1)
        x = x_1 * x_2 * self.bias
        x = self.linear_out(x)
        return self.dropout(x)


class NoisyTopkRouter(nn.Module):
    """Top-k sparse router with noisy logits and Z-loss regularisation.

    Parameters
    ----------
    n_embed : int
        Token embedding dimension.
    num_experts : int
        Total number of expert modules.
    top_k : int
        Number of experts selected per token.
    """

    def __init__(self, n_embed: int, num_experts: int, top_k: int) -> None:
        super().__init__()

        self.top_k = top_k
        self.noisy_logits = NoisyFactorizedLinear(n_embed, num_experts)
        self.dropout = nn.Dropout(0.125)

    def forward(
        self, mh_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute sparse routing weights and auxiliary Z-loss.

        Parameters
        ----------
        mh_output : torch.Tensor
            Shape ``(B, T, n_embed)``.

        Returns
        -------
        router_output : torch.Tensor
            Sparse softmax weights, shape ``(B, T, num_experts)``.
        indices : torch.Tensor
            Top-k expert indices, shape ``(B, T, top_k)``.
        z_loss : torch.Tensor
            Scalar Z-loss regularisation term.
        """
        noisy_logits = self.noisy_logits(mh_output)
        B, T, E = noisy_logits.size()
        noisy_logits = self.dropout(noisy_logits.view(B * T, E)).view(B, T, E)

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices, self._z_loss(sparse_logits)

    def _z_loss(self, sparse_logits: torch.Tensor) -> torch.Tensor:
        """Compute the Z-loss from Stable MoE.

        Encourages the router logits to stay small, reducing routing
        instability.

        Parameters
        ----------
        sparse_logits : torch.Tensor
            Shape ``(B, T, num_experts)``.

        Returns
        -------
        torch.Tensor
            Scalar Z-loss value.
        """
        B, T, E = sparse_logits.size()
        logits_flat = sparse_logits.view(B * T, E)
        return torch.mean(torch.logsumexp(logits_flat, dim=-1) ** 2)


class SMoE(nn.Module):
    """Token-level Sparse Mixture-of-Experts layer.

    Parameters
    ----------
    dim : int
        Token embedding dimension.
    num_experts : int
        Total number of expert modules.
    top_k : int
        Experts selected per token.
    ffn_dropout : float
        Dropout inside each expert.
    activation : type
        Activation function class.
    n_patches : int
        Sequence length (unused in forward but kept for API consistency).
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        ffn_dropout: float,
        activation: type,
        n_patches: int,
    ) -> None:
        super().__init__()

        self.top_k = top_k
        self.router = NoisyTopkRouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList(
            [
                WaveExpert(in_embed=dim, ffn_dropout=ffn_dropout, activation=activation)
                for _ in range(num_experts)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to top-k experts and aggregate their outputs.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, dim)``.

        Returns
        -------
        output : torch.Tensor
            Shape ``(B, T, dim)``.
        z_loss : torch.Tensor
            Scalar routing Z-loss.
        """
        gating_output, indices, z_loss = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_output = expert(flat_x[flat_mask])
                gating_scores = flat_gating[flat_mask, i].unsqueeze(1)
                final_output[expert_mask] += (expert_output * gating_scores).squeeze(1)

        return final_output, z_loss


class SparseMoE(nn.Module):
    """Multi-head Sparse Mixture-of-Experts block.

    Splits the embedding into *heads* of size 32, routes each head
    independently through a shared :class:`SMoE` layer, then merges the
    results back.

    Parameters
    ----------
    dim : int
        Total token embedding dimension (must be divisible by 32).
    top_k : int
        Experts selected per token per head.
    activation : type
        Activation function class.
    n_patches : int
        Sequence length.
    ffn_dropout : float
        Dropout inside each expert.
    num_experts : int
        Total number of experts. Default ``8``.
    num_layers : int
        Number of stacked MoE layers. Default ``1``.
    """

    def __init__(
        self,
        dim: int,
        top_k: int,
        activation: type,
        n_patches: int,
        ffn_dropout: float,
        num_experts: int = 8,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.heads = dim // 32
        self.top_k = top_k
        self.ffn_dropout = ffn_dropout
        self.num_experts = num_experts
        self.num_layers = num_layers

        self.multi_head_layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_layers)]
        )
        self.expert_layers = nn.ModuleList(
            [
                SMoE(32, num_experts, top_k, ffn_dropout, activation, n_patches)
                for _ in range(num_layers)
            ]
        )
        self.merge_layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_layers)]
        )

        for i in range(num_layers):
            nn.init.xavier_uniform_(
                self.multi_head_layers[i].weight, gain=1 / math.sqrt(2)
            )
            nn.init.xavier_uniform_(self.merge_layers[i].weight)
            nn.init.constant_(self.merge_layers[i].bias, 0)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all stacked MoE layers.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, dim)``.

        Returns
        -------
        output : torch.Tensor
            Shape ``(B, T, dim)``.
        loss : torch.Tensor
            Mean Z-loss across all layers.
        """
        total_loss = torch.tensor(0.0, device=x.device)
        for i in range(self.num_layers):
            x, z_loss = self._process_layer(x, i)
            total_loss = total_loss + z_loss
        return x, total_loss / self.num_layers

    def _process_layer(
        self, x: torch.Tensor, layer_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input through a single multi-head MoE layer.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, dim)``.
        layer_index : int
            Index of the layer to process.

        Returns
        -------
        output : torch.Tensor
            Shape ``(B, T, dim)``.
        z_loss : torch.Tensor
            Scalar Z-loss for this layer.
        """
        batch_size, length, _ = x.size()

        x = self.multi_head_layers[layer_index](x)
        x = x.view(batch_size, length, self.heads, 32)
        x = x.permute(0, 2, 1, 3).contiguous()       # (B, H, L, 32)
        x = x.view(batch_size * self.heads, length, 32)

        x, z_loss = self.expert_layers[layer_index](x)

        x = x.view(batch_size, self.heads, length, 32)
        x = x.permute(0, 2, 1, 3).contiguous()       # (B, L, H, 32)
        x = x.view(batch_size, length, self.dim)

        return self.merge_layers[layer_index](x), z_loss
