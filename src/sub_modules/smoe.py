import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear


class WaveExpert(nn.Module):
    def __init__(self, in_embed, ffn_dropout, activation):
        super(WaveExpert, self).__init__()

        self.linear_in = nn.Linear(in_embed, in_embed * 2, bias=False)

        self.activation_1 = activation()

        self.dropout = nn.Dropout1d(ffn_dropout)

        self.bias = nn.Parameter(torch.ones(in_embed, requires_grad=True))

        self.linear_out = NoisyFactorizedLinear(in_embed, in_embed, bias=True)

    def forward(self, inputs):
        x_1, x_2 = self.linear_in(inputs).chunk(2, dim=-1)

        x_1 = self.activation_1(x_1)

        x = x_1 * x_2
        x = x * self.bias
        x = self.linear_out(x)
        x = self.dropout(x)

        return x


"""
Sparsely Gated MoE: https://arxiv.org/pdf/1701.06538
Stable MoE (Used for Expert Layer and Z-loss): https://arxiv.org/pdf/2202.08906
Multihead MoE: https://arxiv.org/pdf/2404.15045

Code adapted from: 
https://huggingface.co/blog/AviSoori1x/seemoe
https://github.com/kyegomez/MHMoE/blob/main/mh_moe/main.py
"""


class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()

        self.top_k = top_k

        # layer for router logits
        self.noisy_logits = NoisyFactorizedLinear(n_embed, num_experts)
        self.dropout = nn.Dropout(0.125)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        # Noise logits
        noisy_logits = self.noisy_logits(mh_output)

        B, T, E = noisy_logits.size()
        noisy_logits = self.dropout(noisy_logits.view(B * T, E))
        noisy_logits = noisy_logits.view(B, T, E)

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)  # These are the gates
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices, self.z_loss(sparse_logits, router_output)

    def z_loss(self, s_logits, probs):
        # Batch size (B), Num Tokens (T), Num Experts (E)
        B, T, E = s_logits.size()

        logits = s_logits.view(B * T, E)

        p = probs.view(B * T, E)

        # Calculate Z-loss
        z_loss = torch.mean(torch.logsumexp(logits.view(B * T, E), -1) ** 2)

        return z_loss


class SMoE(nn.Module):
    def __init__(self, dim, num_experts, top_k, ffn_dropout, activation, n_patches):
        super(SMoE, self).__init__()

        self.top_k = top_k

        self.router = NoisyTopkRouter(dim, num_experts, top_k)

        self.experts = nn.ModuleList([
            WaveExpert(
                in_embed=dim,
                ffn_dropout=ffn_dropout,
                activation=activation
            )
            for _ in range(num_experts)]
        )

    def forward(self, x):
        gating_output, indices, z_loss = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[
                    flat_mask, i
                ].unsqueeze(1)

                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output, z_loss


class SparseMoE(nn.Module):
    def __init__(
            self,
            dim: int,
            top_k: int,
            activation,
            n_patches: int,
            ffn_dropout: float,
            num_experts: int = 8,
            num_layers: int = 1,
    ):
        super(SparseMoE, self).__init__()

        self.dim = dim
        self.heads = dim // 32
        self.top_k = top_k
        self.ffn_dropout = ffn_dropout
        self.num_experts = num_experts
        self.num_layers = num_layers

        # Create a module list for multi-head layers and merge layers
        self.multi_head_layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_layers)]
        )

        self.expert_layers = nn.ModuleList(
            [SMoE(
                32,
                num_experts,
                top_k,
                ffn_dropout,
                activation,
                n_patches
            ) for _ in range(num_layers)]
        )

        self.merge_layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_layers)]
        )

        # Initialize parameters
        for i in range(num_layers):
            nn.init.xavier_uniform_(
                self.multi_head_layers[i].weight,
                gain=1 / math.sqrt(2),
            )
            nn.init.xavier_uniform_(self.merge_layers[i].weight)
            nn.init.constant_(self.merge_layers[i].bias, 0)

    def forward(self, x):
        """
        Forward pass of the MHMoE module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        # Loop through each layer
        loss = 0.0

        for i in range(self.num_layers):
            x, z_loss = self.process_layer(x, i)
            loss += z_loss

        loss /= self.num_layers

        return x, loss

    def process_layer(self, x, layer_index):
        """
        Process a single layer of the MHMoE module.

        Args:
            x (torch.Tensor): The input tensor.
            layer_index (int): The index of the layer.

        Returns:
            torch.Tensor: The output tensor.

        """
        batch_size, length, _ = x.size()

        # Processed by multi-head layer
        x = self.multi_head_layers[layer_index](x)

        # Correcting the reshaping step
        # We need to ensure x is reshaped to (batch_size, heads, length, dim/heads)
        x = x.view(
            batch_size,
            length,
            self.heads,
            32,
        )

        x = x.permute(
            0, 2, 1, 3
        ).contiguous()  # this rearranges to (batch_size, heads, length, dim/heads)

        x = x.view(
            batch_size * self.heads,
            length,
            32,
        )

        x, z_loss = self.expert_layers[layer_index](x)

        # Reshape back to original form after processing
        x = x.view(
            batch_size,
            self.heads,
            length,
            32,
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, length, self.dim)

        # Output processed by merge layer
        x = self.merge_layers[layer_index](x)

        return x, z_loss
