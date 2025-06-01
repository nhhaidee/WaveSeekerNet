import torch
import torch.nn as nn
from torch.nn import RMSNorm
from sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, n_patches, use_attn):
        super().__init__()
        self.norm = RMSNorm(d_ffn, eps=1e-8)
        self.spatial_proj = NoisyFactorizedLinear(n_patches, n_patches)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

        self.use_attn = use_attn

        # For making Q, K, V for conducting efficient attention
        if use_attn:
            self.get_qkv = NoisyFactorizedLinear(d_ffn * 2, 64 * 3)
            self.get_qkv.apply(self.weight_init)
            self.proj_out = nn.Linear(64, d_ffn)
            self.attn_dropout = nn.Dropout1d(0.1)

    def weight_init(self, w):
        if isinstance(w, NoisyFactorizedLinear):
            nn.init.kaiming_uniform_(w.weight)
            nn.init.zeros_(w.bias)

    def attention(self, x):

        x_q, x_k, x_v = self.get_qkv(x).chunk(3, dim=-1)  # (B, L, D) for Qs, Ks, and Vs

        # Efficent attention
        x_qh = x_q.view(x_q.shape[0], x_q.shape[1], 2, x_q.shape[2] // 2).transpose(
            1, 2
        )  # (B, L, D) -> (B, H, L, Dh)
        x_kh = x_k.view(x_k.shape[0], x_k.shape[1], 2, x_k.shape[2] // 2).transpose(
            1, 2
        )  # (B, L, D) -> (B, H, L, Dh)
        x_vh = x_v.view(x_v.shape[0], x_v.shape[1], 2, x_v.shape[2] // 2).transpose(
            1, 2
        )  # (B, L, D) -> (B, H, L, Dh)

        # Softmax Keys and Queries
        x_kh = nn.functional.softmax(x_kh, dim=-2)
        x_qh = nn.functional.softmax(x_qh, dim=-1)

        # Get context vectors
        ch = x_kh.transpose(-1, -2) @ x_vh  # (B, H, Dh, Dh)

        # Get attention scores
        x_attn = x_qh @ ch.transpose(-1, 2)  # (B, H, L, Dh)

        x_attn = x_attn.transpose(1, 2).flatten(
            -2, -1
        )  # (B, H, L, Dh) -> (B, L, H, Dh) - > (B, L, D)

        return self.attn_dropout(self.proj_out(x_attn))

    def forward(self, x):

        u, v = x.chunk(2, dim=-1)

        v = self.norm(v)

        v = torch.transpose(self.spatial_proj(torch.transpose(v, -1, -2)), -1, -2)

        if self.use_attn:
            v = v + self.attention(x)

        return u * v


class gMLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim,
            ffn_dropout,
            n_patches,
            activation=nn.Mish,
            dropout_type="patch",
            use_attn=True,
    ):
        super(gMLPBlock, self).__init__()

        # Channel and spatial projections
        self.channel_proj_1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            activation(),
            nn.Dropout1d(ffn_dropout)
        )

        self.spatial_gate = SpatialGatingUnit(embedding_dim, n_patches, use_attn)

        self.channel_proj_2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            activation(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout1d(ffn_dropout)
        )

    def forward(self, inputs):
        x = self.channel_proj_1(inputs)
        x = self.spatial_gate(x)
        x = self.channel_proj_2(x)

        return x