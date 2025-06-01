import torch
import torch.nn as nn
from torch.nn import RMSNorm
from sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear
import math

"""
"Attention"-Like Mechanisms for "Self-Attention"

FNet: https://arxiv.org/abs/2105.03824
FFT Processing: https://openreview.net/pdf?id=EXHG-A3jlM
Inspiration for shrinkage comes from above.

gMLP: https://arxiv.org/abs/2105.08050
Efficient Attention: https://arxiv.org/abs/1812.01243
"""


class FNETHead(nn.Module):
    def __init__(self, emb_dim, activation):
        super(FNETHead, self).__init__()

        self.heads = emb_dim // 32
        self.dim = emb_dim

        self.d_emb = (emb_dim // 2) + 1

        self.multi_head_proj = nn.Linear(self.d_emb, emb_dim)

        # For self-attention block on frequency components
        self.get_qkv = nn.Linear(32, 32 * 3)

        self.proj_attn = nn.Linear(32, 32)

        self.attn_dropout = nn.Dropout1d(0.125)

        self.norm = RMSNorm(32, eps=1e-8)

        self.out_proj = NoisyFactorizedLinear(32, 32)

        self.merge_proj = nn.Linear(emb_dim, self.d_emb)

        # Process scaled output
        self.process_scale = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            activation(),  # <- This was nn.Mish() before
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Dropout1d(0.125)
        )

        nn.init.xavier_uniform_(
            self.multi_head_proj.weight,
            gain=1 / math.sqrt(2),
        )
        nn.init.xavier_uniform_(self.merge_proj.weight)

        nn.init.constant_(self.multi_head_proj.bias, 0)
        nn.init.constant_(self.merge_proj.bias, 0)

    def create_heads(self, x_in):
        batch_size, length, _ = x_in.size()
        self.batch_size = batch_size
        self.length = length

        # Processed by multi-head layer
        x = self.multi_head_proj(x_in)

        # Correcting the reshaping step
        # We need to ensure x is reshaped to (batch_size, heads, length, 32)
        x = x.view(
            batch_size,
            length,
            self.heads,
            32,
        )

        x = x.permute(
            0, 2, 1, 3
        ).contiguous()  # this rearranges to (batch_size, heads, length, 32)

        x = x.view(
            batch_size * self.heads,
            length,
            32,
        )

        return x

    def restore_dim(self, x, permute=True):
        # Reshape back to original form after processing
        x = x.view(
            self.batch_size,
            self.heads,
            self.length,
            32,
        )

        if permute:
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(self.batch_size, self.length, 32 * self.heads)

        return x

    def attention(self, x):
        x_qh, x_kh, x_vh = self.get_qkv(x).chunk(3, dim=-1)  # (B*H, L, Dh) for Qs, Ks, and Vs

        # Efficent attention: (B*H, L, Dh) -> (B, H, L, Dh)
        x_qh = self.restore_dim(x_qh, False)
        x_kh = self.restore_dim(x_kh, False)
        x_vh = self.restore_dim(x_vh, False)

        # Softmax Keys and Queries
        x_kh = nn.functional.softmax(x_kh, dim=-2)
        x_qh = nn.functional.softmax(x_qh, dim=-1)

        # Get context vectors
        ch = x_kh.transpose(-1, -2) @ x_vh  # (B, H, Dh, Dh)

        # Get attention scores
        x_attn = x_qh @ ch.transpose(-1, 2)  # (B, H, L, Dh)

        x_attn = x_attn.view(
            self.batch_size * self.heads,
            self.length,
            32,
        )

        return self.attn_dropout(self.proj_attn(x_attn))

    def shrinkage(self, x):
        # Shrink frequencies
        x_out = x - torch.arctan(x)

        # Threshold to find all those with value |0.01| or smaller
        x_gate = torch.abs(x_out) - 0.01
        x_gate = torch.where(x_gate > 0, 1, 0)

        return x_out * x_gate

    def forward(self, x_p):
        # Conjugate symmetry property of 2D DFT means half the values are redundant.
        x_fft = torch.real(torch.fft.rfftn(x_p.float(), dim=(-2, -1), norm="forward"))

        x_fft = self.create_heads(x_fft)

        # Process Using Self-Attention
        x_fft = x_fft + self.attention(x_fft)

#        x_fft = x_fft.view(
#            self.batch_size * self.heads,
#            self.length,
#            32,
#        )

        x_fft = self.norm(x_fft)
        x_fft = self.out_proj(x_fft) + x_fft
        x_fft = self.restore_dim(x_fft)
        x_fft = self.merge_proj(x_fft)

        # Thresholding - Encourages sparsity/regularization
        x_fft = self.shrinkage(x_fft)

        # Inverse transform
        x_ifft = torch.fft.irfftn(x_fft.float(), dim=(-2, -1), norm="forward")
        x_ifft = torch.real(x_ifft)

        # Scale tensor to be in range between 5 and -5
        v_min, v_max = x_ifft.min(), x_ifft.max()
        x_ifft_scale = (x_ifft - v_min) / (v_max - v_min) * 10 - 5

        return self.process_scale(x_ifft_scale)
