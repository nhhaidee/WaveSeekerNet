import torch
import torch.nn as nn
from torch.nn import RMSNorm
from sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear

"""
StarNet: https://arxiv.org/pdf/2403.19967
gMLP: https://arxiv.org/pdf/2105.01601
"""


class StarLayer(nn.Module):
    def __init__(self, emb_in, emb_out, n_patches, activation, ffn_dropout=0.1, dropout="1d"):
        super(StarLayer, self).__init__()

        self.emb_in = emb_in
        self.emb_out = emb_out

        # MLP Mixer
        self.norm_mix_in = RMSNorm(emb_in, eps=1e-8)

        if dropout == "2d":
            self.patches_mixer = nn.Sequential(
                nn.Linear(n_patches, n_patches),
                activation(),
                nn.Linear(n_patches, n_patches),
                nn.Dropout2d(ffn_dropout)
            )
        elif dropout == "1d":
            self.patches_mixer = nn.Sequential(
                nn.Linear(n_patches, n_patches),
                activation(),
                nn.Linear(n_patches, n_patches),
                nn.Dropout1d(ffn_dropout)
            )

        self.norm_mix_out = RMSNorm(emb_in, eps=1e-8)

        if dropout == "2d":
            self.hidden_mixer = nn.Sequential(
                nn.Linear(emb_in, emb_in),
                activation(),
                nn.Linear(emb_in, emb_in),
                nn.Dropout2d(ffn_dropout)
            )

        elif dropout == "1d":
            self.hidden_mixer = nn.Sequential(
                nn.Linear(emb_in, emb_in),
                activation(),
                nn.Linear(emb_in, emb_in),
                nn.Dropout1d(ffn_dropout)
            )

        # StarNet
        self.norm_star = RMSNorm(emb_in, eps=1e-8)

        self.project_down = nn.Linear(emb_in, emb_out)

        self.W_in = NoisyFactorizedLinear(emb_out, emb_out * 2)

        self.act = activation()

        self.W_out = nn.Linear(emb_out, emb_out)

    def mlp_mixer(self, x, inputs):

        # Patch Mixing
        x = torch.transpose(x, -2, -1)
        x = self.patches_mixer(x)
        x = torch.transpose(x, -2, -1)

        # Skip connection
        x = x + inputs
        x = self.norm_mix_out(x)

        # Channel Mixing
        x = self.hidden_mixer(x) + x

        return x

    def forward(self, inputs):

        # Normalize inputs
        x = self.norm_mix_in(inputs)

        # MLP Mixer Instead of Depth-wise Convolution
        x = self.mlp_mixer(x, inputs)

        # StarNet Start
        x = self.norm_star(x)
        x = self.project_down(x)

        # Element-wise Multiplication
        w_1, w_2 = self.W_in(x).chunk(2, dim=-1)
        w_1 = self.act(w_1)
        w = w_1 * w_2

        # Output
        return self.W_out(w)