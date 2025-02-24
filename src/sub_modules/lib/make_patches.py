import torch
import torch.nn as nn
import torch.nn.functional as F
from sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear
from sub_modules.lib.pos_encoding import PositionalEncoding


class MakePatches(nn.Module):
    def __init__(
        self, patch_width, patch_height, emb_dim, n_channel=1, patch_mode="patch"
    ):
        super().__init__()

        if patch_mode == "patch" or patch_mode == "compress":
            stride = (patch_width, patch_height)

        else:
            stride = 1

        self.conv = nn.Unfold((patch_width, patch_height), stride=stride)
        self.n_channel = n_channel
        self.pos_encoding = PositionalEncoding(emb_dim)
        #self.drop_out = nn.Dropout1d(0.5)

        # Gated Layer Activation
        self.W1 = NoisyFactorizedLinear(patch_width*patch_height, 128, sigma_zero=0.1)
        self.W2 = NoisyFactorizedLinear(patch_width*patch_height, 128, sigma_zero=0.1)
        self.W3 = nn.Linear(128, emb_dim)

    def forward(self, inputs):

        x = self.conv(inputs)
        x = torch.transpose(x, -1, 1) # (N, C, 128)

        # Gated activation
        x_gate = F.mish(self.W1(x)) # (N, C, 128) -> (N, C, 64)
        x_inner = self.W2(x) # (N, C, 128) -> (N, C, 64)
        x = x_gate * x_inner # (N, C, 64)
        x = self.W3(x) # (N, C, D) -> (N, C, D)

        x = self.pos_encoding(x)
        #x = self.drop_out(x)

        return x

