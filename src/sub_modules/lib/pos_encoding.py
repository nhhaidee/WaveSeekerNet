import torch
import torch.nn as nn
import math
from sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 5000):
        """
        From: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
        Reference for LSPE: https://aclanthology.org/2022.findings-aacl.42.pdf
        """
        # inherit from Module
        super().__init__()

        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)

        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)

        # calc divisor for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)

        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)

        # add dimension
        pe = pe.unsqueeze(0)

        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

        self.lspe = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Mish(),
            NoisyFactorizedLinear(d_model, d_model)
        )

    def forward(self, x):
        # add positional encoding to the embeddings
        x = x + self.lspe(self.pe[:, : x.size(1)].requires_grad_(False))

        return x
