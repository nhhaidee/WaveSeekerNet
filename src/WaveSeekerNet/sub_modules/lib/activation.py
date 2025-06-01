import torch
import torch.nn as nn

"""
Activation Functions

Gaussian Error Function Activation: Need to get paper(s) which
inspired the idea below.

Mish: https://arxiv.org/abs/1908.08681
"""


class ErMish(nn.Module):
    def __init__(self, use_alpha=True):
        super(ErMish, self).__init__()

        self.use_alpha = use_alpha

        if self.use_alpha:
            self.alpha = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        if self.use_alpha:
            return (
                    1.5
                    * inputs
                    * torch.erf(self.alpha + 0.7071 * nn.functional.softplus(inputs).tanh())
            )

        else:
            return (
                    1.5 * inputs * torch.erf(0.7071 * nn.functional.softplus(inputs).tanh())
            )