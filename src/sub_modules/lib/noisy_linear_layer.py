import torch
import math
import torch.nn as nn
import torch.nn.functional as F

"""
Code adapted from:
https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter07/lib/dqn_model.py
"""


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """

    def __init__(self, in_features, out_features, sigma_zero=0.5, bias=True):
        super(NoisyFactorizedLinear, self).__init__(
            in_features, out_features, bias=bias
        )
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init)
        )
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

    def forward(self, input):
        if self.training:
            self.epsilon_input.normal_()
            self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()

        noise_v = torch.mul(eps_in, eps_out)

        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)