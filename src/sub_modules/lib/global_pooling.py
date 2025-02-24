import torch
import torch.nn as nn

class GlobalExpectationPooling(nn.Module):
    """
    Expectation Pooling:
    https://academic.oup.com/bioinformatics/article/36/5/1405/5584233?login=false
    """

    def __init__(self):
        super(GlobalExpectationPooling, self).__init__()

        self.m = nn.Parameter(torch.tensor([[1.0]]))

    def forward(self, inputs):
        now = torch.transpose(inputs, -1, -2)

        now_diff = now - torch.mean(now, dim=-1, keepdim=True)

        now_diff_m = now_diff * self.m

        sgn_now = torch.sign(now_diff_m)

        diff_2 = sgn_now * torch.exp(now_diff_m) + torch.exp(now_diff_m)

        diff_now = diff_2 / 2

        prob = diff_now / torch.sum(diff_now, dim=-1, keepdim=True)

        expectation = torch.sum(now * prob, dim=-1)

        return expectation