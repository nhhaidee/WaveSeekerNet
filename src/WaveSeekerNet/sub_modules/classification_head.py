import torch.nn as nn
from sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear
from sub_modules.lib.kan_layer import KANLinear


class ClassificationHead(nn.Module):
    def __init__(
            self,
            emb_dim,
            n_out,
            activation,
            return_probs,
            use_kan,
            ffn_dropout,
            final_hidden_size,
            grid_size=8,  # 32,
            input_dropout=False,
    ):
        super().__init__()

        self.return_probs = return_probs
        self.input_dropout = input_dropout

        if self.input_dropout:
            self.dropout = nn.Dropout(0.25)

        if not use_kan:
            self.logits = nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2),
                activation(),
                nn.Linear(emb_dim * 2, emb_dim),
                activation(),
                nn.Linear(emb_dim, final_hidden_size),
                activation(),
                nn.Dropout(0.125),
                NoisyFactorizedLinear(final_hidden_size, n_out),
            )

        else:
            self.logits = nn.Sequential(
                KANLinear(
                    emb_dim,
                    emb_dim * 2,
                    grid_size=grid_size,
                    spline_order=3,
                    base_activation=activation,
                ),
                KANLinear(
                    emb_dim * 2,
                    emb_dim,
                    grid_size=grid_size,
                    spline_order=3,
                    base_activation=activation,
                ),
                KANLinear(
                    emb_dim,
                    final_hidden_size,
                    spline_order=3,
                    base_activation=activation,
                ),
                nn.Dropout(0.125),
                NoisyFactorizedLinear(final_hidden_size, n_out),
            )

        if self.return_probs:
            self.sm_out = nn.Softmax(dim=-1)

    def forward(self, x):

        if self.input_dropout:
            x_logit = self.logits(self.dropout(x))

        else:
            x_logit = self.logits(x)

        if self.return_probs:
            sm_out = self.sm_out(x_logit)

            return x_logit, sm_out

        else:
            return x_logit