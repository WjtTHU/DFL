import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """
    This implementation is from https://github.com/QData/spacetimeformer/blob/main/spacetimeformer/time2vec.py#L5.
    Multiple temporal features can be used.

    """

    def __init__(self, input_dim=6, embed_dim=512, act_function=torch.sin):
        assert embed_dim % input_dim == 0
        super(Time2Vec, self).__init__()
        self.enabled = embed_dim > 0
        if self.enabled:
            self.embed_dim = embed_dim // input_dim
            self.input_dim = input_dim
            self.embed_weight = nn.parameter.Parameter(torch.randn(self.input_dim, self.embed_dim))
            self.embed_bias = nn.parameter.Parameter(torch.randn(self.input_dim, self.embed_dim))
            self.act_function = act_function

    def forward(self, x):
        if self.enabled:
            x = torch.diag_embed(x)
            # x.shape = (bs, sequence_length, input_dim, input_dim)
            x_affine = torch.matmul(x, self.embed_weight) + self.embed_bias
            # x_affine.shape = (bs, sequence_length, input_dim, time_embed_dim)
            x_affine_0, x_affine_remain = torch.split(x_affine, [1, self.embed_dim - 1], dim=-1)
            x_affine_remain = self.act_function(x_affine_remain)
            x_output = torch.cat([x_affine_0, x_affine_remain], dim=-1)
            x_output = x_output.view(x_output.size(0), x_output.size(1), -1)
            # x_output.shape = (bs, sequence_length, input_dim * time_embed_dim)
        else:
            x_output = x
        return x_output


class SimpleTime2Vec(nn.Module):
    def __init__(self, k):
        """
        Process only one temporal feature.
        Args:
            k: dimension of temporal feature after encoding

        """
        super().__init__()
        self.k = k
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(k - 1))
        self.b = nn.Parameter(torch.randn(k - 1))

    def forward(self, t):
        if t.dim() == 1:  # ensure the last dimension is 1
            t = t.unsqueeze(-1)
        v0 = self.w0 * t + self.b0  # linear part
        vp = torch.sin(t * self.w + self.b)  # cyclical part
        return torch.cat([v0, vp], dim=-1)
