import torch
import torch.nn as nn
from firelab.utils import cudable


class Dropword(nn.Module):
    def __init__(self, p):
        super(Dropword, self).__init__()
        self.p = p

    def forward(self, x, p:float=None):
        assert x.dim() == 3 # (batch, len, emb_size)

        p = p or self.p
        mask = torch.bernoulli(torch.Tensor(x.size(0), x.size(1)).fill_(1 - p))
        mask = cudable(mask).unsqueeze(-1).repeat(1, 1, x.size(2))

        return x * mask if self.training else x
