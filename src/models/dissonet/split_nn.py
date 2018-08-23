import torch
import torch.nn as nn

from src.models import FFN


class SplitNN(nn.Module):
    def __init__(self, size, style_vec_size, dropout=0):
        super(SplitNN, self).__init__()

        self.size = size
        self.split = FFN([size, size + style_vec_size], dropout=dropout)

    def forward(self, x):
        assert x.dim() == 2

        x = self.split(x)
        content, style = x[:, :self.size], x[:, self.size:]

        return content, style
