import torch
import torch.nn as nn
from firelab.utils.training_utils import cudable

from src.models import FFN


class MergeNN(nn.Module):
    def __init__(self, size, style_vec_size):
        super(MergeNN, self).__init__()

        self.merge = nn.Sequential(
            FFN([size + style_vec_size, size]),
            nn.BatchNorm1d(size)
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor):
        """
        Takes content vector and style vectors, concatenates them,
        passes through an MLP and outputs

        Arguments:
            content (torch.Tensor): content vectors
            style (torch.Tensor): style vectors
        """

        out = self.merge(torch.cat([content, style], dim=1))

        return out
