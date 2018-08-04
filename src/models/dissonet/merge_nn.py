import torch
import torch.nn as nn
from firelab.utils import cudable

from src.models import FFN


class MergeNN(nn.Module):
    def __init__(self, hid_size):
        super(MergeNN, self).__init__()

        self.style_embed = nn.Embedding(2, hid_size)
        self.merge = nn.Sequential(
            FFN([hid_size * 2, hid_size]),
            nn.BatchNorm1d(hid_size)
        )

    def forward(self, content: torch.Tensor, style: int):
        """
        Computes embeddings for style and merges it with content vectors
        Then passes all this through MLP to output a vector of normal size

        Arguments:
            content (torch.Tensor): content vectors
            style (int): binary var, denoting if it is source/target
        """
        style_seq = cudable(torch.ones(content.size(0)) * style).long()
        style_embs = self.style_embed(style_seq)

        out = self.merge(torch.cat([style_embs, content], dim=1))

        return out
