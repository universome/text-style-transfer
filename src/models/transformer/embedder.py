import torch
import torch.nn as nn
import numpy as np
from firelab.utils.training_utils import cudable

from .layers import PositionalEncoding
from .encoder import EncoderLayer
from .utils import pad_mask


class Embedder(nn.Module):
    def __init__(self, n_vecs, config, vocab):
        super(Embedder, self).__init__()

        self.config = config
        self.n_vecs = n_vecs
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab) + n_vecs, config.d_model, padding_idx=vocab.stoi['<pad>'])
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layers)])
        self.pe = PositionalEncoding(config)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = self.extend_x(x)
        mask = pad_mask(x, self.vocab).unsqueeze(1)

        x = self.embed(x)
        x = x * np.sqrt(self.config.d_model)
        x[:, :-self.n_vecs] = self.pe(x[:, :-self.n_vecs])

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        x = x[:, -self.n_vecs:, :]

        return x

    def extend_x(self, x):
        "Extends sequences with EMB tokens"
        assert x.dim() == 2

        idx = torch.arange(len(self.vocab), len(self.vocab) + self.n_vecs)
        idx = idx.unsqueeze(0).repeat(x.size(0), 1)

        out = torch.cat((x, cudable(idx)), dim=1)

        return out
