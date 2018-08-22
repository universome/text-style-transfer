import torch
import torch.nn as nn
import numpy as np

from .encoder import Encoder
from .utils import pad_mask, subsequent_mask

class LM(nn.Module):
    "Transformer-based LM. Just like Decoder, but without encodings"
    def __init__(self, config, vocab):
        super(LM, self).__init__()

        self.vocab = vocab
        self.encoder = Encoder(config, vocab)
        self.linear_out = nn.Linear(config.d_model, len(vocab))

    def forward(self, x, onehot=True):
        x_pad_mask = pad_mask((x if onehot else x.max(dim=-1)[1]), self.vocab)
        mask = x_pad_mask.unsqueeze(1) & subsequent_mask(x.size(1))

        encs, _ = self.encoder(x, mask, onehot=onehot)
        out = self.linear_out(encs)

        return out
