"""
Transformer implementation
Roots from https://github.com/harvardnlp/annotated-transformer
"""

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    """
    Incapsulating Transformer model into a single module
    for efficient data parallelization
    """
    def __init__(self, config, vocab_src, vocab_trg):
        super(Transformer, self).__init__()

        self.encoder = Encoder(config, vocab_src)
        self.decoder = Decoder(config, vocab_trg)

    def forward(self, src, trg):
        encs, mask = self.encoder(src)
        recs = self.decoder(encs, trg[:, :-1], mask)

        return recs
