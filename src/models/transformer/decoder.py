import torch
import torch.nn as nn
import numpy as np

from src.utils.training_utils import embed
from .layers import MultiHeadAttention, SublayerConnection, PositionalEncoding, FeedForward
from .utils import pad_mask, subsequent_mask


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, config, vocab):
        super(Decoder, self).__init__()

        self.config = config
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab), config.d_model, padding_idx=vocab.stoi['<pad>'])
        self.pe = PositionalEncoding(config)
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layers)
        ])
        self.linear_out = nn.Linear(config.d_model, len(vocab))
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, encs, trg, encs_mask=None, onehot=True):
        dec_pad_mask = pad_mask((trg if onehot else trg.max(dim=-1)[1]), self.vocab)
        mask = dec_pad_mask.unsqueeze(1) & subsequent_mask(trg.size(1))

        x = embed(self.embed, trg, onehot)
        x = x * np.sqrt(self.config.d_model)
        x = self.pe(x)

        for layer in self.layers:
            x = layer(encs, x, encs_mask, mask)

        x = self.norm(x)
        x = self.linear_out(x)

        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, encs-attn, and feed forward (defined below)"
    def __init__(self, config):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(config)
        self.self_attn_sl = SublayerConnection(config)
        self.encs_attn = MultiHeadAttention(config)
        self.encs_attn_sl = SublayerConnection(config)
        self.out_ff = FeedForward(config)
        self.out_ff_sl = SublayerConnection(config)

    def forward(self, encs, x, encs_mask, mask):
        """
        Arguments:
            - x — currently available output sequence
            - encs — encoder outputs
        """
        x = self.self_attn_sl(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.encs_attn_sl(x, lambda x: self.encs_attn(x, encs, encs, encs_mask))
        x = self.out_ff_sl(x, self.out_ff)

        return x
