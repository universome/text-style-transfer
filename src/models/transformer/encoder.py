import torch
import torch.nn as nn

from .layers import MultiHeadAttention, SublayerConnection, PositionalEncoding, FeedForward
from .utils import pad_mask


class Encoder(nn.Module):
    def __init__(self, config, vocab_src):
        super(Encoder, self).__init__()

        self.vocab_src = vocab_src
        self.embed = nn.Embedding(len(vocab_src), config.d_model, padding_idx=vocab_src.stoi['<pad>'])
        self.pe = PositionalEncoding(config)
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.n_enc_layers)
        ])
        self.emb_dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        mask = pad_mask(x, self.vocab_src).unsqueeze(1)
        x = self.embed(x)

        for layer in self.layers:
            x = self.emb_dropout(self.pe(x))
            x = layer(x, mask)

        x = self.norm(x)

        return x, mask


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.self_attn_sl = SublayerConnection(config)
        self.out_ff = FeedForward(config)
        self.out_ff_sl = SublayerConnection(config)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.self_attn_sl(x, self.self_attn(x, x, x, mask))
        x = self.out_ff_sl(x, self.out_ff(x))

        return x
