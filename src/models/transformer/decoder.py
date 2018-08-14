import torch
import torch.nn as nn

from .layers import MultiHeadAttention, SublayerConnection, PositionalEncoding, FeedForward
from .utils import pad_mask, subsequent_mask


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, config, vocab_trg):
        super(Decoder, self).__init__()

        self.vocab_trg = vocab_trg
        self.embed = nn.Embedding(len(vocab_trg), config.d_model, padding_idx=vocab_trg.stoi['<pad>'])
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_dec_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.linear_out = nn.Linear(config.d_model, len(vocab_trg))

    def forward(self, encs, trg, encs_mask):
        mask = pad_mask(trg, self.vocab_trg).unsqueeze(1) & subsequent_mask(trg.size(1))
        out = self.embed(trg)

        for layer in self.layers:
            out = layer(encs, out, encs_mask, mask)

        out = self.norm(out)
        out = self.linear_out(out)

        return out


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
        x = self.self_attn_sl(x, self.self_attn(x, x, x, mask))
        x = self.encs_attn_sl(x, self.encs_attn(x, encs, encs, encs_mask))
        x = self.out_ff_sl(x, self.out_ff(x))

        return x
