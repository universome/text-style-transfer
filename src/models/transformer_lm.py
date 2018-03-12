import torch
import torch.nn as nn
from torch.autograd import Variable

from .modules import Encoder
from .layers import Linear
from .helpers import get_positions_for_seqs
from .helpers import get_attn_padding_mask, get_attn_subsequent_mask


class TransformerLM(Encoder):
    def __init__(self, vocab_size, max_len, d_model=512, **kwargs):
        super(TransformerLM, self).__init__(vocab_size, max_len, d_model=d_model, **kwargs)

        self.out_to_logits = Linear(d_model, vocab_size, bias=False)
        self.out_to_logits.weight = self.src_word_emb.weight

    def forward(self, seqs):
        inputs = self.src_word_emb(seqs)
        positions = get_positions_for_seqs(seqs)
        inputs += self.position_enc(positions)

        pad_attn_mask = get_attn_padding_mask(seqs, seqs)
        sub_attn_mask = get_attn_subsequent_mask(seqs)
        attn_mask = torch.gt(pad_attn_mask + sub_attn_mask, 0)

        out = inputs

        for layer in self.layer_stack:
            out, _ = layer(out, slf_attn_mask=attn_mask)

        out = self.out_to_logits(out)

        return out

    def trainable_parameters(self):
        freezed = set(map(id, self.position_enc.parameters()))

        return (p for p in self.parameters() if id(p) not in freezed)

    def inference(self, seqs: Variable, beam_size: int = 1) -> list:
        """
            Generates predictions for each word in the sequence (except the first one)
            :param seqs: batch of sequences to predict.
            :param beam_size: beam size to use.
            :return: batch of sequences
        """

        raise NotImplemented
