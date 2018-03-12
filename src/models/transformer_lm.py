import torch
import torch.nn as nn
from torch.autograd import Variable

from .modules import Encoder
from .helpers import get_positions_for_seqs
from .helpers import get_attn_padding_mask, get_attn_subsequent_mask


class TransformerLM(Encoder):
    def __init__(self, *args, **kwargs):
        super(TransformerLM, self).__init__(*args, **kwargs)

    def forward(self, seqs):
        inputs = self.src_word_emb(seqs)
        positions = get_positions_for_seqs(seqs)
        inputs += self.position_enc(positions)

        pad_attn_mask = get_attn_padding_mask(seqs, seqs)
        sub_attn_mask = get_attn_subsequent_mask(seqs)
        attn_mask = torch.gt(pad_attn_mask + sub_attn_mask, 0)

        outputs = inputs

        for layer in self.layer_stack:
            outputs, _ = layer(outputs, slf_attn_mask=attn_mask)

        return outputs

    def inference(self, seqs: Variable, beam_size: int = 1, n_steps=) -> list:
        """
            Generates predictions for each word in the sequence (except the first one)
            :param seqs: batch of sequences to predict.
            :param beam_size: beam size to use.
            :return: batch of sequences
        """

        raise NotImplemented
