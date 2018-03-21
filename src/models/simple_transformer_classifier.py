from itertools import chain

import torch
import torch.nn as nn
from torch.autograd import Variable

from .helpers import * # TODO: move helper functions somewhere
from .modules import Encoder, Decoder
from src.utils.common import one_hot_seqs_to_seqs, variable
from src.vocab import constants


use_cuda = torch.cuda.is_available()


class SimpleTransformerClassifier(nn.Module):
    '''
        This Transformer Classifier runs over vector-valued sequences
        In such a way it does not use any embeddings.
    '''
    def __init__(self, num_classes, max_len, **transformer_kwargs):
        super(SimpleTransformerClassifier, self).__init__()

        self.d_model = transformer_kwargs['d_model']
        self.encoder = Encoder(self.d_model, max_len, **transformer_kwargs)
        self.decoder = Decoder(4, max_len, **transformer_kwargs)
        self.num_classes = num_classes
        self.dec_out_to_logits = nn.Linear(self.d_model, num_classes-1)

    def trainable_parameters(self):
        freezed = set(map(id, self.encoder.position_enc.parameters()))

        return (p for p in self.parameters() if id(p) not in freezed)

    def forward(self, x, src_seq):
        # Encoding the sequence
        enc_output = x
        positions = get_positions_for_seqs(src_seq)
        enc_output += self.encoder.position_enc(positions)
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)

        for enc_layer in self.encoder.layer_stack:
            enc_output, _ = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)

        # Decoding the sequence
        tgt_seq = variable(torch.zeros(x.size(0)).unsqueeze(1).long())
        dec_output = variable(torch.ones(x.size(0), 1, self.d_model))
        dec_slf_attn_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_mask = torch.zeros(x.size(0), 1, 1).byte()
        if use_cuda: dec_slf_attn_mask = dec_slf_attn_mask.cuda()
        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)

        for dec_layer in self.decoder.layer_stack:
            dec_output = dec_layer(dec_output, enc_output,
                dec_enc_attn_mask=dec_enc_attn_pad_mask,
                slf_attn_mask=dec_slf_attn_mask)

        logits = self.dec_out_to_logits(dec_output).squeeze()

        return logits
