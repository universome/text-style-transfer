import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .layers import EncoderLayer, DecoderLayer, BottleLinear as Linear
from .helpers import *
from src.utils.common import variable, one_hot_seqs_to_seqs, embed
from src.vocab import constants

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, embs=None, one_hot_src=False):
        # Word embedding look up
        enc_input = embed(src_seq, embs or self.src_word_emb, one_hot_input=one_hot_src)

        # Position Encoding addition
        positions = get_positions_for_seqs(src_seq, one_hot_input=one_hot_src)
        enc_input += self.position_enc(positions)

        seq_for_attn = one_hot_seqs_to_seqs(src_seq) if one_hot_src else src_seq
        enc_slf_attn_mask = get_attn_padding_mask(seq_for_attn, seq_for_attn)

        enc_output = enc_input

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)

        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Decoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.n_layers = n_layers

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)
        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=constants.PAD)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, src_seq, enc_output, embs=None, cache=None, one_hot_src=False, one_hot_trg=False):
        """
        Arguments:
            - ...
            - cache — list containing previous decoder state (dec_output for each layer).
              We need this for fast inference.
        """

        # TODO: actually, we can determine one_hot_input by just looking at .dim()
        # Should we guess input format this way?

        # TODO: cache this too
        dec_input = embed(tgt_seq, embs or self.tgt_word_emb, one_hot_input=one_hot_trg)

        # Position Encoding
        positions = get_positions_for_seqs(tgt_seq, one_hot_input=one_hot_trg)
        dec_input += self.position_enc(positions)

        tgt_seq_for_attn = tgt_seq[:,-1].unsqueeze(1) if cache else tgt_seq # Only for the last word
        tgt_seq_for_attn = one_hot_seqs_to_seqs(tgt_seq_for_attn) if one_hot_trg else tgt_seq_for_attn
        tgt_seq_k_for_slf_pad_attn = one_hot_seqs_to_seqs(tgt_seq) if one_hot_trg else tgt_seq
        src_seq_for_attn = one_hot_seqs_to_seqs(src_seq) if one_hot_src else src_seq

        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq_for_attn, tgt_seq_k_for_slf_pad_attn)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq_for_attn)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq_for_attn, src_seq_for_attn)

        dec_output = dec_input

        for i, dec_layer in enumerate(self.layer_stack):
            dec_output = dec_layer(
                dec_output, enc_output, compute_only_last=(not cache is None),
                dec_enc_attn_mask=dec_enc_attn_pad_mask, slf_attn_mask=dec_slf_attn_mask)

            if cache:
                # We have used cached results and computed only the last element
                # Now we should add this element to cache
                if not cache[i] is None: dec_output = torch.cat((cache[i], dec_output), 1)
                cache[i] = dec_output

        return dec_output
