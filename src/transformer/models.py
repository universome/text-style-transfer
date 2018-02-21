''' Define the Transformer model '''
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

import transformer.constants as Constants
from transformer.modules import BottleLinear as Linear
from transformer.layers import EncoderLayer, DecoderLayer
from transformer.beam import Beam


use_cuda = torch.cuda.is_available()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, embs=None, return_attns=False):
        # Word embedding look up
        enc_input = embs(src_seq) if embs else self.src_word_emb(src_seq)

        # Position Encoding addition
        positions = get_positions_for_seqs(src_seq)
        enc_input += self.position_enc(positions)

        if return_attns: enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns: enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
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

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, src_seq, enc_output, embs=None, return_attns=False):
        # Word embedding look up
        dec_input = embs(tgt_seq) if embs else self.tgt_word_emb(tgt_seq)

        # Position Encoding addition
        positions = get_positions_for_seqs(tgt_seq)
        dec_input += self.position_enc(positions)

        # Decode
        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)

        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
            dropout=0.1, proj_share_weight=True, embs_share_weight=False):

        super(Transformer, self).__init__()
        self.encoder = Encoder(
            n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.decoder = Decoder(
            n_tgt_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.src_word_proj = Linear(d_model, n_src_vocab, bias=False)
        self.tgt_word_proj = Linear(d_model, n_tgt_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.prob_projection = nn.LogSoftmax(dim=1)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module output shall be the same.'

        if proj_share_weight:
            # Share the weight matrix between tgt word embedding/projection
            assert d_model == d_word_vec
            self.src_word_proj.weight = self.encoder.src_word_emb.weight
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight

        if embs_share_weight:
            # Share the weight matrix between src/tgt word embeddings
            # assume the src/tgt word vec size are the same
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, trg, use_trg_embs_in_encoder=False, use_src_embs_in_decoder=False, return_encodings=False):
        trg = trg[:, :-1]

        if use_trg_embs_in_encoder:
            enc_output, *_ = self.encoder(src, self.decoder.tgt_word_emb)
        else:
            enc_output, *_ = self.encoder(src)

        if use_src_embs_in_decoder:
            dec_output, *_ = self.decoder(trg, src, enc_output, self.encoder.src_word_emb)
            seq_logit = self.src_word_proj(dec_output)
        else:
            dec_output, *_ = self.decoder(trg, src, enc_output)
            seq_logit = self.tgt_word_proj(dec_output)

        out = seq_logit.view(-1, seq_logit.size(2))

        return (out, enc_output) if return_encodings else out


    def translate_batch(self, src_seq, use_src_embs_in_decoder=False, use_trg_embs_in_encoder=False, beam_size=6, max_len=200):
        # Batch size is in different location depending on data.
        batch_size = src_seq.size(0)

        #- Enocde
        if use_trg_embs_in_encoder:
            enc_output, *_ = self.encoder(src_seq, self.decoder.tgt_word_emb)
        else:
            enc_output, *_ = self.encoder(src_seq)

        # Repeat data for beam
        # We call clone here because there is a bug,
        # which is fixed but not released yet (issue 4054)
        src_seq = Variable(
            src_seq.data.clone().repeat(1, beam_size).view(
                src_seq.size(0) * beam_size, src_seq.size(1)))

        enc_output = Variable(
            enc_output.data.clone().repeat(1, beam_size, 1).view(
                enc_output.size(0) * beam_size, enc_output.size(1), enc_output.size(2)))

        #--- Prepare beams
        beams = [Beam(beam_size, use_cuda) for _ in range(batch_size)]
        beam_seq_idx_map = {beam_idx: seq_idx for seq_idx, beam_idx in enumerate(range(batch_size))}

        n_remaining_sents = batch_size

        #- Decode
        for i in tqdm(range(max_len)):
            len_dec_seq = i + 1

            # -- Preparing decoded data seq -- #
            # size: batch x beam x seq
            dec_partial_seq = torch.stack([b.get_current_state() for b in beams if not b.done])
            # size: (batch * beam) x seq
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            # wrap into a Variable
            dec_partial_seq = Variable(dec_partial_seq, volatile=True)

            if use_cuda: dec_partial_seq = dec_partial_seq.cuda()

            # -- Decoding -- #
            if use_src_embs_in_decoder:
                dec_output, *_ = self.decoder(dec_partial_seq, src_seq, enc_output, self.encoder.src_word_emb)
            else:
                dec_output, *_ = self.decoder(dec_partial_seq, src_seq, enc_output)

            dec_output = dec_output[:, -1, :] # (batch * beam) * d_model

            if use_src_embs_in_decoder:
                dec_output = self.src_word_proj(dec_output)
            else:
                dec_output = self.tgt_word_proj(dec_output)

            out = self.prob_projection(dec_output)

            # batch x beam x n_words
            word_lk = out.view(n_remaining_sents, beam_size, -1).contiguous()

            active_beam_idx_list = []
            for beam_idx in range(batch_size):
                if beams[beam_idx].done: continue

                seq_idx = beam_seq_idx_map[beam_idx]

                if not beams[beam_idx].advance(word_lk.data[seq_idx]):
                    active_beam_idx_list += [beam_idx]

            if not active_beam_idx_list:
                # all sequences have finished their path to <EOS>
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_seq_idxs = torch.LongTensor([beam_seq_idx_map[k] for k in active_beam_idx_list])
            if use_cuda: active_seq_idxs = active_seq_idxs.cuda()

            # update the idx mapping
            beam_seq_idx_map = {beam_idx: seq_idx for seq_idx, beam_idx in enumerate(active_beam_idx_list)}

            def update_active_seq(seq_var, active_seq_idxs):
                ''' Remove the src sequence of finished seqances in one batch. '''

                seq_idx_dim_size, *rest_dim_sizes = seq_var.size()
                seq_idx_dim_size = seq_idx_dim_size * len(active_seq_idxs) // n_remaining_sents
                new_size = (seq_idx_dim_size, *rest_dim_sizes)

                # select the active seqances in batch
                original_seq_data = seq_var.data.view(n_remaining_sents, -1)
                active_seq_data = original_seq_data.index_select(0, active_seq_idxs)
                active_seq_data = active_seq_data.view(*new_size)

                return Variable(active_seq_data, volatile=True)

            def update_active_enc_info(enc_info_var, active_seq_idxs):
                ''' Remove the encoder outputs of finished seqances in one batch. '''

                seq_idx_dim_size, *rest_dim_sizes = enc_info_var.size()
                seq_idx_dim_size = seq_idx_dim_size * len(active_seq_idxs) // n_remaining_sents
                new_size = (seq_idx_dim_size, *rest_dim_sizes)

                # select the active seqances in batch
                original_enc_info_data = enc_info_var.data.view(n_remaining_sents, -1, self.d_model)
                active_enc_info_data = original_enc_info_data.index_select(0, active_seq_idxs)
                active_enc_info_data = active_enc_info_data.view(*new_size)

                return Variable(active_enc_info_data, volatile=True)

            src_seq = update_active_seq(src_seq, active_seq_idxs)
            enc_output = update_active_enc_info(enc_output, active_seq_idxs)

            #- update the remaining size
            n_remaining_sents = len(active_seq_idxs)

        #- Return translations
        translations = []

        for i in range(batch_size):
            beams[i].sort_scores()
            translation = beams[i].get_hypothesis(0) # Best hypothesis lies at index 0
            translations.append(translation)

        return translations


def position_encoding_init(max_len, dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        if pos != 0 else np.zeros(dim) for pos in range(max_len)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1

    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2

    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk

    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2

    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)

    if seq.is_cuda: subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask


def get_positions_for_seq(seq):
    return [i+1 if token != Constants.PAD else 0 for i, token in enumerate(seq.data)]


def get_positions_for_seqs(seqs):
    positions = [get_positions_for_seq(seq) for seq in seqs]
    positions = Variable(torch.LongTensor(positions))
    if use_cuda: positions = positions.cuda()

    return positions
