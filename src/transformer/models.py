from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

import src.transformer.constants as constants
from src.transformer.modules import BottleLinear as Linear
from src.transformer.layers import EncoderLayer, DecoderLayer
from src.transformer.beam import Beam
from src.utils.common import variable


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

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, embs=None, one_hot_src=False):
        # Word embedding look up
        embs = embs or self.src_word_emb
        enc_input = torch.matmul(src_seq, embs.weight) if one_hot_src else embs(src_seq)

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

        # Word embeddings look up
        # TODO: cache this
        embs = embs or self.tgt_word_emb
        dec_input = torch.matmul(tgt_seq, embs.weight) if one_hot_trg else embs(tgt_seq)

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
        self.d_model = d_model
        self.proj_share_weight = proj_share_weight

        # We use LogSoftmax instead of Softmax to avoid
        # numerical issues when computing beam scores
        self.logits_to_logprobs = nn.LogSoftmax(dim=1)

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

    def get_embs_parameters(self):
        enc_embs = set(map(id, self.encoder.src_word_emb.parameters()))
        dec_embs = set(map(id, self.decoder.tgt_word_emb.parameters()))
        src_proj_embs = set(map(id, self.src_word_proj.parameters()))
        trg_proj_embs = set(map(id, self.tgt_word_proj.parameters()))
        embs_ids = enc_embs | dec_embs | src_proj_embs | trg_proj_embs

        return (p for p in self.parameters() if id(p) in embs_ids)

    def get_trainable_params_without_embs(self):
        trainable = self.get_trainable_parameters()
        embs_ids = set(map(id, self.get_embs_parameters()))

        return (p for p in trainable if not id(p) in embs_ids)

    def forward(self, src, trg, use_trg_embs_in_encoder=False,
                use_src_embs_in_decoder=False, return_encodings=False,
                one_hot_src=False, one_hot_trg=False):

        trg = trg[:, :-1]
        encoder_embs = self.decoder.tgt_word_emb if use_trg_embs_in_encoder else None
        decoder_embs = self.encoder.src_word_emb if use_src_embs_in_decoder else None
        proj = self.src_word_proj if use_src_embs_in_decoder else self.tgt_word_proj

        enc_output, *_ = self.encoder(src, embs=encoder_embs, one_hot_src=one_hot_src)
        dec_output = self.decoder(trg, src, enc_output, embs=decoder_embs, one_hot_src=one_hot_src, one_hot_trg=one_hot_trg)
        seq_logit = proj(dec_output)

        out = seq_logit.view(-1, seq_logit.size(2))

        return (out, enc_output) if return_encodings else out

    def translate_batch(self, src_seq, beam_size=6, max_len=50,
                        use_src_embs_in_decoder=False, use_trg_embs_in_encoder=False):

        batch_size = src_seq.size(0)

        #- Enocde
        embs = self.decoder.tgt_word_emb if use_trg_embs_in_encoder else None
        enc_output, *_ = self.encoder(src_seq, embs=embs)

        # Repeating seqs for beam search
        src_seq = repeat_seq_for_beam(src_seq, beam_size)
        enc_output = repeat_seq_for_beam(enc_output, beam_size)

        # Prepare beams
        beams = [Beam(beam_size, use_cuda) for _ in range(batch_size)]
        beam_seq_idx_map = {beam_idx: seq_idx for seq_idx, beam_idx in enumerate(range(batch_size))}

        # Here we are going to keep state of each decoder layer
        # So we are not recompute everything from scratch
        cache = init_cache(batch_size, beam_size, self.decoder.n_layers, self.d_model)

        n_remaining_sents = batch_size

        for i in range(max_len):
            len_dec_seq = i + 1

            # -- Preparing decoded data seq -- #
            # TODO: cache it?
            # size: batch x beam x seq
            dec_partial_seq = torch.stack([b.get_current_state() for b in beams if not b.done])
            # size: (batch * beam) x seq
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            dec_partial_seq = variable(dec_partial_seq, volatile=True)

            # -- Decoding -- #
            embs = self.encoder.src_word_emb if use_src_embs_in_decoder else None
            proj = self.src_word_proj if use_src_embs_in_decoder else self.tgt_word_proj

            dec_output = self.decoder(dec_partial_seq, src_seq, enc_output, embs=embs, cache=cache)
            dec_output = dec_output[:, -1, :] # We need only the last word; (batch * beam) * d_model
            dec_output = proj(dec_output) # Generating logits

            word_logprobs = self.logits_to_logprobs(dec_output)
            # Reshaping from [(batch * beam) x n_words] to [batch x beam x n_words]
            word_logprobs = word_logprobs.view(n_remaining_sents, beam_size, -1).contiguous()

            active_beam_idx_list = []

            for beam_idx in range(batch_size):
                if beams[beam_idx].done: continue

                seq_idx = beam_seq_idx_map[beam_idx]
                is_done = beams[beam_idx].advance(word_logprobs.data[seq_idx])

                if not is_done: active_beam_idx_list.append(beam_idx)

            if not active_beam_idx_list:
                # all sequences have finished their path to EOS
                break

            # In this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_seq_idxs = torch.LongTensor([beam_seq_idx_map[k] for k in active_beam_idx_list])
            if use_cuda: active_seq_idxs = active_seq_idxs.cuda()

            # Update the idx mapping
            beam_seq_idx_map = {beam_idx: seq_idx for seq_idx, beam_idx in enumerate(active_beam_idx_list)}

            # Remove source sentences that are completely translated
            src_seq = update_active_seq(src_seq, active_seq_idxs, n_remaining_sents)
            enc_output = update_layer_outputs(enc_output, active_seq_idxs, n_remaining_sents)

            # Remove decoder states which are not active anymore
            cache = [update_layer_outputs(layer_cache, active_seq_idxs, n_remaining_sents) for layer_cache in cache]

            # Update the remaining size
            n_remaining_sents = len(active_seq_idxs)

        #- Return translations
        return extract_best_translation_from_beams(beams)

    def differentiable_translate(self, src_seq, vocab_trg, max_len=50, temperature=1,
                                 use_src_embs_in_decoder=False, use_trg_embs_in_encoder=False):
        # TODO: can we use beam search here?
        batch_size = src_seq.size(0)
        enc_output, *_ = self.encoder(src_seq)

        # Here we are going to keep state of each decoder layer
        # So we are not recompute everything from scratch
        cache = init_cache(batch_size, 1, self.decoder.n_layers, self.d_model)

        translations = init_translations(len(vocab_trg), batch_size)
        n_remaining_sents = batch_size

        for i in range(max_len-1):
            # Our samples are smooth one hot vectors
            # We run sequences until EOS is reached
            # (we determine EOS by argmax(sample) == EOS)
            active_seq_idxs = [i for i in range(n_remaining_sents) if not (np.argmax(translations[i][-1].data) in (constants.EOS, constants.PAD))]
            active_seq_idxs = torch.LongTensor(active_seq_idxs)
            if use_cuda: active_seq_idxs = active_seq_idxs.cuda()

            # Update the remaining size
            if len(active_seq_idxs) == 0: break

            active_translations = update_layer_outputs(translations, active_seq_idxs, batch_size)

            # Remove source sentences that are completely translated
            src_seq = update_active_seq(src_seq, active_seq_idxs, n_remaining_sents, volatile=False)
            enc_output = update_layer_outputs(enc_output, active_seq_idxs, n_remaining_sents, volatile=False)

            # Remove decoder states which are not active anymore
            if i != 0: cache = [update_layer_outputs(c, active_seq_idxs, n_remaining_sents, volatile=False) for c in cache]

            # -- Decoding -- #
            embs = self.encoder.src_word_emb if use_src_embs_in_decoder else None
            proj = self.src_word_proj if use_src_embs_in_decoder else self.tgt_word_proj

            dec_output = self.decoder(active_translations, src_seq, enc_output, embs=embs, cache=cache, one_hot_trg=True)
            logits = proj(dec_output[:, -1, :]) # We need only the last word
            samples = sample(logits, temperature)
            samples = extend_inactive_with_pads(samples, active_seq_idxs, batch_size)
            translations = torch.cat((translations, samples.unsqueeze(1)), 1)

            n_remaining_sents = active_seq_idxs.size(0)

        #- Return translations
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

    if type(seq_k) == Variable: seq_k = seq_k.data

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.eq(constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k) # bxsqxsk

    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2

    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)

    if seq.is_cuda: subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask


def get_positions_for_seqs(seqs, **kwargs):
    positions = [get_positions_for_seq(seq, **kwargs) for seq in seqs]
    positions = Variable(torch.LongTensor(positions))
    if use_cuda: positions = positions.cuda()

    return positions


def get_positions_for_seq(seq, **kwargs):
    return [token_to_position(token, i, **kwargs) for i, token in enumerate(seq.data)]


def token_to_position(token, index, one_hot_input=False):
    if type(token) == Variable: token = token.data
    if one_hot_input: token = np.argmax(token)

    return 0 if token == constants.PAD else index + 1


def update_active_seq(seq_var, active_seq_idxs, n_remaining_sents, volatile=True):
    '''Remove the src sequence of finished sequences from the batch. '''

    seq_idx_dim_size, *rest_dim_sizes = seq_var.size()
    seq_idx_dim_size = seq_idx_dim_size * len(active_seq_idxs) // n_remaining_sents
    new_size = (seq_idx_dim_size, *rest_dim_sizes)

    # select the active sequences in batch
    original_seq_data = seq_var.data.view(n_remaining_sents, -1)
    active_seq_data = original_seq_data.index_select(0, active_seq_idxs)
    active_seq_data = active_seq_data.view(*new_size)

    return Variable(active_seq_data, volatile=False) # TODO: volatile!


def update_layer_outputs(enc_info_var, active_seq_idxs, n_remaining_sents, volatile=True):
    '''Remove the encoder outputs of finished sequences from the batch. '''
    # TODO(universome): pass beam_size instead of n_remaining_sents
    assert enc_info_var.dim() == 3
    assert active_seq_idxs.dim() == 1

    num_repeated_seqs, seq_len, vec_size = enc_info_var.size()
    new_num_repeated_seqs = num_repeated_seqs * len(active_seq_idxs) // n_remaining_sents

    # select the active sequences in batch
    original_enc_info_data = enc_info_var.data.view(n_remaining_sents, -1, vec_size)
    # print(active_seq_idxs.numpy().tolist(), enc_info_var.size(), original_enc_info_data.size())
    active_enc_info_data = original_enc_info_data.index_select(0, active_seq_idxs)
    active_enc_info_data = active_enc_info_data.view(new_num_repeated_seqs, seq_len, vec_size)

    # print(enc_info_var.size(), active_enc_info_data.size(), active_seq_idxs.size())

    return Variable(active_enc_info_data, volatile=False) # TODO: volatile!


def extract_best_translation_from_beams(beams):
    translations = []
    scores = []

    for i in range(len(beams)):
        beams[i].sort_scores()
        _, idx = beams[i].get_the_best_score_and_idx()
        scores.append(_)
        translation = beams[i].get_hypothesis(idx)
        translations.append(translation)

    return translations


def repeat_seq_for_beam(seq, beam_size):
    # Repeat data for beam
    # We call .clone() here because there is a bug,
    # which is fixed but not released yet (issue 4054)
    repeated_seq = seq.data.clone()

    if seq.ndimension() == 2:
        repeated_seq = repeated_seq.repeat(1, beam_size)
        repeated_seq = repeated_seq.view(seq.size(0) * beam_size, seq.size(1))
    elif seq.ndimension() == 3:
        repeated_seq = repeated_seq.repeat(1, beam_size, 1)
        repeated_seq = repeated_seq.view(seq.size(0) * beam_size, seq.size(1), seq.size(2))
    else:
        assert False, "Not implemented"

    return Variable(repeated_seq, volatile=True)


def init_cache(batch_size, beam_size, n_layers, d_model):
    # TODO: torch squeezes this array to the size of [batch_size * beam_size] :|
    # return [torch.zeros(batch_size * beam_size, 0, d_model) for _ in range(n_layers)]
    return [None for _ in range(n_layers)]


def init_translations(vocab_size, batch_size):
    # Creating array of one hot vectors
    translations = np.zeros((batch_size, 1, vocab_size))
    translations[:, :, constants.BOS] = 1

    # Convert it to pytorch Variable
    translations = Variable(torch.FloatTensor(translations))
    if use_cuda: translations = translations.cuda()

    return translations


def one_hot_seqs_to_seqs(seqs):
    seqs = torch.LongTensor([[np.argmax(t.data) for t in s] for s in seqs])
    if use_cuda: seqs = seqs.cuda()

    return seqs


def sample(logits, temperature=1):
    return F.softmax(logits / temperature, dim=1)


def extend_inactive_with_pads(samples, active_seq_idx, batch_size):
    assert samples.dim() == 2
    assert samples.size(0) == len(active_seq_idx)

    # We should distribute our samples according their indices (active_seq_idx)
    # And other staff we should fill with constants.PAD (one-hotted)
    pad_idx = Variable(torch.LongTensor([constants.PAD]))
    outputs = Variable(torch.zeros(batch_size, samples.size(1)))

    if use_cuda:
        pad_idx = pad_idx.cuda()
        outputs = outputs.cuda()

    outputs.index_fill_(1, pad_idx, 1)
    outputs[active_seq_idx] = samples

    return outputs
