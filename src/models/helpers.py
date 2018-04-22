import numpy as np
import torch
from torch.autograd import Variable

from src.vocab import constants


use_cuda = torch.cuda.is_available()


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
    positions = Variable(torch.LongTensor(positions), requires_grad=False)
    if use_cuda: positions = positions.cuda()

    return positions


def get_positions_for_seq(seq, **kwargs):
    return [token_to_position(token, i, **kwargs) for i, token in enumerate(seq.data)]


def token_to_position(token, index, one_hot_input=False):
    if type(token) == Variable: token = token.data
    if one_hot_input: token = np.argmax(token)

    return 0 if token == constants.PAD else (index+1)


def update_active_seq(seq_var, active_seq_idxs, n_remaining_sents, volatile=True):
    '''Remove the src sequence of finished sequences from the batch. '''

    seq_idx_dim_size, *rest_dim_sizes = seq_var.size()
    seq_idx_dim_size = seq_idx_dim_size * len(active_seq_idxs) // n_remaining_sents
    new_size = (seq_idx_dim_size, *rest_dim_sizes)

    # select the active sequences in batch
    original_seq_data = seq_var.data.view(n_remaining_sents, -1)
    active_seq_data = original_seq_data.index_select(0, active_seq_idxs)
    active_seq_data = active_seq_data.view(*new_size)

    return Variable(active_seq_data, volatile=volatile)


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

    return Variable(active_enc_info_data, volatile=volatile)


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


def extend_inactive_with_pads(samples, active_seq_idx, batch_size):
    assert samples.dim() == 2
    assert samples.size(0) == len(active_seq_idx)

    # We should distribute our samples according their indices (active_seq_idx)
    # And other staff we should fill with constants.PAD (one-hotted)
    pad_idx = Variable(torch.LongTensor([constants.PAD]), requires_grad=False)
    outputs = Variable(torch.zeros(batch_size, samples.size(1)))

    if use_cuda:
        pad_idx = pad_idx.cuda()
        outputs = outputs.cuda()

    outputs.index_fill_(1, pad_idx, 1)
    outputs[active_seq_idx] = samples

    return outputs
