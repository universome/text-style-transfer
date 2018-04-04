import numpy as np
import torch
import torch.nn as nn

from src.vocab import constants
from src.beam import Beam
from src.utils.common import variable

from .modules import Encoder, Decoder
from .helpers import *
from .layers import Linear


use_cuda = torch.cuda.is_available()


class TransformerWithTwoDecoders(nn.Module):
    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
            dropout=0.1):

        super(TransformerWithTwoDecoders, self).__init__()
        self.d_model = d_model
        self.encoder = Encoder(
            n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)
        
        init_decoder = lambda: Decoder(
            n_tgt_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.decoder_src = init_decoder()
        self.decoder_trg = init_decoder()
        
        self.src_word_proj = Linear(d_model, n_src_vocab, bias=False) # For decoder_src
        self.tgt_word_proj = Linear(d_model, n_tgt_vocab, bias=False) # For decoder_trg

        # We use LogSoftmax instead of Softmax to avoid
        # numerical issues when computing beam scores
        self.logits_to_logprobs = nn.LogSoftmax(dim=1)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module output shall be the same.'

    def trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed = set(map(id, self.encoder.position_enc.parameters()))
        dec_src_freezed = set(map(id, self.decoder_src.position_enc.parameters()))
        dec_trg_freezed = set(map(id, self.decoder_trg.position_enc.parameters()))
        freezed = enc_freezed | dec_src_freezed | dec_trg_freezed

        return (p for p in self.parameters() if id(p) not in freezed)

    def forward(self, src, trg, domain):
        assert domain in ['src', 'trg']

        trg = trg[:, :-1]
        decoder = self.decoder_src if domain == 'src' else self.decoder_trg
        proj = self.src_word_proj if domain == 'src' else self.tgt_word_proj

        enc_output, *_ = self.encoder(src)
        dec_output = decoder(trg, src, enc_output)
        seq_logit = proj(dec_output)

        out = seq_logit.view(-1, seq_logit.size(2))

        return out

    def translate_batch(self, src_seq, domain, beam_size=6, max_len=50):
        assert domain in ['src', 'trg']

        batch_size = src_seq.size(0)
        
        # Choosing corresponding decoder
        decoder = self.decoder_src if domain == 'src' else self.decoder_trg
        proj = self.src_word_proj if domain == 'src' else self.tgt_word_proj

        #- Enocde
        enc_output, *_ = self.encoder(src_seq)

        # Repeating seqs for beam search
        src_seq = repeat_seq_for_beam(src_seq, beam_size)
        enc_output = repeat_seq_for_beam(enc_output, beam_size)

        # Prepare beams
        beams = [Beam(beam_size) for _ in range(batch_size)]
        beam_seq_idx_map = {beam_idx: seq_idx for seq_idx, beam_idx in enumerate(range(batch_size))}

        # Here we are going to keep state of each decoder layer
        # So we are not recompute everything from scratch
        cache = init_cache(batch_size, beam_size, decoder.n_layers, self.d_model)

        n_remaining_sents = batch_size

        for i in range(max_len-1):
            len_dec_seq = i + 1

            # -- Preparing decoded data seq -- #
            # TODO: cache it?
            # size: batch x beam x seq
            dec_partial_seq = torch.stack([b.get_current_state() for b in beams if not b.done])
            # size: (batch * beam) x seq
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            dec_partial_seq = variable(dec_partial_seq, volatile=True)

            # -- Decoding -- #
            dec_output = decoder(dec_partial_seq, src_seq, enc_output, cache=cache)
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

        translations = extract_best_translation_from_beams(beams)
        
        return translations
