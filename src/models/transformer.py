import numpy as np
import torch
import torch.nn as nn

from src.vocab import constants
from src.beam import Beam
from src.utils.common import variable
from src.utils.gumbel import gumbel_softmax

from .modules import Encoder, Decoder
from .helpers import *
from .layers import Linear


use_cuda = torch.cuda.is_available()


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
        beams = [Beam(beam_size) for _ in range(batch_size)]
        beam_seq_idx_map = {beam_idx: seq_idx for seq_idx, beam_idx in enumerate(range(batch_size))}

        # Here we are going to keep state of each decoder layer
        # So we are not recompute everything from scratch
        cache = init_cache(batch_size, beam_size, self.decoder.n_layers, self.d_model)

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
        embs = self.decoder.tgt_word_emb if use_trg_embs_in_encoder else None
        enc_output, *_ = self.encoder(src_seq, embs=embs)

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

            active_translations = update_layer_outputs(translations, active_seq_idxs, batch_size, volatile=False)

            # Remove source sentences that are completely translated
            src_seq = update_active_seq(src_seq, active_seq_idxs, n_remaining_sents)
            enc_output = update_layer_outputs(enc_output, active_seq_idxs, n_remaining_sents, volatile=False)

            # Remove decoder states which are not active anymore
            if i != 0: cache = [update_layer_outputs(c, active_seq_idxs, n_remaining_sents, volatile=False) for c in cache]

            # -- Decoding -- #
            embs = self.encoder.src_word_emb if use_src_embs_in_decoder else None
            proj = self.src_word_proj if use_src_embs_in_decoder else self.tgt_word_proj

            dec_output = self.decoder(active_translations, src_seq, enc_output, embs=embs, cache=cache, one_hot_trg=True)
            logits = proj(dec_output[:, -1, :]) # We need only the last word
            samples_one_hot = gumbel_softmax(logits, temperature)
            samples = extend_inactive_with_pads(samples_one_hot, active_seq_idxs, batch_size)
            translations = torch.cat((translations, samples.unsqueeze(1)), 1)

            n_remaining_sents = active_seq_idxs.size(0)

        #- Return translations
        return translations
