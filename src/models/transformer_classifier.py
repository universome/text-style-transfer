from itertools import chain

import torch
import torch.nn as nn

from . import Transformer
from src.utils.common import variable, one_hot_seqs_to_seqs


class TransformerClassifier(nn.Module):
    '''
        We use Transformer as a classifier the following way:
        Encoder is normal, and decoder always produces only 1 token
        which is the corresponding class.
    '''
    def __init__(self, vocab_size, num_classes, max_len, **transformer_kwargs):
        super(TransformerClassifier, self).__init__()

        # We set n_tgt_vocab=4 to Transformer because we only have just 1 BOS token
        # We can't pass n_tgt_vocab=1 (although it's enough) becuase of legacy code
        # (we are passing constants.PAD there for padding_idx, and it's index is 3)
        self.transformer = Transformer(vocab_size, 4, max_len, **transformer_kwargs)
        self.num_classes = num_classes
        self.dec_out_to_logits = nn.Linear(transformer_kwargs['d_model'], num_classes-1)

    def get_trainable_parameters(self):
        return chain(self.transformer.get_trainable_parameters(), self.dec_out_to_logits.parameters())

    def forward(self, x, use_trg_embs_in_encoder=False, one_hot_input=False):
        pseudo_trg_seq = variable(torch.zeros(x.size(0)).unsqueeze(1).long())

        encoder_embs = self.decoder.tgt_word_emb if use_trg_embs_in_encoder else None
        src_seq = one_hot_seqs_to_seqs(x) if one_hot_input else x

        enc_output, *_ = self.transformer.encoder(x, embs=encoder_embs, one_hot_src=one_hot_input)
        dec_output = self.transformer.decoder(pseudo_trg_seq, src_seq, enc_output).squeeze(1)
        logits = self.dec_out_to_logits(dec_output).squeeze()

        return logits
