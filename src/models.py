from itertools import chain

import torch
import torch.nn as nn
from torch.autograd import Variable

from src.transformer.models import * # TODO: move helper functions somewhere
from src.utils.common import one_hot_seqs_to_seqs, variable
import src.transformer.constants as constants


use_cuda = torch.cuda.is_available()


class FFN(nn.Module):
    def __init__(self, input_size, n_hid_layers, hid_size=1024):
        super(FFN, self).__init__()

        self.n_hid_layers = n_hid_layers
        self.hid_size = hid_size
        self.input_size = input_size

        self.nn = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.input_size, self.hid_size),
            nn.ReLU(),
            self._init_hidden(),
            nn.Linear(self.hid_size, 1),
            nn.Sigmoid()
        )

    def _init_hidden(self):
        layers = []

        for _ in range(self.n_hid_layers):
            layers += [
                nn.Dropout(0.3),
                nn.Linear(self.hid_size, self.hid_size),
                nn.ReLU()
            ]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


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
