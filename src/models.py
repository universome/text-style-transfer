import torch
import torch.nn as nn
from torch.autograd import Variable

from src.transformer.models import Transformer
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
        self.dec_out_to_logits = nn.Linear(transformer_kwargs['d_model'], num_classes)

    def forward(self, x, use_trg_embs_in_encoder=False, one_hot_input=False):
        pseudo_trg = Variable(torch.LongTensor([[0] for _ in range(x.size(0))]))
        if use_cuda: pseudo_trg.cuda()

        encoder_embs = self.decoder.tgt_word_emb if use_trg_embs_in_encoder else None

        enc_output, *_ = self.transformer.encoder(x, embs=encoder_embs, one_hot_input=one_hot_input)
        dec_output = self.transformer.decoder(pseudo_trg, x, enc_output, one_hot_input=one_hot_input).squeeze(1)
        logits = self.dec_out_to_logits(dec_output)

        return logits
