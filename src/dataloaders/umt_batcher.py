import random

import numpy as np
import torch
from torch.autograd import Variable

from src.seq_noise import seq_noise_many
from src.utils.data_utils import pad_to_longest
from src.vocab import constants

use_cuda = torch.cuda.is_available()


class UMTBatcher(object):
    def __init__(self, src, trg, vocab_src, vocab_trg,
                 batch_size=32, shuffle=False, seq_noise_options={}):
        """
        This shuffler is supposed to be used for Unsupervised Machine Translation
        The logic here is different from normal batcher,
        because we need to train a discriminator and apply noise for autoencoding task.

        Arguments:
            - src, trg — lists of lists of tokens
            - token2id_src, token2id_trg — dicts, which maps token to id
            - batch_size
            - shuffle
        """

        assert batch_size > 0
        assert len(src) >= batch_size and len(trg) >= batch_size

        self._n_batch = int(np.ceil(min(len(src), len(trg)) / batch_size))

        self._batch_size = batch_size

        self._src = src
        self._trg = trg

        self._vocab_src = vocab_src
        self._vocab_trg = vocab_trg

        self._iter_count = 0
        self._should_shuffle = shuffle
        self._seq_noise_options = seq_noise_options

        if self._should_shuffle:
            self.shuffle()

    def shuffle(self):
        # TODO(universome): pass seed as an argument and disable it by default
        random.seed(42)

        random.shuffle(self._src)
        random.shuffle(self._trg)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            src = self._src[start_idx:end_idx]
            trg = self._trg[start_idx:end_idx]

            src_noised = seq_noise_many(src, **self._seq_noise_options)
            trg_noised = seq_noise_many(trg, **self._seq_noise_options)

            src = [[constants.BOS] + seq + [constants.EOS] for seq in src]
            trg = [[constants.BOS] + seq + [constants.EOS] for seq in trg]
            src_noised = [[constants.BOS] + seq + [constants.EOS] for seq in src_noised]
            trg_noised = [[constants.BOS] + seq + [constants.EOS] for seq in trg_noised]

            # This is the main data for DAE
            src_noised = pad_to_longest(src_noised)
            trg_noised = pad_to_longest(trg_noised)
            src = pad_to_longest(src)
            trg = pad_to_longest(trg)

            self._iter_count += 1

            return src_noised, trg_noised, src, trg
        else:
            # Whoa, we have done one epoch! That's cool. Let's reset.
            if self._should_shuffle: self.shuffle()
            self._iter_count = 0

            raise StopIteration()
