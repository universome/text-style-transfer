import random

import numpy as np

from src.utils.data_utils import pad_to_longest
from src.vocab import constants
from .base_dataloader import BaseDataloader


class OneSidedDataloader(BaseDataloader):
    """
        This dataloader loads batches for a single corpus (not two parallel ones)
        It is used to train language models.
    """
    def __init__(self, seqs, batch_size=16, shuffle=False):
        assert batch_size > 0
        assert len(seqs) >= batch_size

        self._seqs = seqs
        self._n_batch = int(np.ceil(len(seqs)) / batch_size)
        self._batch_size = batch_size
        self._iter_count = 0
        self._should_shuffle = shuffle

        if self._should_shuffle:
            self.shuffle()

    def shuffle(self):
        random.shuffle(self._seqs)

    def step(self):
        batch_idx = self._iter_count
        self._iter_count += 1

        start_idx = batch_idx * self._batch_size
        end_idx = (batch_idx + 1) * self._batch_size

        seqs = self._seqs[start_idx:end_idx]
        seqs = [[constants.BOS] + s + [constants.EOS] for s in seqs]
        seqs = pad_to_longest(seqs)

        return seqs

