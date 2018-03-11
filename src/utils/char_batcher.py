from src.utils.data_utils import * # TODO: be specific
import transformer.constants as constants


class CharBatcher(object):
    def __init__(self, seqs, batch_size=16, shuffle=False):
        assert batch_size > 0
        assert len(seqs) >= batch_size
        
        self._seqs = seqs
        self._n_batch = int(np.ceil(len(seqs)) / batch_size)
        self._batch_size = batch_size
        self._iter_count = 0
        self._should_shuffle = shuffle

        if self._should_shuffle: random.shuffle(self._seqs)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            seqs = self._seqs[start_idx:end_idx]
            seqs = [[constants.BOS] + s + [constants.EOS] for s in seqs]
            seqs = pad_to_longest(seqs)

            return seqs
        else:
            if self._should_shuffle: random.shuffle(self._src_seqs)

            self._iter_count = 0
            raise StopIteration()
