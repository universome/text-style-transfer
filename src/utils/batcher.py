from src.utils.data_utils import * # TODO: be specific
import transformer.constants as constants


class Batcher(object):
    def __init__(
            self, src_seqs, tgt_seqs, src_word2idx, tgt_word2idx,
            batch_size=64, shuffle=False):

        assert batch_size > 0
        assert len(src_seqs) >= batch_size
        assert len(src_seqs) == len(tgt_seqs)

        self._n_batch = int(np.ceil(len(src_seqs) / batch_size))

        self._batch_size = batch_size

        self._src_seqs = src_seqs
        self._tgt_seqs = tgt_seqs

        self._iter_count = 0

        self._need_shuffle = shuffle

        if self._need_shuffle:
            self.shuffle()

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        if self._tgt_seqs:
            paired_insts = list(zip(self._src_seqs, self._tgt_seqs))
            random.shuffle(paired_insts)
            self._src_seqs, self._tgt_seqs = zip(*paired_insts)
        else:
            random.shuffle(self._src_seqs)

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

            src_seqs = self._src_seqs[start_idx:end_idx]
            tgt_seqs = self._tgt_seqs[start_idx:end_idx]

            src_seqs = [[constants.BOS] + seq + [constants.EOS] for seq in src_seqs]
            tgt_seqs = [[constants.BOS] + seq + [constants.EOS] for seq in tgt_seqs]

            src_data = pad_to_longest(src_seqs)
            tgt_data = pad_to_longest(tgt_seqs)

            return src_data, tgt_data
        else:
            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()
