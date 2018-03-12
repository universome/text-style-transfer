from src.utils.data_utils import * # TODO: be specific
from src.vocab import constants
from .base_dataloader import BaseDataloader


class Batcher(BaseDataloader):
    def __init__(
            self, src_seqs, tgt_seqs, src_word2idx, tgt_word2idx,
            batch_size=32, shuffle=False, parallel=True):

        assert batch_size > 0
        assert len(src_seqs) >= batch_size
        assert (len(src_seqs) == len(tgt_seqs)) or not parallel

        self._src_seqs = src_seqs
        self._tgt_seqs = tgt_seqs

        self._n_batch = int(np.ceil(min(len(src_seqs), len(tgt_seqs)) / batch_size))
        self._batch_size = batch_size
        self._iter_count = 0
        self._should_shuffle = shuffle
        self._parallel = parallel

        if self._should_shuffle:
            self.shuffle()

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        if self._parallel:
            paired_insts = list(zip(self._src_seqs, self._tgt_seqs))
            random.shuffle(paired_insts)

            self._src_seqs, self._tgt_seqs = zip(*paired_insts)
        else:
            random.shuffle(self._src_seqs)
            random.shuffle(self._trg_seqs)

    def step(self):
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
