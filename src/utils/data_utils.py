import random
import numpy as np
import torch
from torch.autograd import Variable
import transformer.constants as constants

use_cuda = torch.cuda.is_available()

class Batcher(object):
    def __init__(
            self, src_insts, tgt_insts, src_word2idx, tgt_word2idx,
            batch_size=64, shuffle=False):

        assert batch_size > 0
        assert len(src_insts) >= batch_size
        assert len(src_insts) == len(tgt_insts)

        self._n_batch = int(np.ceil(len(src_insts) / batch_size))

        self._batch_size = batch_size

        self._src_insts = src_insts
        self._tgt_insts = tgt_insts

        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}

        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word

        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word

        self._iter_count = 0

        self._need_shuffle = shuffle

        if self._need_shuffle:
            self.shuffle()

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        if self._tgt_insts:
            paired_insts = list(zip(self._src_insts, self._tgt_insts))
            random.shuffle(paired_insts)
            self._src_insts, self._tgt_insts = zip(*paired_insts)
        else:
            random.shuffle(self._src_insts)


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

            src_insts = self._src_insts[start_idx:end_idx]
            src_data = pad_to_longest(src_insts)

            tgt_insts = self._tgt_insts[start_idx:end_idx]
            tgt_data = pad_to_longest(tgt_insts)

            return src_data, tgt_data
        else:
            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()

def pad_to_longest(sentences):
    ''' Pad the instance to the max seq length in batch '''
    max_len = max(len(sentence) for sentence in sentences)
    seqs = np.array([seq + [constants.PAD] * (max_len - len(seq)) for seq in sentences])
    seqs = Variable(torch.LongTensor(seqs))

    if use_cuda: seqs = seqs.cuda()

    return seqs
