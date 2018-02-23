import random
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import transformer.constants as constants

use_cuda = torch.cuda.is_available()


def load_embeddings(embeddings_path):
    embeddings = {}

    with open(embeddings_path, 'r', encoding='utf-8') as f:
        next(f) # Skipping first line, because it's header info
        for line in tqdm(f):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            embeddings[word] = np.asarray(values[1:], dtype='float32')

    return embeddings


def init_emb_matrix(emb_matrix, emb_dict, token2id):
    emb_size = emb_matrix.size(1)

    for word, idx in token2id.items():
        if not word in emb_dict:
            # print('Skipping ', word)
            continue

        emb_matrix[idx] = torch.FloatTensor(emb_dict[word])
        
        
def token_ids_to_sents(token_ids, vocab):
    if type(token_ids) == Variable: token_ids = token_ids.data.cpu()
    sents = remove_pads(token_ids)

    for s in sents:
        if s[0] != constants.BOS: s.insert(0, constants.BOS)
        if s[-1] != constants.EOS: s.append(constants.EOS)
            
    sents = [vocab.remove_bpe(vocab.detokenize(s)) for s in sents]
    sents = [' '.join(s.split()) for s in sents]

    return sents


def remove_pads(ids_seqs):
    return [[t for t in s if t != constants.PAD] for s in ids_seqs]
    

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


def pad_to_longest(seqs):
    ''' Pads the instance to the max seq length in batch '''
    max_len = max(len(seq) for seq in seqs)

    padded_seqs = np.array([seq + [constants.PAD] * (max_len - len(seq)) for seq in seqs])
    padded_seqs = Variable(torch.LongTensor(padded_seqs))

    if use_cuda: padded_seqs = padded_seqs.cuda()

    return padded_seqs
