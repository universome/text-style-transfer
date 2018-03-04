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
        if not word in emb_dict: continue

        emb_matrix[idx] = torch.FloatTensor(emb_dict[word])


def token_ids_to_sents(token_ids, vocab):
    if type(token_ids) == Variable: token_ids = token_ids.data.cpu()
    sents = remove_spec_symbols(token_ids)
    sents = [vocab.remove_bpe(vocab.detokenize(s)) for s in sents]
    sents = [' '.join(s.split()) for s in sents]

    return sents


def remove_spec_symbols(ids_seqs):
    spec_symbols = set([constants.PAD, constants.BOS, constants.EOS])
    return [[t for t in s if not t in spec_symbols] for s in ids_seqs]


def pad_to_longest(seqs):
    ''' Pads the instance to the max seq length in batch '''
    max_len = max(len(seq) for seq in seqs)

    padded_seqs = np.array([seq + [constants.PAD] * (max_len - len(seq)) for seq in seqs])
    padded_seqs = Variable(torch.LongTensor(padded_seqs))

    if use_cuda: padded_seqs = padded_seqs.cuda()

    return padded_seqs
