import random
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from src.vocab import constants
from src.utils.common import variable
from src.utils.constants import SPECIAL_TOKENS


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


def remove_spec_symbols(ids_seqs):
    spec_symbols = set([constants.PAD, constants.BOS, constants.EOS])
    return [[t for t in s if not t in spec_symbols] for s in ids_seqs]


def pad_to_longest(seqs, volatile=False):
    ''' Pads the instance to the max seq length in batch '''
    max_len = max(len(seq) for seq in seqs)

    padded_seqs = np.array([seq + [constants.PAD] * (max_len - len(seq)) for seq in seqs])
    padded_seqs = variable(torch.LongTensor(padded_seqs), volatile=volatile)

    return padded_seqs


def itos_many(seqs, vocab):
    """
    Converts sequences of token ids to normal strings
    """

    sents = [[vocab.itos[i] for i in seq] for seq in seqs]
    sents = [[t for t in s if not t in SPECIAL_TOKENS] for s in sents]
    sents = [' '.join(s).replace('@@ ', '') for s in sents]

    return sents
