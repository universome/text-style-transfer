import random
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

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


def remove_special_symbols(sentences):
    return [[t for t in s if not t in SPECIAL_TOKENS] for s in sentences]


def itos_many(seqs, vocab, remove_special=True, sep=' '):
    """
    Converts sequences of token ids to normal strings
    """

    sents = [[vocab.itos[i] for i in seq] for seq in seqs]

    if remove_special:
        sents = remove_special_symbols(sents)

    sents = [sep.join(s).replace('@@ ', '') for s in sents]

    return sents


def stoi_many(sentences, vocab, sep=' '):
    if sep == '':
        return [[vocab.stoi[t] for t in s.split(sep)] for s in sentences]
    else:
        return [[vocab.stoi[t] for t in s.split(sep)] for s in sentences]


def split(s, sep):
    return list(s) if sep == '' else s.split(sep)


def char_tokenize(s):
    "Splits string into chars. We need it here to make pickle work :|"
    return split(s, '')
