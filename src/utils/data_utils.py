import math
import random

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from src.utils.constants import SPECIAL_TOKENS, PREFIXES


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


def k_middle_chars(word: str, k: int) -> str:
    "Extracts k middle chars from the word"
    if len(word) <= k: return word

    left_part = word[:round(len(word) / 2)]
    right_part = word[round(len(word) / 2):] # Right part is always longer or equal

    return left_part[-(k // 2):] + right_part[: (k-1) // 2 + 1]


def word_base(word: str, k: int) -> str:
    "Extracts word base for ru lang veeeery heuristically :|"
    for p in PREFIXES:
        if word.startswith(p):
            return word[len(p) : len(p) + k]

    return word[:k]


def repeat_str_to_longest(str_x, str_y, eos=None):
    """
    Takes two strings (str_x and str_y) and repeats smaller one such
    that both strings have equal lengths
    """
    if len(str_x) < len(str_y):
        str_x = repeat_str_to_size(str_x, len(str_y), eos)
    elif len(str_x) > len(str_y):
        str_y = repeat_str_to_size(str_y, len(str_x), eos)

    return str_x, str_y


def repeat_str_to_size(s, size, eos=''):
    if len(s) >= size: return s

    num_repeats = (size - len(s)) // len(s)
    s = eos.join([s for _ in range(num_repeats)])

    assert (size - len(s)) // len(s) == 1

    if len(s) != size:
        s = s + eos + s[:size-len(s)]

    return s


def split_in_batches(s:str, size:int):
    "Splits string into batches of length `size`"
    return [s[i*size : (i+1)*size] for i in range(math.ceil(len(s) / size))]
