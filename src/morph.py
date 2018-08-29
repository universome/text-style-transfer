#!/usr/bin/env python

from typing import List

import numpy as np
from pymorphy2 import MorphAnalyzer

from src.utils.data_utils import itos_many


morph = MorphAnalyzer()

_MORPH_ATTRS = [
    'animacy', 'aspect', 'case', 'gender', 'involvement', 'mood',
    'number', 'POS', 'person', 'tense', 'transitivity',  'voice',
]

# Hardcoding possible morphological features
# to preserve order from run to run (yes, it differs)
_MORPH_ATTRS_VALS = {
    'animacy': ['anim', 'inan', None],
    'aspect': ['perf', 'impf', None],
    'case': ['gen1', 'loc1', 'gent', 'voct', 'nomn', 'acc2', 'accs', 'datv', 'loc2', 'loct', 'ablt', 'gen2', None],
    'gender': ['masc', 'femn', 'neut', None],
    'involvement': ['excl', 'incl', None],
    'mood': ['indc', 'impr', None],
    'number': ['sing', 'plur', None],
    'POS': ['GRND', 'ADVB', 'PRTF', 'NOUN', 'PRED', 'ADJF', 'PRTS', 'CONJ', 'INTJ', 'PREP', 'NPRO', 'INFN', 'ADJS', 'COMP', 'PRCL', 'NUMR', 'VERB', None],
    'person': ['2per', '3per', '1per', None],
    'tense': ['futr', 'past', 'pres', None],
    'transitivity': ['tran', 'intr', None],
    'voice': ['pssv', 'actv', None]
}

MORPHS_SIZE = sum(len(v) for v in _MORPH_ATTRS_VALS.values())


def morph_chars_idx(chars_idx, vocab):
    words = itos_many(chars_idx, vocab, sep='')
    out = [word_to_onehot_features(w) for w in words]

    return np.stack(out)


def word_to_onehot_features(word):
    morph_info = morph.parse(word)[0].tag
    tag_vals = [getattr(morph_info, a) for a in _MORPH_ATTRS]
    onehot = morph_info_to_onehot(tag_vals)

    return onehot


def morph_info_to_onehot(x: List[str]):
    # Each dimension is a categorical variable of specific len
    # We first convert each variable into one-hot, then concatenate them
    onehots = []

    for a, v in zip(_MORPH_ATTRS, x):
        onehot = np.zeros(len(_MORPH_ATTRS_VALS[a]), dtype=int)
        onehot[_MORPH_ATTRS_VALS[a].index(v)] = 1
        onehots.append(onehot)

    return np.concatenate(onehots)
