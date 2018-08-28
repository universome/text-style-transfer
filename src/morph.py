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

_ATTRS_TO_SET_NAMES = [
    'ANIMACY', 'ASPECTS', 'CASES', 'GENDERS', 'INVOLVEMENT', 'MOODS',
    'NUMBERS', 'PARTS_OF_SPEECH', 'PERSONS', 'TENSES', 'TRANSITIVITY', 'VOICES',
]

_MORPH_ATTRS_VALS = {}
for a, c in zip(_MORPH_ATTRS, _ATTRS_TO_SET_NAMES):
    _MORPH_ATTRS_VALS[a] = list(getattr(morph.TagClass, c))
    _MORPH_ATTRS_VALS[a].append(None)

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
