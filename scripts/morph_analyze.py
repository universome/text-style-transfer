#!/usr/bin/env python

import sys
from typing import List

import numpy as np
from pymorphy2 import MorphAnalyzer
from tqdm import tqdm

from utils import read_corpus


morph = MorphAnalyzer()


def _prepare_morph_attrs_vals():
    names = [
        'ANIMACY',
        'ASPECTS',
        'CASES',
        'GENDERS',
        'INVOLVEMENT',
        'MOODS',
        'NUMBERS',
        'PARTS_OF_SPEECH',
        'PERSONS',
        'TENSES',
        'TRANSITIVITY',
        'VOICES',
    ]

    vals = {}

    for a, c in zip(MORPH_ATTRS, names):
        vals[a] = list(getattr(morph.TagClass, c))
        vals[a].append(None)

    return vals


MORPH_ATTRS = [
    'animacy',
    'aspect',
    'case',
    'gender',
    'involvement',
    'mood',
    'number',
    'POS',
    'person',
    'tense',
    'transitivity',
    'voice'
]
MORPH_ATTRS_VALS = _prepare_morph_attrs_vals()


def main(input_path: str, output_path: str):
    words = read_corpus(input_path)
    result = []

    for word in tqdm(words):
        morph_info = morph.parse(word)[0].tag
        tag_vals = [getattr(morph_info, a) for a in MORPH_ATTRS]
        onehot = morph_info_to_onehot(tag_vals)
        result.append(onehot)

    np.save(output_path, np.stack(result))


def morph_info_to_onehot(x: List[str]):
    # Each dimension is a categorical variable of specific len
    # We first convert each variable into one-hot, then concatenate them
    onehots = []

    for a, v in zip(MORPH_ATTRS, x):
        onehot = np.zeros(len(MORPH_ATTRS_VALS[a]), dtype=int)
        onehot[MORPH_ATTRS_VALS[a].index(v)] = 1
        onehots.append(onehot)

    return np.concatenate(onehots)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
