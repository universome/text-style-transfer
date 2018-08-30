#!/usr/bin/env python

import  re
import sys
from typing import List

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import read_corpus, save_corpus


def prepare_subs_for_open_nmt(data_path):
    print('Preparing subs for open-nmt')
    print('Reading data...')
    data = read_corpus(data_path)
    # data = [s for s in data if 5 <= len(s.split()) <= 20] # Removing noise
    src = data[:-1]
    trg = data[1:]

    print('Splitting into train/val...')
    splits = train_test_split(src, trg, test_size=5000, random_state=42)
    src_train, src_val, trg_train, trg_val = splits

    print('Saving...')
    save_corpus(src_train, data_path + '.open-nmt.train.src')
    save_corpus(trg_train, data_path + '.open-nmt.train.trg')
    save_corpus(src_val, data_path + '.open-nmt.val.src')
    save_corpus(trg_val, data_path + '.open-nmt.val.trg')

    print('Done!')


def prepare_word_filling(corpus_data_path: str, words_data_path: str):
    print('Preparing word filling')
    DROP = '__DROP__'
    CONTEXT_SIZE = 3 # Limiting context for char-level encoder

    print('Reading data...')
    lines = read_corpus(corpus_data_path)
    words = set(read_corpus(words_data_path))
    src, trg = [], []

    for s in tqdm(lines):
        tokens = s.split()

        for i, t in enumerate(tokens):
            if not (t in words): continue

            context_left = tokens[max(i - CONTEXT_SIZE, 0) : i]
            context_right = tokens[i + 1 : i + CONTEXT_SIZE + 1]
            src.append(' '.join(context_left + [DROP] + context_right))
            trg.append(t)

    print('Saving...')
    save_corpus(src, 'data/generated/classics-word-filling.src')
    save_corpus(trg, 'data/generated/classics-word-filling.trg')

    print('Done!')


def filter_subs(input_data_path: str, output_data_path: str):
    print('Reading data...')
    subs = open(input_data_path).read().splitlines()
    pattern = re.compile('^(?:[A-z]|[А-я]|[ёЁ\d\s.,!:?\-––\'"%$()`])+$')
    print('Filtering...')
    filtered = [s for s in tqdm(subs) if pattern.match(s)]
    print('Removing too long sentences...')
    short = [s for s in tqdm(filtered) if len(s.split()) <= 50 and len(s) <= 250]
    print('Saving...')
    save_corpus(short, output_data_path)
    print('Done!')


def main(cmd, *args):
    if cmd == 'subs-open-nmt':
        prepare_subs_for_open_nmt(*args)
    elif cmd == 'word-filling':
        prepare_word_filling(*args)
    elif cmd == 'filter-subs':
        filter_subs(*args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main(sys.argv[1], *sys.argv[2:])
