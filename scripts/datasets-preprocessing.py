#!/usr/bin/env python

import sys
from typing import List

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def prepare_subs_for_open_nmt(data_path):
    print('Preparing subs for open-nmt')
    print('Reading data...')
    data = read_corpus(data_path)
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

    print('Reading data...')
    lines = read_corpus(corpus_data_path)
    words = set(read_corpus(words_data_path))
    src, trg = [], []

    for s in tqdm(lines):
        tokens = s.split()

        for i, t in enumerate(tokens):
            if not (t in words): continue

            src.append(' '.join(tokens[:i] + [DROP] + tokens[i+1:]))
            trg.append(t)

    print('Saving...')
    save_corpus(src, 'data/generated/classics-dropped-words.src')
    save_corpus(trg, 'data/generated/classics-dropped-words.trg')

    print('Done!')


def read_corpus(data_path: str) -> List[str]:
    with open(data_path) as f:
        lines = f.read().splitlines()

    return lines


def save_corpus(corpus: List[str], path: str):
    with open(path, 'w') as f:
        for line in corpus:
            f.write(line + '\n')


def main(cmd, *args):
    if cmd == 'subs-open-nmt':
        prepare_subs_for_open_nmt(*args)
    elif cmd == 'word-filling':
        prepare_word_filling(*args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main(sys.argv[1], *sys.argv[2:])
