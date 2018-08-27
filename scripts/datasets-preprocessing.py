#!/usr/bin/env python

import sys
from sklearn.model_selection import train_test_split


def prepare_subs_for_open_nmt(data_path):
    print('Reading data...')
    data = open(data_path).read().splitlines()
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


def save_corpus(corpus, path):
    with open(path, 'w') as f:
        for line in corpus:
            f.write(line + '\n')


def main(cmd, *args):
    if cmd == 'subs-open-nmt':
        prepare_subs_for_open_nmt(*args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
