#!/usr/bin/env python

import sys

import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm

from utils import read_corpus, save_corpus

nltk.download('punkt')
detokenizer = TreebankWordDetokenizer()


def detokenize(input_file, output_file):
    print('Reading data...')
    texts = read_corpus(input_file)
    print('Data reading finished!')

    print('Detokenizing...')
    detokenized = [detokenizer.detokenize(s.split()) for s in tqdm(texts)]
    print('Detokenized!')

    print('Saving...')
    save_corpus(detokenized, output_file)
    print('Saved!')


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    detokenize(input_file, output_file)
