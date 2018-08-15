#!/usr/bin/env python

import sys

import nltk
from tqdm import tqdm

nltk.download('punkt')


def tokenize(input_file, output_file):
    texts = open(input_file).read().splitlines()

    with open(output_file, 'w') as f:
        for text in tqdm(texts):
            for sent in nltk.sent_tokenize(text):
                sent_tok = ' '.join(nltk.word_tokenize(sent))
                f.write(sent_tok + '\n')


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    tokenize(input_file, output_file)
