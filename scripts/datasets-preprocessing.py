#!/usr/bin/env python

import os
import re
import sys
import random
from typing import List
from collections import Counter

random.seed(42)

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import read_corpus, save_corpus

SPEC_DASH = '–' # Medium dash char. It is different from (normal) dash "-".
# DIALOG_SEPARATORS = ('–', '—') # medium / long dashes (they are different symbols, although looks the same in IDE)

def filter_direct_speech(s):
    parts = s.split(SPEC_DASH)
    s =  ' '.join([parts[i] for i in range(len(parts)) if i % 2 == 1]).strip()
    s = s if s[-1] != ',' else s[:-1] + '.' # Replacing last ',' with '.'

    return s


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
    subs = read_corpus(input_data_path)
    pattern = re.compile('^(?:[A-z]|[А-я]|[ёЁ\d\s.,!:?\-––\'"%$()`])+$')
    print('Filtering...')
    filtered = [s for s in tqdm(subs) if pattern.match(s)]
    print('Removing too long sentences...')
    short = [s for s in tqdm(filtered) if len(s.split()) <= 50 and len(s) <= 250]
    print('Saving...')
    save_corpus(short, output_data_path)
    print('Done!')


def filter_dialogs_in_classics(input_data_path: str, output_data_path: str):
    print('Reading data...')
    classics = read_corpus(input_data_path)

    print('Finding dialogs...')
    dialogs = [s for s in tqdm(classics) if s.strip().startswith(SPEC_DASH)]

    print('Removing markup chars from dialogs...')
    dialogs = [s.replace('\xa0', ' ') for s in tqdm(dialogs)]

    print('Removing degenerate lines...')
    dialogs = [s for s in tqdm(dialogs) if s != SPEC_DASH]

    print('Filtering direct speech...')
    dialogs = [filter_direct_speech(s) for s in tqdm(dialogs)]

    print('Saving...')
    save_corpus(dialogs, output_data_path)
    print('Done!')


def dialogs_from_lines(input_data_path:str, output_data_path:str, n_lines: int, eos:str, n_dialogs:int):
    n_lines, n_dialogs = int(n_lines), int(n_dialogs) # TODO: argparse?

    print('Reading data...')
    lines = read_corpus(input_data_path)
    lines = lines[:n_dialogs * n_lines]

    print('Generating dialogs')
    dialogs = [lines[i:i+n_lines] for i in range(0, len(lines) - n_lines)]
    dialogs = [eos.join(d) for d in dialogs]

    print('Saving corpus')
    save_corpus(dialogs, output_data_path)
    print('Done!')


def generate_sentiment_words(neg_input_path:str, pos_input_path:str,
                             neg_output_path:str, pos_output_path:str,
                             keep_n_most_popular_words:int=3000):
    print('Reading data...')
    neg_lines = read_corpus(neg_input_path)
    pos_lines = read_corpus(pos_input_path)

    print('Counting words')
    neg_counter = Counter([w.lower() for s in tqdm(neg_lines) for w in s.split()])
    pos_counter = Counter([w.lower() for s in tqdm(pos_lines) for w in s.split()])

    print('Getting most popular')
    neg_top_words = set(w for w, _ in neg_counter.most_common(keep_n_most_popular_words))
    pos_top_words = set(w for w, _ in pos_counter.most_common(keep_n_most_popular_words))

    only_neg_top_words = neg_top_words - pos_top_words
    only_pos_top_words = pos_top_words - neg_top_words

    print('Saving')
    save_corpus(list(only_neg_top_words), neg_output_path)
    save_corpus(list(only_pos_top_words), pos_output_path)
    print('Done!')


def cut_long_dialogs(data_path:str, output_path:str, max_len:int=128):
    print('Reading data...')
    data = read_corpus(data_path)

    print('Splitting dialogs...')
    dialogs = [d.split('|') for d in data]
    print('Cutting data...')
    dialogs = [cut_dialog(d, max_len) for d in tqdm(dialogs)]
    num_dialogs_before = len(dialogs)

    print('Saving data...')
    dialogs = list(filter(len, dialogs))
    save_corpus(dialogs, output_path)
    print('Done! Num dialogs reduced: {} -> {}'.format(num_dialogs_before, len(dialogs)))


def cut_dialog(dialog:List[str], max_len:int) -> str:
    dialog = [cut_line(s, max_len) for s in dialog]
    dialog = dialog if all(dialog) else []
    dialog = '|'.join(dialog) + '|'

    return dialog


def cut_line(line:str, max_len:int) -> str:
    if len(line) <= max_len: return line

    ending_tokens = ('.', '!', '?')
    end = max([line.rfind(t) for t in ending_tokens])

    if end == -1:
        print('Going to empty line:', line)

    return '' if end == -1 else line[:end + 1]


def extract_dialogs_texts(data_dir:str, out_path:str):
    text_names = os.listdir(os.path.join(data_dir))

    with open(out_path, 'w') as out_file:
        for text_name in tqdm(text_names):
            text = open(os.path.join(data_dir, text_name)).read().splitlines()
            dialogs = extract_dialogs(text)

            for d in dialogs:
                out_file.write(d + '\n')


def extract_dialogs(corpus:List[str]) -> List[str]:
    dialogs = []
    curr_dialog_start = 0
    dialog_is_active = False

    for i, line in enumerate(corpus):
        if line.strip().startswith(SPEC_DASH):
            if dialog_is_active:
                continue
            else:
                # Activate dialog
                curr_dialog_start = i
                dialog_is_active = True
        else:
            if dialog_is_active:
                # Stopping dialog
                dialog = corpus[curr_dialog_start:i]
                dialog = [filter_direct_speech(s) for s in dialog]

                # We separate dialog lines with pipes so we can separate dialogs with "\n"
                # It's better than separating dialogs with double "\n\n"
                dialog = '|'.join(dialog)
                dialogs.append(dialog)
                dialog_is_active = False
            else:
                continue

    return dialogs


def restructure_taiga_proza_ru_texts(texts_path:str, out_dir:str):
    years = os.listdir(texts_path)

    for year in years:
        for month in tqdm(os.listdir(os.path.join())):
            dir = os.path.join(texts_path, year, month)

            for text_name in os.listdir(dir):
                path_from = os.path.join(dir, text_name)
                path_to = os.path.join(out_dir, text_name)
                os.rename(path_from, path_to)

    print('Done!')


def main(cmd:str, *args):
    if cmd == 'subs-open-nmt':
        prepare_subs_for_open_nmt(*args)
    elif cmd == 'word-filling':
        prepare_word_filling(*args)
    elif cmd == 'filter-subs':
        filter_subs(*args)
    elif cmd == 'filter-dialogs-in-classics':
        filter_dialogs_in_classics(*args)
    elif cmd == 'dialogs-from-lines':
        dialogs_from_lines(*args)
    elif cmd == 'generate-sentiment-words':
        generate_sentiment_words(*args)
    elif cmd == 'cut-long-dialogs':
        cut_long_dialogs(*args)
    elif cmd == 'restructure-taiga-proza-ru-texts':
        restructure_taiga_proza_ru_texts(*args)
    elif cmd == 'extract-dialogs-texts':
        extract_dialogs_texts(*args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main(sys.argv[1], *sys.argv[2:])
