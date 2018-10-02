#!/usr/bin/env python

"""
This is the dirtiest file in the whole project
It contains nasty scripts of generating examples for some datasets
and will not be used in production
"""

import sys
import time
import argparse

import numpy as np
from tqdm import tqdm

from news import NEWS
from dialog_model import build_predict_fn
from dialog_model import predict as generate_dialog
# from dialog_model import predict_next_word as predict_next_word
from style_model import predict as restyle

N_LINES = 7
titles = [n['title'] for n in NEWS]
AVAILABLE_SCHEMES = ['just-on-titles', 'from-golds', 'with-style', 'next-words', 'score-models']

parser = argparse.ArgumentParser()
parser.add_argument('--scheme', '-s', dest='scheme', type=str,
    help='What kind of examples we should generate?', choices=AVAILABLE_SCHEMES)
parser.add_argument('--output_file_path', type=str,
    help='Where should we save results?', dest='output_file_path')


def generate_articles_from_news(news_list):
    "Articles is just a list of all sentences in somewhat good format"
    articles = [[s + '.' for p in news['text'].splitlines() for s in p.split('. ')] for news in news_list]

    for article in articles:
        article[-1] = article[-1][:-1] # Removing last dot from double dot

    return articles

articles = generate_articles_from_news(NEWS)
temperatures = [0.0001, 0.01, 0.1, 0.25, 0.5, 1., 2., 5.]


def main(scheme, output_file_path=None):
    if scheme in {'just-on-titles', 'from-golds', 'with-style'}:
        generate_dialogs(scheme, generate_dialog, output_file_path)
    elif scheme == 'next-words':
        predict_next_words(scheme, predict_next_word, output_file_path)
    elif scheme == 'score-models':
        score_models()
    else:
        raise NotImplementedError


def score_models():
    MODELS_DIRS = [
        ('conditional_lm', 'ConditionalLM', {'style': 1}),
        ('char_lm_from_embs', 'CharLMFromEmbs', {}),
        ('subs_lm', 'RNNLM', {}),
        ('fine_tuned_classic_lm', 'RNNLM', {}),
        ('classic_lm', 'RNNLM', {}),
    ]

    for dir, model_cls_name, kwargs in MODELS_DIRS:
        predict_fn = build_predict_fn(dir, model_cls_name, kwargs)
        output_file_path = 'examples/%s.txt' % dir
        generate_dialogs('just-on-titles', predict_fn, output_file_path)


def generate_dialogs(scheme, predict_fn, output_file_path):
    with open(output_file_path, 'w') as f:
        for t in tqdm(temperatures):
            start_time = time.time()

            if scheme == 'just-on-titles':
                dialogs = generate_on_titles(t, predict_fn)
            elif scheme == 'from-golds':
                dialogs = generate_from_golds(t, predict_fn)
            elif scheme == 'with-style':
                dialogs = generate_on_titles(t, predict_fn)
            else:
                raise NotImplementedError

            elapsed = time.time() - start_time

            f.write('\n\n====== Temperature: {}. (Took seconds: {:.03f}) ======\n\n'.format(t, elapsed))

            for dialog in dialogs:
                for line in dialog:
                    assert type(line) is str
                    f.write(line + '\n')
                f.write('---------------------------------\n')

            if scheme == 'with-style':
                restyled = [restyle(d, predict_fn) for d in dialogs]

                f.write('\n <= RESTYLINGS => \n')

                for dialog in restyled:
                    for line in dialog:
                        assert type(line) is str
                        f.write(line + '\n')
                    f.write('---------------------------------\n')

            f.flush()


def predict_next_words(scheme, predict_fn, output_file_path):
    with open(output_file_path, 'w') as f:
        sentences = [s for a in articles for s in a]
        predicted = continue_sentences(sentences, predict_fn)

        for s,p in zip(sentences, predicted):
            f.write(s + ' => ' + p + '\n')
    print('Done!')


def generate_on_titles(t, predict_fn):
    dialogs = predict_fn(titles, N_LINES, t)
    dialogs = [[l['text'] for l in d] for d in dialogs]

    return dialogs


def generate_from_golds(t, predict_fn):
    dialogs = []

    for article in articles:
        dialog = [article[0]]

        for i in range(1, len(article)):
            condition = '|'.join(article[:i])
            predicted = predict_fn([condition], 1, t)
            next_line = predicted[0][-1]['text']
            dialog.append(next_line)

        dialogs.append(dialog)

    return dialogs


def continue_sentences(sentences, predict_fn):
    # sentences = [s for a in articles for s in a]
    n_words = [len(s.split()) for s in sentences]

    max_len = max(n_words)
    sentences = [s.split() for s in sentences]
    sentences = [pad_sentence(s, max_len, 'PAD') for s in sentences]
    predictions = []

    for i in tqdm(range(1, max_len - 1)):
        conditions = [' '.join(s[:i]) for s in sentences]
        next_words = predict_fn(conditions)
        print('Next words', next_words)
        predictions.append(next_words)


    predictions = np.array(predictions).T.tolist()
    predictions = [p[:n] for p,n in zip(predictions, n_words)]
    predictions = [[s[0]] + p for s,p in zip(sentences, predictions)] # Adding the first word
    predictions = [' '.join(p) for p in predictions]

    print('Predictions:')
    for p in predictions:
        print(p)

    return predictions


def pad_sentence(s, size, word):
    if len(s) == size: return s

    return s + [word for _ in range(size - len(s))]


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.scheme, args.output_file_path)
