import os
from os import path
import random
from itertools import islice

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np; np.random.seed(42)
from tqdm import tqdm
from hasheroku import hasheroku

from src.models import Transformer
from src.dataloaders import WordRecoveryDataloader, OneSidedDataloader
from src.vocab import constants
from src.trainers import WordRecoveryTrainer
from src.utils.data_utils import token_ids_to_sents, pad_to_longest
from src.utils.training_utils import cross_entropy_without_pads
from src.vocab import Vocab, constants
from src.dataloaders.word_recovery_dataloader import drop_each_word, group_bpes_into_words


class WordRecoveryRunner:
    def __init__(self, config, results_dir):
        self.training_config = config.get('training_config', {})
        self.data_path = config['data_path']
        self.results_dir = results_dir

        print('Results will be saved into', results_dir)

    def load_data(self, data_path):
        # We set rather large min_len, because our BPEs are very short
        min_len = 50
        max_len = 150

        classics_path = os.path.join(data_path, 'classics.tok.bpe')
        news_path = os.path.join(data_path, 'news.ru.tok.bpe')

        #classics = open(classics_path, 'r', encoding='utf-8').read().splitlines()
        #news = open(news_path, 'r', encoding='utf-8').read().splitlines()
        with open(classics_path, encoding='utf-8') as classics_file, open(news_path, encoding='utf-8') as news_file:
            classics = list(islice(classics_file, 10**4))
            news = list(islice(news_file, 10**4))

        classics = [s for s in classics if min_len < len(s.split()) < (max_len - 2)]
        news = [s for s in news if min_len < len(s.split()) < (max_len - 2)]

        # There are sentences which do not contain a single normal
        # (i.e. fully alphabetic) word. Let's remove them.
        classics = [s for s in classics if any([w.isalpha() for w in s.replace('@@ ', '').split()])]
        news = [s for s in news if any([w.isalpha() for w in s.replace('@@ ', '').split()])]

        vocab = Vocab.from_sequences(classics + news)
        vocab.token2id['__DROP__'] = len(vocab)
        vocab.tokens.append('__DROP__')

        np.random.shuffle(classics)
        np.random.shuffle(news)

        classics, classics_val = classics[:-1000], classics[-1000:]
        news, news_val = news[:-1000], news[-1000:]

        return classics, classics_val, news, news_val, vocab

    def evaluate(self):
        sentences = self.val_data[5:10]
        sentences = [s for s in sentences if len(s.split()) < max_len]
        results = []

        self.transformer.eval()

        for sentence in tqdm(sentences):
            seqs, _, words_idx = drop_each_word(sentence)
            seqs_idx = [[vocab.token2id[t] for t in s.split()] for s in seqs]
            seqs_idx = [[constants.BOS] + s + [constants.EOS] for s in seqs_idx]
            seqs_idx = pad_to_longest(seqs_idx, volatile=True)
            predictions = self.transformer.translate_batch(seqs_idx, beam_size=6, max_len=20)
            predictions = token_ids_to_sents(predictions, vocab)

            # Now we should construct initial sentence with some care,
            # because not each word was dropped, but only normal ones (fully alphabetical)
            words = group_bpes_into_words(sentence)
            for i, w in enumerate(predictions): words[words_idx[i]] = w
            sentence = ' '.join(words)

            results.append(sentence)

        for i in range(len(sentences)):
            print('Source: ', sentences[i].replace('@@ ', ''))
            print('Result: ', results[i].replace('@@ ', ''))
            print()

    def train(self):
        max_len = 150
        model_path = path.join(self.results_dir, '/word_recovery.pth')
        classics, classics_val, news, news_val, vocab = self.load_data(self.data_path)

        # Let's clean log file
        if os.path.exists(log_file_path): os.remove(log_file_path)

        self.transformer = Transformer(len(vocab), len(vocab), max_len)

        if resume_training:
            self.transformer.load_state_dict(torch.load(model_path))

        lr = 1e-5 if resume_training else 1e-4
        optimizer = Adam(self.transformer.get_trainable_parameters(), lr=lr)
        criterion = cross_entropy_without_pads(len(vocab))

        trainer = WordRecoveryTrainer(transformer, optimizer, criterion, vocab, config)
        training_data = WordRecoveryDataloader(classics, news, vocab, trainer.mixing_coef, batch_size=32, shuffle=True)
        val_data = WordRecoveryDataloader(classics_val[:512], news_val[:512], vocab, batch_size=512)

        # Making validation data deterministic
        val_data = np.array(next(val_data)).transpose().tolist()
        val_data = OneSidedDataloader(val_data, batch_size=32, unpack=True, pad=False)

        trainer.run_training(training_data, val_data)

        torch.save(self.transformer.state_dict(), model_path)

    def start(self):
        self.train()
        self.evaluate()
