import random

import numpy as np

from .base_dataloader import BaseDataloader
from src.utils.data_utils import pad_to_longest
from src.vocab import constants


DROP_WORD = '__DROP__'


class WordRecoveryDataloader(BaseDataloader):
    """
        We want to reconstruct the missing word from the sentence.
        Sentence is a BPEed string.
    """
    def __init__(self, seqs, seqs_to_mix, vocab,
                 mixing_coef_fn=None, batch_size=32, shuffle=False):
        """
        :param mixing_coef_fn: function which produces mixing coef value
        """
        assert batch_size > 0
        assert len(seqs) >= batch_size and len(seqs_to_mix) >= batch_size

        self._seqs = seqs
        self._seqs_to_mix = seqs_to_mix

        self._n_batch = int(np.ceil(min(len(seqs), len(seqs_to_mix))) / batch_size)
        self._batch_size = batch_size
        self._iter_count = 0
        self._should_shuffle = shuffle
        self.vocab = vocab
        self._mixing_coef_fn = mixing_coef_fn if mixing_coef_fn else lambda: 0.5

        if self._should_shuffle:
            self.shuffle()

    def shuffle(self):
        random.shuffle(self._seqs)
        random.shuffle(self._seqs_to_mix)

    def step(self):
        batch_idx = self._iter_count
        self._iter_count += 1

        start_idx = batch_idx * self._batch_size
        end_idx = (batch_idx + 1) * self._batch_size

        main_seqs = self._seqs[start_idx:end_idx]
        seqs_to_mix = self._seqs_to_mix[start_idx:end_idx]

        # Choosing appropriate amount of each style
        main_seqs, seqs_to_mix = sample_seqs(main_seqs, seqs_to_mix, self._mixing_coef_fn())

        # Dropping random words from sequences
        main_seqs, main_seqs_trg = drop_random_word_many(main_seqs)
        seqs_to_mix, seqs_to_mix_trg = drop_random_word_many(seqs_to_mix)

        # Tokenizing sequences
        main_seqs = [[self.vocab.token2id[t] for t in s.split()] for s in main_seqs]
        main_seqs_trg = [[self.vocab.token2id[t] for t in s.split()] for s in main_seqs_trg]
        seqs_to_mix = [[self.vocab.token2id[t] for t in s.split()] for s in seqs_to_mix]
        seqs_to_mix_trg = [[self.vocab.token2id[t] for t in s.split()] for s in seqs_to_mix_trg]

        # Adding BOS and EOS
        main_seqs = [[constants.BOS] + s + [constants.EOS] for s in main_seqs]
        main_seqs_trg = [[constants.BOS] + s + [constants.EOS] for s in main_seqs_trg]
        seqs_to_mix = [[constants.BOS] + s + [constants.EOS] for s in seqs_to_mix]
        seqs_to_mix_trg = [[constants.BOS] + s + [constants.EOS] for s in seqs_to_mix_trg]

        return main_seqs, seqs_to_mix, main_seqs_trg, seqs_to_mix_trg


def drop_random_word(seq):
    words = group_bpes_into_words(seq)
    i = random.choice(range(len(words)))

    sentence = ' '.join(words[:i] + [DROP_WORD] + words[i+1:])

    return sentence, words[i]


def drop_random_word_many(seqs):
    seqs, dropped_words = zip(*[drop_random_word(s) for s in seqs])

    return list(seqs), list(dropped_words)


def group_bpes_into_words(sentence:str):
    """
    We are given a sentence of BPEs
    and here we split it into words
    :param sentence: sentence to split into words
    """
    groups = []
    curr_group = []

    for bpe in sentence.split():
        if bpe[-2:] == '@@':
            curr_group.append(bpe)
            continue
        
        curr_group.append(bpe)
        groups.append(' '.join(curr_group))
        curr_group = []

    # Sentence can't be finished with "prefixed" BPE, like "wh@@";
    # only with "terminating" BPE, like "ary" or "isible"
    assert len(curr_group) == 0

    return groups


def sample_seqs(main_seqs:list, seqs_to_mix:list, mixing_coef:float):
    """
    Takes (1-q)*len main sequences and q*len auxiliary ones.
    """
    # We do not know beforehand how much sentences of each type
    # we'll have to use to generate the batch
    assert 0 <= mixing_coef <= 1
    assert len(main_seqs) == len(seqs_to_mix)

    # Calculating how much sentences of each type to use
    batch_size = len(main_seqs)
    num_seqs_to_mix = int(mixing_coef * batch_size)
    num_main_seqs_to_keep = batch_size - num_seqs_to_mix

    # Choosing appropriate sentences
    main_seqs = random.sample(main_seqs, num_main_seqs_to_keep)
    seqs_to_mix = random.sample(seqs_to_mix, num_seqs_to_mix)

    return main_seqs, seqs_to_mix