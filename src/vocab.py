import numpy as np


class Vocab:
    """
    Vocab converts between strings, tokens and token indices.
    It should normally be treated as immutable
    It should be saved with pickle.dump
    Here's a tour to what it does: http://bit.ly/2BDupuH
    """
    _default_tokens = ("__BOS__", "__EOS__", "__UNK__")

    def __init__(self, tokens):
        tokens = tuple(tokens)
        assert len(tokens) == len(set(tokens)), "tokens must be unique"
        for i, t in enumerate(self._default_tokens):
            assert t in tokens and tokens.index(t) == i, "token must have %s at index %i" % (t, i)

        self.tokens = tokens
        self.token2id = {token: i for i, token in enumerate(self.tokens)}
        self.bos = 0
        self.eos = 1
        self.unk = 2

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, item):
        return item in self.token2id

    def tokenize(self, sentence, separator=' '):
        """ Converts sentence into a sequence of ids """
        if isinstance(sentence, str):
            sentence = sentence.split(separator)

        sentence = list(sentence)
        if "__EOS__" not in sentence:
            sentence.append("__EOS__")
        if sentence[0] != "__BOS__":
            sentence.insert(0, "__BOS__")

        return [self.token2id.get(token, self.unk) for token in sentence]

    def detokenize(self, indices, crop=True, sep=' ', unbpe=False, deprocess=False):
        """ converts indices to words. If separator is not None, joins them over it """
        indices = tuple(indices)
        if self.eos in indices:
            indices = indices[:indices.index(self.eos) + 1]

        tokens = [self.tokens[token] for token in indices]
        if deprocess:
            tokens = [t for t in tokens if t not in self._default_tokens]

        if sep is None:
            return tokens
        else:
            line = sep.join(tokens)
            if unbpe:
                line = self.remove_bpe(line)
            return line

    def tokenize_many(self, lines, max_len=None, sep=' '):
        """
        convert variable length token sequences into fixed size matrix
        pads short sequences with self.EOS
        example usage:
        >>>sentences = ... # a list of strings
        >>>vocab = Vocab.from_sequences(sentences)
        >>>print(vocab.tokenize_many(sentences[:3]))
        [[15 22 21 28 27 13  1  1  1  1  1]
         [30 21 15 15 21 14 28 27 13  1  1]
         [25 37 31 34 21 20 37 21 28 19 13]]
        """
        max_len = max_len or max(map(lambda s: len(s.split(sep)), lines)) + 2  # 2 for bos and eos

        matrix = np.zeros((len(lines), max_len), dtype='int32') + self.eos
        for i, seq in enumerate(lines):
            tokens = self.tokenize(seq)[:max_len]
            matrix[i, :len(tokens)] = tokens

        return matrix

    def detokenize_many(self, matrix, crop=True, sep=' ', unbpe=False, deprocess=False):
        """
        Convert matrix of token ids into strings
        :param matrix: matrix of tokens of int32, shape=[batch,time]
        :param crop: if True, crops BOS and EOS from line
        :param sep: if not None, joins tokens over that string
        :param unbpe: if True, merges BPE into words
        :param deprocess: if True, removes all unknowns
        :return: a list of strings of
        """
        return [self.detokenize(sent, crop, sep, unbpe, deprocess) for sent in matrix]

    @classmethod
    def from_file(cls, voc_path):
        """ Parses vocab from a .voc file """
        tokens = set()
        with open(voc_path, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.split(" ")[0]
                tokens.update([token])

        return Vocab(list(cls._default_tokens) + sorted(tokens))

    @staticmethod
    def remove_bpe(s):
        return s.replace('@@ ', '').replace("&apos;", "'").replace(" '", "'")  # TODO: make this generic

    @classmethod
    def remove_bpe_many(cls, sentences):
        return [cls.remove_bpe(s) for s in sentences]

    @classmethod
    def deprocess(cls, tokens, sep=' '):
        tokens = [t for t in tokens if t not in cls._default_tokens]
        return tokens if sep is None else sep.join(tokens)

    @classmethod
    def deprocess_many(cls, sentences, sep=' '):
        return [cls.deprocess(s, sep) for s in sentences]

    @classmethod
    def from_sequences(cls, sentences, separator=' '):
        """ Infers tokens from a corpora of sentences (tokens separated by separator) """
        tokens = set()
        for s in sentences:
            tokens.update(s.split(separator))
        return Vocab(list(cls._default_tokens) + sorted(tokens))

    @classmethod
    def merge(cls, first, *others):
        """
        Constructs vocab out of several different vocabulatries.
        Maintains existing token ids by first vocab
        """
        for vocab in (first,) + others:
            assert isinstance(vocab, Vocab)

        # get all tokens from others that are not in first
        other_tokens = set()
        for vocab in others:
            other_tokens.update(set(vocab.tokens))

        # inplace substract first.tokens from other_tokens
        other_tokens.difference_update(set(first.tokens))

        return Vocab(first.tokens + tuple(sorted(other_tokens)))
