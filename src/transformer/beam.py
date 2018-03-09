""" Manage beam search info structure.

    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

from time import time

import torch
import numpy as np
import transformer.constants as constants

class Beam(object):
    ''' Store the neccesary info for beam search. '''

    def __init__(self, size, cuda=False):
        
        assert size > 0

        self.size = size
        self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        # Here we keep which beams from the previous time-step we used
        # to continue the current beam
        self.beam_ancestors = []

        # The outputs at each time-step.
        self.next_words = [self.tt.LongTensor(size).fill_(constants.PAD)]
        self.next_words[0][0] = constants.BOS

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.beam_ancestors[-1]

    def advance(self, word_lk):
        "Update the status and check for finished or not."
        vocab_size = word_lk.size(1)

        # Sum the previous scores.
        if len(self.beam_ancestors) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0] # Taking seq, which we started with BOS

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_ids = flat_beam_lk.topk(self.size)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_ids is flattened [beam_size x vocab_size] array,
        # so calculate which word and beam each best score came from
        prev_k = best_scores_ids / vocab_size
        self.beam_ancestors.append(prev_k)
        self.next_words.append(best_scores_ids - prev_k * vocab_size)

        # End condition is when top-of-beam is EOS.
        if self.next_words[-1][0] == constants.EOS:
            self.done = True
            self.all_scores.append(self.scores)

        return self.done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[0], ids[0]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_words) == 1:
            dec_seq = self.next_words[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[constants.BOS] + h for h in hyps]
            dec_seq = torch.from_numpy(np.array(hyps))

        return dec_seq

    def get_hypothesis(self, k):
        """
        Walk back to construct the full hypothesis.

        Parameters:
            * `k` - the position in the beam to construct.
        Returns:
            * The hypothesis
        """
        hyp = []
        for j in range(len(self.beam_ancestors) - 1, -1, -1):
            hyp.append(self.next_words[j+1][k])
            k = self.beam_ancestors[j][k]

        hyp = hyp[::-1] # reversing

        return hyp
