import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
from torch.autograd import Variable

from src.utils.common import variable
from src.utils.data_utils import pad_to_longest, token_ids_to_sents
from .base_trainer import BaseTrainer
from .helpers import compute_param_by_scheme


use_cuda = torch.cuda.is_available()


class WordRecoveryTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, vocab, config):
        super(WordRecoveryTrainer, self).__init__(config)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.vocab = vocab

        if use_cuda:
            self.model.cuda()
            self.criterion.cuda()

        self.num_iters_done = 0
        self.num_epochs_done = 0
        self.max_num_epochs = config.get('max_num_epochs', 0)
        self.mixing_scheme = config.get('mixing_scheme', (1,1,0))
        self.max_len = config.get('max_len', 20)
        self.beam_size = config.get('beam_size', 1)
        self.loss_history = []

        self.val_scores = {
            'main_seqs_accuracy': [],
            'seqs_to_mix_accuracy': []
        }

    def train_on_batch(self, batch):
        main_seqs, seqs_to_mix, main_seqs_trg, seqs_to_mix_trg = batch

        seqs = main_seqs + seqs_to_mix
        trg = main_seqs_trg + seqs_to_mix_trg

        seqs = pad_to_longest(seqs)
        trg = pad_to_longest(trg)

        predictions = self.model(seqs, trg)
        loss = self.criterion(predictions, trg[:, 1:].contiguous().view(-1))
        self.loss_history.append(loss.data[0])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_scores(self):
        clear_output(True)
        plt.figure(figsize=[16,8])

        # Plot training score
        plt.subplot(211)
        plt.title('Batch loss history')
        plt.plot(self.loss_history)
        plt.plot(pd.DataFrame(self.loss_history).ewm(span=300).mean())
        plt.grid()

        # Plot validation score
        num_vals = len(self.val_scores['main_seqs_accuracy'])
        if num_vals > 0:
            val_iters = np.arange(num_vals) * (self.num_iters_done / num_vals)
            plt.subplot(212)
            plt.title('Validation accuracy')
            plt.plot(val_iters, self.val_scores['main_seqs_accuracy'], label='Accuracy on main corpus')
            plt.plot(val_iters, self.val_scores['seqs_to_mix_accuracy'], label='Accuracy on mixing corpus')
            plt.legend()
            plt.grid()

        plt.show()

    def validate(self, val_data, return_results=False):
        self.eval_mode()

        all_main_seqs = []
        all_seqs_to_mix = []
        all_main_seqs_preds = []
        all_seqs_to_mix_preds = []

        for batch in val_data:
            main_seqs, seqs_to_mix, main_seqs_trg, seqs_to_mix_trg = batch

            # Converting strings to token indices
            main_seqs_idx = pad_to_longest(main_seqs, volatile=True)
            seqs_to_mix_idx = pad_to_longest(seqs_to_mix, volatile=True)

            main_seqs_preds = self.model.translate_batch(
                main_seqs_idx, max_len=self.max_len, beam_size=self.beam_size)
            seqs_to_mix_preds = self.model.translate_batch(
                seqs_to_mix_idx, max_len=self.max_len, beam_size=self.beam_size)

            # Converting predicted token indices to strings
            main_seqs = token_ids_to_sents(main_seqs, self.vocab)
            seqs_to_mix = token_ids_to_sents(seqs_to_mix, self.vocab)
            main_seqs_trg = token_ids_to_sents(main_seqs_trg, self.vocab)
            seqs_to_mix_trg = token_ids_to_sents(seqs_to_mix_trg, self.vocab)
            main_seqs_preds = token_ids_to_sents(main_seqs_preds, self.vocab)
            seqs_to_mix_preds = token_ids_to_sents(seqs_to_mix_preds, self.vocab)

            # Let's construct full sentences
            main_seqs_preds = [s.replace('__DROP__', main_seqs_preds[i]) for i, s in enumerate(main_seqs)]
            seqs_to_mix_preds = [s.replace('__DROP__', seqs_to_mix_preds[i]) for i, s in enumerate(seqs_to_mix)]
            main_seqs = [s.replace('__DROP__', main_seqs_trg[i]) for i, s in enumerate(main_seqs)]
            seqs_to_mix = [s.replace('__DROP__', seqs_to_mix_trg[i]) for i, s in enumerate(seqs_to_mix)]

            # Let's get normal sentences instead of BPEs
            main_seqs = [s.replace('@@ ', '') for s in main_seqs]
            seqs_to_mix = [s.replace('@@ ', '') for s in seqs_to_mix]
            main_seqs_preds = [s.replace('@@ ', '') for s in main_seqs_preds]
            seqs_to_mix_preds = [s.replace('@@ ', '') for s in seqs_to_mix_preds]

            # Saving results
            all_main_seqs += main_seqs
            all_seqs_to_mix += seqs_to_mix
            all_main_seqs_preds += main_seqs_preds
            all_seqs_to_mix_preds += seqs_to_mix_preds
            
        num_main_seqs_hits = sum([all_main_seqs[i] == all_main_seqs_preds[i] for i in range(len(all_main_seqs))])
        num_seqs_to_mix_hits = sum([all_seqs_to_mix[i] == all_seqs_to_mix_preds[i] for i in range(len(all_seqs_to_mix))])

        main_seqs_acc = num_main_seqs_hits / len(all_main_seqs)
        seqs_to_mix_acc = num_seqs_to_mix_hits / len(all_seqs_to_mix)

        scores = (main_seqs_acc, seqs_to_mix_acc)
        
        if self.log_file:
            self.log(scores, all_main_seqs + all_seqs_to_mix, all_main_seqs_preds + all_seqs_to_mix_preds)

        if return_results:
            predictions = {
                'main_seqs_src': all_main_seqs,
                'seqs_to_mix_src': all_seqs_to_mix,
                'main_seqs_preds': all_main_seqs_preds,
                'seqs_to_mix_preds': all_seqs_to_mix_preds
            }

            return scores, predictions
        else:
            self.val_scores['main_seqs_accuracy'].append(main_seqs_acc)
            self.val_scores['seqs_to_mix_accuracy'].append(seqs_to_mix_acc)

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def mixing_coef(self):
        """
        This coef determines the proportion of seqs_to_mix to be added
        """
        return compute_param_by_scheme(self.mixing_scheme, self.num_iters_done)

    def log(self, scores, src, preds):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write('Epochs done: {}. Iters done: {}\n'.format(self.num_epochs_done, self.num_iters_done))
            f.write('[main_seqs] Accuracy: {:.04f}. [seqs_to_mix] Accuracy: {:.04f}.\n'.format(scores[0], scores[1]))

            for i in range(len(src)):
                f.write('Source:     ' + src[i] + '\n')
                f.write('Prediction: ' + preds[i] + '\n')

            f.write('=======================================================================================\n')
