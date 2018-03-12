import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
from torch.autograd import Variable

from src.utils.common import variable
from .base_trainer import BaseTrainer


use_cuda = torch.cuda.is_available()


class LMTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, vocab, config):
        super(LMTrainer, self).__init__(config)
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
        self.loss_history = []

    def train_on_batch(self, seqs):
        predictions = self.model(seqs[:, :-1])
        loss = self.criterion(predictions.view(-1, len(self.vocab)), seqs[:, 1:].contiguous().view(-1))
        self.loss_history.append(loss.data[0])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_scores(self):
        clear_output(True)
        plt.figure(figsize=[10,5])
        plt.title("Batch loss history")
        plt.plot(self.loss_history)
        plt.plot(pd.DataFrame(np.array(self.loss_history)).ewm(span=100).mean())
        plt.grid()
        plt.show()

    def train_mode(self):
        self.model.train()
