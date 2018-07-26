import torch
import torch.nn as nn
from firelab.utils import cudable

from src.vocab import constants


def cross_entropy_without_pads(vocab):
    ''' With PAD token zero weight '''
    weight = cudable(torch.ones(len(vocab)))
    weight[vocab.stoi['<pad>']] = 0

    return nn.CrossEntropyLoss(weight)
