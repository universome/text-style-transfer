import torch
import torch.nn as nn
from firelab.utils.training_utils import cudable


def cross_entropy_without_pads(vocab):
    "CE with zero weight for PAD token"
    weight = torch.ones(len(vocab))
    weight[vocab.stoi['<pad>']] = 0

    return cudable(nn.CrossEntropyLoss(weight))
