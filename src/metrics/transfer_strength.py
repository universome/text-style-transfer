import os
import pickle
from typing import List

import torch
import torch.nn as nn
from torchtext.data import Field, Dataset, Example, BucketIterator
import numpy as np
from firelab.config import Config

from src.models import RNNClassifer


class TransferStrength:
    def __init__(self, config:Config):
        self.config = config
        self.load_model()
        self.load_vocab()

    def score(self, sentences: List[str]) -> np.ndarray:
        results = []

        fields = [('data', self.Field)]
        examples = [Example.fromlist([s], fields) for s in sentences]
        dataset = Dataset(examples, fields)
        dataloader = BucketIterator(
            self.val_ds, self.config.classifier_metric.batch_size, repeat=False, shuffle=False)

        for batch in dataloader:
            scores = dataloader(batch.data.to(self.config.device_name))
            scores = torch.sigmoid(scores)
            scores = scores.detach().cpu().numpy().tolist()
            results.extend(scores)

        return scores

    def load_model(self):
        model_path = os.path.join(self.config.firelab.project_path,
                                  self.config.classifier_metric.model_path)

        self.model = RNNClassifer(self.config.classifier_metric.hp)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.config.device_name)

    def load_vocab(self):
        vocab_path = os.path.join(self.config.firelab.project_path,
                                  self.config.classifier_metric.vocab_path)

        self.field = Field(init_token='<bos>', eos_token='<eos>', batch_first=True)
        self.field.vocab = pickle.load(vocab_path, 'rb')
