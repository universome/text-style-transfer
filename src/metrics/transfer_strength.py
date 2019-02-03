import os
import pickle
from typing import List

import torch
import torch.nn as nn
from torchtext.data import Field, Dataset, Example, BucketIterator
import numpy as np
from firelab.config import Config

from src.models import RNNClassifier


class TransferStrength:
    def __init__(self, config:Config):
        self.config = config
        self.load_vocab()
        self.load_model()

    def score(self, sentences: List[str]) -> np.ndarray:
        results = []

        fields = [('data', self.field)]
        examples = [Example.fromlist([s], fields) for s in sentences]
        dataset = Dataset(examples, fields)
        dataloader = BucketIterator(
            dataset, self.config.metrics.classifier.batch_size, repeat=False,
            shuffle=False, device=self.config.device_name)

        for batch in dataloader:
            scores = self.model(batch.data)
            scores = torch.sigmoid(scores)
            scores = scores.detach().cpu().numpy().tolist()
            results.extend(scores)

        return np.mean(results)

    def load_model(self):
        model_path = os.path.join(self.config.firelab.project_path,
                                  self.config.metrics.classifier.model_path)

        self.model = RNNClassifier(self.config.metrics.classifier.hp.model_size, self.field.vocab)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.config.device_name)

    def load_vocab(self):
        vocab_path = os.path.join(self.config.firelab.project_path,
                                  self.config.metrics.classifier.vocab_path)

        self.field = Field(init_token='<bos>', eos_token='<eos>', batch_first=True)
        self.field.vocab = pickle.load(open(vocab_path, 'rb'))
