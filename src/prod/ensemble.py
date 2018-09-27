from typing import List

import torch
import torch.nn as nn


class WeightedEnsemble(nn.Module):
    "Simple weighted ensemble"
    def __init__(self, models:List[nn.Module], weights:List[float]):
        assert len(models) == len(weights)

        self.models = nn.ModuleList(models)
        self.weights = weights

    def forward(self, *args, **kwargs):
        predictions = [m(*args, **kwargs) for m in self.models]
        predictions = [w * p for w,p in zip(self.weights, predictions)]
        predictions = sum(predictions)

        return predictions
