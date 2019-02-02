import os
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from firelab.config import Config
from firelab.training_utils import cudable


INFERSENT_PARAMS = {
    'bsize': 64,
    'word_emb_dim': 300,
    'enc_lstm_dim': 2048,
    'pool_type': 'max',
    'dpout_model': 0.0,
    'version': 2
}
VOCAB_SIZE = 100000


class ContentPreservation:
    """
    Metric to measure content preservation when doing style transfer.
    Works by comparing cosine distances of InferSent sentence embeddings.
    """
    def __init__(self, config: Config):
        self.config = config
        self.model = self.load_infersent_model()

    def load_infersent_model(self) -> nn.Module:
        # TODO: This module should not know such high-level info...
        project_path = self.config.firelab.project_path
        repo_path = os.path.join(project_path, self.config.infersent.repo_path)
        model_pkl_path = os.path.join(project_path, self.config.infersent.model_pkl_path)
        fasttext_w2v_path = os.path.join(project_path, self.config.infersent.fasttext_w2v_path)

        # TODO: Is there any other way to load model class?
        sys.path.append(os.path.dirname(repo_path))
        from infersent.models import InferSent

        print('Loading InferSent model', end='')
        model = InferSent(INFERSENT_PARAMS).to(self.config.device_name)
        model.load_state_dict(torch.load(model_pkl_path))
        print('Done!')

        print('Loading fastText embeddings', end='')
        model.set_w2v_path(fasttext_w2v_path)
        print('Done!')

        print('Building vocab...', end='')
        model.build_vocab_k_words(K=VOCAB_SIZE)
        print('Done!')

        return model

    def score(self, predicted: List[str], targets: List[str]):
        """
        Compares two arrays of sentences by cosine distance
        between their sentence embeddings.
        """
        assert len(predicted) == len(targets)

        embeddings_pred = self.model.encode(predicted, tokenize=True)
        embeddings_trg = self.model.encode(targets, tokenize=True)

        return 1 - F.cosine_similarity(embeddings_pred, embeddings_trg)
