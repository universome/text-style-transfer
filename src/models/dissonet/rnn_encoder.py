import torch.nn as nn

from src.models.layers import Dropword


class RNNEncoder(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_size, dropword_p=0):
        super(RNNEncoder, self).__init__()

        self.hid_size = hid_size
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.dropword = Dropword(dropword_p)
        self.gru = nn.GRU(emb_size, hid_size, batch_first=True)

    def forward(self, sentence):
        embs = self.embeddings(sentence)
        embs = self.dropword(embs)
        _, last_hidden_state = self.gru(embs)
        state = last_hidden_state.squeeze(0)

        return state
