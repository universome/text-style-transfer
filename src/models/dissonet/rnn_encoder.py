import torch.nn as nn

from src.models.layers import Dropword


class RNNEncoder(nn.Module):
    def __init__(self, emb_size, hid_size, vocab, dropword_p=0):
        super(RNNEncoder, self).__init__()

        self.embeddings = nn.Embedding(len(vocab), emb_size, padding_idx=vocab.stoi['<pad>'])
        self.dropword = Dropword(dropword_p)
        self.gru = nn.GRU(emb_size, hid_size, batch_first=True)

    def forward(self, sentence):
        embs = self.embeddings(sentence)
        embs = self.dropword(embs)
        self.gru.flatten_parameters()
        _, last_hidden_state = self.gru(embs)
        state = last_hidden_state.squeeze(0)

        return state
