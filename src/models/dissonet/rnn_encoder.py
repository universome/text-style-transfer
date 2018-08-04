import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_size):
        super(RNNEncoder, self).__init__()

        self.hid_size = hid_size
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(emb_size, hid_size, batch_first=True)

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        _, last_hidden_state = self.gru(embeds)
        state = last_hidden_state.squeeze(0)

        return state
