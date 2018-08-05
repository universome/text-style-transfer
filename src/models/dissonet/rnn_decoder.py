import torch.nn as nn

from src.models.layers import Dropword


class RNNDecoder(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_size, dropword_p=0):
        super(RNNDecoder, self).__init__()

        self.hid_size = hid_size
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.dropword = Dropword(dropword_p)
        self.gru = nn.GRU(emb_size, hid_size, batch_first=True)
        self.embs_to_logits = nn.Linear(hid_size, vocab_size)
        # self.embs_to_logits.weight = self.embeddings.weight # Sharing weights

    def forward(self, z, sentences):
        embs = self.embeddings(sentences)
        embs = self.dropword(embs)
        hid_states, _ = self.gru(embs, z.unsqueeze(0))
        logits = self.embs_to_logits(hid_states)

        return logits
